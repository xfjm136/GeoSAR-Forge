"""Inference for MintPy forecasting v2."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np

from insar_utils.hardware import recommend_torch_batch_size

from .baselines import apply_residual_quantiles, harmonic_trend_predict, linear_trend_predict, persistence_predict, seasonal_naive_predict
from .dataset import apply_forecast_normalizer, build_latest_windows, load_forecast_context
from .train import _apply_cqr_calibration, _predict_model_path, _resolve_device


def _future_dates(dates: list[str], day_offsets: np.ndarray, horizon: int) -> list[str]:
    if not dates:
        return []
    if len(dates) < 2:
        return [dates[-1]] * horizon
    gap = int(np.median(np.diff(day_offsets)[-min(len(day_offsets) - 1, 5) :])) if len(day_offsets) > 1 else 12
    current = np.datetime64(dates[-1], "D")
    out: list[str] = []
    for _ in range(horizon):
        current = current + np.timedelta64(max(gap, 1), "D")
        out.append(str(current).replace("-", ""))
    return out


def _normalize_width(width: np.ndarray) -> np.ndarray:
    width = np.asarray(width, dtype=np.float32)
    finite = np.isfinite(width)
    if not np.any(finite):
        return np.zeros_like(width, dtype=np.float32)
    p5 = float(np.nanpercentile(width[finite], 5))
    p95 = float(np.nanpercentile(width[finite], 95))
    if np.isclose(p5, p95):
        return np.zeros_like(width, dtype=np.float32)
    return np.clip((width - p5) / (p95 - p5), 0.0, 1.0).astype(np.float32)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_normalizers(path: Path) -> dict[str, Any]:
    payload = joblib.load(path)
    if isinstance(payload, dict) and "generic" in payload:
        return payload
    return {"generic": payload, "hazard": None, "active_model": "generic"}


def _run_baseline(
    baseline_name: str,
    samples: dict[str, np.ndarray],
    ctx,
    horizon: int,
    baseline_intervals: dict[str, Any],
) -> np.ndarray:
    if baseline_name == "persistence":
        result = persistence_predict(samples, horizon=horizon)
    elif baseline_name == "linear_trend":
        result = linear_trend_predict(samples, ctx.day_offsets, horizon=horizon)
    elif baseline_name == "seasonal_naive":
        result = seasonal_naive_predict(samples, ctx.day_offsets, ctx.doy, horizon=horizon)
    elif baseline_name == "harmonic_trend":
        result = harmonic_trend_predict(samples, ctx.day_offsets, horizon=horizon)
    else:
        raise ValueError(f"Unsupported baseline: {baseline_name}")
    pred_p50 = np.asarray(result["pred_offset_p50"], dtype=np.float32)
    bands = baseline_intervals.get(baseline_name, {}).get("residual_bands", {})
    pred_p10, pred_p90 = apply_residual_quantiles(pred_p50, bands)
    return np.stack([pred_p10, pred_p50, pred_p90], axis=-1).astype(np.float32)


def _extract_calibration(calibration_payload: dict[str, Any], model_name: str) -> tuple[dict[str, Any], float]:
    models = calibration_payload.get("models", {})
    info = models.get(model_name, calibration_payload)
    reliability = float(info.get("calibration_reliability_factor", calibration_payload.get("calibration_reliability_factor", 1.0)))
    return info, reliability


def _confidence_from_intervals(
    interval_width_cal: np.ndarray,
    *,
    ps_score: np.ndarray,
    jump_risk: np.ndarray,
    anomaly_exposure: np.ndarray,
    reliability_factor: float,
) -> np.ndarray:
    width_norm = _normalize_width(interval_width_cal)
    score = (
        0.60 * (1.0 - width_norm)
        + 0.20 * np.clip(ps_score, 0.0, 1.0)
        + 0.10 * (1.0 - np.clip(jump_risk, 0.0, 1.0))
        + 0.10 * (1.0 - np.clip(anomaly_exposure, 0.0, 1.0))
    )
    return np.clip(score * float(reliability_factor), 0.0, 1.0).astype(np.float32)


def _active_model_path(model_dir: Path, active_model: str, train_summary: dict[str, Any]) -> Path | None:
    artifacts = train_summary.get("model_artifacts", {})
    if active_model in artifacts:
        return Path(artifacts[active_model]).resolve()
    fallback_map = {
        "generic": model_dir / "generic_forecaster.pt",
        "hazard": model_dir / "hazard_forecaster.pt",
    }
    path = fallback_map.get(active_model)
    if path is not None and path.exists():
        return path
    return None


def run_forecast_inference(
    mintpy_dir: str | Path,
    *,
    qc_report_dir: str | Path,
    model_dir: str | Path,
    output_path: str | Path,
    summary_csv_path: str | Path,
    point_inventory_path: str | Path,
    batch_size: int = 512,
    device: str = "auto",
    zone_mask_path: str | Path | None = None,
    forecast_point_scope: str = "all_high_confidence",
) -> dict[str, Any]:
    mintpy_dir = Path(mintpy_dir).resolve()
    qc_report_dir = Path(qc_report_dir).resolve()
    model_dir = Path(model_dir).resolve()
    output_path = Path(output_path).resolve()
    summary_csv_path = Path(summary_csv_path).resolve()
    point_inventory_path = Path(point_inventory_path).resolve()

    train_summary = _load_json(model_dir / "forecast_train_summary.json")
    calibration_payload = _load_json(model_dir / "forecast_calibration.json")
    explainability_payload = _load_json(model_dir / "forecast_explainability.json")
    baseline_intervals = _load_json(model_dir / "baseline_intervals.json")
    normalizers = _load_normalizers(model_dir / "forecast_normalizer.joblib")

    lookback = int(train_summary.get("lookback", 12))
    horizon = int(train_summary.get("forecast_horizon", 3))
    graph_k = int(train_summary.get("graph_k", 8))
    decomposition = str(train_summary.get("decomposition_method", "robust_harmonic_v1"))
    active_model = str(train_summary.get("active_model", normalizers.get("active_model", "generic")))
    forecast_mode_requested = str(train_summary.get("forecast_mode_requested", "generic"))
    forecast_mode_actual = str(train_summary.get("forecast_mode_actual", "generic"))
    zone_mask_path = zone_mask_path or train_summary.get("zone_mask_path") or None
    forecast_point_scope = str(forecast_point_scope or train_summary.get("zone_filter_mode", "all_high_confidence"))
    device_obj = _resolve_device(device)
    resolved_batch_size = recommend_torch_batch_size("forecast_infer", requested=batch_size, device=device_obj.type)

    ctx = load_forecast_context(
        mintpy_dir,
        qc_report_dir=qc_report_dir,
        lookback=lookback,
        forecast_horizon=horizon,
        graph_k=graph_k,
        decomposition=decomposition,
        zone_mask_path=zone_mask_path,
        forecast_point_scope=forecast_point_scope,
    )
    samples = build_latest_windows(ctx, lookback=lookback)
    if len(samples["seq"]) == 0:
        raise RuntimeError("没有足够的最新窗口可供预测。")

    point_idx = samples["point_index"]
    ps_score = ctx.points["ps_score"][point_idx].astype(np.float32)
    jump_risk = ctx.points["jump_risk"][point_idx].astype(np.float32)
    anomaly_exposure = ctx.points["anomaly_exposure"][point_idx].astype(np.float32)

    attention_mean = np.zeros((len(point_idx), 0), dtype=np.float32)
    attention_topk_idx = np.zeros((len(point_idx), 0), dtype=np.int32)
    attention_topk_weight = np.zeros((len(point_idx), 0), dtype=np.float32)

    if active_model.startswith("baseline:") or active_model in {"persistence", "linear_trend", "seasonal_naive", "harmonic_trend"}:
        baseline_name = active_model.split("baseline:", 1)[-1]
        pred_offset_raw = _run_baseline(baseline_name, samples, ctx, horizon, baseline_intervals)
        pred_offset_cal = pred_offset_raw.copy()
        reliability_factor = 1.0
        confidence_mode = "baseline_interval_v1"
    else:
        normalizer_key = "hazard" if active_model == "hazard" and normalizers.get("hazard") is not None else "generic"
        normalizer = normalizers[normalizer_key]
        samples_norm = apply_forecast_normalizer(samples, normalizer)
        model_path = _active_model_path(model_dir, active_model, train_summary)
        if model_path is None:
            raise FileNotFoundError(f"未找到 active_model={active_model} 的模型文件。")
        return_aux = active_model == "hazard"
        pred_result = _predict_model_path(
            model_path,
            samples_norm,
            batch_size=resolved_batch_size,
            device=device_obj,
            return_aux=return_aux,
        )
        if return_aux:
            pred_offset_raw, aux = pred_result
            attention_mean = np.asarray(aux.get("neighbor_attention_mean", attention_mean), dtype=np.float32)
            if attention_mean.ndim == 1:
                attention_mean = attention_mean[:, None]
            if attention_mean.size:
                k = min(5, attention_mean.shape[1])
                attention_topk_idx = np.argsort(-attention_mean, axis=1)[:, :k].astype(np.int32)
                attention_topk_weight = np.take_along_axis(attention_mean, attention_topk_idx, axis=1).astype(np.float32)
        else:
            pred_offset_raw = pred_result
        pred_offset_raw = pred_offset_raw.astype(np.float32) * float(normalizer["target_scale"])
        calibration_info, reliability_factor = _extract_calibration(calibration_payload, active_model)
        pred_offset_cal = _apply_cqr_calibration(pred_offset_raw, calibration_info)
        confidence_mode = "calibrated_v1"

    last_rel0 = samples["last_rel0"][:, None, None].astype(np.float32)
    pred_rel0_raw = last_rel0 + pred_offset_raw
    pred_rel0_cal = last_rel0 + pred_offset_cal
    interval_width_raw = np.nanmean(pred_rel0_raw[:, :, 2] - pred_rel0_raw[:, :, 0], axis=1).astype(np.float32)
    interval_width_cal = np.nanmean(pred_rel0_cal[:, :, 2] - pred_rel0_cal[:, :, 0], axis=1).astype(np.float32)
    c_pred = _confidence_from_intervals(
        interval_width_cal,
        ps_score=ps_score,
        jump_risk=jump_risk,
        anomaly_exposure=anomaly_exposure,
        reliability_factor=reliability_factor,
    )

    future_dates = _future_dates(ctx.dates, ctx.day_offsets, horizon)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        meta = f.create_group("meta")
        meta.create_dataset("mintpy_dir", data=np.bytes_(str(mintpy_dir)))
        meta.create_dataset("qc_report_dir", data=np.bytes_(str(qc_report_dir)))
        meta.create_dataset("forecast_mode_requested", data=np.bytes_(forecast_mode_requested))
        meta.create_dataset("forecast_mode_actual", data=np.bytes_(forecast_mode_actual))
        meta.create_dataset("uncertainty_mode", data=np.bytes_(str(train_summary.get("uncertainty_mode", "cqr_conformal_v1"))))
        meta.create_dataset("active_model", data=np.bytes_(active_model))
        meta.create_dataset("selection_mode", data=np.bytes_(ctx.selection_mode))
        meta.create_dataset("zone_detection_status", data=np.bytes_(ctx.zone_detection_status))
        meta.create_dataset("zone_filter_mode", data=np.bytes_(ctx.zone_filter_mode))
        meta.create_dataset("zone_mask_path", data=np.bytes_(ctx.zone_mask_path))
        meta.create_dataset("accepted_pass", data=np.bytes_(ctx.accepted_pass))
        meta.create_dataset("confidence_mode", data=np.bytes_(confidence_mode))
        meta.create_dataset("lookback", data=np.int32(lookback))
        meta.create_dataset("forecast_horizon", data=np.int32(horizon))
        meta.create_dataset("history_dates", data=np.asarray([np.bytes_(d) for d in ctx.dates]))
        meta.create_dataset("future_dates", data=np.asarray([np.bytes_(d) for d in future_dates]))
        meta.create_dataset("cumulative_reference_date", data=np.bytes_(ctx.dates[0]))

        points_group = f.create_group("points")
        points_group.create_dataset("point_index", data=point_idx, compression="gzip")
        points_group.create_dataset("row", data=ctx.points["row"][point_idx], compression="gzip")
        points_group.create_dataset("col", data=ctx.points["col"][point_idx], compression="gzip")
        points_group.create_dataset("longitude", data=ctx.points["longitude"][point_idx], compression="gzip")
        points_group.create_dataset("latitude", data=ctx.points["latitude"][point_idx], compression="gzip")
        points_group.create_dataset("ps_score", data=ps_score, compression="gzip")
        points_group.create_dataset("jump_risk", data=jump_risk, compression="gzip")
        points_group.create_dataset("anomaly_exposure", data=anomaly_exposure, compression="gzip")
        points_group.create_dataset("strict_or_relaxed", data=np.asarray([np.bytes_(x) for x in samples["strict_or_relaxed"]]))
        points_group.create_dataset("zone_id", data=np.asarray([np.bytes_(str(ctx.point_zone_ids[int(i)])) for i in point_idx]))
        points_group.create_dataset("in_zone_mask", data=ctx.point_in_zone_mask[point_idx].astype(np.uint8), compression="gzip")
        points_group.create_dataset("n_windows", data=ctx.window_count_per_point[point_idx], compression="gzip")

        pred = f.create_group("predictions")
        pred.create_dataset("pred_offset_p10", data=pred_offset_cal[:, :, 0], compression="gzip")
        pred.create_dataset("pred_offset_p50", data=pred_offset_cal[:, :, 1], compression="gzip")
        pred.create_dataset("pred_offset_p90", data=pred_offset_cal[:, :, 2], compression="gzip")
        pred.create_dataset("pred_rel0_p10", data=pred_rel0_cal[:, :, 0], compression="gzip")
        pred.create_dataset("pred_rel0_p50", data=pred_rel0_cal[:, :, 1], compression="gzip")
        pred.create_dataset("pred_rel0_p90", data=pred_rel0_cal[:, :, 2], compression="gzip")
        pred.create_dataset("pred_offset_p10_raw", data=pred_offset_raw[:, :, 0], compression="gzip")
        pred.create_dataset("pred_offset_p50_raw", data=pred_offset_raw[:, :, 1], compression="gzip")
        pred.create_dataset("pred_offset_p90_raw", data=pred_offset_raw[:, :, 2], compression="gzip")
        pred.create_dataset("pred_rel0_p10_raw", data=pred_rel0_raw[:, :, 0], compression="gzip")
        pred.create_dataset("pred_rel0_p50_raw", data=pred_rel0_raw[:, :, 1], compression="gzip")
        pred.create_dataset("pred_rel0_p90_raw", data=pred_rel0_raw[:, :, 2], compression="gzip")
        pred.create_dataset("interval_width_raw", data=interval_width_raw, compression="gzip")
        pred.create_dataset("interval_width_calibrated", data=interval_width_cal, compression="gzip")
        pred.create_dataset("c_pred", data=c_pred, compression="gzip")

        explain = f.create_group("explainability")
        explain.create_dataset("neighbor_attention_mean", data=attention_mean, compression="gzip")
        explain.create_dataset("neighbor_attention_topk_index", data=attention_topk_idx, compression="gzip")
        explain.create_dataset("neighbor_attention_topk_weight", data=attention_topk_weight, compression="gzip")

    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = [
            "row",
            "col",
            "longitude",
            "latitude",
            "forecast_mode_requested",
            "forecast_mode_actual",
            "active_model",
            "strict_or_relaxed",
            "zone_id",
            "c_pred",
        ]
        for h in range(horizon):
            header.extend([f"pred_rel0_p50_h{h + 1}", f"pred_offset_p50_h{h + 1}"])
        writer.writerow(header)
        for i in range(len(point_idx)):
            row = [
                int(ctx.points["row"][point_idx[i]]),
                int(ctx.points["col"][point_idx[i]]),
                float(ctx.points["longitude"][point_idx[i]]),
                float(ctx.points["latitude"][point_idx[i]]),
                forecast_mode_requested,
                forecast_mode_actual,
                active_model,
                str(samples["strict_or_relaxed"][i]),
                str(ctx.point_zone_ids[int(point_idx[i])]),
                float(c_pred[i]),
            ]
            for h in range(horizon):
                row.extend([float(pred_rel0_cal[i, h, 1]), float(pred_offset_cal[i, h, 1])])
            writer.writerow(row)

    point_inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with point_inventory_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row",
                "col",
                "lon",
                "lat",
                "ps_score",
                "jump_risk",
                "anomaly_exposure",
                "c_pred",
                "strict_or_relaxed",
                "zone_id",
                "n_windows",
            ]
        )
        for i in range(len(point_idx)):
            writer.writerow(
                [
                    int(ctx.points["row"][point_idx[i]]),
                    int(ctx.points["col"][point_idx[i]]),
                    float(ctx.points["longitude"][point_idx[i]]),
                    float(ctx.points["latitude"][point_idx[i]]),
                    float(ps_score[i]),
                    float(jump_risk[i]),
                    float(anomaly_exposure[i]),
                    float(c_pred[i]),
                    str(samples["strict_or_relaxed"][i]),
                    str(ctx.point_zone_ids[int(point_idx[i])]),
                    int(ctx.window_count_per_point[point_idx[i]]),
                ]
            )

    return {
        "output_path": str(output_path),
        "summary_csv_path": str(summary_csv_path),
        "point_inventory_csv": str(point_inventory_path),
        "selection_mode": ctx.selection_mode,
        "zone_detection_status": ctx.zone_detection_status,
        "zone_filter_mode": ctx.zone_filter_mode,
        "zone_mask_path": ctx.zone_mask_path,
        "n_detected_zones": int(ctx.n_detected_zones),
        "n_zone_points_used": int(np.sum(ctx.point_in_zone_mask[point_idx])) if len(point_idx) else 0,
        "zone_ids_used": list(ctx.zone_ids_used),
        "zone_filter_fallback_triggered": bool(ctx.zone_filter_fallback_triggered),
        "zone_filter_fallback_reason": ctx.zone_filter_fallback_reason,
        "forecast_mode_requested": forecast_mode_requested,
        "forecast_mode_actual": forecast_mode_actual,
        "n_points": int(len(point_idx)),
        "active_model": active_model,
        "confidence_mode": confidence_mode,
        "forecast_horizon": int(horizon),
        "requested_batch_size": int(batch_size),
        "resolved_batch_size": int(resolved_batch_size),
        "explainability_path": str(model_dir / "forecast_explainability.json"),
        "notes": {
            "hazard_attention_exported": bool(attention_mean.size),
            "explainability_summary_available": bool(explainability_payload),
        },
    }


__all__ = ["run_forecast_inference"]
