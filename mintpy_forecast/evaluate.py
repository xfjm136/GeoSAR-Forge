"""Evaluation for MintPy forecasting v2."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from insar_utils.hardware import recommend_torch_batch_size

from .baselines import apply_residual_quantiles, harmonic_trend_predict, linear_trend_predict, metric_summary, persistence_predict, seasonal_naive_predict
from .dataset import apply_forecast_normalizer, build_window_samples, load_forecast_context
from .train import TARGET_COVERAGE, _apply_cqr_calibration, _predict_model_path, _resolve_device


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_normalizers(path: Path) -> dict[str, Any]:
    payload = joblib.load(path)
    if isinstance(payload, dict) and "generic" in payload:
        return payload
    return {"generic": payload, "hazard": None, "active_model": "generic"}


def _baseline_predictions(
    name: str,
    samples: dict[str, np.ndarray],
    ctx,
    baseline_intervals: dict[str, Any],
) -> np.ndarray:
    horizon = samples["target_offset"].shape[1]
    if name == "persistence":
        result = persistence_predict(samples, horizon=horizon)
    elif name == "linear_trend":
        result = linear_trend_predict(samples, ctx.day_offsets, horizon=horizon)
    elif name == "seasonal_naive":
        result = seasonal_naive_predict(samples, ctx.day_offsets, ctx.doy, horizon=horizon)
    elif name == "harmonic_trend":
        result = harmonic_trend_predict(samples, ctx.day_offsets, horizon=horizon)
    else:
        raise ValueError(f"Unsupported baseline: {name}")
    pred_p50 = np.asarray(result["pred_offset_p50"], dtype=np.float32)
    pred_p10, pred_p90 = apply_residual_quantiles(pred_p50, baseline_intervals.get(name, {}).get("residual_bands", {}))
    return np.stack([pred_p10, pred_p50, pred_p90], axis=-1).astype(np.float32)


def _slice_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    target_coverage: float = TARGET_COVERAGE,
) -> dict[str, Any]:
    if pred.size == 0 or target.size == 0 or not np.any(mask):
        return {
            "n_samples": 0,
            "metrics": {
                "mae_mm": float("nan"),
                "rmse_mm": float("nan"),
                "pinball_loss": float("nan"),
                "coverage": float("nan"),
                "coverage_error": float("nan"),
                "mean_interval_width_mm": float("nan"),
                "wis": float("nan"),
                "per_horizon": [],
            },
        }
    pred_slice = pred[mask]
    target_slice = target[mask]
    return {
        "n_samples": int(np.sum(mask)),
        "metrics": metric_summary(
            pred_slice[:, :, 1],
            target_slice,
            pred_p10=pred_slice[:, :, 0],
            pred_p90=pred_slice[:, :, 2],
            target_coverage=target_coverage,
        ),
    }


def _method_rows(method: str, subset: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in metrics.get("per_horizon", []):
        rows.append(
            {
                "method": method,
                "subset": subset,
                "horizon": int(item["horizon"]),
                "mae_mm": float(item["mae_mm"]),
                "rmse_mm": float(item["rmse_mm"]),
                "pinball_loss": float(item.get("pinball_loss", float("nan"))),
                "coverage": float(item.get("coverage", float("nan"))),
                "coverage_error": float(item.get("coverage_error", float("nan"))),
                "mean_interval_width_mm": float(item.get("mean_interval_width_mm", float("nan"))),
                "wis": float(item.get("wis", float("nan"))),
            }
        )
    return rows


def evaluate_forecast_predictions(
    mintpy_dir: str | Path,
    *,
    qc_report_dir: str | Path,
    model_dir: str | Path,
    output_path: str | Path,
    baseline_csv_path: str | Path,
    batch_size: int = 512,
    device: str = "auto",
    zone_mask_path: str | Path | None = None,
    forecast_point_scope: str = "all_high_confidence",
) -> dict[str, Any]:
    mintpy_dir = Path(mintpy_dir).resolve()
    qc_report_dir = Path(qc_report_dir).resolve()
    model_dir = Path(model_dir).resolve()
    output_path = Path(output_path).resolve()
    baseline_csv_path = Path(baseline_csv_path).resolve()

    train_summary = _load_json(model_dir / "forecast_train_summary.json")
    calibration_payload = _load_json(model_dir / "forecast_calibration.json")
    baseline_intervals = _load_json(model_dir / "baseline_intervals.json")
    normalizers = _load_normalizers(model_dir / "forecast_normalizer.joblib")
    active_model = str(train_summary.get("active_model", "generic"))
    lookback = int(train_summary.get("lookback", 12))
    horizon = int(train_summary.get("forecast_horizon", 3))
    graph_k = int(train_summary.get("graph_k", 8))
    decomposition = str(train_summary.get("decomposition_method", "robust_harmonic_v1"))
    zone_mask_path = zone_mask_path or train_summary.get("zone_mask_path") or None
    forecast_point_scope = str(forecast_point_scope or train_summary.get("zone_filter_mode", "all_high_confidence"))

    device_obj = _resolve_device(device)
    resolved_batch_size = recommend_torch_batch_size("forecast_eval", requested=batch_size, device=device_obj.type)

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
    test_samples = build_window_samples(
        ctx,
        ctx.selected_indices,
        end_indices=ctx.split_end_indices["test"],
        lookback=lookback,
        forecast_horizon=horizon,
        max_samples=50000,
        seed=42,
    )
    if len(test_samples["seq"]) == 0:
        raise RuntimeError("测试窗口为空，无法评估预测结果。")
    target = test_samples["target_offset"].astype(np.float32)
    event_mask = np.asarray(test_samples["target_abnormal_flag"], dtype=bool)
    stable_mask = ~event_mask

    method_predictions: dict[str, np.ndarray] = {}
    method_metrics: dict[str, Any] = {}

    for baseline_name in ["persistence", "linear_trend", "seasonal_naive", "harmonic_trend"]:
        pred = _baseline_predictions(baseline_name, test_samples, ctx, baseline_intervals)
        method_predictions[baseline_name] = pred

    for model_name in ["generic", "hazard"]:
        artifact_path = train_summary.get("model_artifacts", {}).get(model_name)
        normalizer = normalizers.get(model_name) if model_name == "hazard" else normalizers.get("generic")
        if not artifact_path or normalizer is None:
            continue
        model_path = Path(artifact_path)
        if not model_path.exists():
            continue
        samples_norm = apply_forecast_normalizer(test_samples, normalizer)
        pred_raw = _predict_model_path(model_path, samples_norm, batch_size=resolved_batch_size, device=device_obj)
        pred_raw = pred_raw.astype(np.float32) * float(normalizer["target_scale"])
        calibration = calibration_payload.get("models", {}).get(model_name, calibration_payload)
        pred_cal = _apply_cqr_calibration(pred_raw, calibration)
        method_predictions[model_name] = pred_cal

    if active_model not in method_predictions and active_model.startswith("baseline:"):
        base = active_model.split("baseline:", 1)[-1]
        if base in method_predictions:
            method_predictions[active_model] = method_predictions[base]
    elif active_model in method_predictions:
        method_predictions["active_model"] = method_predictions[active_model]

    rows: list[dict[str, Any]] = []
    for method_name, pred in method_predictions.items():
        overall = _slice_metrics(pred, target, np.ones(len(target), dtype=bool))
        event = _slice_metrics(pred, target, event_mask)
        stable = _slice_metrics(pred, target, stable_mask)
        method_metrics[method_name] = {
            "overall": overall,
            "event_windows": event,
            "stable_windows": stable,
        }
        rows.extend(_method_rows(method_name, "overall", overall["metrics"]))
        rows.extend(_method_rows(method_name, "event_windows", event["metrics"]))
        rows.extend(_method_rows(method_name, "stable_windows", stable["metrics"]))

    baseline_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "subset",
                "horizon",
                "mae_mm",
                "rmse_mm",
                "pinball_loss",
                "coverage",
                "coverage_error",
                "mean_interval_width_mm",
                "wis",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    active_key = "active_model" if "active_model" in method_metrics else active_model
    summary = {
        "mintpy_dir": str(mintpy_dir),
        "qc_report_dir": str(qc_report_dir),
        "model_dir": str(model_dir),
        "selection_mode": ctx.selection_mode,
        "zone_detection_status": ctx.zone_detection_status,
        "zone_filter_mode": ctx.zone_filter_mode,
        "zone_mask_path": ctx.zone_mask_path,
        "n_detected_zones": int(ctx.n_detected_zones),
        "n_zone_points_used": int(np.sum(ctx.point_in_zone_mask[test_samples["point_index"]])) if len(test_samples["point_index"]) else 0,
        "zone_ids_used": list(ctx.zone_ids_used),
        "zone_filter_fallback_triggered": bool(ctx.zone_filter_fallback_triggered),
        "zone_filter_fallback_reason": ctx.zone_filter_fallback_reason,
        "forecast_mode_requested": str(train_summary.get("forecast_mode_requested", "generic")),
        "forecast_mode_actual": str(train_summary.get("forecast_mode_actual", "generic")),
        "active_model": active_model,
        "n_test_samples": int(len(test_samples["seq"])),
        "n_event_windows": int(np.sum(event_mask)),
        "n_stable_windows": int(np.sum(stable_mask)),
        "active_model_metrics": method_metrics.get(active_key, {}),
        "method_metrics": method_metrics,
        "forecast_baseline_comparison_csv": str(baseline_csv_path),
        "requested_batch_size": int(batch_size),
        "resolved_batch_size": int(resolved_batch_size),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


__all__ = ["evaluate_forecast_predictions"]
