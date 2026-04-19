"""Training pipeline for MintPy forecasting v2."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from insar_utils.hardware import recommend_torch_batch_size

from .baselines import (
    apply_residual_quantiles,
    fit_residual_quantiles,
    harmonic_trend_predict,
    linear_trend_predict,
    metric_summary,
    persistence_predict,
    seasonal_naive_predict,
)
from .dataset import (
    CHANNEL_GROUPS,
    STATIC_FEATURE_NAMES,
    SEQUENCE_FEATURE_NAMES,
    apply_forecast_normalizer,
    build_window_samples,
    fit_forecast_normalizer,
    load_forecast_context,
)
from .models import (
    DecompTCNGRUConfig,
    DecompTCNGRUQuantileForecaster,
    GraphTCNAttnConfig,
    GraphTCNAttnQuantileForecaster,
    TemporalFusionConfig,
    TemporalFusionForecaster,
)
from .models.decomp_tcn_gru_forecaster import config_to_dict as generic_config_to_dict
from .models.graph_tcn_attn_forecaster import config_to_dict as hazard_config_to_dict
from .models.temporal_fusion_forecaster import config_to_dict as tft_config_to_dict

TARGET_COVERAGE = 0.80
INTERVAL_QUANTILES = (0.1, 0.5, 0.9)
MONOTONIC_PENALTY = 0.1


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _empty_metric_summary(target_coverage: float = TARGET_COVERAGE) -> dict[str, Any]:
    item = {
        "mae_mm": float("nan"),
        "rmse_mm": float("nan"),
        "pinball_loss": float("nan"),
        "coverage": float("nan"),
        "coverage_error": float("nan"),
        "mean_interval_width_mm": float("nan"),
        "wis": float("nan"),
    }
    item["per_horizon"] = []
    item["target_coverage"] = float(target_coverage)
    return item


def _quantile_loss(pred: torch.Tensor, target: torch.Tensor, monotonic_penalty: float = MONOTONIC_PENALTY) -> torch.Tensor:
    quantiles = torch.tensor(INTERVAL_QUANTILES, dtype=pred.dtype, device=pred.device).view(1, 1, 3)
    err = target.unsqueeze(-1) - pred
    base = torch.maximum(quantiles * err, (quantiles - 1.0) * err).mean()
    monotonic = torch.relu(pred[:, :, 0] - pred[:, :, 1]).mean() + torch.relu(pred[:, :, 1] - pred[:, :, 2]).mean()
    return base + monotonic_penalty * monotonic


def _config_to_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError(f"Unsupported config: {type(config)!r}")


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _instantiate_model_from_payload(payload: dict[str, Any]) -> torch.nn.Module:
    model_name = str(payload.get("model_name", ""))
    config = dict(payload.get("config", {}))
    if "TemporalFusion" in model_name:
        return TemporalFusionForecaster(TemporalFusionConfig(**config))
    if "GraphTCNAttn" in model_name:
        return GraphTCNAttnQuantileForecaster(GraphTCNAttnConfig(**config))
    if "DecompTCNGRU" in model_name:
        return DecompTCNGRUQuantileForecaster(DecompTCNGRUConfig(**config))
    raise ValueError(f"Unsupported forecast model payload: {model_name!r}")


def _batch_slice(samples: dict[str, np.ndarray], index: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in samples.items():
        if isinstance(value, np.ndarray) and len(value) == len(index) if value.ndim > 0 else False:
            out[key] = value[index]
        else:
            out[key] = value
    return out


def _sample_subset(samples: dict[str, np.ndarray], max_samples: int | None, seed: int) -> dict[str, np.ndarray]:
    n = int(len(samples["seq"]))
    if max_samples is None or n <= max_samples:
        return samples
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(np.arange(n), size=max_samples, replace=False))
    out: dict[str, np.ndarray] = {}
    for key, value in samples.items():
        if isinstance(value, np.ndarray) and value.shape[0] == n:
            out[key] = value[keep]
        else:
            out[key] = value
    return out


def _make_loader(samples_norm: dict[str, np.ndarray], batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(samples_norm["seq_norm"]).float(),
        torch.from_numpy(samples_norm["static_norm"]).float(),
        torch.from_numpy(samples_norm["target_norm"]).float(),
        torch.from_numpy(samples_norm["neighbor_seq_norm"]).float(),
        torch.from_numpy(samples_norm["neighbor_static_norm"]).float(),
        torch.from_numpy(samples_norm["edge_features_norm"]).float(),
        torch.from_numpy(samples_norm["neighbor_mask"]).bool(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _predict_torch_batches(
    model: torch.nn.Module,
    samples_norm: dict[str, np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
    return_aux: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    preds: list[np.ndarray] = []
    aux_store: dict[str, list[np.ndarray]] = {}
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for start in range(0, len(samples_norm["seq_norm"]), batch_size):
            stop = min(start + batch_size, len(samples_norm["seq_norm"]))
            seq = torch.from_numpy(samples_norm["seq_norm"][start:stop]).to(device)
            static = torch.from_numpy(samples_norm["static_norm"][start:stop]).to(device)
            neighbor_seq = torch.from_numpy(samples_norm["neighbor_seq_norm"][start:stop]).to(device)
            neighbor_static = torch.from_numpy(samples_norm["neighbor_static_norm"][start:stop]).to(device)
            edge_features = torch.from_numpy(samples_norm["edge_features_norm"][start:stop]).to(device)
            neighbor_mask = torch.from_numpy(samples_norm["neighbor_mask"][start:stop]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(
                    seq,
                    static,
                    neighbor_seq=neighbor_seq,
                    neighbor_static=neighbor_static,
                    edge_features=edge_features,
                    neighbor_mask=neighbor_mask,
                    return_attention=return_aux,
                )
            if return_aux:
                pred, aux = out
            else:
                pred = out
                aux = {}
            preds.append(pred.detach().float().cpu().numpy())
            for key, value in aux.items():
                aux_store.setdefault(key, []).append(value.detach().float().cpu().numpy())
    pred_arr = np.concatenate(preds, axis=0).astype(np.float32)
    if not return_aux:
        return pred_arr
    aux_concat = {key: np.concatenate(value, axis=0).astype(np.float32) for key, value in aux_store.items()}
    return pred_arr, aux_concat


def _predict_model_path(
    model_path: Path,
    samples_norm: dict[str, np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
    return_aux: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    payload = _torch_load(model_path)
    model = _instantiate_model_from_payload(payload)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return _predict_torch_batches(model, samples_norm, batch_size=batch_size, device=device, return_aux=return_aux)


def _apply_monotonic_fix(pred: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(pred, dtype=np.float32), axis=-1).astype(np.float32)


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def _fit_cqr_calibration(
    pred_raw: np.ndarray,
    target: np.ndarray,
    *,
    target_coverage: float = TARGET_COVERAGE,
) -> dict[str, Any]:
    pred_raw = _apply_monotonic_fix(pred_raw)
    alpha = 1.0 - float(target_coverage)
    horizon = target.shape[1]
    corrections: list[float] = []
    per_horizon: list[dict[str, Any]] = []
    raw_coverages: list[float] = []
    cal_coverages: list[float] = []
    for h in range(horizon):
        q10 = pred_raw[:, h, 0]
        q50 = pred_raw[:, h, 1]
        q90 = pred_raw[:, h, 2]
        y = target[:, h]
        valid = np.isfinite(y) & np.isfinite(q10) & np.isfinite(q50) & np.isfinite(q90)
        if not np.any(valid):
            corrections.append(0.0)
            per_horizon.append(
                {
                    "horizon": int(h + 1),
                    "score_quantile": float("nan"),
                    "raw_coverage": float("nan"),
                    "calibrated_coverage": float("nan"),
                    "mean_raw_width_mm": float("nan"),
                    "mean_calibrated_width_mm": float("nan"),
                }
            )
            raw_coverages.append(float("nan"))
            cal_coverages.append(float("nan"))
            continue
        score = np.maximum.reduce([q10[valid] - y[valid], y[valid] - q90[valid], np.zeros(np.sum(valid), dtype=np.float32)])
        score_quantile = min(1.0, math.ceil((score.size + 1) * (1.0 - alpha)) / max(score.size, 1))
        correction = _safe_quantile(score, score_quantile)
        corrections.append(correction)
        raw_cover = float(np.mean((y[valid] >= q10[valid]) & (y[valid] <= q90[valid])))
        q10_cal = q10[valid] - correction
        q90_cal = q90[valid] + correction
        cal_cover = float(np.mean((y[valid] >= q10_cal) & (y[valid] <= q90_cal)))
        raw_width = float(np.mean(q90[valid] - q10[valid]))
        cal_width = float(np.mean(q90_cal - q10_cal))
        per_horizon.append(
            {
                "horizon": int(h + 1),
                "score_quantile": float(score_quantile),
                "raw_coverage": raw_cover,
                "calibrated_coverage": cal_cover,
                "mean_raw_width_mm": raw_width,
                "mean_calibrated_width_mm": cal_width,
            }
        )
        raw_coverages.append(raw_cover)
        cal_coverages.append(cal_cover)
    coverage_error = float(np.nanmean(np.abs(np.asarray(cal_coverages) - target_coverage))) if cal_coverages else float("nan")
    reliability = float(np.clip(1.0 - coverage_error / max(target_coverage, 1e-6), 0.0, 1.0)) if np.isfinite(coverage_error) else 0.0
    return {
        "uncertainty_mode": "cqr_conformal_v1",
        "interval_quantiles": list(INTERVAL_QUANTILES),
        "target_coverage": float(target_coverage),
        "corrections": corrections,
        "per_horizon": per_horizon,
        "raw_coverage_mean": float(np.nanmean(raw_coverages)) if raw_coverages else float("nan"),
        "calibrated_coverage_mean": float(np.nanmean(cal_coverages)) if cal_coverages else float("nan"),
        "calibration_reliability_factor": reliability,
    }


def _apply_cqr_calibration(pred_raw: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    pred_raw = _apply_monotonic_fix(pred_raw)
    out = pred_raw.copy()
    corrections = np.asarray(calibration.get("corrections", []), dtype=np.float32)
    if corrections.size == 0:
        return out
    out[:, :, 0] -= corrections[None, :]
    out[:, :, 2] += corrections[None, :]
    return _apply_monotonic_fix(out)


def _interval_to_metrics(pred_raw: np.ndarray, pred_cal: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    raw_metrics = metric_summary(
        pred_raw[:, :, 1],
        target,
        pred_p10=pred_raw[:, :, 0],
        pred_p90=pred_raw[:, :, 2],
        target_coverage=TARGET_COVERAGE,
    )
    calibrated_metrics = metric_summary(
        pred_cal[:, :, 1],
        target,
        pred_p10=pred_cal[:, :, 0],
        pred_p90=pred_cal[:, :, 2],
        target_coverage=TARGET_COVERAGE,
    )
    return {"raw": raw_metrics, "calibrated": calibrated_metrics}


def _run_baselines(
    samples: dict[str, np.ndarray],
    *,
    ctx,
    horizon: int,
) -> dict[str, dict[str, Any]]:
    baselines = {
        "persistence": persistence_predict(samples, horizon=horizon),
        "linear_trend": linear_trend_predict(samples, ctx.day_offsets, horizon=horizon),
        "seasonal_naive": seasonal_naive_predict(samples, ctx.day_offsets, ctx.doy, horizon=horizon),
        "harmonic_trend": harmonic_trend_predict(samples, ctx.day_offsets, horizon=horizon),
    }
    return baselines


def _baseline_with_intervals(
    baseline_result: dict[str, Any],
    *,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    pred_p50 = np.asarray(baseline_result["pred_offset_p50"], dtype=np.float32)
    bands = fit_residual_quantiles(pred_p50, target)
    pred_p10, pred_p90 = apply_residual_quantiles(pred_p50, bands)
    metrics = metric_summary(pred_p50, target, pred_p10=pred_p10, pred_p90=pred_p90, target_coverage=TARGET_COVERAGE)
    metrics["status"] = str(baseline_result.get("status", "ok"))
    metrics["reason"] = str(baseline_result.get("reason", ""))
    return pred_p10.astype(np.float32), pred_p90.astype(np.float32), bands, metrics


def _channel_group_importance(
    model_path: Path,
    samples_norm: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    target_scale: float,
) -> dict[str, float]:
    subset = _sample_subset(samples_norm, min(4096, len(samples_norm["seq_norm"])), seed=seed)
    subset_target = np.asarray(subset["target_offset"], dtype=np.float32)
    base_pred = _predict_model_path(model_path, subset, batch_size=batch_size, device=device) * float(target_scale)
    base_rmse = float(metric_summary(base_pred[:, :, 1], subset_target)["rmse_mm"])
    importance: dict[str, float] = {}
    for group_name, feature_names in CHANNEL_GROUPS.items():
        group_idx = [int(name) if isinstance(name, (int, np.integer)) else SEQUENCE_FEATURE_NAMES.index(name) for name in feature_names]
        mutated = dict(subset)
        seq = subset["seq_norm"].copy()
        seq[:, :, group_idx] = 0.0
        mutated["seq_norm"] = seq
        if "neighbor_seq_norm" in subset:
            neighbor_seq = subset["neighbor_seq_norm"].copy()
            neighbor_seq[:, :, :, group_idx] = 0.0
            mutated["neighbor_seq_norm"] = neighbor_seq
        pred = _predict_model_path(model_path, mutated, batch_size=batch_size, device=device) * float(target_scale)
        rmse = float(metric_summary(pred[:, :, 1], subset_target)["rmse_mm"])
        importance[group_name] = max(rmse - base_rmse, 0.0)
    return importance


def _static_permutation_importance(
    model_path: Path,
    samples_norm: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    target_scale: float,
) -> dict[str, float]:
    subset = _sample_subset(samples_norm, min(4096, len(samples_norm["seq_norm"])), seed=seed)
    subset_target = np.asarray(subset["target_offset"], dtype=np.float32)
    base_pred = _predict_model_path(model_path, subset, batch_size=batch_size, device=device) * float(target_scale)
    base_rmse = float(metric_summary(base_pred[:, :, 1], subset_target)["rmse_mm"])
    rng = np.random.default_rng(seed)
    importance: dict[str, float] = {}
    for idx, feature_name in enumerate(STATIC_FEATURE_NAMES):
        mutated = dict(subset)
        static = subset["static_norm"].copy()
        perm = rng.permutation(static.shape[0])
        static[:, idx] = static[perm, idx]
        mutated["static_norm"] = static
        if "neighbor_static_norm" in subset and subset["neighbor_static_norm"].size:
            neighbor_static = subset["neighbor_static_norm"].copy()
            flat = neighbor_static.reshape(-1, neighbor_static.shape[-1])
            perm_nb = rng.permutation(flat.shape[0])
            flat[:, idx] = flat[perm_nb, idx]
            mutated["neighbor_static_norm"] = flat.reshape(neighbor_static.shape)
        pred = _predict_model_path(model_path, mutated, batch_size=batch_size, device=device) * float(target_scale)
        rmse = float(metric_summary(pred[:, :, 1], subset_target)["rmse_mm"])
        importance[feature_name] = max(rmse - base_rmse, 0.0)
    return importance


def _hazard_attention_summary(
    model_path: Path,
    samples_norm: dict[str, np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    subset = _sample_subset(samples_norm, min(4096, len(samples_norm["seq_norm"])), seed=seed)
    _, aux = _predict_model_path(model_path, subset, batch_size=batch_size, device=device, return_aux=True)
    attn = np.asarray(aux.get("neighbor_attention_mean", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    if attn.size == 0:
        return {
            "mean_attention_strength": float("nan"),
            "topk_neighbor_rank_mean": [],
            "topk_neighbor_weight_mean": [],
            "n_samples_used": 0,
        }
    k = min(5, attn.shape[1])
    topk_idx = np.argsort(-attn, axis=1)[:, :k]
    topk_weight = np.take_along_axis(attn, topk_idx, axis=1)
    return {
        "mean_attention_strength": float(np.nanmean(attn)),
        "topk_neighbor_rank_mean": [float(np.nanmean(topk_idx[:, i])) for i in range(k)],
        "topk_neighbor_weight_mean": [float(np.nanmean(topk_weight[:, i])) for i in range(k)],
        "n_samples_used": int(attn.shape[0]),
    }


def _train_one_model(
    *,
    model_name: str,
    model: torch.nn.Module,
    config: Any,
    train_norm: dict[str, np.ndarray],
    val_norm: dict[str, np.ndarray],
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
    device: torch.device,
    config_to_dict_fn,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device=device.type, enabled=device.type == "cuda")
    loader = _make_loader(train_norm, batch_size)
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    wait = patience
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for seq, static, target, neighbor_seq, neighbor_static, edge_features, neighbor_mask in loader:
            seq = seq.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            neighbor_seq = neighbor_seq.to(device, non_blocking=True)
            neighbor_static = neighbor_static.to(device, non_blocking=True)
            edge_features = edge_features.to(device, non_blocking=True)
            neighbor_mask = neighbor_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pred = model(
                    seq,
                    static,
                    neighbor_seq=neighbor_seq,
                    neighbor_static=neighbor_static,
                    edge_features=edge_features,
                    neighbor_mask=neighbor_mask,
                )
                loss = _quantile_loss(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val_pred = _predict_torch_batches(model, val_norm, batch_size=batch_size, device=device)
        val_loss = float(np.mean(np.abs(val_pred[:, :, 1] - val_norm["target_norm"])))
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "val_loss": val_loss,
            }
        )
        if best_state is None or val_loss < best_loss:
            best_loss = val_loss
            wait = patience
            best_state = {
                "state_dict": model.state_dict(),
                "config": config_to_dict_fn(config),
                "model_name": model_name,
                "history": history,
            }
        else:
            wait -= 1
            if wait <= 0:
                break

    if best_state is None:
        raise RuntimeError(f"{model_name} failed to train.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)
    model.load_state_dict(best_state["state_dict"])
    return {
        "artifact_path": str(output_path),
        "history": history,
        "best_val_loss": float(best_loss),
    }


def _prepare_samples(
    ctx,
    *,
    point_indices: np.ndarray,
    lookback: int,
    forecast_horizon: int,
    max_windows_train: int,
    max_windows_val: int,
    max_windows_test: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        "train": build_window_samples(
            ctx,
            point_indices,
            end_indices=ctx.split_end_indices["train"],
            lookback=lookback,
            forecast_horizon=forecast_horizon,
            max_samples=max_windows_train,
            seed=seed,
        ),
        "val": build_window_samples(
            ctx,
            point_indices,
            end_indices=ctx.split_end_indices["val"],
            lookback=lookback,
            forecast_horizon=forecast_horizon,
            max_samples=max_windows_val,
            seed=seed + 1,
        ),
        "test": build_window_samples(
            ctx,
            point_indices,
            end_indices=ctx.split_end_indices["test"],
            lookback=lookback,
            forecast_horizon=forecast_horizon,
            max_samples=max_windows_test,
            seed=seed + 2,
        ),
    }


def train_forecast_models(
    mintpy_dir: str | Path,
    *,
    qc_report_dir: str | Path,
    output_dir: str | Path,
    lookback: int = 12,
    forecast_horizon: int = 3,
    min_points_for_training: int = 500,
    min_train_windows: int = 3000,
    min_val_windows: int = 500,
    max_windows_train: int = 180000,
    max_windows_val: int = 50000,
    max_windows_test: int = 50000,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
    device: str = "auto",
    forecast_mode: str = "generic",
    graph_k: int = 8,
    decomposition: str = "robust_harmonic_v1",
    calibrate_intervals: bool = True,
    zone_mask_path: str | Path | None = None,
    forecast_point_scope: str = "all_high_confidence",
) -> dict[str, Any]:
    forecast_mode = str(forecast_mode or "generic").strip().lower()
    if forecast_mode not in {"generic", "hazard"}:
        raise ValueError(f"不支持的 forecast_mode: {forecast_mode}")

    mintpy_dir = Path(mintpy_dir).resolve()
    qc_report_dir = Path(qc_report_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device_obj = _resolve_device(device)
    resolved_batch_size = recommend_torch_batch_size("forecast_train", requested=batch_size, device=device_obj.type)
    ctx = load_forecast_context(
        mintpy_dir,
        qc_report_dir=qc_report_dir,
        lookback=lookback,
        forecast_horizon=forecast_horizon,
        min_points_for_training=min_points_for_training,
        min_train_windows=min_train_windows,
        min_val_windows=min_val_windows,
        graph_k=graph_k,
        decomposition=decomposition,
        zone_mask_path=zone_mask_path,
        forecast_point_scope=forecast_point_scope,
    )

    generic_samples = _prepare_samples(
        ctx,
        point_indices=ctx.selected_indices,
        lookback=lookback,
        forecast_horizon=forecast_horizon,
        max_windows_train=max_windows_train,
        max_windows_val=max_windows_val,
        max_windows_test=max_windows_test,
        seed=seed,
    )
    hazard_samples = _prepare_samples(
        ctx,
        point_indices=ctx.hazard_indices,
        lookback=lookback,
        forecast_horizon=forecast_horizon,
        max_windows_train=max_windows_train,
        max_windows_val=max_windows_val,
        max_windows_test=max_windows_test,
        seed=seed + 13,
    )

    if len(generic_samples["train"]["seq"]) == 0 or len(generic_samples["val"]["seq"]) == 0:
        raise RuntimeError("generic 模式训练/验证窗口为空，无法训练预测模型。")

    normalizers: dict[str, dict[str, Any]] = {}
    normalizers["generic"] = fit_forecast_normalizer(generic_samples["train"])
    generic_norm = {split: apply_forecast_normalizer(sample, normalizers["generic"]) for split, sample in generic_samples.items()}

    hazard_requested = forecast_mode == "hazard"
    structural_fallback_reasons = list(ctx.graph.get("graph_stats", {}).get("fallback_reasons", []))
    can_train_hazard = (
        hazard_requested
        and ctx.hazard_available
        and len(hazard_samples["train"]["seq"]) > 0
        and len(hazard_samples["val"]["seq"]) > 0
    )
    if can_train_hazard:
        normalizers["hazard"] = fit_forecast_normalizer(hazard_samples["train"])
        hazard_norm = {split: apply_forecast_normalizer(sample, normalizers["hazard"]) for split, sample in hazard_samples.items()}
    else:
        hazard_norm = {}

    trained_models: dict[str, Any] = {}
    calibration_by_model: dict[str, Any] = {}
    explainability: dict[str, Any] = {"generic": {}, "hazard": {}}
    model_artifacts: dict[str, str] = {}
    histories: dict[str, Any] = {}

    # --- Train TFT (new primary generic model) ---
    tft_config = TemporalFusionConfig(
        input_dim=len(SEQUENCE_FEATURE_NAMES),
        static_dim=len(STATIC_FEATURE_NAMES),
        hidden_dim=96,
        n_heads=4,
        n_attn_layers=2,
        dropout_rate=0.12,
        horizon=forecast_horizon,
        edge_dim=4,
        use_neighbor_context=True,
    )
    tft_model = TemporalFusionForecaster(tft_config)
    tft_result = _train_one_model(
        model_name="TemporalFusionForecaster",
        model=tft_model,
        config=tft_config,
        train_norm=generic_norm["train"],
        val_norm=generic_norm["val"],
        output_path=output_dir / "tft_forecaster.pt",
        epochs=max(epochs, 30),
        batch_size=resolved_batch_size,
        learning_rate=learning_rate * 0.5,
        patience=max(patience, 8),
        seed=seed,
        device=device_obj,
        config_to_dict_fn=tft_config_to_dict,
    )

    # --- Train DecompTCNGRU (previous generic model, now competitor) ---
    generic_config = DecompTCNGRUConfig(
        input_dim=len(SEQUENCE_FEATURE_NAMES),
        static_dim=len(STATIC_FEATURE_NAMES),
        long_term_indices=tuple(int(idx) for idx in (CHANNEL_GROUPS["raw"] + CHANNEL_GROUPS["decomposition"])),
        short_term_indices=tuple(SEQUENCE_FEATURE_NAMES.index(name) for name in ("residual_component", "local_event_persistence", "abnormal_date_flag", "neighbor_mean_delta", "neighbor_delta_std")),
        horizon=forecast_horizon,
    )
    generic_model = DecompTCNGRUQuantileForecaster(generic_config)
    generic_result = _train_one_model(
        model_name="DecompTCNGRUQuantileForecaster",
        model=generic_model,
        config=generic_config,
        train_norm=generic_norm["train"],
        val_norm=generic_norm["val"],
        output_path=output_dir / "generic_forecaster.pt",
        epochs=epochs,
        batch_size=resolved_batch_size,
        learning_rate=learning_rate,
        patience=patience,
        seed=seed,
        device=device_obj,
        config_to_dict_fn=generic_config_to_dict,
    )

    # --- Pick best generic model (TFT vs DecompTCNGRU) ---
    tft_val_raw = _predict_model_path(output_dir / "tft_forecaster.pt", generic_norm["val"], batch_size=resolved_batch_size, device=device_obj)
    tft_val_raw *= float(normalizers["generic"]["target_scale"])
    tft_val_mae = float(np.mean(np.abs(tft_val_raw[:, :, 1] - generic_samples["val"]["target_offset"])))

    tcn_val_raw = _predict_model_path(output_dir / "generic_forecaster.pt", generic_norm["val"], batch_size=resolved_batch_size, device=device_obj)
    tcn_val_raw *= float(normalizers["generic"]["target_scale"])
    tcn_val_mae = float(np.mean(np.abs(tcn_val_raw[:, :, 1] - generic_samples["val"]["target_offset"])))

    if tft_val_mae <= tcn_val_mae:
        best_generic_path = output_dir / "tft_forecaster.pt"
        best_generic_name = "TemporalFusionForecaster"
        best_generic_config = tft_config
        best_generic_config_to_dict = tft_config_to_dict
        generic_model_selection = f"TFT selected (val MAE {tft_val_mae:.3f} <= TCN {tcn_val_mae:.3f})"
        histories["generic_tft"] = tft_result["history"]
        histories["generic_tcn"] = generic_result["history"]
    else:
        best_generic_path = output_dir / "generic_forecaster.pt"
        best_generic_name = "DecompTCNGRUQuantileForecaster"
        best_generic_config = generic_config
        best_generic_config_to_dict = generic_config_to_dict
        generic_model_selection = f"TCN selected (val MAE {tcn_val_mae:.3f} < TFT {tft_val_mae:.3f})"
        histories["generic_tft"] = tft_result["history"]
        histories["generic_tcn"] = generic_result["history"]

    # Copy best generic model to canonical path if needed
    if best_generic_path != output_dir / "generic_forecaster.pt":
        import shutil
        shutil.copy2(best_generic_path, output_dir / "generic_forecaster.pt")
    model_artifacts["generic"] = str(output_dir / "generic_forecaster.pt")
    model_artifacts["tft"] = tft_result["artifact_path"]
    histories["generic"] = histories.get("generic_tft" if "TemporalFusion" in best_generic_name else "generic_tcn", [])
    generic_val_raw = _predict_model_path(output_dir / "generic_forecaster.pt", generic_norm["val"], batch_size=resolved_batch_size, device=device_obj)
    generic_val_raw *= float(normalizers["generic"]["target_scale"])
    calibration_by_model["generic"] = _fit_cqr_calibration(generic_val_raw, generic_samples["val"]["target_offset"]) if calibrate_intervals else {
        "uncertainty_mode": "raw_quantile_v1",
        "interval_quantiles": list(INTERVAL_QUANTILES),
        "target_coverage": float(TARGET_COVERAGE),
        "corrections": [0.0] * forecast_horizon,
        "per_horizon": [],
        "calibration_reliability_factor": 1.0,
    }
    generic_val_cal = _apply_cqr_calibration(generic_val_raw, calibration_by_model["generic"])
    generic_test_raw = _predict_model_path(output_dir / "generic_forecaster.pt", generic_norm["test"], batch_size=resolved_batch_size, device=device_obj)
    generic_test_raw *= float(normalizers["generic"]["target_scale"])
    generic_test_cal = _apply_cqr_calibration(generic_test_raw, calibration_by_model["generic"])
    trained_models["generic"] = {
        "artifact_path": generic_result["artifact_path"],
        "config": _config_to_dict(generic_config),
        "validation": _interval_to_metrics(generic_val_raw, generic_val_cal, generic_samples["val"]["target_offset"]),
        "test": _interval_to_metrics(generic_test_raw, generic_test_cal, generic_samples["test"]["target_offset"]),
    }

    explainability["generic"]["channel_group_importance"] = _channel_group_importance(
        output_dir / "generic_forecaster.pt",
        generic_norm["val"],
        generic_samples["val"]["target_offset"],
        batch_size=resolved_batch_size,
        device=device_obj,
        seed=seed,
        target_scale=float(normalizers["generic"]["target_scale"]),
    )
    explainability["generic"]["static_feature_permutation_importance"] = _static_permutation_importance(
        output_dir / "generic_forecaster.pt",
        generic_norm["val"],
        generic_samples["val"]["target_offset"],
        batch_size=resolved_batch_size,
        device=device_obj,
        seed=seed + 1,
        target_scale=float(normalizers["generic"]["target_scale"]),
    )

    if can_train_hazard:
        hazard_config = GraphTCNAttnConfig(
            input_dim=len(SEQUENCE_FEATURE_NAMES),
            static_dim=len(STATIC_FEATURE_NAMES),
            edge_dim=4,
            horizon=forecast_horizon,
            message_passing_steps=2,
        )
        hazard_model = GraphTCNAttnQuantileForecaster(hazard_config)
        hazard_result = _train_one_model(
            model_name="GraphTCNAttnQuantileForecaster",
            model=hazard_model,
            config=hazard_config,
            train_norm=hazard_norm["train"],
            val_norm=hazard_norm["val"],
            output_path=output_dir / "hazard_forecaster.pt",
            epochs=epochs,
            batch_size=resolved_batch_size,
            learning_rate=learning_rate,
            patience=patience,
            seed=seed + 29,
            device=device_obj,
            config_to_dict_fn=hazard_config_to_dict,
        )
        model_artifacts["hazard"] = hazard_result["artifact_path"]
        histories["hazard"] = hazard_result["history"]
        hazard_val_raw = _predict_model_path(output_dir / "hazard_forecaster.pt", hazard_norm["val"], batch_size=resolved_batch_size, device=device_obj)
        hazard_val_raw *= float(normalizers["hazard"]["target_scale"])
        calibration_by_model["hazard"] = _fit_cqr_calibration(hazard_val_raw, hazard_samples["val"]["target_offset"]) if calibrate_intervals else {
            "uncertainty_mode": "raw_quantile_v1",
            "interval_quantiles": list(INTERVAL_QUANTILES),
            "target_coverage": float(TARGET_COVERAGE),
            "corrections": [0.0] * forecast_horizon,
            "per_horizon": [],
            "calibration_reliability_factor": 1.0,
        }
        hazard_val_cal = _apply_cqr_calibration(hazard_val_raw, calibration_by_model["hazard"])
        hazard_test_raw = _predict_model_path(output_dir / "hazard_forecaster.pt", hazard_norm["test"], batch_size=resolved_batch_size, device=device_obj)
        hazard_test_raw *= float(normalizers["hazard"]["target_scale"])
        hazard_test_cal = _apply_cqr_calibration(hazard_test_raw, calibration_by_model["hazard"])
        trained_models["hazard"] = {
            "artifact_path": hazard_result["artifact_path"],
            "config": _config_to_dict(hazard_config),
            "validation": _interval_to_metrics(hazard_val_raw, hazard_val_cal, hazard_samples["val"]["target_offset"]),
            "test": _interval_to_metrics(hazard_test_raw, hazard_test_cal, hazard_samples["test"]["target_offset"]),
        }
        explainability["hazard"]["channel_group_importance"] = _channel_group_importance(
            output_dir / "hazard_forecaster.pt",
            hazard_norm["val"],
            hazard_samples["val"]["target_offset"],
            batch_size=resolved_batch_size,
            device=device_obj,
            seed=seed + 3,
            target_scale=float(normalizers["hazard"]["target_scale"]),
        )
        explainability["hazard"]["static_feature_permutation_importance"] = _static_permutation_importance(
            output_dir / "hazard_forecaster.pt",
            hazard_norm["val"],
            hazard_samples["val"]["target_offset"],
            batch_size=resolved_batch_size,
            device=device_obj,
            seed=seed + 4,
            target_scale=float(normalizers["hazard"]["target_scale"]),
        )
        explainability["hazard"]["neighbor_attention_summary"] = _hazard_attention_summary(
            output_dir / "hazard_forecaster.pt",
            hazard_norm["val"],
            batch_size=resolved_batch_size,
            device=device_obj,
            seed=seed + 5,
        )

    baseline_validation = _run_baselines(generic_samples["val"], ctx=ctx, horizon=forecast_horizon)
    baseline_test = _run_baselines(generic_samples["test"], ctx=ctx, horizon=forecast_horizon)
    baseline_intervals: dict[str, Any] = {}
    baseline_metrics: dict[str, Any] = {}
    for name, baseline_result in baseline_validation.items():
        p10_val, p90_val, bands, metrics_val = _baseline_with_intervals(
            baseline_result,
            target=generic_samples["val"]["target_offset"],
        )
        pred_test_p50 = np.asarray(baseline_test[name]["pred_offset_p50"], dtype=np.float32)
        pred_test_p10, pred_test_p90 = apply_residual_quantiles(pred_test_p50, bands)
        metrics_test = metric_summary(
            pred_test_p50,
            generic_samples["test"]["target_offset"],
            pred_p10=pred_test_p10,
            pred_p90=pred_test_p90,
            target_coverage=TARGET_COVERAGE,
        )
        metrics_test["status"] = str(baseline_test[name].get("status", "ok"))
        metrics_test["reason"] = str(baseline_test[name].get("reason", ""))
        baseline_intervals[name] = {"residual_bands": bands}
        baseline_metrics[name] = {"validation": metrics_val, "test": metrics_test}

    active_model = "generic"
    model_selection_reason = "generic_is_default_mode"
    forecast_mode_actual = "generic"
    fallback_triggered = bool(structural_fallback_reasons) and hazard_requested
    fallback_reasons = list(structural_fallback_reasons)

    if forecast_mode == "generic":
        active_model = "generic"
        forecast_mode_actual = "generic"
    elif "hazard" not in trained_models:
        active_model = "generic"
        forecast_mode_actual = "generic"
        if not fallback_reasons:
            fallback_reasons.append("hazard_model_unavailable")
        model_selection_reason = "hazard_requested_but_fallback_to_generic"
        fallback_triggered = True
    else:
        generic_ref_pred = _predict_model_path(output_dir / "generic_forecaster.pt", hazard_norm["val"], batch_size=resolved_batch_size, device=device_obj)
        generic_ref_pred *= float(normalizers["generic"]["target_scale"])
        generic_ref_cal = _apply_cqr_calibration(generic_ref_pred, calibration_by_model["generic"])
        hazard_ref_pred = _predict_model_path(output_dir / "hazard_forecaster.pt", hazard_norm["val"], batch_size=resolved_batch_size, device=device_obj)
        hazard_ref_pred *= float(normalizers["hazard"]["target_scale"])
        hazard_ref_cal = _apply_cqr_calibration(hazard_ref_pred, calibration_by_model["hazard"])
        generic_ref_metrics = metric_summary(
            generic_ref_cal[:, :, 1],
            hazard_samples["val"]["target_offset"],
            pred_p10=generic_ref_cal[:, :, 0],
            pred_p90=generic_ref_cal[:, :, 2],
            target_coverage=TARGET_COVERAGE,
        )
        hazard_ref_metrics = metric_summary(
            hazard_ref_cal[:, :, 1],
            hazard_samples["val"]["target_offset"],
            pred_p10=hazard_ref_cal[:, :, 0],
            pred_p90=hazard_ref_cal[:, :, 2],
            target_coverage=TARGET_COVERAGE,
        )
        if hazard_ref_metrics["coverage_error"] > generic_ref_metrics["coverage_error"] + 0.05:
            active_model = "generic"
            forecast_mode_actual = "generic"
            model_selection_reason = "hazard_coverage_error_worse_than_generic"
        elif hazard_ref_metrics["rmse_mm"] <= generic_ref_metrics["rmse_mm"] * 0.98:
            active_model = "hazard"
            forecast_mode_actual = "hazard"
            model_selection_reason = "hazard_val_rmse_improved_by_2pct"
        else:
            active_model = "generic"
            forecast_mode_actual = "generic"
            model_selection_reason = "generic_val_rmse_more_stable"

    calibration_active = calibration_by_model.get(active_model, calibration_by_model["generic"])
    calibration_summary = {
        **calibration_active,
        "active_model": active_model,
        "models": calibration_by_model,
    }
    explainability_summary = {
        "generic": explainability.get("generic", {}),
        "hazard": explainability.get("hazard", {}),
    }

    joblib.dump(
        {
            "generic": normalizers["generic"],
            "hazard": normalizers.get("hazard"),
            "active_model": active_model,
        },
        output_dir / "forecast_normalizer.joblib",
    )
    np.savez(
        output_dir / "forecast_split_info.npz",
        selected_indices=ctx.selected_indices,
        hazard_indices=ctx.hazard_indices,
        generic_train_end=ctx.split_end_indices["train"],
        generic_val_end=ctx.split_end_indices["val"],
        generic_test_end=ctx.split_end_indices["test"],
    )
    (output_dir / "forecast_data_summary.json").write_text(
        json.dumps(_to_serializable(ctx.data_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "baseline_intervals.json").write_text(
        json.dumps(_to_serializable(baseline_intervals), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "forecast_calibration.json").write_text(
        json.dumps(_to_serializable(calibration_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "forecast_explainability.json").write_text(
        json.dumps(_to_serializable(explainability_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_summary = {
        "mintpy_dir": str(mintpy_dir),
        "qc_report_dir": str(qc_report_dir),
        "output_dir": str(output_dir),
        "device": device_obj.type,
        "requested_batch_size": int(batch_size),
        "resolved_batch_size": int(resolved_batch_size),
        "lookback": int(lookback),
        "forecast_horizon": int(forecast_horizon),
        "forecast_mode_requested": forecast_mode,
        "forecast_mode_actual": forecast_mode_actual,
        "decomposition_method": decomposition,
        "graph_k": int(graph_k),
        "graph_stats": _to_serializable(ctx.graph.get("graph_stats", {})),
        "channel_groups": {
            key: [SEQUENCE_FEATURE_NAMES[int(idx)] if isinstance(idx, (int, np.integer)) else str(idx) for idx in value]
            for key, value in CHANNEL_GROUPS.items()
        },
        "static_feature_names": list(STATIC_FEATURE_NAMES),
        "uncertainty_mode": "cqr_conformal_v1" if calibrate_intervals else "raw_quantile_v1",
        "trained_models": sorted(list(trained_models.keys()) + list(baseline_metrics.keys())),
        "active_model": active_model,
        "model_selection_reason": model_selection_reason,
        "generic_model_selection": generic_model_selection,
        "generic_model_name": best_generic_name,
        "fallback_triggered": bool(fallback_triggered),
        "fallback_reasons": fallback_reasons,
        "selection_mode": ctx.selection_mode,
        "zone_detection_status": ctx.zone_detection_status,
        "zone_filter_mode": ctx.zone_filter_mode,
        "zone_mask_path": ctx.zone_mask_path,
        "n_detected_zones": int(ctx.n_detected_zones),
        "n_zone_points_used": int(np.sum(ctx.point_in_zone_mask[ctx.selected_indices])) if len(ctx.selected_indices) else 0,
        "zone_ids_used": list(ctx.zone_ids_used),
        "zone_filter_fallback_triggered": bool(ctx.zone_filter_fallback_triggered),
        "zone_filter_fallback_reason": ctx.zone_filter_fallback_reason,
        "accepted_pass": ctx.accepted_pass,
        "n_points_used": int(len(ctx.selected_indices)),
        "n_hazard_points": int(len(ctx.hazard_indices)),
        "n_train_samples_generic": int(len(generic_samples["train"]["seq"])),
        "n_val_samples_generic": int(len(generic_samples["val"]["seq"])),
        "n_test_samples_generic": int(len(generic_samples["test"]["seq"])),
        "n_train_samples_hazard": int(len(hazard_samples["train"]["seq"])),
        "n_val_samples_hazard": int(len(hazard_samples["val"]["seq"])),
        "n_test_samples_hazard": int(len(hazard_samples["test"]["seq"])),
        "model_artifacts": model_artifacts,
        "training_histories": histories,
        "models": _to_serializable(trained_models),
        "baselines": _to_serializable(baseline_metrics),
        "calibration_path": str(output_dir / "forecast_calibration.json"),
        "explainability_path": str(output_dir / "forecast_explainability.json"),
        "data_summary_path": str(output_dir / "forecast_data_summary.json"),
    }
    (output_dir / "forecast_train_summary.json").write_text(
        json.dumps(_to_serializable(train_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return train_summary


__all__ = [
    "TARGET_COVERAGE",
    "_apply_cqr_calibration",
    "_apply_monotonic_fix",
    "_fit_cqr_calibration",
    "_instantiate_model_from_payload",
    "_predict_model_path",
    "_predict_torch_batches",
    "_resolve_device",
    "_torch_load",
    "train_forecast_models",
]
