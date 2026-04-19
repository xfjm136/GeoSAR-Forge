"""Forecast baselines and shared metrics for MintPy forecasting v2."""

from __future__ import annotations

from typing import Any

import numpy as np


def _future_time_matrix(window_end: np.ndarray, day_offsets: np.ndarray, horizon: int) -> np.ndarray:
    return np.stack([day_offsets[window_end + h + 1] for h in range(horizon)], axis=1).astype(np.float32)


def persistence_predict(samples: dict[str, np.ndarray], *, horizon: int) -> dict[str, np.ndarray]:
    n = len(samples["seq"])
    pred_offset = np.zeros((n, horizon), dtype=np.float32)
    last_rel0 = samples["last_rel0"][:, None].astype(np.float32)
    pred_rel0 = np.repeat(last_rel0, horizon, axis=1).astype(np.float32)
    return {
        "pred_offset_p50": pred_offset,
        "pred_rel0_p50": pred_rel0,
        "available_mask": np.ones(n, dtype=bool),
        "status": "ok",
        "reason": "",
    }


def linear_trend_predict(samples: dict[str, np.ndarray], day_offsets: np.ndarray, *, horizon: int) -> dict[str, np.ndarray]:
    seq = samples["seq"]
    window_end = samples["window_end"]
    n, lookback, _ = seq.shape
    pred_rel0 = np.full((n, horizon), np.nan, dtype=np.float32)
    hist_times = np.zeros((n, lookback), dtype=np.float32)
    for i, end in enumerate(window_end):
        end = int(end)
        hist_times[i] = day_offsets[end - lookback + 1 : end + 1]
    future_times = _future_time_matrix(window_end.astype(np.int64), day_offsets, horizon)
    hist_rel0 = seq[:, :, 0].astype(np.float64)

    for i in range(n):
        x = hist_times[i].astype(np.float64)
        y = hist_rel0[i].astype(np.float64)
        if not np.isfinite(y).all():
            continue
        x_center = x - np.nanmean(x)
        denom = float(np.sum(x_center**2))
        if denom <= 1e-6:
            slope = 0.0
        else:
            slope = float(np.sum(x_center * (y - np.nanmean(y))) / denom)
        intercept = float(np.nanmean(y) - slope * np.nanmean(x))
        pred_rel0[i] = (intercept + slope * future_times[i]).astype(np.float32)

    pred_offset = (pred_rel0 - samples["last_rel0"][:, None]).astype(np.float32)
    return {
        "pred_offset_p50": pred_offset,
        "pred_rel0_p50": pred_rel0,
        "available_mask": np.isfinite(pred_rel0).all(axis=1),
        "status": "ok",
        "reason": "",
    }


def seasonal_naive_predict(
    samples: dict[str, np.ndarray],
    day_offsets: np.ndarray,
    doy: np.ndarray,
    *,
    horizon: int,
    min_history_span_days: float = 180.0,
    max_doy_distance: float = 45.0,
) -> dict[str, np.ndarray]:
    seq = samples["seq"]
    window_end = samples["window_end"].astype(np.int64)
    n, lookback, _ = seq.shape
    pred_rel0 = np.full((n, horizon), np.nan, dtype=np.float32)

    for i, end in enumerate(window_end):
        hist_days = day_offsets[end - lookback + 1 : end + 1].astype(np.float32)
        hist_doy = doy[end - lookback + 1 : end + 1].astype(np.float32)
        hist_rel0 = seq[i, :, 0].astype(np.float32)
        if float(hist_days[-1] - hist_days[0]) < float(min_history_span_days):
            continue
        for h in range(horizon):
            fut_idx = int(end + h + 1)
            target_doy = float(doy[fut_idx])
            cyc_dist = np.abs(hist_doy - target_doy)
            cyc_dist = np.minimum(cyc_dist, 365.25 - cyc_dist)
            best = int(np.argmin(cyc_dist))
            if float(cyc_dist[best]) > float(max_doy_distance):
                continue
            pred_rel0[i, h] = float(hist_rel0[best])

    available_mask = np.isfinite(pred_rel0).all(axis=1)
    if not np.any(available_mask):
        return {
            "pred_offset_p50": np.full((n, horizon), np.nan, dtype=np.float32),
            "pred_rel0_p50": pred_rel0,
            "available_mask": available_mask,
            "status": "unavailable",
            "reason": "insufficient_seasonal_history",
        }

    pred_offset = (pred_rel0 - samples["last_rel0"][:, None]).astype(np.float32)
    return {
        "pred_offset_p50": pred_offset,
        "pred_rel0_p50": pred_rel0,
        "available_mask": available_mask,
        "status": "ok",
        "reason": "",
    }


def harmonic_trend_predict(samples: dict[str, np.ndarray], day_offsets: np.ndarray, *, horizon: int) -> dict[str, np.ndarray]:
    seq = samples["seq"]
    window_end = samples["window_end"].astype(np.int64)
    n, lookback, _ = seq.shape
    pred_rel0 = np.full((n, horizon), np.nan, dtype=np.float32)
    future_times = _future_time_matrix(window_end, day_offsets, horizon).astype(np.float64) / 365.25

    for i, end in enumerate(window_end):
        hist_days = day_offsets[end - lookback + 1 : end + 1].astype(np.float64) / 365.25
        hist_rel0 = seq[i, :, 0].astype(np.float64)
        if not np.isfinite(hist_rel0).all():
            continue
        x = np.stack(
            [
                np.ones_like(hist_days),
                hist_days,
                np.sin(2.0 * np.pi * hist_days),
                np.cos(2.0 * np.pi * hist_days),
            ],
            axis=1,
        )
        try:
            beta, *_ = np.linalg.lstsq(x, hist_rel0, rcond=None)
        except np.linalg.LinAlgError:
            continue
        fut_x = np.stack(
            [
                np.ones((horizon,), dtype=np.float64),
                future_times[i],
                np.sin(2.0 * np.pi * future_times[i]),
                np.cos(2.0 * np.pi * future_times[i]),
            ],
            axis=1,
        )
        pred_rel0[i] = (fut_x @ beta).astype(np.float32)

    pred_offset = (pred_rel0 - samples["last_rel0"][:, None]).astype(np.float32)
    return {
        "pred_offset_p50": pred_offset,
        "pred_rel0_p50": pred_rel0,
        "available_mask": np.isfinite(pred_rel0).all(axis=1),
        "status": "ok",
        "reason": "",
    }


def fit_residual_quantiles(
    pred_p50: np.ndarray,
    target: np.ndarray,
    *,
    quantiles: tuple[float, float] = (0.1, 0.9),
) -> dict[str, Any]:
    residual = target - pred_p50
    out: dict[str, Any] = {
        "q10": [],
        "q90": [],
        "status": "ok",
        "reason": "",
    }
    for h in range(target.shape[1]):
        valid = np.isfinite(residual[:, h])
        if not np.any(valid):
            out["q10"].append(float("nan"))
            out["q90"].append(float("nan"))
            continue
        out["q10"].append(float(np.nanquantile(residual[valid, h], quantiles[0])))
        out["q90"].append(float(np.nanquantile(residual[valid, h], quantiles[1])))
    return out


def apply_residual_quantiles(pred_p50: np.ndarray, residual_bands: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    q10 = np.asarray(residual_bands.get("q10", []), dtype=np.float32)
    q90 = np.asarray(residual_bands.get("q90", []), dtype=np.float32)
    if q10.size == 0 or q90.size == 0:
        nan = np.full_like(pred_p50, np.nan, dtype=np.float32)
        return nan, nan
    pred_p10 = pred_p50 + q10[None, :]
    pred_p90 = pred_p50 + q90[None, :]
    return pred_p10.astype(np.float32), pred_p90.astype(np.float32)


def pinball_loss(pred: np.ndarray, target: np.ndarray, quantile: float) -> float:
    err = target - pred
    return float(np.nanmean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def weighted_interval_score(
    target: np.ndarray,
    pred_p10: np.ndarray,
    pred_p50: np.ndarray,
    pred_p90: np.ndarray,
    *,
    alpha: float = 0.2,
) -> float:
    width = pred_p90 - pred_p10
    lower_penalty = np.maximum(pred_p10 - target, 0.0)
    upper_penalty = np.maximum(target - pred_p90, 0.0)
    wis = width + (2.0 / alpha) * lower_penalty + (2.0 / alpha) * upper_penalty + np.abs(pred_p50 - target)
    return float(np.nanmean(wis))


def metric_summary(
    pred_p50: np.ndarray,
    target: np.ndarray,
    *,
    pred_p10: np.ndarray | None = None,
    pred_p90: np.ndarray | None = None,
    target_coverage: float = 0.80,
) -> dict[str, Any]:
    diff = pred_p50 - target
    summary: dict[str, Any] = {
        "mae_mm": float(np.nanmean(np.abs(diff))) if diff.size else float("nan"),
        "rmse_mm": float(np.sqrt(np.nanmean(diff**2))) if diff.size else float("nan"),
        "pinball_loss": float(pinball_loss(pred_p50, target, 0.5)) if diff.size else float("nan"),
    }
    if pred_p10 is not None and pred_p90 is not None:
        cover = (target >= pred_p10) & (target <= pred_p90)
        width = pred_p90 - pred_p10
        summary["coverage"] = float(np.nanmean(cover)) if cover.size else float("nan")
        summary["coverage_error"] = float(abs(summary["coverage"] - target_coverage)) if np.isfinite(summary["coverage"]) else float("nan")
        summary["mean_interval_width_mm"] = float(np.nanmean(width)) if width.size else float("nan")
        summary["wis"] = weighted_interval_score(target, pred_p10, pred_p50, pred_p90)
        summary["pinball_loss"] = float(
            np.nanmean(
                [
                    pinball_loss(pred_p10, target, 0.1),
                    pinball_loss(pred_p50, target, 0.5),
                    pinball_loss(pred_p90, target, 0.9),
                ]
            )
        )
    else:
        summary["coverage"] = float("nan")
        summary["coverage_error"] = float("nan")
        summary["mean_interval_width_mm"] = float("nan")
        summary["wis"] = float("nan")
    per_horizon: list[dict[str, float]] = []
    for h in range(target.shape[1]):
        diff_h = diff[:, h]
        item = {
            "horizon": int(h + 1),
            "mae_mm": float(np.nanmean(np.abs(diff_h))) if diff_h.size else float("nan"),
            "rmse_mm": float(np.sqrt(np.nanmean(diff_h**2))) if diff_h.size else float("nan"),
            "pinball_loss": float(pinball_loss(pred_p50[:, h], target[:, h], 0.5)) if diff_h.size else float("nan"),
        }
        if pred_p10 is not None and pred_p90 is not None:
            cover_h = (target[:, h] >= pred_p10[:, h]) & (target[:, h] <= pred_p90[:, h])
            width_h = pred_p90[:, h] - pred_p10[:, h]
            item["coverage"] = float(np.nanmean(cover_h)) if cover_h.size else float("nan")
            item["coverage_error"] = float(abs(item["coverage"] - target_coverage)) if np.isfinite(item["coverage"]) else float("nan")
            item["mean_interval_width_mm"] = float(np.nanmean(width_h)) if width_h.size else float("nan")
            item["wis"] = float(
                np.nanmean(
                    width_h
                    + (2.0 / 0.2) * np.maximum(pred_p10[:, h] - target[:, h], 0.0)
                    + (2.0 / 0.2) * np.maximum(target[:, h] - pred_p90[:, h], 0.0)
                    + np.abs(pred_p50[:, h] - target[:, h])
                )
            ) if width_h.size else float("nan")
            item["pinball_loss"] = float(
                np.nanmean(
                    [
                        pinball_loss(pred_p10[:, h], target[:, h], 0.1),
                        pinball_loss(pred_p50[:, h], target[:, h], 0.5),
                        pinball_loss(pred_p90[:, h], target[:, h], 0.9),
                    ]
                )
            )
        else:
            item["coverage"] = float("nan")
            item["coverage_error"] = float("nan")
            item["mean_interval_width_mm"] = float("nan")
            item["wis"] = float("nan")
        per_horizon.append(item)
    summary["per_horizon"] = per_horizon
    return summary
