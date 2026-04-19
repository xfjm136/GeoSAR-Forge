"""Dataset preparation for MintPy downstream forecasting v2."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio


FINAL_TS_CANDIDATES = [
    "timeseries_SET_GACOS_ramp_demErr.h5",
    "timeseries_SET_ERA5_ramp_demErr.h5",
    "timeseries_SET_GACOS_ramp.h5",
    "timeseries_SET_ERA5_ramp.h5",
    "timeseries_SET_GACOS.h5",
    "timeseries_SET_ERA5.h5",
    "timeseries_SET.h5",
    "timeseries.h5",
]

SEQUENCE_FEATURE_NAMES = [
    "rel0",
    "delta",
    "day_gap",
    "sin_doy",
    "cos_doy",
    "trend_component",
    "seasonal_component",
    "residual_component",
    "delta2",
    "rolling_velocity_3step",
    "rolling_residual_std",
    "neighbor_mean_rel0",
    "neighbor_mean_delta",
    "neighbor_delta_std",
    "local_event_persistence",
    "abnormal_date_flag",
]

CHANNEL_GROUPS = {
    "raw": [0, 1, 2, 3, 4],
    "decomposition": [5, 6, 7, 8, 9, 10],
    "neighborhood": [11, 12, 13],
    "event": [14, 15],
}

STATIC_FEATURE_NAMES = [
    "ps_score",
    "tcoh",
    "valid_pair_ratio",
    "mainCC_ratio",
    "jump_risk",
    "anomaly_exposure",
    "velocity_mm_yr",
    "height_m",
]


def _normalize_forecast_point_scope(scope: str | None) -> str:
    scope_norm = str(scope or "all_high_confidence").strip().lower()
    aliases = {
        "depsi_like_high_confidence": "all_high_confidence",
        "depsi_high_confidence": "all_high_confidence",
        "all_depsi_like_high_confidence": "all_high_confidence",
        "all_high_confidence": "all_high_confidence",
        "zone_high_confidence_only": "zone_high_confidence_only",
    }
    return aliases.get(scope_norm, "all_high_confidence")


@dataclass
class ForecastContext:
    mintpy_dir: Path
    qc_report_dir: Path
    accepted_pass: str
    final_timeseries_path: Path
    dates: list[str]
    day_offsets: np.ndarray
    doy: np.ndarray
    rel0_mm: np.ndarray
    delta_mm: np.ndarray
    valid_mask: np.ndarray
    selected_indices: np.ndarray
    selected_labels: np.ndarray
    selection_mode: str
    zone_filter_mode: str
    zone_detection_status: str
    zone_mask_path: str
    zone_ids_used: list[str]
    n_detected_zones: int
    zone_filter_fallback_triggered: bool
    zone_filter_fallback_reason: str
    point_zone_ids: np.ndarray
    point_in_zone_mask: np.ndarray
    hazard_indices: np.ndarray
    hazard_available: bool
    points: dict[str, np.ndarray]
    static_feature_names: list[str]
    static_features: np.ndarray
    sequence_feature_names: list[str]
    channel_groups: dict[str, list[int]]
    sequence_array: np.ndarray
    anomaly_exposure: np.ndarray
    window_count_per_point: np.ndarray
    split_end_indices: dict[str, np.ndarray]
    graph: dict[str, Any]
    decomposition_method: str
    data_summary: dict[str, Any]


def _pick_final_timeseries(mintpy_dir: Path) -> Path:
    mintpy_dir = Path(mintpy_dir)
    for name in FINAL_TS_CANDIDATES:
        path = mintpy_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到 MintPy 时序文件: {mintpy_dir}")


def _load_mintpy_timeseries(mintpy_dir: Path) -> tuple[Path, list[str], np.ndarray]:
    ts_path = _pick_final_timeseries(mintpy_dir)
    with h5py.File(ts_path, "r") as f:
        raw_dates = f["date"][:]
        dates = [d.decode("utf-8") if hasattr(d, "decode") else d.tobytes().decode("utf-8") for d in raw_dates]
        ts = np.asarray(f["timeseries"][:], dtype=np.float32) * 1000.0
    rel0 = ts - ts[0:1]
    return ts_path, dates, rel0.astype(np.float32)


def _read_tif_bool(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1).astype(bool)


def _read_tif_int(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.int32)


def _load_geometry(mintpy_dir: Path) -> dict[str, np.ndarray]:
    geom_path = mintpy_dir / "inputs" / "geometryRadar.h5"
    with h5py.File(geom_path, "r") as f:
        out = {
            "latitude": np.asarray(f["latitude"][:], dtype=np.float32),
            "longitude": np.asarray(f["longitude"][:], dtype=np.float32),
        }
        if "height" in f:
            out["height"] = np.asarray(f["height"][:], dtype=np.float32)
        else:
            out["height"] = np.full_like(out["latitude"], np.nan, dtype=np.float32)
    return out


def _load_velocity(mintpy_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(mintpy_dir / "velocity.h5", "r") as f:
        vel = np.asarray(f["velocity"][:], dtype=np.float32) * 1000.0
        if "velocityStd" in f:
            vstd = np.asarray(f["velocityStd"][:], dtype=np.float32) * 1000.0
        else:
            vstd = np.full_like(vel, np.nan, dtype=np.float32)
    with h5py.File(mintpy_dir / "temporalCoherence.h5", "r") as f:
        tcoh = np.asarray(f["temporalCoherence"][:], dtype=np.float32)
    return vel, vstd, tcoh


def _load_metrics(qc_report_dir: Path) -> dict[str, np.ndarray]:
    metrics_path = qc_report_dir / "ps_model_metrics.h5"
    with h5py.File(metrics_path, "r") as f:
        return {
            "model_rms": np.asarray(f["model_rms"][:], dtype=np.float32),
            "valid_pair_ratio": np.asarray(f["valid_pair_ratio"][:], dtype=np.float32),
            "mainCC_ratio": np.asarray(f["mainCC_ratio"][:], dtype=np.float32),
            "jump_risk": np.asarray(f["jump_risk"][:], dtype=np.float32),
        }


def _load_ps_score(qc_report_dir: Path) -> np.ndarray:
    with rasterio.open(qc_report_dir / "ps_score.tif") as ds:
        return ds.read(1).astype(np.float32)


def _load_date_qc(qc_report_dir: Path, dates: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    path = qc_report_dir / "date_qc.csv"
    flags = np.zeros(len(dates), dtype=bool)
    rows: dict[str, Any] = {}
    if not path.exists():
        return flags, rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["date"]] = row
    for i, date in enumerate(dates):
        row = rows.get(date)
        if row and str(row.get("final_abnormal_flag", "")).strip().lower() in {"1", "true", "yes"}:
            flags[i] = True
    return flags, rows


def _normalize_interval(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    if not finite.any():
        return out
    p5 = float(np.nanpercentile(arr[finite], 5))
    p95 = float(np.nanpercentile(arr[finite], 95))
    if np.isclose(p5, p95):
        out[finite] = 0.0
    else:
        out[finite] = (arr[finite] - p5) / (p95 - p5)
    return np.clip(out, 0.0, 1.0)


def _compute_window_counts(valid_mask: np.ndarray, lookback: int, horizon: int) -> np.ndarray:
    n_points, n_dates = valid_mask.shape
    counts = np.zeros(n_points, dtype=np.int32)
    for idx in range(n_points):
        mask = valid_mask[idx]
        total = 0
        for end in range(lookback - 1, n_dates - horizon):
            start = end - lookback + 1
            if mask[start : end + 1 + horizon].all():
                total += 1
        counts[idx] = total
    return counts


def _split_end_indices(n_dates: int, lookback: int, horizon: int) -> dict[str, np.ndarray]:
    usable = np.arange(lookback - 1, n_dates - horizon, dtype=np.int32)
    if usable.size <= 2:
        return {"train": usable[:1], "val": usable[-1:], "test": usable[-1:]} if usable.size else {"train": usable, "val": usable, "test": usable}
    n = usable.size
    train_end = max(1, int(np.floor(n * 0.7)))
    val_end = max(train_end + 1, int(np.floor(n * 0.85)))
    val_end = min(val_end, n - 1)
    return {
        "train": usable[:train_end],
        "val": usable[train_end:val_end],
        "test": usable[val_end:],
    }


def _estimate_total_windows(window_counts: np.ndarray, indices: np.ndarray, split_end_indices: dict[str, np.ndarray]) -> tuple[int, int, int]:
    total = int(np.sum(window_counts[indices])) if len(indices) else 0
    n_ends = {name: max(len(vals), 1) for name, vals in split_end_indices.items()}
    total_end = sum(n_ends.values())
    train = int(round(total * n_ends["train"] / total_end))
    val = int(round(total * n_ends["val"] / total_end))
    test = max(0, total - train - val)
    return train, val, test


def _build_selection(
    strict_idx: np.ndarray,
    relaxed_idx: np.ndarray,
    window_counts: np.ndarray,
    split_end_indices: dict[str, np.ndarray],
    *,
    min_points_for_training: int,
    min_train_windows: int,
    min_val_windows: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    def _enough(indices: np.ndarray) -> bool:
        if len(indices) < int(min_points_for_training):
            return False
        n_train, n_val, _ = _estimate_total_windows(window_counts, indices, split_end_indices)
        return n_train >= int(min_train_windows) and n_val >= int(min_val_windows)

    if _enough(strict_idx):
        return strict_idx, np.array(["strict"] * len(strict_idx), dtype=object), "strict_only"

    mixed_idx = np.concatenate([strict_idx, relaxed_idx]).astype(np.int64)
    if mixed_idx.size:
        mixed_idx = np.unique(mixed_idx)
    if _enough(mixed_idx):
        labels = np.where(np.isin(mixed_idx, strict_idx), "strict", "relaxed").astype(object)
        return mixed_idx, labels, "strict_plus_relaxed"

    labels = np.where(np.isin(mixed_idx, strict_idx), "strict", "relaxed").astype(object) if mixed_idx.size else np.empty((0,), dtype=object)
    return mixed_idx, labels, "fallback_baseline_only"


def _feature_summary(names: list[str], values: np.ndarray) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        col = values[:, idx]
        finite = np.isfinite(col)
        if not finite.any():
            summary[name] = {"mean": float("nan"), "std": float("nan"), "p05": float("nan"), "p95": float("nan")}
            continue
        summary[name] = {
            "mean": float(np.nanmean(col[finite])),
            "std": float(np.nanstd(col[finite])),
            "p05": float(np.nanpercentile(col[finite], 5)),
            "p95": float(np.nanpercentile(col[finite], 95)),
        }
    return summary


def _load_zone_filter_metadata(zone_mask_path: Path, expected_shape: tuple[int, int]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "zone_detection_status": "not_requested",
        "zone_mask_path": "",
        "n_detected_zones": 0,
        "zone_ids_all": [],
        "point_in_zone_mask": np.zeros(expected_shape, dtype=bool),
        "point_zone_ids_raster": np.zeros(expected_shape, dtype=np.int32),
        "zone_filter_fallback_triggered": False,
        "zone_filter_fallback_reason": "",
    }
    if not zone_mask_path:
        return meta
    zone_mask_path = Path(zone_mask_path).resolve()
    meta["zone_mask_path"] = str(zone_mask_path)
    if not zone_mask_path.exists():
        meta["zone_detection_status"] = "mask_missing"
        meta["zone_filter_fallback_triggered"] = True
        meta["zone_filter_fallback_reason"] = "zone_mask_missing"
        return meta

    zone_mask = _read_tif_bool(zone_mask_path)
    if zone_mask.shape != expected_shape:
        meta["zone_detection_status"] = "mask_shape_mismatch"
        meta["zone_filter_fallback_triggered"] = True
        meta["zone_filter_fallback_reason"] = "zone_mask_shape_mismatch"
        return meta

    meta["point_in_zone_mask"] = zone_mask.astype(bool)
    summary_path = zone_mask_path.parent / "deformation_zone_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            meta["zone_detection_status"] = str(summary.get("status", "loaded"))
            meta["n_detected_zones"] = int(summary.get("n_detected_zones", 0))
            meta["zone_ids_all"] = [str(v) for v in summary.get("zone_ids", [])]
        except Exception:
            meta["zone_detection_status"] = "summary_unreadable"
    else:
        meta["zone_detection_status"] = "loaded"

    zone_id_path = zone_mask_path.parent / "deformation_zone_id.tif"
    if zone_id_path.exists():
        try:
            zone_id_raster = _read_tif_int(zone_id_path)
            if zone_id_raster.shape == expected_shape:
                meta["point_zone_ids_raster"] = zone_id_raster.astype(np.int32)
                if not meta["zone_ids_all"]:
                    zone_vals = np.unique(zone_id_raster[zone_id_raster > 0])
                    meta["zone_ids_all"] = [f"Z{int(v)}" for v in zone_vals]
            else:
                meta["zone_filter_fallback_triggered"] = True
                meta["zone_filter_fallback_reason"] = "zone_id_shape_mismatch"
        except Exception:
            pass
    return meta


def _build_design_matrix(day_offsets: np.ndarray) -> np.ndarray:
    t_years = np.asarray(day_offsets, dtype=np.float64) / 365.25
    angle = 2.0 * np.pi * t_years
    return np.stack(
        [
            np.ones_like(t_years),
            t_years,
            np.sin(angle),
            np.cos(angle),
        ],
        axis=1,
    ).astype(np.float64)


def _fit_robust_harmonic(
    series_mm: np.ndarray,
    valid_mask: np.ndarray,
    day_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _build_design_matrix(day_offsets)
    y = np.asarray(series_mm, dtype=np.float64)
    weights = valid_mask.astype(np.float64)

    def _solve(cur_weights: np.ndarray) -> np.ndarray:
        xtx = np.einsum("nt,ti,tj->nij", cur_weights, x, x, optimize=True)
        xty = np.einsum("nt,ti,nt->ni", cur_weights, x, y, optimize=True)
        beta = np.einsum("nij,nj->ni", np.linalg.pinv(xtx), xty, optimize=True)
        return beta

    beta = _solve(weights)
    pred = beta @ x.T
    resid = np.where(valid_mask, y - pred, np.nan)
    scale = np.nanmedian(np.abs(resid), axis=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, 1.0)
    huber = np.abs(resid) / (1.5 * scale[:, None])
    robust_weights = weights * np.where(np.isnan(huber), 0.0, np.where(huber <= 1.0, 1.0, 1.0 / np.maximum(huber, 1e-6)))
    beta = _solve(robust_weights)

    intercept = beta[:, 0:1]
    slope = beta[:, 1:2]
    sin_coef = beta[:, 2:3]
    cos_coef = beta[:, 3:4]
    t_years = (np.asarray(day_offsets, dtype=np.float64) / 365.25)[None, :]
    angle = 2.0 * np.pi * t_years
    trend = intercept + slope * t_years
    seasonal = sin_coef * np.sin(angle) + cos_coef * np.cos(angle)
    residual = y - trend - seasonal
    residual = np.where(valid_mask, residual, np.nan)
    return trend.astype(np.float32), seasonal.astype(np.float32), residual.astype(np.float32)


def _rolling_velocity_3step(series_mm: np.ndarray, day_offsets: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.full_like(series_mm, np.nan, dtype=np.float32)
    day_offsets = np.asarray(day_offsets, dtype=np.float32)
    for idx in range(series_mm.shape[1]):
        start = max(0, idx - 2)
        delta_days = float(day_offsets[idx] - day_offsets[start])
        if delta_days <= 0:
            continue
        ok = valid_mask[:, start] & valid_mask[:, idx]
        out[ok, idx] = ((series_mm[ok, idx] - series_mm[ok, start]) / delta_days) * 365.25
    return out


def _rolling_residual_std(residual_mm: np.ndarray) -> np.ndarray:
    out = np.full_like(residual_mm, np.nan, dtype=np.float32)
    for idx in range(residual_mm.shape[1]):
        start = max(0, idx - 2)
        out[:, idx] = np.nanstd(residual_mm[:, start : idx + 1], axis=1).astype(np.float32)
    return out


def _compute_event_persistence(delta_mm: np.ndarray, abnormal_flags: np.ndarray) -> np.ndarray:
    abnormal = abnormal_flags.astype(np.float32)[None, :]
    event_strength = np.abs(np.nan_to_num(delta_mm, nan=0.0)) * abnormal
    out = np.zeros_like(event_strength, dtype=np.float32)
    for idx in range(event_strength.shape[1]):
        start = max(0, idx - 2)
        out[:, idx] = np.nanmean(event_strength[:, start : idx + 1], axis=1).astype(np.float32)
    scale = np.nanpercentile(np.abs(np.nan_to_num(delta_mm, nan=0.0)), 95, axis=1).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1.0), scale, 1.0).astype(np.float32)
    out = out / scale[:, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _latlon_to_local_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat0 = float(np.nanmean(lat))
    lon0 = float(np.nanmean(lon))
    x = (lon - lon0) * (111320.0 * np.cos(np.radians(lat0)))
    y = (lat - lat0) * 110540.0
    return x.astype(np.float32), y.astype(np.float32)


def _connected_component_ratio(neighbor_index: np.ndarray, neighbor_mask: np.ndarray) -> float:
    n_nodes = neighbor_index.shape[0]
    if n_nodes == 0:
        return 0.0
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        rows = np.repeat(np.arange(n_nodes), neighbor_index.shape[1])[neighbor_mask.reshape(-1)]
        cols = neighbor_index[neighbor_mask]
        if rows.size == 0:
            return 0.0
        data = np.ones_like(rows, dtype=np.int8)
        graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        graph = graph.maximum(graph.transpose())
        n_comp, labels = connected_components(graph, directed=False, return_labels=True)
        if n_comp == 0:
            return 0.0
        counts = np.bincount(labels, minlength=n_comp)
        return float(counts.max() / max(n_nodes, 1))
    except Exception:
        return 0.0


def _build_neighbor_graph(
    lat: np.ndarray,
    lon: np.ndarray,
    height: np.ndarray,
    velocity: np.ndarray,
    ps_score: np.ndarray,
    *,
    graph_k: int,
) -> dict[str, Any]:
    n_nodes = len(lat)
    if n_nodes == 0:
        empty = np.empty((0, graph_k), dtype=np.int64)
        return {
            "neighbor_index": empty,
            "neighbor_mask": np.zeros((0, graph_k), dtype=bool),
            "edge_features": np.empty((0, graph_k, 4), dtype=np.float32),
            "graph_stats": {
                "effective_nodes": 0,
                "graph_k": int(graph_k),
                "mean_degree": 0.0,
                "largest_component_ratio": 0.0,
            },
        }

    from scipy.spatial import cKDTree

    x, y = _latlon_to_local_xy(lat, lon)
    coords = np.column_stack([x, y])
    tree = cKDTree(coords)
    query_k = min(max(int(graph_k) + 1, 2), n_nodes)
    dist, idx = tree.query(coords, k=query_k)
    if query_k == 2:
        dist = dist[:, None] if dist.ndim == 1 else dist
        idx = idx[:, None] if idx.ndim == 1 else idx
    neighbor_index = np.full((n_nodes, int(graph_k)), 0, dtype=np.int64)
    neighbor_mask = np.zeros((n_nodes, int(graph_k)), dtype=bool)
    edge_features = np.zeros((n_nodes, int(graph_k), 4), dtype=np.float32)
    for row in range(n_nodes):
        cand_idx = idx[row].reshape(-1)
        cand_dist = dist[row].reshape(-1)
        keep = cand_idx != row
        cand_idx = cand_idx[keep][: int(graph_k)]
        cand_dist = cand_dist[keep][: int(graph_k)]
        count = len(cand_idx)
        if count == 0:
            continue
        neighbor_index[row, :count] = cand_idx.astype(np.int64)
        neighbor_mask[row, :count] = True
        edge_features[row, :count, 0] = cand_dist.astype(np.float32)
        edge_features[row, :count, 1] = (height[cand_idx] - height[row]).astype(np.float32)
        edge_features[row, :count, 2] = (velocity[cand_idx] - velocity[row]).astype(np.float32)
        edge_features[row, :count, 3] = (ps_score[cand_idx] - ps_score[row]).astype(np.float32)

    graph_stats = {
        "effective_nodes": int(n_nodes),
        "graph_k": int(graph_k),
        "mean_degree": float(np.mean(neighbor_mask.sum(axis=1))) if n_nodes else 0.0,
        "largest_component_ratio": _connected_component_ratio(neighbor_index, neighbor_mask),
    }
    return {
        "neighbor_index": neighbor_index,
        "neighbor_mask": neighbor_mask,
        "edge_features": edge_features,
        "graph_stats": graph_stats,
    }


def _compute_neighbor_summaries(
    rel0_mm: np.ndarray,
    delta_mm: np.ndarray,
    valid_mask: np.ndarray,
    neighbor_index: np.ndarray,
    neighbor_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes, n_dates = rel0_mm.shape
    k = neighbor_index.shape[1] if neighbor_index.ndim == 2 else 0
    if n_nodes == 0 or k == 0:
        shape = (n_nodes, n_dates)
        return (
            np.full(shape, np.nan, dtype=np.float32),
            np.full(shape, np.nan, dtype=np.float32),
            np.full(shape, np.nan, dtype=np.float32),
        )

    safe_idx = np.clip(neighbor_index, 0, max(n_nodes - 1, 0))
    nbr_rel0 = rel0_mm[safe_idx]
    nbr_delta = delta_mm[safe_idx]
    nbr_valid = valid_mask[safe_idx]
    valid_neighbor = neighbor_mask[:, :, None] & nbr_valid

    rel0_masked = np.where(valid_neighbor, nbr_rel0, 0.0).astype(np.float64)
    delta_masked = np.where(valid_neighbor, nbr_delta, 0.0).astype(np.float64)
    counts = valid_neighbor.sum(axis=1).astype(np.float64)
    valid_any = counts > 0

    rel0_out = np.full((n_nodes, n_dates), np.nan, dtype=np.float32)
    delta_mean = np.full((n_nodes, n_dates), np.nan, dtype=np.float32)
    delta_std = np.full((n_nodes, n_dates), np.nan, dtype=np.float32)
    rel0_out[valid_any] = (rel0_masked.sum(axis=1)[valid_any] / counts[valid_any]).astype(np.float32)
    delta_mean[valid_any] = (delta_masked.sum(axis=1)[valid_any] / counts[valid_any]).astype(np.float32)
    delta_sumsq = np.where(valid_neighbor, np.square(nbr_delta), 0.0).astype(np.float64).sum(axis=1)
    variance = np.full((n_nodes, n_dates), np.nan, dtype=np.float64)
    variance[valid_any] = np.maximum(delta_sumsq[valid_any] / counts[valid_any] - np.square(delta_mean[valid_any]), 0.0)
    delta_std[valid_any] = np.sqrt(variance[valid_any]).astype(np.float32)
    return rel0_out, delta_mean, delta_std


def load_forecast_context(
    mintpy_dir: str | Path,
    *,
    qc_report_dir: str | Path,
    lookback: int = 12,
    forecast_horizon: int = 3,
    min_points_for_training: int = 500,
    min_train_windows: int = 3000,
    min_val_windows: int = 500,
    graph_k: int = 8,
    decomposition: str = "robust_harmonic_v1",
    zone_mask_path: str | Path | None = None,
    forecast_point_scope: str = "all_high_confidence",
) -> ForecastContext:
    if decomposition != "robust_harmonic_v1":
        raise ValueError(f"不支持的 decomposition: {decomposition}")

    mintpy_dir = Path(mintpy_dir).resolve()
    qc_report_dir = Path(qc_report_dir).resolve()
    ts_path, dates, rel0 = _load_mintpy_timeseries(mintpy_dir)
    geom = _load_geometry(mintpy_dir)
    vel, vstd, tcoh = _load_velocity(mintpy_dir)
    ps_score = _load_ps_score(qc_report_dir)
    strict_mask = _read_tif_bool(qc_report_dir / "mask_ps_strict.tif")
    relaxed_mask = _read_tif_bool(qc_report_dir / "mask_ps_relaxed.tif")
    metrics = _load_metrics(qc_report_dir)

    day_offsets = np.asarray(
        [(np.datetime64(d) - np.datetime64(dates[0])).astype(int) for d in dates],
        dtype=np.int32,
    )
    doy = np.asarray(
        [int((np.datetime64(d, "D") - np.datetime64(f"{d[:4]}-01-01", "D")).astype(int)) + 1 for d in dates],
        dtype=np.int32,
    )
    abnormal_flags, _ = _load_date_qc(qc_report_dir, dates)

    rel0_pf = np.moveaxis(rel0, 0, -1)
    valid_mask_pf = np.isfinite(rel0_pf)
    delta_pf = np.full_like(rel0_pf, np.nan, dtype=np.float32)
    delta_pf[..., 1:] = rel0_pf[..., 1:] - rel0_pf[..., :-1]

    local_velocity_gradient = np.sqrt(np.sum(np.square(np.gradient(np.nan_to_num(vel, nan=0.0))), axis=0)).astype(np.float32)
    abnormal_inc_idx = np.where(abnormal_flags[1:])[0] + 1
    if abnormal_inc_idx.size:
        exposure_raw = np.nanmean(np.abs(delta_pf[..., abnormal_inc_idx]), axis=-1).astype(np.float32)
        anomaly_exposure = _normalize_interval(exposure_raw)
    else:
        anomaly_exposure = np.zeros_like(vel, dtype=np.float32)

    valid_geo = (
        np.isfinite(geom["latitude"])
        & np.isfinite(geom["longitude"])
        & (geom["latitude"] > 0.1)
        & np.isfinite(ps_score)
        & np.isfinite(tcoh)
        & np.isfinite(metrics["model_rms"])
        & np.isfinite(metrics["valid_pair_ratio"])
        & np.isfinite(metrics["mainCC_ratio"])
        & np.isfinite(metrics["jump_risk"])
    )
    candidate_mask = valid_geo & (strict_mask | relaxed_mask)
    rows, cols = np.where(candidate_mask)
    if len(rows) == 0:
        raise RuntimeError("未找到可用于未来预测的高可信点。")

    point_rel0 = rel0_pf[rows, cols, :].astype(np.float32)
    point_valid = valid_mask_pf[rows, cols, :].astype(bool)
    point_delta = np.full_like(point_rel0, np.nan, dtype=np.float32)
    point_delta[:, 1:] = point_rel0[:, 1:] - point_rel0[:, :-1]

    trend_component, seasonal_component, residual_component = _fit_robust_harmonic(point_rel0, point_valid, day_offsets)
    delta2 = np.full_like(point_delta, np.nan, dtype=np.float32)
    delta2[:, 1:] = point_delta[:, 1:] - point_delta[:, :-1]
    rolling_velocity = _rolling_velocity_3step(point_rel0, day_offsets, point_valid)
    rolling_residual_std = _rolling_residual_std(residual_component)
    local_event_persistence = _compute_event_persistence(point_delta, abnormal_flags)
    abnormal_date_flag = np.broadcast_to(abnormal_flags.astype(np.float32)[None, :], point_rel0.shape).astype(np.float32)

    height_points = geom["height"][rows, cols].astype(np.float32)
    velocity_points = vel[rows, cols].astype(np.float32)
    ps_score_points = ps_score[rows, cols].astype(np.float32)
    graph = _build_neighbor_graph(
        geom["latitude"][rows, cols].astype(np.float32),
        geom["longitude"][rows, cols].astype(np.float32),
        height_points,
        velocity_points,
        ps_score_points,
        graph_k=graph_k,
    )
    neighbor_mean_rel0, neighbor_mean_delta, neighbor_delta_std = _compute_neighbor_summaries(
        point_rel0,
        point_delta,
        point_valid,
        graph["neighbor_index"],
        graph["neighbor_mask"],
    )
    neighbor_nan_ratio = float(np.mean(~np.isfinite(neighbor_mean_rel0))) if neighbor_mean_rel0.size else 1.0
    graph["graph_stats"]["neighbor_nan_ratio"] = neighbor_nan_ratio

    day_gaps = np.diff(day_offsets, prepend=day_offsets[0]).astype(np.float32)
    seasonal_sin = np.sin(2.0 * np.pi * (doy.astype(np.float32) / 365.25)).astype(np.float32)
    seasonal_cos = np.cos(2.0 * np.pi * (doy.astype(np.float32) / 365.25)).astype(np.float32)
    sequence_array = np.stack(
        [
            point_rel0,
            np.nan_to_num(point_delta, nan=0.0),
            np.broadcast_to(day_gaps[None, :], point_rel0.shape),
            np.broadcast_to(seasonal_sin[None, :], point_rel0.shape),
            np.broadcast_to(seasonal_cos[None, :], point_rel0.shape),
            trend_component,
            seasonal_component,
            residual_component,
            np.nan_to_num(delta2, nan=0.0),
            np.nan_to_num(rolling_velocity, nan=0.0),
            np.nan_to_num(rolling_residual_std, nan=0.0),
            np.nan_to_num(neighbor_mean_rel0, nan=0.0),
            np.nan_to_num(neighbor_mean_delta, nan=0.0),
            np.nan_to_num(neighbor_delta_std, nan=0.0),
            np.nan_to_num(local_event_persistence, nan=0.0),
            abnormal_date_flag,
        ],
        axis=-1,
    ).astype(np.float32)

    point_window_counts = _compute_window_counts(point_valid, lookback, forecast_horizon)
    split_end_indices = _split_end_indices(len(dates), lookback, forecast_horizon)
    enough_span_mask = point_window_counts > 0
    strict_point_mask = strict_mask[rows, cols].astype(bool)
    relaxed_point_mask = relaxed_mask[rows, cols].astype(bool)
    strict_idx = np.where(strict_point_mask & enough_span_mask)[0].astype(np.int64)
    relaxed_idx = np.where(relaxed_point_mask & enough_span_mask)[0].astype(np.int64)
    full_selected_indices, full_selected_labels, full_selection_mode = _build_selection(
        strict_idx,
        relaxed_idx,
        point_window_counts,
        split_end_indices,
        min_points_for_training=min_points_for_training,
        min_train_windows=min_train_windows,
        min_val_windows=min_val_windows,
    )
    if len(full_selected_indices) == 0:
        raise RuntimeError("strict 与 relaxed 点集都不足以形成任何预测样本。")

    zone_meta = _load_zone_filter_metadata(Path(zone_mask_path).resolve() if zone_mask_path else None, strict_mask.shape)
    point_in_zone_mask = zone_meta["point_in_zone_mask"][rows, cols].astype(bool)
    point_zone_ids_raw = zone_meta["point_zone_ids_raster"][rows, cols].astype(np.int32)
    point_zone_ids = np.asarray([f"Z{int(v)}" if int(v) > 0 else "" for v in point_zone_ids_raw], dtype=object)

    selected_indices = full_selected_indices
    selected_labels = full_selected_labels
    selection_mode = full_selection_mode
    zone_filter_mode = "all_high_confidence"
    zone_filter_fallback_triggered = bool(zone_meta["zone_filter_fallback_triggered"])
    zone_filter_fallback_reason = str(zone_meta["zone_filter_fallback_reason"])
    zone_ids_used: list[str] = []
    forecast_point_scope = _normalize_forecast_point_scope(forecast_point_scope)

    if zone_mask_path is not None and forecast_point_scope == "zone_high_confidence_only":
        zone_strict_idx = np.where(strict_point_mask & enough_span_mask & point_in_zone_mask)[0].astype(np.int64)
        zone_relaxed_idx = np.where(relaxed_point_mask & enough_span_mask & point_in_zone_mask)[0].astype(np.int64)
        zone_selected_indices, zone_selected_labels, zone_selection_mode = _build_selection(
            zone_strict_idx,
            zone_relaxed_idx,
            point_window_counts,
            split_end_indices,
            min_points_for_training=min_points_for_training,
            min_train_windows=min_train_windows,
            min_val_windows=min_val_windows,
        )
        if len(zone_selected_indices) > 0 and zone_selection_mode != "fallback_baseline_only":
            selected_indices = zone_selected_indices
            selected_labels = zone_selected_labels
            selection_mode = zone_selection_mode
            zone_filter_mode = "zone_high_confidence_only"
            zone_filter_fallback_triggered = False
            zone_filter_fallback_reason = ""
            zone_ids_used = sorted({str(point_zone_ids[int(idx)]) for idx in selected_indices if str(point_zone_ids[int(idx)])})
        else:
            zone_filter_fallback_triggered = True
            if not point_in_zone_mask.any():
                zone_filter_fallback_reason = "zone_mask_empty"
            elif len(zone_strict_idx) + len(zone_relaxed_idx) == 0:
                zone_filter_fallback_reason = "zone_points_without_complete_windows"
            else:
                zone_filter_fallback_reason = "zone_points_insufficient_for_training"
    elif zone_mask_path is not None:
        zone_filter_mode = forecast_point_scope

    if not zone_ids_used:
        zone_ids_used = sorted({str(point_zone_ids[int(idx)]) for idx in selected_indices if str(point_zone_ids[int(idx)]) and point_in_zone_mask[int(idx)]})

    static_features = np.column_stack(
        [
            ps_score_points,
            tcoh[rows, cols].astype(np.float32),
            metrics["valid_pair_ratio"][rows, cols].astype(np.float32),
            metrics["mainCC_ratio"][rows, cols].astype(np.float32),
            metrics["jump_risk"][rows, cols].astype(np.float32),
            anomaly_exposure[rows, cols].astype(np.float32),
            velocity_points,
            height_points,
        ]
    ).astype(np.float32)

    hazard_point_mask = enough_span_mask & graph["neighbor_mask"].any(axis=1)
    hazard_indices = selected_indices[np.isin(selected_indices, np.where(hazard_point_mask)[0])]
    generic_train_est, generic_val_est, _ = _estimate_total_windows(point_window_counts, selected_indices, split_end_indices)
    hazard_train_est, hazard_val_est, _ = _estimate_total_windows(point_window_counts, hazard_indices, split_end_indices)

    fallback_reasons: list[str] = []
    if graph["graph_stats"]["effective_nodes"] < max(64, 4 * int(graph_k)):
        fallback_reasons.append("effective_nodes_below_threshold")
    if float(graph["graph_stats"]["mean_degree"]) < 3.0:
        fallback_reasons.append("mean_degree_below_threshold")
    if float(graph["graph_stats"]["largest_component_ratio"]) < 0.60:
        fallback_reasons.append("largest_component_too_small")
    if hazard_train_est < int(round(generic_train_est * 0.5)):
        fallback_reasons.append("hazard_train_samples_too_small")
    if neighbor_nan_ratio > 0.20:
        fallback_reasons.append("neighbor_feature_nan_ratio_too_high")
    graph["graph_stats"]["hazard_train_windows_est"] = int(hazard_train_est)
    graph["graph_stats"]["hazard_val_windows_est"] = int(hazard_val_est)
    graph["graph_stats"]["generic_train_windows_est"] = int(generic_train_est)
    graph["graph_stats"]["generic_val_windows_est"] = int(generic_val_est)
    graph["graph_stats"]["fallback_reasons"] = list(fallback_reasons)

    accepted_pass = "unknown"
    feedback_json = qc_report_dir / "mintpy_feedback_roundtrip.json"
    if feedback_json.exists():
        try:
            accepted_pass = json.loads(feedback_json.read_text(encoding="utf-8")).get("accepted_pass", "unknown")
        except Exception:
            accepted_pass = "unknown"

    n_train, n_val, n_test = _estimate_total_windows(point_window_counts, selected_indices, split_end_indices)
    data_summary = {
        "mintpy_dir": str(mintpy_dir),
        "qc_report_dir": str(qc_report_dir),
        "final_timeseries_path": str(ts_path),
        "accepted_pass": accepted_pass,
        "selection_mode": selection_mode,
        "zone_detection_status": str(zone_meta["zone_detection_status"]),
        "zone_filter_mode": zone_filter_mode,
        "zone_mask_path": str(zone_meta["zone_mask_path"]),
        "n_detected_zones": int(zone_meta["n_detected_zones"]),
        "n_zone_points_used": int(np.sum(point_in_zone_mask[selected_indices])) if len(selected_indices) else 0,
        "zone_ids_used": list(zone_ids_used),
        "zone_filter_fallback_triggered": bool(zone_filter_fallback_triggered),
        "zone_filter_fallback_reason": zone_filter_fallback_reason,
        "n_points_used": int(len(selected_indices)),
        "n_strict_points": int(len(strict_idx)),
        "n_relaxed_points": int(len(relaxed_idx)),
        "n_hazard_points": int(len(hazard_indices)),
        "n_samples_train": int(n_train),
        "n_samples_val": int(n_val),
        "n_samples_test": int(n_test),
        "lookback": int(lookback),
        "forecast_horizon": int(forecast_horizon),
        "date_range": {"start": dates[0], "end": dates[-1], "n_dates": len(dates)},
        "time_split": {
            "train_window_end_dates": [dates[i] for i in split_end_indices["train"]],
            "val_window_end_dates": [dates[i] for i in split_end_indices["val"]],
            "test_window_end_dates": [dates[i] for i in split_end_indices["test"]],
        },
        "abnormal_date_ratio": float(np.mean(abnormal_flags)) if len(abnormal_flags) else 0.0,
        "abnormal_dates": [date for date, flag in zip(dates, abnormal_flags) if flag],
        "sequence_feature_names": list(SEQUENCE_FEATURE_NAMES),
        "channel_groups": {k: list(v) for k, v in CHANNEL_GROUPS.items()},
        "static_feature_names": list(STATIC_FEATURE_NAMES),
        "static_feature_summary": _feature_summary(STATIC_FEATURE_NAMES, static_features[selected_indices]),
        "decomposition_method": decomposition,
        "graph_stats": graph["graph_stats"],
    }

    points = {
        "row": rows.astype(np.int32),
        "col": cols.astype(np.int32),
        "longitude": geom["longitude"][rows, cols].astype(np.float32),
        "latitude": geom["latitude"][rows, cols].astype(np.float32),
        "height_m": height_points,
        "velocity_mm_yr": velocity_points,
        "velocity_std_mm_yr": vstd[rows, cols].astype(np.float32),
        "ps_score": ps_score_points,
        "tcoh": tcoh[rows, cols].astype(np.float32),
        "valid_pair_ratio": metrics["valid_pair_ratio"][rows, cols].astype(np.float32),
        "mainCC_ratio": metrics["mainCC_ratio"][rows, cols].astype(np.float32),
        "model_rms": metrics["model_rms"][rows, cols].astype(np.float32),
        "jump_risk": metrics["jump_risk"][rows, cols].astype(np.float32),
        "anomaly_exposure": anomaly_exposure[rows, cols].astype(np.float32),
        "local_velocity_gradient": local_velocity_gradient[rows, cols].astype(np.float32),
    }

    return ForecastContext(
        mintpy_dir=mintpy_dir,
        qc_report_dir=qc_report_dir,
        accepted_pass=accepted_pass,
        final_timeseries_path=ts_path,
        dates=dates,
        day_offsets=day_offsets,
        doy=doy,
        rel0_mm=point_rel0,
        delta_mm=point_delta,
        valid_mask=point_valid.astype(bool),
        selected_indices=selected_indices.astype(np.int64),
        selected_labels=selected_labels.astype(object),
        selection_mode=selection_mode,
        zone_filter_mode=zone_filter_mode,
        zone_detection_status=str(zone_meta["zone_detection_status"]),
        zone_mask_path=str(zone_meta["zone_mask_path"]),
        zone_ids_used=list(zone_ids_used),
        n_detected_zones=int(zone_meta["n_detected_zones"]),
        zone_filter_fallback_triggered=bool(zone_filter_fallback_triggered),
        zone_filter_fallback_reason=zone_filter_fallback_reason,
        point_zone_ids=point_zone_ids.astype(object),
        point_in_zone_mask=point_in_zone_mask.astype(bool),
        hazard_indices=hazard_indices.astype(np.int64),
        hazard_available=len(fallback_reasons) == 0 and len(hazard_indices) > 0,
        points=points,
        static_feature_names=list(STATIC_FEATURE_NAMES),
        static_features=static_features,
        sequence_feature_names=list(SEQUENCE_FEATURE_NAMES),
        channel_groups={k: list(v) for k, v in CHANNEL_GROUPS.items()},
        sequence_array=sequence_array,
        anomaly_exposure=points["anomaly_exposure"],
        window_count_per_point=point_window_counts.astype(np.int32),
        split_end_indices=split_end_indices,
        graph=graph,
        decomposition_method=decomposition,
        data_summary=data_summary,
    )


def _sample_window_neighbors(
    ctx: ForecastContext,
    point_idx: int,
    hist_slice: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    neighbor_index = ctx.graph["neighbor_index"][point_idx]
    neighbor_mask = ctx.graph["neighbor_mask"][point_idx]
    edge_features = ctx.graph["edge_features"][point_idx].astype(np.float32)
    k = neighbor_index.shape[0]
    hist_len = hist_slice.stop - hist_slice.start
    seq_channels = ctx.sequence_array.shape[-1]
    static_dim = ctx.static_features.shape[1]

    out_seq = np.zeros((hist_len, k, seq_channels), dtype=np.float32)
    out_static = np.zeros((k, static_dim), dtype=np.float32)
    out_mask = neighbor_mask.astype(bool).copy()
    if not out_mask.any():
        return out_seq, out_static, np.zeros_like(edge_features), out_mask

    safe_idx = np.clip(neighbor_index, 0, max(len(ctx.static_features) - 1, 0))
    valid_window = ctx.valid_mask[safe_idx, hist_slice].all(axis=1)
    out_mask &= valid_window
    if not out_mask.any():
        return out_seq, out_static, np.zeros_like(edge_features), out_mask

    out_seq[:, out_mask, :] = np.transpose(ctx.sequence_array[safe_idx[out_mask], hist_slice, :], (1, 0, 2))
    out_static[out_mask] = ctx.static_features[safe_idx[out_mask]]
    edge = np.zeros_like(edge_features, dtype=np.float32)
    edge[out_mask] = edge_features[out_mask]
    return out_seq, out_static, edge, out_mask


def build_window_samples(
    ctx: ForecastContext,
    point_indices: np.ndarray,
    *,
    end_indices: np.ndarray,
    lookback: int,
    forecast_horizon: int,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    seq_list: list[np.ndarray] = []
    static_list: list[np.ndarray] = []
    target_offset_list: list[np.ndarray] = []
    target_rel0_list: list[np.ndarray] = []
    point_list: list[int] = []
    label_list: list[str] = []
    end_list: list[int] = []
    last_rel0_list: list[float] = []
    target_abnormal_list: list[bool] = []
    neighbor_seq_list: list[np.ndarray] = []
    neighbor_static_list: list[np.ndarray] = []
    edge_feature_list: list[np.ndarray] = []
    neighbor_mask_list: list[np.ndarray] = []

    label_map = {int(idx): str(label) for idx, label in zip(ctx.selected_indices, ctx.selected_labels)}

    for point_idx in point_indices:
        mask = ctx.valid_mask[point_idx]
        for end in end_indices:
            end = int(end)
            start = end - lookback + 1
            target_slice = slice(end + 1, end + 1 + forecast_horizon)
            hist_slice = slice(start, end + 1)
            if forecast_horizon > 0 and not mask[start : end + 1 + forecast_horizon].all():
                continue
            if forecast_horizon == 0 and not mask[hist_slice].all():
                continue
            seq = ctx.sequence_array[point_idx, hist_slice, :].astype(np.float32)
            last_rel0 = float(ctx.rel0_mm[point_idx, end])
            if forecast_horizon > 0:
                target_rel0 = ctx.rel0_mm[point_idx, target_slice].astype(np.float32)
                target_offset = (target_rel0 - last_rel0).astype(np.float32)
                target_abnormal = bool(np.any(np.asarray(ctx.sequence_array[point_idx, target_slice, CHANNEL_GROUPS["event"][-1]], dtype=np.float32) > 0.5))
            else:
                target_rel0 = np.empty((0,), dtype=np.float32)
                target_offset = np.empty((0,), dtype=np.float32)
                target_abnormal = False
            neighbor_seq, neighbor_static, edge_features, neighbor_window_mask = _sample_window_neighbors(ctx, int(point_idx), hist_slice)

            seq_list.append(seq)
            static_list.append(ctx.static_features[point_idx].astype(np.float32))
            target_offset_list.append(target_offset)
            target_rel0_list.append(target_rel0)
            point_list.append(int(point_idx))
            label_list.append(label_map.get(int(point_idx), "unknown"))
            end_list.append(int(end))
            last_rel0_list.append(last_rel0)
            target_abnormal_list.append(target_abnormal)
            neighbor_seq_list.append(neighbor_seq)
            neighbor_static_list.append(neighbor_static)
            edge_feature_list.append(edge_features.astype(np.float32))
            neighbor_mask_list.append(neighbor_window_mask.astype(bool))

    if not seq_list:
        k = ctx.graph["neighbor_index"].shape[1] if ctx.graph["neighbor_index"].ndim == 2 else 0
        return {
            "seq": np.empty((0, lookback, len(ctx.sequence_feature_names)), dtype=np.float32),
            "static": np.empty((0, len(ctx.static_feature_names)), dtype=np.float32),
            "target_offset": np.empty((0, forecast_horizon), dtype=np.float32),
            "target_rel0": np.empty((0, forecast_horizon), dtype=np.float32),
            "point_index": np.empty((0,), dtype=np.int64),
            "strict_or_relaxed": np.empty((0,), dtype=object),
            "window_end": np.empty((0,), dtype=np.int64),
            "last_rel0": np.empty((0,), dtype=np.float32),
            "target_abnormal_flag": np.empty((0,), dtype=bool),
            "neighbor_seq": np.empty((0, lookback, k, len(ctx.sequence_feature_names)), dtype=np.float32),
            "neighbor_static": np.empty((0, k, len(ctx.static_feature_names)), dtype=np.float32),
            "edge_features": np.empty((0, k, 4), dtype=np.float32),
            "neighbor_mask": np.empty((0, k), dtype=bool),
        }

    seq_arr = np.stack(seq_list).astype(np.float32)
    static_arr = np.stack(static_list).astype(np.float32)
    target_offset = np.stack(target_offset_list).astype(np.float32)
    target_rel0 = np.stack(target_rel0_list).astype(np.float32)
    point_arr = np.asarray(point_list, dtype=np.int64)
    label_arr = np.asarray(label_list, dtype=object)
    end_arr = np.asarray(end_list, dtype=np.int64)
    last_rel0 = np.asarray(last_rel0_list, dtype=np.float32)
    target_abnormal_flag = np.asarray(target_abnormal_list, dtype=bool)
    neighbor_seq = np.stack(neighbor_seq_list).astype(np.float32)
    neighbor_static = np.stack(neighbor_static_list).astype(np.float32)
    edge_features = np.stack(edge_feature_list).astype(np.float32)
    neighbor_mask = np.stack(neighbor_mask_list).astype(bool)

    if max_samples is not None and len(seq_arr) > max_samples:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(np.arange(len(seq_arr)), size=max_samples, replace=False))
        seq_arr = seq_arr[keep]
        static_arr = static_arr[keep]
        target_offset = target_offset[keep]
        target_rel0 = target_rel0[keep]
        point_arr = point_arr[keep]
        label_arr = label_arr[keep]
        end_arr = end_arr[keep]
        last_rel0 = last_rel0[keep]
        target_abnormal_flag = target_abnormal_flag[keep]
        neighbor_seq = neighbor_seq[keep]
        neighbor_static = neighbor_static[keep]
        edge_features = edge_features[keep]
        neighbor_mask = neighbor_mask[keep]

    return {
        "seq": seq_arr,
        "static": static_arr,
        "target_offset": target_offset,
        "target_rel0": target_rel0,
        "point_index": point_arr,
        "strict_or_relaxed": label_arr,
        "window_end": end_arr,
        "last_rel0": last_rel0,
        "target_abnormal_flag": target_abnormal_flag,
        "neighbor_seq": neighbor_seq,
        "neighbor_static": neighbor_static,
        "edge_features": edge_features,
        "neighbor_mask": neighbor_mask,
    }


def build_inference_windows(ctx: ForecastContext, *, lookback: int) -> dict[str, np.ndarray]:
    point_indices = ctx.selected_indices
    end_index = len(ctx.dates) - 1
    if end_index < lookback - 1:
        raise RuntimeError("MintPy 时序长度不足以构建预测窗口。")
    return build_window_samples(
        ctx,
        point_indices,
        end_indices=np.asarray([end_index], dtype=np.int32),
        lookback=lookback,
        forecast_horizon=0,
        max_samples=None,
    )


def build_latest_windows(ctx: ForecastContext, *, lookback: int) -> dict[str, np.ndarray]:
    return build_inference_windows(ctx, lookback=lookback)


def fit_forecast_normalizer(train_samples: dict[str, np.ndarray]) -> dict[str, Any]:
    seq = train_samples["seq"]
    static = train_samples["static"]
    target = train_samples["target_offset"]
    neighbor_seq = train_samples["neighbor_seq"]
    neighbor_static = train_samples["neighbor_static"]
    edge_features = train_samples["edge_features"]

    seq_mean = np.nanmean(seq, axis=(0, 1)).astype(np.float32)
    seq_std = np.nanstd(seq, axis=(0, 1)).astype(np.float32)
    seq_std = np.where(seq_std < 1e-6, 1.0, seq_std).astype(np.float32)
    static_mean = np.nanmean(static, axis=0).astype(np.float32)
    static_std = np.nanstd(static, axis=0).astype(np.float32)
    static_std = np.where(static_std < 1e-6, 1.0, static_std).astype(np.float32)
    edge_mean = np.nanmean(edge_features, axis=(0, 1)).astype(np.float32) if edge_features.size else np.zeros((4,), dtype=np.float32)
    edge_std = np.nanstd(edge_features, axis=(0, 1)).astype(np.float32) if edge_features.size else np.ones((4,), dtype=np.float32)
    edge_std = np.where(edge_std < 1e-6, 1.0, edge_std).astype(np.float32)
    target_scale = max(float(np.nanpercentile(np.abs(target[np.isfinite(target)]), 95)) if np.isfinite(target).any() else 1.0, 1.0)

    return {
        "seq_mean": seq_mean,
        "seq_std": seq_std,
        "static_mean": static_mean,
        "static_std": static_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "target_scale": float(target_scale),
        "neighbor_seq_uses_seq_stats": True,
        "neighbor_static_uses_static_stats": True,
    }


def apply_forecast_normalizer(samples: dict[str, np.ndarray], normalizer: dict[str, Any]) -> dict[str, np.ndarray]:
    seq_norm = np.nan_to_num(
        (samples["seq"] - normalizer["seq_mean"]) / normalizer["seq_std"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    static_norm = np.nan_to_num(
        (samples["static"] - normalizer["static_mean"]) / normalizer["static_std"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    target = samples.get("target_offset", np.empty((samples["seq"].shape[0], 0), dtype=np.float32))
    target_norm = np.nan_to_num(target / float(normalizer["target_scale"]), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    neighbor_seq_norm = np.nan_to_num(
        (samples["neighbor_seq"] - normalizer["seq_mean"]) / normalizer["seq_std"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    neighbor_static_norm = np.nan_to_num(
        (samples["neighbor_static"] - normalizer["static_mean"]) / normalizer["static_std"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    edge_features_norm = np.nan_to_num(
        (samples["edge_features"] - normalizer["edge_mean"]) / normalizer["edge_std"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    return {
        **samples,
        "seq_norm": seq_norm,
        "static_norm": static_norm,
        "target_norm": target_norm,
        "neighbor_seq_norm": neighbor_seq_norm,
        "neighbor_static_norm": neighbor_static_norm,
        "edge_features_norm": edge_features_norm,
    }
