"""Automatic deformation-zone detection before forecasting."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import rasterio
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
from rasterio.features import shapes
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    binary_propagation,
    convolve,
    distance_transform_edt,
    label,
)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg
from .config import logger
from .viz import (
    _CMAP_VEL,
    _add_uncertainty_legend,
    _add_north_arrow,
    _add_water_legend,
    _draw_hillshade_background,
    _draw_lowq_hatch,
    _draw_water_overlay,
    _format_degree_axis,
    _load_mintpy_data,
    _prepare_geo_canvas,
    _save_figure,
    _subfig_label,
    _symmetric_vlim,
)


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

FEATURE_NAMES = [
    "velocity_mm_yr",
    "abs_velocity_mm_yr",
    "velocity_sign",
    "local_velocity_mean_11x11",
    "local_velocity_anomaly",
    "abs_local_velocity_anomaly",
    "local_velocity_gradient",
    "local_velocity_std_5x5",
    "tcoh",
    "ps_score",
    "valid_pair_ratio",
    "mainCC_ratio",
    "jump_risk",
    "anomaly_exposure",
    "strict_flag",
    "relaxed_flag",
]

PROBABILITY_THRESHOLD = 0.55
MIN_ZONE_PIXELS = 25
MIN_ZONE_AREA_KM2 = 0.01
MIN_ZONE_SUPPORT_PIXELS = 32
MIN_ZONE_AREA_FRACTION_AOI = 1.2e-4
MIN_ZONE_BBOX_FILL_RATIO = 0.12
MIN_ZONE_SUPPORT_RATIO = 0.15
MIN_ZONE_SIGN_COHERENCE = 0.75
MIN_ZONE_MEDIAN_ABS_VELOCITY_MM_YR = 8.0
MIN_ZONE_TEMPORAL_NET_DISP_MM = 10.0
MIN_ZONE_TEMPORAL_CONTINUITY = 0.55
MIN_ZONE_TIMESERIES_POINTS = 32
MAX_ZONE_CURVES = 6
MAX_PRINCIPAL_ZONES = 1
SUPPORT_GRAPH_K = 12
SUPPORT_GRAPH_EDGE_CORR = 0.58
SUPPORT_GRAPH_EDGE_RADIUS_MULT = 4.5
SUPPORT_GRAPH_EDGE_RADIUS_FLOOR_KM = 0.10
SUPPORT_GRAPH_CONCAVE_RATIO = 0.22
SUPPORT_GRAPH_COMPACTNESS_RATIO = 0.10
SUPPORT_GRAPH_ACTIVE_SALIENCY_FLOOR = 0.50
SUPPORT_GRAPH_CORE_SALIENCY_FLOOR = 0.62
SUPPORT_GRAPH_ACTIVE_COHERENCE_MIN = 0.48
SUPPORT_GRAPH_CORE_COHERENCE_MIN = 0.55
SUPPORT_GRAPH_QC_SUPPORT_MIN = 0.50
SUPPORT_GRAPH_AREA_FRACTION_MIN = 8.0e-4
SUPPORT_GRAPH_POINT_RATIO_MIN = 0.003
SUPPORT_GRAPH_INTERNAL_TRACE_CORR_MIN = 0.60
SUPPORT_GRAPH_TEMPORAL_COHERENCE_MIN = 0.55
SUPPORT_GRAPH_COMPACTNESS_MIN = 0.20
SUPPORT_GRAPH_REGION_SCORE_RATIO_MIN = 0.35
SUPPORT_GRAPH_SALIENT_POINT_RATIO_MIN = 0.0015
SUPPORT_GRAPH_SALIENT_MIN_POINTS = 96
SUPPORT_GRAPH_SALIENT_REGION_SCORE_RATIO = 0.75
SUPPORT_GRAPH_SALIENT_TEMPORAL_COHERENCE_MIN = 0.60
SUPPORT_GRAPH_SALIENT_BOUNDARY_CONTRAST_MIN = 0.25
SUPPORT_GRAPH_SALIENT_ACTIVITY_PERCENTILE = 80.0
SUPPORT_GRAPH_MERGE_TRACE_CORR_MIN = 0.95
SUPPORT_GRAPH_MERGE_DISTANCE_KM = 2.0
SUPPORT_GRAPH_MERGE_SCORE_RATIO_MIN = 0.20
SUPPORT_GRAPH_MERGE_SCORE_FLOOR = 0.002
ZONE_RASTER_GROW_CORR_MIN = 0.90
ZONE_RASTER_GROW_PEAK_MIN_MM = 10.0
ZONE_RASTER_GROW_NET_MIN_MM = 10.0
ZONE_RASTER_GROW_VEL_MIN_MM_YR = 8.0


def _pick_final_timeseries(mintpy_dir: Path) -> Path:
    for name in FINAL_TS_CANDIDATES:
        path = mintpy_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到 MintPy 时序文件: {mintpy_dir}")


def _resolve_velocity_raster(export_dir: Path) -> Path:
    candidates = [
        export_dir / "velocity.tif",
        export_dir / "raster" / "velocity.tif",
        export_dir / "velocity_high_confidence.tif",
        export_dir / "raster" / "velocity_high_confidence.tif",
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "未找到速率栅格文件。请先完成导出单元，或检查 export 目录。\n"
        f"已检查路径:\n{searched}"
    )


def _read_tif(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
        profile["transform"] = ds.transform
        profile["crs"] = ds.crs
        profile["width"] = ds.width
        profile["height"] = ds.height
    return arr, profile


def _write_tif(path: Path, array: np.ndarray, ref_profile: dict[str, Any], *, dtype: str, nodata: float | int) -> Path:
    profile = ref_profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    if path.exists():
        path.unlink()
    profile.update(
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        tiled=False,
    )
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(array.astype(dtype), 1)
    return path


def _normalize_interval(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    if not finite.any():
        return out
    p5 = float(np.nanpercentile(arr[finite], 5))
    p95 = float(np.nanpercentile(arr[finite], 95))
    if np.isclose(p5, p95):
        return out
    out[finite] = (arr[finite] - p5) / (p95 - p5)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _load_abnormal_flags(qc_report_dir: Path, dates: list[str]) -> np.ndarray:
    flags = np.zeros(len(dates), dtype=bool)
    path = qc_report_dir / "date_qc.csv"
    if not path.exists():
        return flags
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[str(row.get("date", ""))] = row
    for idx, date in enumerate(dates):
        row = rows.get(date)
        if row and str(row.get("final_abnormal_flag", "")).strip().lower() in {"1", "true", "yes"}:
            flags[idx] = True
    return flags


def _load_anomaly_exposure(mintpy_dir: Path, qc_report_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    ts_path = _pick_final_timeseries(mintpy_dir)
    with h5py.File(ts_path, "r") as f:
        raw_dates = f["date"][:]
        dates = [d.decode("utf-8") if hasattr(d, "decode") else d.tobytes().decode("utf-8") for d in raw_dates]
        abnormal_flags = _load_abnormal_flags(qc_report_dir, dates)
        abnormal_inc_idx = np.where(abnormal_flags[1:])[0] + 1
        if abnormal_inc_idx.size == 0:
            return np.zeros(shape, dtype=np.float32)
        ts_ds = f["timeseries"]
        first = np.asarray(ts_ds[0], dtype=np.float32) * 1000.0
        deltas: list[np.ndarray] = []
        for idx in abnormal_inc_idx:
            curr = np.asarray(ts_ds[int(idx)], dtype=np.float32) * 1000.0 - first
            prev = np.asarray(ts_ds[int(idx) - 1], dtype=np.float32) * 1000.0 - first
            deltas.append(np.abs(curr - prev).astype(np.float32))
        if not deltas:
            return np.zeros(shape, dtype=np.float32)
        exposure_raw = np.nanmean(np.stack(deltas, axis=0), axis=0).astype(np.float32)
        return _normalize_interval(exposure_raw)


def _load_rel0_timeseries(mintpy_dir: Path) -> tuple[list[str], np.ndarray]:
    ts_path = _pick_final_timeseries(mintpy_dir)
    with h5py.File(ts_path, "r") as f:
        raw_dates = f["date"][:]
        dates = [d.decode("utf-8") if hasattr(d, "decode") else d.tobytes().decode("utf-8") for d in raw_dates]
        ts = np.asarray(f["timeseries"][:], dtype=np.float32) * 1000.0
    rel0 = ts - ts[0:1]
    return dates, np.moveaxis(rel0, 0, -1).astype(np.float32)


def _load_metrics(qc_report_dir: Path) -> dict[str, np.ndarray]:
    metrics_path = qc_report_dir / "ps_model_metrics.h5"
    with h5py.File(metrics_path, "r") as f:
        return {
            "valid_pair_ratio": np.asarray(f["valid_pair_ratio"][:], dtype=np.float32),
            "mainCC_ratio": np.asarray(f["mainCC_ratio"][:], dtype=np.float32),
            "jump_risk": np.asarray(f["jump_risk"][:], dtype=np.float32),
        }


def _local_std_5x5(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), dtype=np.float32)
    value0 = np.where(valid_mask, values, 0.0).astype(np.float32)
    sum1 = convolve(value0, kernel, mode="nearest")
    sum2 = convolve(value0 * value0, kernel, mode="nearest")
    count = convolve(valid_mask.astype(np.float32), kernel, mode="nearest")
    mean = np.divide(sum1, count, out=np.zeros_like(sum1), where=count > 0)
    var = np.divide(sum2, count, out=np.zeros_like(sum2), where=count > 0) - mean * mean
    std = np.sqrt(np.clip(var, 0.0, None)).astype(np.float32)
    std[~valid_mask] = np.nan
    return std


def _local_mean_window(values: np.ndarray, valid_mask: np.ndarray, size: int = 11) -> np.ndarray:
    kernel = np.ones((size, size), dtype=np.float32)
    value0 = np.where(valid_mask, values, 0.0).astype(np.float32)
    sum1 = convolve(value0, kernel, mode="nearest")
    count = convolve(valid_mask.astype(np.float32), kernel, mode="nearest")
    mean = np.divide(sum1, count, out=np.zeros_like(sum1), where=count > 0)
    mean[~valid_mask] = np.nan
    return mean.astype(np.float32)


def _local_gradient(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    filled = np.where(valid_mask, values, np.nanmedian(values[valid_mask]) if np.any(valid_mask) else 0.0).astype(np.float32)
    gy, gx = np.gradient(filled)
    grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    grad[~valid_mask] = np.nan
    return grad


def _pixel_area_km2(profile: dict[str, Any], latitude: np.ndarray) -> np.ndarray:
    transform = profile.get("transform")
    crs = profile.get("crs")
    if crs is not None and getattr(crs, "is_projected", False):
        xres = abs(float(transform.a))
        yres = abs(float(transform.e))
        return np.full_like(latitude, (xres * yres) / 1_000_000.0, dtype=np.float32)
    xdeg = abs(float(transform.a)) if transform is not None else 1.0
    ydeg = abs(float(transform.e)) if transform is not None else 1.0
    dx_km = xdeg * 111.320 * np.cos(np.radians(latitude))
    dy_km = ydeg * 110.540
    return (dx_km * dy_km).astype(np.float32)


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    if not finite.any():
        return out
    med = float(np.nanmedian(arr[finite]))
    mad = float(np.nanmedian(np.abs(arr[finite] - med)))
    scale = max(mad * 1.4826, 1.0e-3)
    out[finite] = (arr[finite] - med) / scale
    return np.clip(out, -5.0, 8.0).astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-arr.astype(np.float64)))).astype(np.float32)


def _smooth_series_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return arr.astype(np.float32)
    padded = np.pad(arr, ((0, 0), (1, 1)), mode="edge")
    return (
        0.25 * padded[:, :-2]
        + 0.50 * padded[:, 1:-1]
        + 0.25 * padded[:, 2:]
    ).astype(np.float32)


def _trimmed_mean_axis0(values: np.ndarray, trim_fraction: float = 0.20) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros(arr.shape[1] if arr.ndim == 2 else 0, dtype=np.float32)
    out = np.full(arr.shape[1], np.nan, dtype=np.float32)
    for idx in range(arr.shape[1]):
        col = arr[:, idx]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        vals = np.sort(col[finite].astype(np.float32))
        trim_n = int(np.floor(trim_fraction * vals.size))
        if trim_n * 2 >= vals.size:
            trim_n = 0
        core = vals[trim_n: vals.size - trim_n] if trim_n > 0 else vals
        if core.size > 0:
            out[idx] = float(np.mean(core))
    return out


def _date_offsets_days(dates: list[str]) -> np.ndarray:
    dt = np.asarray([datetime.strptime(str(d), "%Y%m%d") for d in dates], dtype=object)
    start = dt[0]
    return np.asarray([(d - start).days for d in dt], dtype=np.float32)


def _fit_linear_trend_mm_yr(series: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float32)
    t = np.asarray(day_offsets, dtype=np.float32) / 365.25
    if arr.ndim != 2 or arr.shape[1] != t.size:
        return np.zeros(arr.shape[0] if arr.ndim == 2 else 0, dtype=np.float32)
    t_center = t - float(np.mean(t))
    t_ss = float(np.sum(t_center * t_center))
    if t_ss <= 1.0e-6:
        return np.zeros(arr.shape[0], dtype=np.float32)
    y_mean = np.nanmean(arr, axis=1, keepdims=True)
    num = np.nansum((arr - y_mean) * t_center[None, :], axis=1)
    return (num / t_ss).astype(np.float32)


def _project_lonlat_km(longitude: np.ndarray, latitude: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    lat_c = float(np.nanmean(latitude)) if np.size(latitude) else 0.0
    x_km = longitude.astype(np.float32) * (111.320 * np.cos(np.radians(lat_c)))
    y_km = latitude.astype(np.float32) * 110.540
    return x_km.astype(np.float32), y_km.astype(np.float32), lat_c


def _support_graph_sign(net_disp: np.ndarray, trend_mm_yr: np.ndarray) -> np.ndarray:
    sign = np.sign(np.asarray(net_disp, dtype=np.float32)).astype(np.int8)
    weak = np.abs(net_disp) < 1.0
    sign[weak] = np.sign(trend_mm_yr[weak]).astype(np.int8)
    return sign


def _series_profile(zone_series: np.ndarray) -> dict[str, Any] | None:
    arr = np.asarray(zone_series, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < MIN_ZONE_TIMESERIES_POINTS:
        return None
    p25 = np.nanpercentile(arr, 25, axis=0).astype(np.float32)
    p50 = np.nanmedian(arr, axis=0).astype(np.float32)
    p75 = np.nanpercentile(arr, 75, axis=0).astype(np.float32)
    center = _trimmed_mean_axis0(arr, trim_fraction=0.20).astype(np.float32)
    finite = np.isfinite(center)
    if int(np.sum(finite)) < max(6, min(8, center.size)):
        return None
    center_filled = center.copy()
    if not np.all(finite):
        keep = np.flatnonzero(finite)
        miss = np.flatnonzero(~finite)
        center_filled[miss] = np.interp(miss, keep, center[finite]).astype(np.float32)
    center_smooth = _smooth_series(center_filled)
    diffs = np.diff(center_filled).astype(np.float32)
    meaningful = np.abs(diffs) > 0.25
    diffs_use = diffs[meaningful]
    net_disp = float(center_filled[-1] - center_filled[0])
    if diffs_use.size == 0:
        path_length = abs(net_disp)
        sign_consistency = 1.0 if not np.isclose(net_disp, 0.0) else 0.0
    else:
        path_length = float(np.sum(np.abs(diffs_use)))
        if np.isclose(net_disp, 0.0):
            sign_consistency = 0.0
        else:
            sign_consistency = float(np.mean(np.sign(diffs_use) == np.sign(net_disp)))
    net_to_path = float(abs(net_disp) / max(path_length, 1.0e-3))
    continuity = 0.5 * net_to_path + 0.5 * sign_consistency
    counts = np.sum(np.isfinite(arr), axis=0).astype(np.int32)
    return {
        "timeseries_p25_mm": p25,
        "timeseries_p50_mm": p50,
        "timeseries_p75_mm": p75,
        "timeseries_center_mm": center_filled.astype(np.float32),
        "timeseries_center_smooth_mm": center_smooth.astype(np.float32),
        "timeseries_point_count_by_date": counts,
        "timeseries_support_points": int(arr.shape[0]),
        "temporal_net_disp_mm": net_disp,
        "temporal_peak_to_peak_mm": float(np.nanmax(center_filled) - np.nanmin(center_filled)),
        "temporal_path_length_mm": path_length,
        "temporal_sign_consistency": sign_consistency,
        "temporal_net_to_path_ratio": net_to_path,
        "temporal_continuity_score": float(continuity),
    }


def _build_support_graph_context(
    *,
    rel0_cube: np.ndarray,
    dates: list[str],
    latitude: np.ndarray,
    longitude: np.ndarray,
    pixel_area_km2: np.ndarray,
    strict_flag: np.ndarray,
    relaxed_flag: np.ndarray,
    tcoh: np.ndarray,
    ps_score: np.ndarray,
    valid_pair_ratio: np.ndarray,
    maincc_ratio: np.ndarray,
    velocity: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    support_mask = (
        valid_mask
        & (strict_flag | relaxed_flag)
        & np.isfinite(latitude)
        & np.isfinite(longitude)
        & np.all(np.isfinite(rel0_cube), axis=-1)
    )
    rows, cols = np.where(support_mask)
    if rows.size == 0:
        return {
            "support_mask": support_mask,
            "n_support_points": 0,
            "summary": {"graph_k": SUPPORT_GRAPH_K, "graph_median_edge_km": 0.0},
        }

    series = rel0_cube[support_mask].astype(np.float32)
    series = _smooth_series_matrix(series)
    day_offsets = _date_offsets_days(dates)
    trend_mm_yr = _fit_linear_trend_mm_yr(series, day_offsets)
    net_disp_mm = (series[:, -1] - series[:, 0]).astype(np.float32)
    peak_abs_disp_mm = np.nanmax(np.abs(series), axis=1).astype(np.float32)
    diffs = np.diff(series, axis=1).astype(np.float32)
    meaningful = np.abs(diffs) > 0.25
    path_length_mm = np.sum(np.abs(diffs) * meaningful, axis=1).astype(np.float32)
    pos_steps = np.sum((diffs > 0) & meaningful, axis=1).astype(np.float32)
    neg_steps = np.sum((diffs < 0) & meaningful, axis=1).astype(np.float32)
    total_steps = np.maximum(pos_steps + neg_steps, 1.0)
    sign_consistency = np.divide(
        np.maximum(pos_steps, neg_steps),
        total_steps,
        out=np.zeros_like(total_steps, dtype=np.float32),
        where=total_steps > 0,
    ).astype(np.float32)
    net_to_path_ratio = np.divide(
        np.abs(net_disp_mm),
        np.maximum(path_length_mm, 1.0e-3),
        out=np.zeros_like(net_disp_mm, dtype=np.float32),
        where=path_length_mm > 0,
    ).astype(np.float32)
    temporal_coherence = np.clip(0.5 * net_to_path_ratio + 0.5 * sign_consistency, 0.0, 1.0).astype(np.float32)

    demeaned = (series - np.nanmean(series, axis=1, keepdims=True)).astype(np.float32)
    norms = np.linalg.norm(demeaned, axis=1).astype(np.float32)
    series_unit = np.divide(
        demeaned,
        np.maximum(norms[:, None], 1.0e-6),
        out=np.zeros_like(demeaned, dtype=np.float32),
        where=norms[:, None] > 1.0e-6,
    ).astype(np.float32)

    n_components = int(min(3, series.shape[1], max(series.shape[0], 1)))
    if n_components > 0:
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        pcs = pca.fit_transform(demeaned).astype(np.float32)
        pca_evr = [float(v) for v in pca.explained_variance_ratio_]
    else:
        pcs = np.zeros((series.shape[0], 0), dtype=np.float32)
        pca_evr = []
    pc1 = pcs[:, 0] if pcs.shape[1] > 0 else np.zeros(series.shape[0], dtype=np.float32)
    pc2 = pcs[:, 1] if pcs.shape[1] > 1 else np.zeros(series.shape[0], dtype=np.float32)
    pc3 = pcs[:, 2] if pcs.shape[1] > 2 else np.zeros(series.shape[0], dtype=np.float32)

    lon_support = longitude[support_mask].astype(np.float32)
    lat_support = latitude[support_mask].astype(np.float32)
    x_km, y_km, lat_c = _project_lonlat_km(lon_support, lat_support)
    coords_km = np.column_stack([x_km, y_km]).astype(np.float32)
    k_query = int(min(SUPPORT_GRAPH_K + 1, max(series.shape[0], 1)))
    if k_query <= 1:
        neighbor_idx = np.zeros((series.shape[0], 0), dtype=np.int32)
        neighbor_dist_km = np.zeros((series.shape[0], 0), dtype=np.float32)
        median_edge_km = 0.0
    else:
        tree = cKDTree(coords_km)
        dists, nbrs = tree.query(coords_km, k=k_query)
        if dists.ndim == 1:
            dists = dists[:, None]
            nbrs = nbrs[:, None]
        neighbor_idx = nbrs[:, 1:].astype(np.int32)
        neighbor_dist_km = dists[:, 1:].astype(np.float32)
        median_edge_km = float(np.nanmedian(neighbor_dist_km[:, 0])) if neighbor_dist_km.size else 0.0

    if neighbor_idx.size > 0:
        neighbor_mean_series = np.zeros_like(series, dtype=np.float32)
        local_net_mean = np.zeros(series.shape[0], dtype=np.float32)
        local_peak_mean = np.zeros(series.shape[0], dtype=np.float32)
        for col in range(neighbor_idx.shape[1]):
            nbr = neighbor_idx[:, col]
            neighbor_mean_series += series[nbr]
            local_net_mean += net_disp_mm[nbr]
            local_peak_mean += peak_abs_disp_mm[nbr]
        neighbor_mean_series /= float(neighbor_idx.shape[1])
        local_net_mean /= float(neighbor_idx.shape[1])
        local_peak_mean /= float(neighbor_idx.shape[1])
        neighbor_mean_demeaned = neighbor_mean_series - np.mean(neighbor_mean_series, axis=1, keepdims=True)
        neighbor_norm = np.linalg.norm(neighbor_mean_demeaned, axis=1).astype(np.float32)
        neighbor_mean_unit = np.divide(
            neighbor_mean_demeaned,
            np.maximum(neighbor_norm[:, None], 1.0e-6),
            out=np.zeros_like(neighbor_mean_demeaned, dtype=np.float32),
            where=neighbor_norm[:, None] > 1.0e-6,
        ).astype(np.float32)
        neighbor_trace_corr = np.sum(series_unit * neighbor_mean_unit, axis=1).astype(np.float32)
        local_contrast_net = np.abs(net_disp_mm - local_net_mean).astype(np.float32)
        local_contrast_peak = np.abs(peak_abs_disp_mm - local_peak_mean).astype(np.float32)
        density_raw = np.divide(
            max(median_edge_km, 1.0e-3),
            np.maximum(np.mean(neighbor_dist_km[:, : min(4, neighbor_dist_km.shape[1])], axis=1), 1.0e-3),
            out=np.zeros(series.shape[0], dtype=np.float32),
            where=True,
        ).astype(np.float32)
    else:
        neighbor_trace_corr = np.zeros(series.shape[0], dtype=np.float32)
        local_contrast_net = np.zeros(series.shape[0], dtype=np.float32)
        local_contrast_peak = np.zeros(series.shape[0], dtype=np.float32)
        density_raw = np.zeros(series.shape[0], dtype=np.float32)

    support_density = _sigmoid(_robust_zscore(density_raw))
    qc_support = np.clip(
        0.35 * ps_score[support_mask].astype(np.float32)
        + 0.25 * tcoh[support_mask].astype(np.float32)
        + 0.20 * valid_pair_ratio[support_mask].astype(np.float32)
        + 0.20 * maincc_ratio[support_mask].astype(np.float32),
        0.0,
        1.0,
    ).astype(np.float32)
    activity_logit = (
        0.45 * _robust_zscore(peak_abs_disp_mm)
        + 0.35 * _robust_zscore(np.abs(net_disp_mm))
        + 0.20 * _robust_zscore(np.abs(trend_mm_yr))
    ).astype(np.float32)
    contrast_logit = (
        0.60 * _robust_zscore(local_contrast_peak)
        + 0.40 * _robust_zscore(local_contrast_net)
    ).astype(np.float32)
    activity = _sigmoid(activity_logit)
    saliency_logit = (
        0.42 * activity_logit
        + 0.24 * contrast_logit
        + 1.10 * (temporal_coherence - 0.45)
        + 0.55 * (neighbor_trace_corr - 0.35)
        + 0.35 * (support_density - 0.45)
        + 0.25 * (qc_support - 0.55)
    ).astype(np.float32)
    saliency = _sigmoid(saliency_logit)
    main_sign = _support_graph_sign(net_disp_mm, trend_mm_yr)

    return {
        "support_mask": support_mask,
        "rows": rows.astype(np.int32),
        "cols": cols.astype(np.int32),
        "series": series,
        "series_unit": series_unit,
        "net_disp_mm": net_disp_mm,
        "peak_abs_disp_mm": peak_abs_disp_mm,
        "trend_mm_yr": trend_mm_yr,
        "path_length_mm": path_length_mm,
        "net_to_path_ratio": net_to_path_ratio,
        "sign_consistency": sign_consistency,
        "temporal_coherence": temporal_coherence,
        "pc1": pc1.astype(np.float32),
        "pc2": pc2.astype(np.float32),
        "pc3": pc3.astype(np.float32),
        "local_contrast_net": local_contrast_net,
        "local_contrast_peak": local_contrast_peak,
        "neighbor_trace_corr": neighbor_trace_corr,
        "support_density": support_density,
        "qc_support": qc_support,
        "saliency": saliency,
        "activity": activity,
        "main_sign": main_sign,
        "velocity": velocity[support_mask].astype(np.float32),
        "ps_score": ps_score[support_mask].astype(np.float32),
        "tcoh": tcoh[support_mask].astype(np.float32),
        "valid_pair_ratio": valid_pair_ratio[support_mask].astype(np.float32),
        "mainCC_ratio": maincc_ratio[support_mask].astype(np.float32),
        "strict_flag": strict_flag[support_mask].astype(bool),
        "relaxed_flag": relaxed_flag[support_mask].astype(bool),
        "pixel_area_km2": pixel_area_km2[support_mask].astype(np.float32),
        "longitude": lon_support,
        "latitude": lat_support,
        "x_km": x_km,
        "y_km": y_km,
        "coords_km": coords_km,
        "neighbor_idx": neighbor_idx,
        "neighbor_dist_km": neighbor_dist_km,
        "median_edge_km": float(median_edge_km),
        "edge_radius_km": float(max(SUPPORT_GRAPH_EDGE_RADIUS_MULT * median_edge_km, SUPPORT_GRAPH_EDGE_RADIUS_FLOOR_KM)),
        "graph_k": int(max(k_query - 1, 0)),
        "lat_c": lat_c,
        "aoi_area_km2": float(np.nansum(pixel_area_km2[valid_mask])),
        "n_support_points": int(rows.size),
        "pca_explained_variance_ratio": pca_evr,
    }


def _build_support_graph_candidates(ctx: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_points = int(ctx.get("n_support_points", 0))
    if n_points == 0 or ctx["graph_k"] <= 0:
        return [], {
            "graph_k": int(ctx.get("graph_k", 0)),
            "graph_median_edge_km": float(ctx.get("median_edge_km", 0.0)),
            "active_threshold": 0.0,
            "core_threshold": 0.0,
            "activity_threshold": 0.0,
            "candidate_region_count": 0,
        }

    saliency = np.asarray(ctx["saliency"], dtype=np.float32)
    activity = np.asarray(ctx["activity"], dtype=np.float32)
    coherence = np.asarray(ctx["temporal_coherence"], dtype=np.float32)
    qc_support = np.asarray(ctx["qc_support"], dtype=np.float32)
    main_sign = np.asarray(ctx["main_sign"], dtype=np.int8)
    neighbor_idx = np.asarray(ctx["neighbor_idx"], dtype=np.int32)
    neighbor_dist_km = np.asarray(ctx["neighbor_dist_km"], dtype=np.float32)
    series_unit = np.asarray(ctx["series_unit"], dtype=np.float32)

    active_threshold = float(max(SUPPORT_GRAPH_ACTIVE_SALIENCY_FLOOR, np.nanpercentile(saliency, 80)))
    core_threshold = float(max(SUPPORT_GRAPH_CORE_SALIENCY_FLOOR, np.nanpercentile(saliency, 92)))
    activity_threshold = float(np.nanpercentile(activity, 75))
    active = (
        (main_sign != 0)
        & (saliency >= active_threshold)
        & (coherence >= SUPPORT_GRAPH_ACTIVE_COHERENCE_MIN)
        & (qc_support >= SUPPORT_GRAPH_QC_SUPPORT_MIN)
    )
    core = active & (saliency >= core_threshold) & (activity >= activity_threshold) & (coherence >= SUPPORT_GRAPH_CORE_COHERENCE_MIN)
    active_idx = np.flatnonzero(active)
    if active_idx.size == 0:
        return [], {
            "graph_k": int(ctx["graph_k"]),
            "graph_median_edge_km": float(ctx["median_edge_km"]),
            "active_threshold": active_threshold,
            "core_threshold": core_threshold,
            "activity_threshold": activity_threshold,
            "candidate_region_count": 0,
        }

    active_lookup = -np.ones(n_points, dtype=np.int32)
    active_lookup[active_idx] = np.arange(active_idx.size, dtype=np.int32)
    rows_acc: list[np.ndarray] = []
    cols_acc: list[np.ndarray] = []
    edge_corr_samples: list[np.ndarray] = []
    for col in range(neighbor_idx.shape[1]):
        nbr = neighbor_idx[active_idx, col]
        keep = (
            active[nbr]
            & (main_sign[active_idx] == main_sign[nbr])
            & (neighbor_dist_km[active_idx, col] <= float(ctx["edge_radius_km"]))
        )
        if not np.any(keep):
            continue
        left = active_idx[keep]
        right = nbr[keep]
        corr = np.sum(series_unit[left] * series_unit[right], axis=1).astype(np.float32)
        good = corr >= SUPPORT_GRAPH_EDGE_CORR
        if not np.any(good):
            continue
        rows_acc.append(active_lookup[left[good]])
        cols_acc.append(active_lookup[right[good]])
        edge_corr_samples.append(corr[good])

    if rows_acc:
        rows = np.concatenate(rows_acc)
        cols = np.concatenate(cols_acc)
        graph = coo_matrix(
            (np.ones(rows.size, dtype=np.int8), (rows, cols)),
            shape=(active_idx.size, active_idx.size),
        )
        _, labels = connected_components(graph.tocsr(), directed=False)
        edge_corr_median = float(np.median(np.concatenate(edge_corr_samples))) if edge_corr_samples else 0.0
    else:
        labels = np.arange(active_idx.size, dtype=np.int32)
        edge_corr_median = 0.0

    candidate_rows: list[dict[str, Any]] = []
    n_components = int(labels.max()) + 1 if labels.size else 0
    for comp_idx in range(n_components):
        members = active_idx[labels == comp_idx]
        if members.size == 0 or not np.any(core[members]):
            continue
        candidate_rows.append({
            "support_indices": members.astype(np.int32),
            "core_point_count": int(np.sum(core[members])),
        })

    summary = {
        "graph_k": int(ctx["graph_k"]),
        "graph_median_edge_km": float(ctx["median_edge_km"]),
        "edge_radius_km": float(ctx["edge_radius_km"]),
        "active_threshold": active_threshold,
        "core_threshold": core_threshold,
        "activity_threshold": activity_threshold,
        "active_point_count": int(active_idx.size),
        "core_point_count": int(np.sum(core)),
        "candidate_region_count": int(len(candidate_rows)),
        "edge_corr_threshold": float(SUPPORT_GRAPH_EDGE_CORR),
        "edge_corr_median": edge_corr_median,
    }
    return candidate_rows, summary


def _build_feature_stack(
    *,
    velocity: np.ndarray,
    tcoh: np.ndarray,
    ps_score: np.ndarray,
    valid_pair_ratio: np.ndarray,
    maincc_ratio: np.ndarray,
    jump_risk: np.ndarray,
    anomaly_exposure: np.ndarray,
    strict_flag: np.ndarray,
    relaxed_flag: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    valid_mask = (
        np.isfinite(velocity)
        & np.isfinite(tcoh)
        & np.isfinite(ps_score)
        & np.isfinite(valid_pair_ratio)
        & np.isfinite(maincc_ratio)
        & np.isfinite(jump_risk)
        & np.isfinite(anomaly_exposure)
    )
    abs_velocity = np.abs(velocity).astype(np.float32)
    velocity_sign = np.sign(velocity).astype(np.float32)
    local_velocity_mean = _local_mean_window(velocity, valid_mask, size=11)
    local_velocity_anomaly = (velocity - local_velocity_mean).astype(np.float32)
    abs_local_velocity_anomaly = np.abs(local_velocity_anomaly).astype(np.float32)
    local_velocity_gradient = _local_gradient(velocity, valid_mask)
    local_velocity_std = _local_std_5x5(velocity, valid_mask)
    features = {
        "velocity_mm_yr": velocity.astype(np.float32),
        "abs_velocity_mm_yr": abs_velocity,
        "velocity_sign": velocity_sign,
        "local_velocity_mean_11x11": local_velocity_mean,
        "local_velocity_anomaly": local_velocity_anomaly,
        "abs_local_velocity_anomaly": abs_local_velocity_anomaly,
        "local_velocity_gradient": local_velocity_gradient,
        "local_velocity_std_5x5": local_velocity_std,
        "tcoh": tcoh.astype(np.float32),
        "ps_score": ps_score.astype(np.float32),
        "valid_pair_ratio": valid_pair_ratio.astype(np.float32),
        "mainCC_ratio": maincc_ratio.astype(np.float32),
        "jump_risk": jump_risk.astype(np.float32),
        "anomaly_exposure": anomaly_exposure.astype(np.float32),
        "strict_flag": strict_flag.astype(np.float32),
        "relaxed_flag": relaxed_flag.astype(np.float32),
    }
    return features, valid_mask


def _seed_threshold(values: np.ndarray, *, percentile: float, floor: float) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float(floor)
    return float(max(np.nanpercentile(values[finite], percentile), floor))


def _build_weak_supervision_seeds(
    features: dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    abs_velocity = features["abs_velocity_mm_yr"]
    abs_local_anomaly = features["abs_local_velocity_anomaly"]
    gradient = features["local_velocity_gradient"]
    local_std = features["local_velocity_std_5x5"]
    tcoh = features["tcoh"]
    ps_score = features["ps_score"]
    strict_flag = features["strict_flag"] > 0.5
    relaxed_flag = features["relaxed_flag"] > 0.5
    velocity = features["velocity_mm_yr"]

    pos_thr = _seed_threshold(abs_velocity[valid_mask], percentile=95, floor=5.0)
    anomaly_thr = _seed_threshold(abs_local_anomaly[valid_mask], percentile=94, floor=3.0)
    neg_vel_thr = float(max(np.nanpercentile(abs_velocity[valid_mask], 40), 1.5)) if np.any(valid_mask) else 1.5
    neg_anom_thr = float(max(np.nanpercentile(abs_local_anomaly[valid_mask], 45), 1.2)) if np.any(valid_mask) else 1.2
    grad_low_thr = _seed_threshold(gradient[valid_mask], percentile=45, floor=0.5)
    std_low_thr = _seed_threshold(local_std[valid_mask], percentile=45, floor=0.75)
    trusted = (tcoh >= 0.65) | (ps_score >= 0.50) | strict_flag | relaxed_flag

    kernel = np.ones((3, 3), dtype=np.int16)
    same_sign_level = np.maximum(pos_thr * 0.60, 3.0)
    pos_support = convolve((valid_mask & (velocity > 0) & (abs_velocity >= same_sign_level)).astype(np.int16), kernel, mode="constant", cval=0)
    neg_support = convolve((valid_mask & (velocity < 0) & (abs_velocity >= same_sign_level)).astype(np.int16), kernel, mode="constant", cval=0)
    contiguous_support = (pos_support >= 4) | (neg_support >= 4)

    positive_seed = valid_mask & trusted & contiguous_support & (abs_velocity >= pos_thr) & (abs_local_anomaly >= anomaly_thr)
    far_enough = np.ones_like(valid_mask, dtype=bool)
    if np.any(positive_seed):
        far_enough = distance_transform_edt(~positive_seed) >= 4.0
    negative_seed = (
        valid_mask
        & far_enough
        & (abs_velocity <= neg_vel_thr)
        & (abs_local_anomaly <= neg_anom_thr)
        & (gradient <= grad_low_thr)
        & (local_std <= std_low_thr)
        & (tcoh >= 0.70)
    )
    negative_seed &= ~positive_seed
    return positive_seed, negative_seed, {
        "positive_abs_velocity_threshold_mm_yr": float(pos_thr),
        "positive_local_anomaly_threshold_mm_yr": float(anomaly_thr),
        "negative_abs_velocity_threshold_mm_yr": float(neg_vel_thr),
        "negative_local_anomaly_threshold_mm_yr": float(neg_anom_thr),
        "negative_gradient_threshold": float(grad_low_thr),
        "negative_local_std_threshold": float(std_low_thr),
    }


def _heuristic_probability(features: dict[str, np.ndarray], valid_mask: np.ndarray) -> np.ndarray:
    prob = (
        0.24 * _normalize_interval(features["abs_velocity_mm_yr"])
        + 0.28 * _normalize_interval(features["abs_local_velocity_anomaly"])
        + 0.12 * _normalize_interval(features["local_velocity_gradient"])
        + 0.10 * _normalize_interval(features["local_velocity_std_5x5"])
        + 0.12 * np.clip(features["ps_score"], 0.0, 1.0)
        + 0.10 * np.clip(features["tcoh"], 0.0, 1.0)
        + 0.07 * np.clip(features["valid_pair_ratio"], 0.0, 1.0)
        + 0.05 * np.clip(features["mainCC_ratio"], 0.0, 1.0)
        + 0.04 * (features["strict_flag"] > 0.5).astype(np.float32)
    )
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
    prob[~valid_mask] = np.nan
    return prob


class _DilatedResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.act(self.bn1(self.conv1(x)))
        y = self.dropout(y)
        y = self.bn2(self.conv2(y))
        return self.act(residual + y)


class _SpatialFeatureNet(nn.Module):
    """Fully convolutional network with multi-scale dilated convolutions.

    Captures spatial context at multiple scales for deformation zone probability
    estimation, replacing the pixel-independent HistGradientBoosting classifier.
    """

    def __init__(self, in_channels: int, hidden: int = 32, dropout: float = 0.10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            _DilatedResBlock(hidden, dilation=1, dropout=dropout),
            _DilatedResBlock(hidden, dilation=2, dropout=dropout),
            _DilatedResBlock(hidden, dilation=4, dropout=dropout),
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        multi_scale = []
        for block in self.blocks:
            x = block(x)
            multi_scale.append(x)
        fused = self.fuse(torch.cat(multi_scale, dim=1))
        return self.head(fused).squeeze(1)


def _zone_detect_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        free_mem = torch.cuda.mem_get_info()[0]
        if free_mem > 512 * 1024 * 1024:
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


def _focal_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for class-imbalanced pixel classification."""
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    alpha_t = torch.where(targets > 0.5, alpha, 1.0 - alpha)
    focal = alpha_t * ((1.0 - pt) ** gamma) * ce
    return (focal * mask).sum() / mask.sum().clamp(min=1.0)


def _build_feature_tensor(features: dict[str, np.ndarray], valid_mask: np.ndarray) -> torch.Tensor:
    """Stack features into a (1, C, H, W) tensor with NaN replaced by 0."""
    channels = []
    for name in FEATURE_NAMES:
        arr = features[name].astype(np.float32).copy()
        arr[~valid_mask] = 0.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        channels.append(arr)
    stacked = np.stack(channels, axis=0)[None]
    return torch.from_numpy(stacked).float()


def _extract_training_patches(
    feat_tensor: torch.Tensor,
    label_map: np.ndarray,
    valid_mask: np.ndarray,
    positive_seed: np.ndarray,
    negative_seed: np.ndarray,
    *,
    patch_size: int = 64,
    n_patches: int = 256,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract random patches centered on seed pixels for efficient training."""
    H, W = valid_mask.shape
    half = patch_size // 2
    pos_locs = np.argwhere(positive_seed)
    neg_locs = np.argwhere(negative_seed)
    all_locs = np.concatenate([pos_locs, neg_locs], axis=0)
    if len(all_locs) == 0:
        return torch.empty(0), torch.empty(0), torch.empty(0)
    n_each = n_patches
    chosen_idx = rng.choice(len(all_locs), size=min(n_each, len(all_locs)), replace=len(all_locs) < n_each)
    patches_x, patches_y, patches_m = [], [], []
    for idx in chosen_idx:
        cy, cx = int(all_locs[idx, 0]), int(all_locs[idx, 1])
        y0 = max(0, min(cy - half, H - patch_size))
        x0 = max(0, min(cx - half, W - patch_size))
        y1, x1 = y0 + patch_size, x0 + patch_size
        patches_x.append(feat_tensor[0, :, y0:y1, x0:x1])
        label_patch = label_map[y0:y1, x0:x1].copy()
        mask_patch = (label_patch >= 0).astype(np.float32) * valid_mask[y0:y1, x0:x1].astype(np.float32)
        patches_y.append(torch.from_numpy(label_patch.clip(0, 1).astype(np.float32)))
        patches_m.append(torch.from_numpy(mask_patch))
    return torch.stack(patches_x), torch.stack(patches_y), torch.stack(patches_m)


def _train_probability_model_cnn(
    features: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    positive_seed: np.ndarray,
    negative_seed: np.ndarray,
    *,
    self_training_rounds: int = 2,
    epochs_per_round: int = 40,
    lr: float = 3e-3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train a spatial CNN for deformation probability, with self-training refinement."""
    pos_count = int(np.sum(positive_seed))
    neg_count = int(np.sum(negative_seed))
    seed_summary: dict[str, Any] = {
        "positive_seed_count": pos_count,
        "negative_seed_count": neg_count,
        "classifier": "SpatialFeatureNet_CNN",
        "training_mode": "spatial_cnn_self_training_v1",
    }
    if pos_count < 16 or neg_count < 16:
        seed_summary["training_status"] = "heuristic_fallback"
        return _heuristic_probability(features, valid_mask), seed_summary

    device = _zone_detect_device()
    feat_tensor = _build_feature_tensor(features, valid_mask)

    current_pos = positive_seed.copy()
    current_neg = negative_seed.copy()
    round_summaries: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)

    for rnd in range(self_training_rounds + 1):
        label_map = np.full(valid_mask.shape, -1.0, dtype=np.float32)
        label_map[current_pos] = 1.0
        label_map[current_neg] = 0.0

        model = _SpatialFeatureNet(in_channels=len(FEATURE_NAMES)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        n_epochs = epochs_per_round if rnd == 0 else max(epochs_per_round // 2, 20)
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None

        for epoch in range(n_epochs):
            px, py, pm = _extract_training_patches(
                feat_tensor, label_map, valid_mask, current_pos, current_neg,
                patch_size=64, n_patches=128, rng=rng,
            )
            if px.shape[0] == 0:
                break
            px, py, pm = px.to(device), py.to(device), pm.to(device)
            model.train()
            optimizer.zero_grad()
            logits = model(px)
            loss = _focal_loss(logits, py, pm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_val = float(loss.detach().cpu())
            if loss_val < best_loss:
                best_loss = loss_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        H, W = valid_mask.shape
        with torch.no_grad():
            logits = model(feat_tensor.to(device)).squeeze(0).cpu().numpy()
        prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64))).astype(np.float32)
        prob[~valid_mask] = np.nan

        round_summaries.append({
            "round": rnd,
            "n_epochs": n_epochs,
            "best_loss": float(best_loss),
            "pos_seeds": int(np.sum(current_pos)),
            "neg_seeds": int(np.sum(current_neg)),
        })

        if rnd < self_training_rounds:
            high_conf_pos = valid_mask & (prob >= 0.85) & ~current_neg
            high_conf_neg = valid_mask & (prob <= 0.15) & ~current_pos
            current_pos = current_pos | high_conf_pos
            current_neg = current_neg | high_conf_neg

    prob[positive_seed] = np.maximum(prob[positive_seed], 0.85)
    prob[negative_seed] = np.minimum(prob[negative_seed], 0.15)
    prob[~valid_mask] = np.nan
    seed_summary["training_status"] = "trained_cnn"
    seed_summary["n_self_training_rounds"] = self_training_rounds
    seed_summary["round_summaries"] = round_summaries
    seed_summary["final_pos_seeds"] = int(np.sum(current_pos))
    seed_summary["final_neg_seeds"] = int(np.sum(current_neg))
    del model, feat_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prob, seed_summary


def _train_probability_model(
    features: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    positive_seed: np.ndarray,
    negative_seed: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        return _train_probability_model_cnn(features, valid_mask, positive_seed, negative_seed)
    except Exception as exc:
        logger.warning(f"CNN probability model failed ({type(exc).__name__}: {exc}), falling back to HistGBT")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _train_probability_model_histgbt(features, valid_mask, positive_seed, negative_seed)


def _train_probability_model_histgbt(
    features: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    positive_seed: np.ndarray,
    negative_seed: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    pos_idx = np.flatnonzero(positive_seed)
    neg_idx = np.flatnonzero(negative_seed)
    seed_summary = {
        "positive_seed_count": int(pos_idx.size),
        "negative_seed_count": int(neg_idx.size),
        "classifier": "HistGradientBoostingClassifier",
        "training_mode": "weak_ml_v1",
    }
    if pos_idx.size < 16 or neg_idx.size < 16:
        seed_summary["training_status"] = "heuristic_fallback"
        return _heuristic_probability(features, valid_mask), seed_summary

    rng = np.random.default_rng(42)
    sample_n = int(min(pos_idx.size, neg_idx.size, 50000))
    if pos_idx.size > sample_n:
        pos_idx = np.sort(rng.choice(pos_idx, size=sample_n, replace=False))
    if neg_idx.size > sample_n:
        neg_idx = np.sort(rng.choice(neg_idx, size=sample_n, replace=False))
    train_idx = np.concatenate([pos_idx, neg_idx])
    y = np.concatenate([np.ones(len(pos_idx), dtype=np.int8), np.zeros(len(neg_idx), dtype=np.int8)])

    feature_matrix = np.column_stack([features[name].reshape(-1) for name in FEATURE_NAMES]).astype(np.float32)
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=220,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=48,
        l2_regularization=0.10,
        random_state=42,
    )
    clf.fit(feature_matrix[train_idx], y)
    probability = clf.predict_proba(feature_matrix)[:, 1].reshape(valid_mask.shape).astype(np.float32)
    probability[positive_seed] = np.maximum(probability[positive_seed], 0.85)
    probability[negative_seed] = np.minimum(probability[negative_seed], 0.15)
    probability[~valid_mask] = np.nan
    seed_summary["training_status"] = "trained"
    seed_summary["n_train_samples"] = int(train_idx.size)
    return probability, seed_summary


def _build_region_grow_mask(
    features: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    probability: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    velocity = np.asarray(features["velocity_mm_yr"], dtype=np.float32)
    tcoh = np.asarray(features["tcoh"], dtype=np.float32)
    ps_score = np.asarray(features["ps_score"], dtype=np.float32)
    strict_flag = np.asarray(features["strict_flag"], dtype=np.float32) > 0.5
    relaxed_flag = np.asarray(features["relaxed_flag"], dtype=np.float32) > 0.5
    support_flag = (strict_flag | relaxed_flag) & valid_mask

    support_density = _local_mean_window(support_flag.astype(np.float32), valid_mask, size=9)
    vel_core = _local_mean_window(velocity, valid_mask, size=15)
    vel_region = _local_mean_window(velocity, valid_mask, size=21)
    vel_background = _local_mean_window(velocity, valid_mask, size=41)
    regional_anomaly = np.abs(vel_region - vel_background).astype(np.float32)
    tcoh_region = _local_mean_window(tcoh, valid_mask, size=9)
    ps_region = _local_mean_window(ps_score, valid_mask, size=9)

    finite_support = support_flag & np.isfinite(vel_core)
    core_thr = _seed_threshold(np.abs(vel_core[finite_support]), percentile=90, floor=9.0)
    grow_thr = float(max(core_thr * 0.62, 6.0))
    anom_thr = _seed_threshold(regional_anomaly[finite_support], percentile=78, floor=0.55)

    structure = np.ones((3, 3), dtype=bool)
    zone_mask = np.zeros(valid_mask.shape, dtype=bool)
    grown_components: list[dict[str, Any]] = []

    for sign_name, sign in (("uplift", 1.0), ("subsidence", -1.0)):
        signed_core = valid_mask & (vel_core * sign >= core_thr)
        signed_region = valid_mask & (vel_region * sign >= grow_thr)
        same_sign_support = convolve((support_flag & (vel_region * sign >= grow_thr)).astype(np.int16), np.ones((3, 3), dtype=np.int16), mode="constant", cval=0)

        core_mask = (
            signed_core
            & support_flag
            & (same_sign_support >= 5)
            & ((np.nan_to_num(probability, nan=0.0) >= 0.52) | (regional_anomaly >= anom_thr))
        )
        allowed_mask = (
            signed_region
            & (support_density >= 0.08)
            & (tcoh_region >= 0.48)
            & ((ps_region >= 0.32) | (support_density >= 0.18))
            & ((np.nan_to_num(probability, nan=0.0) >= 0.32) | (regional_anomaly >= anom_thr * 0.75))
        )
        if not np.any(core_mask) or not np.any(allowed_mask):
            continue
        grown = binary_propagation(core_mask, structure=structure, mask=allowed_mask)
        grown = binary_closing(grown, structure=structure, iterations=2)
        grown = binary_opening(grown, structure=structure, iterations=1)
        grown = binary_fill_holes(grown)
        grown &= valid_mask
        if not np.any(grown):
            continue
        zone_mask |= grown
        grown_components.append(
            {
                "sign": sign_name,
                "core_threshold_mm_yr": float(core_thr),
                "grow_threshold_mm_yr": float(grow_thr),
                "regional_anomaly_threshold_mm_yr": float(anom_thr),
                "core_pixels": int(np.sum(core_mask)),
                "grown_pixels": int(np.sum(grown)),
            }
        )

    summary = {
        "segmentation_mode": "region_grow_v2",
        "core_velocity_threshold_mm_yr": float(core_thr),
        "grow_velocity_threshold_mm_yr": float(grow_thr),
        "regional_anomaly_threshold_mm_yr": float(anom_thr),
        "support_density_threshold": 0.08,
        "tcoh_region_threshold": 0.48,
        "ps_region_threshold": 0.32,
        "grown_components": grown_components,
    }
    return zone_mask.astype(bool), summary


def _build_principal_component_mask(
    features: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    probability: np.ndarray,
    *,
    rel0_cube: np.ndarray,
    support_flag: np.ndarray,
    pixel_area_km2: np.ndarray,
    min_zone_area_dynamic: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    velocity = np.asarray(features["velocity_mm_yr"], dtype=np.float32)
    tcoh = np.asarray(features["tcoh"], dtype=np.float32)
    abs_local_anomaly = np.asarray(features["abs_local_velocity_anomaly"], dtype=np.float32)

    vel_core = _local_mean_window(velocity, valid_mask, size=15)
    support_density = _local_mean_window(support_flag.astype(np.float32), valid_mask, size=9)
    tcoh_region = _local_mean_window(tcoh, valid_mask, size=9)
    prob_region = _local_mean_window(np.nan_to_num(probability, nan=0.0).astype(np.float32), valid_mask, size=9)

    finite_support = support_flag & np.isfinite(vel_core)
    if not np.any(finite_support):
        return np.zeros(valid_mask.shape, dtype=bool), {
            "segmentation_mode": "principal_component_v3",
            "candidate_count": 0,
            "selected_candidate_count": 0,
            "selected_candidates": [],
            "fallback_triggered": True,
            "fallback_reason": "no_support_pixels",
        }

    base_thr = _seed_threshold(np.abs(vel_core[finite_support]), percentile=90, floor=10.0)
    threshold_candidates = sorted(
        {
            float(max(base_thr * 0.60, 7.0)),
            float(max(base_thr * 0.75, 9.0)),
            float(max(base_thr * 0.90, 11.0)),
        }
    )
    structure = np.ones((3, 3), dtype=bool)
    candidate_rows: list[dict[str, Any]] = []

    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = int(np.sum(a & b))
        if inter == 0:
            return 0.0
        union = int(np.sum(a | b))
        return float(inter / max(union, 1))

    for sign_name, sign in (("uplift", 1.0), ("subsidence", -1.0)):
        for thr in threshold_candidates:
            mask = (
                valid_mask
                & (vel_core * sign >= thr)
                & (support_density >= 0.05)
                & (tcoh_region >= 0.45)
                & (prob_region >= 0.18)
            )
            if not np.any(mask):
                continue
            mask = binary_closing(mask, structure=structure, iterations=2)
            mask = binary_opening(mask, structure=structure, iterations=1)
            mask = binary_fill_holes(mask)
            mask &= valid_mask
            labeled, n_comp = _connected_components(mask)
            for comp_id in range(1, int(n_comp) + 1):
                comp_mask = labeled == comp_id
                pixel_count = int(np.sum(comp_mask))
                if pixel_count < max(MIN_ZONE_PIXELS, 120):
                    continue
                area_km2 = float(np.nansum(pixel_area_km2[comp_mask]))
                if area_km2 < float(min_zone_area_dynamic * 0.80):
                    continue
                temporal_profile = _zone_timeseries_profile(rel0_cube, comp_mask, support_flag)
                if temporal_profile is None:
                    continue
                continuity = float(temporal_profile["temporal_continuity_score"])
                net_disp = float(temporal_profile["temporal_net_disp_mm"])
                if abs(net_disp) < 6.0 or continuity < 0.40:
                    continue
                median_velocity = float(np.nanmedian(velocity[comp_mask]))
                mean_probability = float(np.nanmean(probability[comp_mask]))
                median_anomaly = float(np.nanmedian(abs_local_anomaly[comp_mask]))
                score = (
                    float(area_km2)
                    * max(abs(median_velocity), 1.0)
                    * max(abs(net_disp), 1.0)
                    * max(continuity, 0.05)
                    * max(mean_probability, 0.10)
                )
                candidate_rows.append(
                    {
                        "sign": sign_name,
                        "threshold_mm_yr": float(thr),
                        "mask": comp_mask,
                        "pixel_count": pixel_count,
                        "area_km2": area_km2,
                        "median_velocity_mm_yr": median_velocity,
                        "median_abs_local_anomaly_mm_yr": median_anomaly,
                        "temporal_net_disp_mm": net_disp,
                        "temporal_continuity_score": continuity,
                        "probability_mean": mean_probability,
                        "score": float(score),
                    }
                )

    candidate_rows.sort(key=lambda item: float(item["score"]), reverse=True)
    selected_rows: list[dict[str, Any]] = []
    principal_mask = np.zeros(valid_mask.shape, dtype=bool)
    for candidate in candidate_rows:
        comp_mask = np.asarray(candidate["mask"], dtype=bool)
        if any(_mask_iou(comp_mask, np.asarray(row["mask"], dtype=bool)) > 0.15 for row in selected_rows):
            continue
        selected_rows.append(candidate)
        principal_mask |= comp_mask
        if len(selected_rows) >= MAX_PRINCIPAL_ZONES:
            break

    summary = {
        "segmentation_mode": "principal_component_v3",
        "base_velocity_threshold_mm_yr": float(base_thr),
        "threshold_candidates_mm_yr": [float(v) for v in threshold_candidates],
        "candidate_count": int(len(candidate_rows)),
        "selected_candidate_count": int(len(selected_rows)),
        "selected_candidates": [
            {
                key: value
                for key, value in row.items()
                if key != "mask"
            }
            for row in selected_rows
        ],
        "fallback_triggered": False,
        "fallback_reason": "",
    }
    if not np.any(principal_mask):
        summary["fallback_triggered"] = True
        summary["fallback_reason"] = "no_principal_component"
    return principal_mask.astype(bool), summary


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    structure = np.ones((3, 3), dtype=np.int8)
    return label(mask.astype(np.uint8), structure=structure)


def _smooth_series(values: np.ndarray) -> np.ndarray:
    series = np.asarray(values, dtype=np.float32)
    if series.size < 3:
        return series
    padded = np.pad(series, (1, 1), mode="edge")
    kernel = np.asarray([0.25, 0.50, 0.25], dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _zone_timeseries_profile(
    rel0_cube: np.ndarray,
    zone_mask: np.ndarray,
    support_mask: np.ndarray,
) -> dict[str, Any] | None:
    zone_support = zone_mask & support_mask
    if int(np.sum(zone_support)) < MIN_ZONE_TIMESERIES_POINTS:
        zone_support = zone_mask
    if int(np.sum(zone_support)) < MIN_ZONE_TIMESERIES_POINTS:
        return None
    zone_series = rel0_cube[zone_support, :]
    return _series_profile(zone_series)


def _classify_zone_type(zone_velocity: np.ndarray) -> str:
    finite = np.isfinite(zone_velocity)
    if not finite.any():
        return "mixed"
    vals = zone_velocity[finite]
    pos_ratio = float(np.mean(vals > 0))
    neg_ratio = float(np.mean(vals < 0))
    median = float(np.nanmedian(vals))
    if pos_ratio > 0.25 and neg_ratio > 0.25:
        return "mixed"
    if median < -1.0:
        return "subsidence"
    if median > 1.0:
        return "uplift"
    return "mixed"


def _zone_plot_records(zone_records_map: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    items = sorted(
        zone_records_map.values(),
        key=lambda item: (
            float(item.get("area_km2", 0.0)),
            abs(float(item.get("median_velocity_mm_yr", 0.0))),
            float(item.get("temporal_continuity_score", 0.0)),
        ),
        reverse=True,
    )
    return items[:MAX_ZONE_CURVES]


def _record_public_view(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if key.startswith("timeseries_") or key in {"geometry", "geometry_proj", "support_indices"}:
            continue
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        elif isinstance(value, (np.integer,)):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def _estimate_zone_buffer_radius(longitude: np.ndarray, latitude: np.ndarray, valid_mask: np.ndarray) -> float:
    lon_step = np.abs(np.diff(longitude, axis=1))
    lon_ok = valid_mask[:, 1:] & valid_mask[:, :-1] & np.isfinite(lon_step) & (lon_step > 0)
    lat_step = np.abs(np.diff(latitude, axis=0))
    lat_ok = valid_mask[1:, :] & valid_mask[:-1, :] & np.isfinite(lat_step) & (lat_step > 0)
    lon_med = float(np.nanmedian(lon_step[lon_ok])) if np.any(lon_ok) else 0.0
    lat_med = float(np.nanmedian(lat_step[lat_ok])) if np.any(lat_ok) else 0.0
    step = max(lon_med, lat_med, 3.0e-4)
    return float(step * 0.62)


def _clean_zone_geometry(geom):
    try:
        from shapely.geometry import MultiPolygon, Polygon
    except Exception:
        return geom
    if geom is None or geom.is_empty:
        return geom
    geom = geom.buffer(0)
    if geom.is_empty:
        return geom
    if isinstance(geom, Polygon):
        holes = [ring for ring in geom.interiors if Polygon(ring).area >= 0.10 * geom.area]
        geom = Polygon(geom.exterior.coords, holes=[ring.coords for ring in holes]).buffer(0)
        return geom
    if isinstance(geom, MultiPolygon):
        geoms = [g for g in geom.geoms if g.area >= 0.10 * max(geom.area, 1.0e-9)]
        if not geoms:
            geoms = [max(geom.geoms, key=lambda g: g.area)]
        return MultiPolygon(geoms).buffer(0)
    return geom


def _zone_geom_from_points(
    comp_mask: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    buffer_radius: float,
    ratio: float = SUPPORT_GRAPH_CONCAVE_RATIO,
):
    try:
        from shapely import concave_hull
        from shapely.geometry import MultiPoint, Point
        from shapely.ops import unary_union
    except Exception:
        return None

    pts = np.column_stack([longitude[comp_mask], latitude[comp_mask]])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return None
    if len(pts) == 1:
        return Point(float(pts[0, 0]), float(pts[0, 1])).buffer(buffer_radius * 1.2)
    max_points = 3200
    if len(pts) > max_points:
        step = int(np.ceil(len(pts) / max_points))
        pts = pts[::step]
    geom = None
    try:
        geom = concave_hull(MultiPoint([(float(x), float(y)) for x, y in pts]), ratio=float(ratio))
    except Exception:
        geom = None
    if geom is None or geom.is_empty or geom.geom_type not in {"Polygon", "MultiPolygon"}:
        buffers = [Point(float(x), float(y)).buffer(buffer_radius, resolution=2) for x, y in pts]
        geom = unary_union(buffers).buffer(buffer_radius * 0.55).buffer(-buffer_radius * 0.30)
    if geom.is_empty:
        geom = MultiPoint([(float(x), float(y)) for x, y in pts]).convex_hull.buffer(buffer_radius * 1.1)
    return _clean_zone_geometry(geom)


def _shape_records(
    zone_id_raster: np.ndarray,
    profile: dict[str, Any],
    zone_records: dict[int, dict[str, Any]],
    *,
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> tuple[list[dict[str, Any]], Any]:
    try:
        import geopandas as gpd
        from shapely.geometry import mapping
    except Exception as exc:
        logger.warning(f"deformation zone polygon export skipped: {exc}")
        return [], None

    features_out: list[dict[str, Any]] = []
    geoms = []
    props = []
    valid_mask = np.isfinite(longitude) & np.isfinite(latitude) & (zone_id_raster > 0)
    buffer_radius = _estimate_zone_buffer_radius(longitude, latitude, valid_mask)
    for zid in sorted(zone_records):
        record = zone_records.get(zid)
        if not record:
            continue
        geom = record.get("geometry")
        if geom is None:
            comp_mask = zone_id_raster == int(zid)
            geom = _zone_geom_from_points(comp_mask, longitude, latitude, buffer_radius=buffer_radius)
        if geom is None or geom.is_empty:
            continue
        public = _record_public_view(record)
        geoms.append(geom)
        features_out.append({"type": "Feature", "geometry": mapping(geom), "properties": public})
        props.append(
            {
                "zone_id": str(public["zone_id"]),
                "zone_type": str(public["zone_type"]),
                "area_km2": float(public["area_km2"]),
                "pixel_cnt": int(public["pixel_count"]),
                "bbox_w": float(public["bbox_wsen"][0]),
                "bbox_s": float(public["bbox_wsen"][1]),
                "bbox_e": float(public["bbox_wsen"][2]),
                "bbox_n": float(public["bbox_wsen"][3]),
                "cent_lon": float(public["centroid_lonlat"][0]),
                "cent_lat": float(public["centroid_lonlat"][1]),
                "mean_vel": float(public["mean_velocity_mm_yr"]),
                "median_vel": float(public["median_velocity_mm_yr"]),
                "p10_vel": float(public["p10_velocity_mm_yr"]),
                "p90_vel": float(public["p90_velocity_mm_yr"]),
                "max_abs_v": float(public["max_abs_velocity_mm_yr"]),
                "prob_mean": float(public["probability_mean"]),
                "strict_cnt": int(public["strict_point_count"]),
                "relaxed_cn": int(public["relaxed_point_count"]),
                "fore_ct": int(public["forecast_point_count"]),
            }
        )
    if not geoms:
        return [], None
    gdf = gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")
    return features_out, gdf


def _write_zone_vectors(
    *,
    features_out: list[dict[str, Any]],
    gdf,
    export_dir: Path,
) -> dict[str, str | None]:
    geojson_path = export_dir / "deformation_zones.geojson"
    shp_path = export_dir / "deformation_zones.shp"
    kmz_path = export_dir / "deformation_zones.kmz"

    geojson = {"type": "FeatureCollection", "features": features_out}
    geojson_path.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")

    shp_out: str | None = None
    if gdf is not None:
        try:
            gdf.to_file(shp_path)
            shp_out = str(shp_path)
        except Exception as exc:
            logger.warning(f"deformation zone shapefile export failed: {exc}")

    kmz_out: str | None = None
    try:
        import simplekml

        kml = simplekml.Kml()
        color_map = {
            "subsidence": simplekml.Color.changealphaint(180, simplekml.Color.red),
            "uplift": simplekml.Color.changealphaint(180, simplekml.Color.blue),
            "mixed": simplekml.Color.changealphaint(180, simplekml.Color.orange),
        }
        for feature in features_out:
            props = feature["properties"]
            geom = feature["geometry"]
            rings = []
            if geom["type"] == "Polygon":
                rings = [geom["coordinates"][0]]
            elif geom["type"] == "MultiPolygon":
                rings = [poly[0] for poly in geom["coordinates"]]
            for ring in rings:
                pol = kml.newpolygon(
                    name=str(props["zone_id"]),
                    outerboundaryis=[(float(x), float(y)) for x, y in ring],
                )
                color = color_map.get(str(props.get("zone_type", "mixed")), simplekml.Color.changealphaint(180, simplekml.Color.orange))
                pol.style.linestyle.color = color
                pol.style.linestyle.width = 2
                pol.style.polystyle.color = simplekml.Color.changealphaint(38, color)
                pol.description = (
                    f"zone_type={props['zone_type']}\n"
                    f"area_km2={props['area_km2']:.3f}\n"
                    f"median_velocity_mm_yr={props['median_velocity_mm_yr']:.2f}\n"
                    f"forecast_point_count={props['forecast_point_count']}"
                )
        kml.savekmz(str(kmz_path))
        kmz_out = str(kmz_path)
    except Exception as exc:
        logger.warning(f"deformation zone KMZ export failed: {exc}")

    return {
        "geojson": str(geojson_path),
        "shp": shp_out,
        "kmz": kmz_out,
    }


def _plot_geometry_outline(
    ax,
    geom,
    *,
    color: str,
    linewidth: float = 1.4,
    linestyle: str | tuple = "-",
    alpha: float = 1.0,
    zorder: float = 6.0,
) -> None:
    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        ax.plot(
            x,
            y,
            color=color,
            lw=linewidth,
            ls=linestyle,
            alpha=alpha,
            zorder=zorder,
            solid_capstyle="round",
            solid_joinstyle="round",
            dash_capstyle="round",
            dash_joinstyle="round",
        )
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            _plot_geometry_outline(ax, poly, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder)


def _prepare_display_geometry(geom):
    try:
        from shapely import BufferJoinStyle
        from shapely.affinity import scale as scale_geom
    except Exception:
        return geom
    if geom is None or geom.is_empty:
        return geom
    minx, miny, maxx, maxy = geom.bounds
    span = max(float(maxx - minx), float(maxy - miny), 1.0e-4)
    smooth = max(span * 0.022, 2.6e-4)
    simplify_tol = max(span * 0.032, 4.0e-4)
    candidate = geom.buffer(smooth * 1.25, join_style=BufferJoinStyle.round).buffer(-smooth * 1.10, join_style=BufferJoinStyle.round)
    candidate = candidate.simplify(simplify_tol, preserve_topology=True)
    candidate = _clean_zone_geometry(candidate)
    if candidate is None or candidate.is_empty:
        return geom
    orig_area = float(getattr(geom, "area", 0.0))
    cand_area = float(getattr(candidate, "area", 0.0))
    hull = geom.convex_hull if getattr(geom, "convex_hull", None) is not None else None
    hull_area = float(getattr(hull, "area", 0.0)) if hull is not None else 0.0
    compactness = float(orig_area / max(hull_area, 1.0e-9)) if orig_area > 0.0 else 1.0
    if hull is not None and hull_area > 0.0 and compactness < 0.72:
        scale = float(np.sqrt(min(max(orig_area * 1.10, 1.0e-9), hull_area) / hull_area))
        envelope = scale_geom(hull, xfact=scale, yfact=scale, origin="center")
        envelope = envelope.buffer(smooth * 0.70, join_style=BufferJoinStyle.round).buffer(
            -smooth * 0.48,
            join_style=BufferJoinStyle.round,
        )
        envelope = envelope.simplify(simplify_tol * 0.85, preserve_topology=True)
        envelope = _clean_zone_geometry(envelope)
        env_area = float(getattr(envelope, "area", 0.0)) if envelope is not None else 0.0
        if envelope is not None and not envelope.is_empty and env_area >= 0.82 * orig_area:
            candidate = envelope
            cand_area = env_area
    if orig_area > 0.0 and (cand_area < 0.55 * orig_area or cand_area > 1.80 * orig_area):
        fallback = _clean_zone_geometry(geom.simplify(simplify_tol * 0.60, preserve_topology=True))
        if fallback is not None and not fallback.is_empty:
            return fallback
        return geom
    return candidate


def _build_zone_display_items(zone_records: list[dict[str, Any]], gdf) -> list[dict[str, Any]]:
    records_by_id = {str(item["zone_id"]): item for item in zone_records}
    gdf_by_id: dict[str, Any] = {}
    if gdf is not None and len(gdf) > 0:
        for _, row in gdf.iterrows():
            gdf_by_id[str(row["zone_id"])] = row.geometry
    items: list[dict[str, Any]] = []
    if not records_by_id:
        return items
    for zone_id, record in records_by_id.items():
        geom = record.get("geometry")
        if geom is None or getattr(geom, "is_empty", True):
            geom = gdf_by_id.get(zone_id)
        if geom is None or getattr(geom, "is_empty", True):
            continue
        display_geom = _prepare_display_geometry(geom)
        if display_geom is None or getattr(display_geom, "is_empty", True):
            continue
        rep = display_geom.representative_point()
        items.append(
            {
                "zone_id": zone_id,
                "record": record,
                "geom": display_geom,
                "anchor_x": float(rep.x),
                "anchor_y": float(rep.y),
                "bounds": tuple(float(v) for v in display_geom.bounds),
            }
        )
    return items


def _assign_zone_callout_positions(items: list[dict[str, Any]], canvas: dict[str, Any]) -> list[dict[str, Any]]:
    x_span = float(canvas["lon_max"] - canvas["lon_min"])
    y_span = float(canvas["lat_max"] - canvas["lat_min"])
    x_mid = 0.5 * (canvas["lon_min"] + canvas["lon_max"])
    left_items = [dict(item) for item in items if item["anchor_x"] <= x_mid]
    right_items = [dict(item) for item in items if item["anchor_x"] > x_mid]
    min_gap = max(0.038 * y_span, 0.0045)
    y_min = canvas["lat_min"] + 0.06 * y_span
    y_max = canvas["lat_max"] - 0.06 * y_span

    def _place(side_items: list[dict[str, Any]], side: str) -> list[dict[str, Any]]:
        if not side_items:
            return []
        side_items.sort(key=lambda item: item["anchor_y"], reverse=True)
        placed: list[dict[str, Any]] = []
        prev_y = y_max + min_gap
        for item in side_items:
            ly = min(float(item["anchor_y"]), prev_y - min_gap)
            ly = max(ly, y_min)
            item["label_y"] = ly
            if side == "left":
                item["label_x"] = canvas["lon_min"] + 0.065 * x_span
                item["elbow_x"] = item["label_x"] + 0.022 * x_span
                item["ha"] = "left"
            else:
                item["label_x"] = canvas["lon_max"] - 0.065 * x_span
                item["elbow_x"] = item["label_x"] - 0.022 * x_span
                item["ha"] = "right"
            item["va"] = "center"
            prev_y = ly
            placed.append(item)
        if placed and placed[-1]["label_y"] <= y_min + 1e-9 and len(placed) > 1:
            target_ys = np.linspace(y_max, y_min, len(placed))
            for item, ty in zip(placed, target_ys):
                item["label_y"] = float(ty)
        return placed

    return _place(left_items, "left") + _place(right_items, "right")


def _draw_zone_callout(ax, *, geom, zone_id: str, color, label_x: float, label_y: float, elbow_x: float, ha: str) -> None:
    rep = geom.representative_point()
    x0 = float(rep.x)
    y0 = float(rep.y)
    ax.plot(
        [x0, elbow_x, label_x],
        [y0, label_y, label_y],
        color=color,
        lw=1.05,
        zorder=7.5,
        alpha=0.96,
        path_effects=[pe.Stroke(linewidth=2.6, foreground="white", alpha=0.9), pe.Normal()],
    )
    txt = ax.text(
        label_x,
        label_y,
        zone_id,
        fontsize=7.5,
        fontweight="bold",
        ha=ha,
        va="center",
        color=color,
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=color, lw=0.95, alpha=0.97),
        zorder=8.2,
    )
    txt.set_path_effects([pe.withStroke(linewidth=1.0, foreground="white", alpha=0.92)])


def _render_zone_figure(
    *,
    mintpy_dir: Path,
    export_dir: Path,
    velocity: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    dates: list[str],
    zone_records: list[dict[str, Any]],
    gdf,
) -> tuple[str | None, str | None]:
    if gdf is None or not zone_records:
        return None, None
    unique_zone_records: list[dict[str, Any]] = []
    seen_zone_ids: set[str] = set()
    for record in zone_records:
        zone_id = str(record["zone_id"])
        if zone_id in seen_zone_ids:
            continue
        seen_zone_ids.add(zone_id)
        unique_zone_records.append(record)
    zone_records = unique_zone_records

    data = _load_mintpy_data(mintpy_dir)
    lat = data.get("lat")
    lon = data.get("lon")
    raw_velocity = np.asarray(data.get("vel"), dtype=np.float32) if data.get("vel") is not None else None
    if lat is None or lon is None:
        return None, None
    plot_velocity = np.asarray(velocity, dtype=np.float32)
    if raw_velocity is not None and raw_velocity.shape == lat.shape:
        plot_velocity = raw_velocity
    geo_valid = np.isfinite(lat) & np.isfinite(lon) & (lat > 0.1)
    aoi = cfg._AOI_BBOX
    if aoi is not None:
        s, n, w, e = aoi
        geo_valid &= (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
    canvas = _prepare_geo_canvas(data, geo_valid, target_cols=760)
    display_mask = np.isfinite(plot_velocity) & (plot_velocity != 0) & geo_valid
    tcoh = np.asarray(data.get("tcoh"), dtype=np.float32) if data.get("tcoh") is not None else None
    vstd = np.asarray(data.get("vstd"), dtype=np.float32) if data.get("vstd") is not None else None
    if tcoh is not None and tcoh.shape == plot_velocity.shape:
        stats_mask = display_mask & np.isfinite(tcoh) & (tcoh >= 0.5)
    else:
        stats_mask = display_mask.copy()
    if vstd is not None and vstd.shape == plot_velocity.shape:
        stats_mask &= np.isfinite(vstd) & (vstd < 80)
    lowq_mask = display_mask & ~stats_mask
    valid_vel = stats_mask
    if not np.any(display_mask):
        return None, None
    vmax = _symmetric_vlim(plot_velocity[valid_vel] if np.any(valid_vel) else plot_velocity[display_mask], pct=95)
    vel_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    aspect = (
        (canvas["lat_max"] - canvas["lat_min"]) /
        max((canvas["lon_max"] - canvas["lon_min"]) * np.cos(np.radians(canvas["lat_c"])), 0.001)
    )
    fig_h = float(np.clip(4.35 + aspect * 0.82, 5.3, 6.8))
    fig = plt.figure(figsize=(12.8, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.96, 1.20], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[0, 1])
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.42)
    _draw_hillshade_background(ax, canvas["hs_grid"], None, canvas["extent"], alpha=0.26)
    _draw_water_overlay(ax, canvas["water_grid"], canvas["extent"])
    sc = ax.scatter(
        lon[valid_vel],
        lat[valid_vel],
        c=plot_velocity[valid_vel],
        cmap=_CMAP_VEL,
        norm=vel_norm,
        s=4.0,
        linewidths=0.0,
        alpha=0.90,
        rasterized=True,
        zorder=4,
    )
    display_items = _build_zone_display_items(zone_records, gdf)
    if not display_items:
        return None, None
    line_records = _zone_plot_records(
        {
            int(str(item["zone_id"]).lstrip("Z")): item["record"]
            for item in display_items
        }
    )
    line_zone_ids = [str(item["zone_id"]) for item in line_records]
    palette = plt.cm.get_cmap("tab10", max(len(line_zone_ids), 1))
    zone_colors = {zone_id: palette(i) for i, zone_id in enumerate(line_zone_ids)}
    default_outline = (0.45, 0.45, 0.45, 0.75)

    placed_items = _assign_zone_callout_positions(display_items, canvas)
    for item in placed_items:
        zone_id = str(item["zone_id"])
        color = zone_colors.get(zone_id, default_outline)
        geom = item["geom"]
        _plot_geometry_outline(ax, geom, color="white", linewidth=3.2, linestyle="-", zorder=6.2)
        _plot_geometry_outline(ax, geom, color=color, linewidth=2.0, linestyle="-", zorder=6.4)
        _draw_zone_callout(
            ax,
            geom=geom,
            zone_id=zone_id,
            color=color,
            label_x=float(item["label_x"]),
            label_y=float(item["label_y"]),
            elbow_x=float(item["elbow_x"]),
            ha=str(item["ha"]),
        )
    ax.set_xlim(canvas["lon_min"], canvas["lon_max"])
    ax.set_ylim(canvas["lat_min"], canvas["lat_max"])
    ax.set_aspect(1.0 / np.cos(np.radians(canvas["lat_c"])))
    ax.set_title("Velocity Map With Regional Deformation Zones", fontsize=8.2, loc="left", fontweight="bold", pad=4)
    _subfig_label(ax, "a", x=0.02, y=0.96)
    ax.set_xlabel("Longitude (°E)", fontsize=7.5)
    ax.set_ylabel("Latitude (°N)", fontsize=7.5)
    _format_degree_axis(ax, 4)
    _add_north_arrow(ax, x=0.94, y=0.90, size=0.07)
    if canvas["water_grid"] is not None and np.any(canvas["water_grid"]):
        _add_water_legend(ax, loc="lower left", fontsize=5.4)
    if np.any(lowq_mask):
        has_hatch = _draw_lowq_hatch(ax, canvas, lowq_mask)
        if has_hatch:
            _add_uncertainty_legend(ax, loc="lower right", fontsize=5.2)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, extend="both")
    cbar.set_label("mm yr$^{-1}$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.3)
    ax.text(
        0.985,
        0.985,
        f"zones = {len(display_items)}\n"
        f"area = {sum(float(item['record']['area_km2']) for item in display_items):.2f} km²",
        transform=ax.transAxes,
        fontsize=5.0,
        ha="right",
        va="top",
        color="0.40",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.88", lw=0.35, alpha=0.86),
    )

    ax_ts.set_facecolor("white")
    for sp in ax_ts.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.42)
    date_ticks = [datetime.strptime(str(d), "%Y%m%d") for d in dates]
    for record in line_records:
        zone_id = str(record["zone_id"])
        color = zone_colors[zone_id]
        p25 = np.asarray(record.get("timeseries_p25_mm", []), dtype=np.float32)
        p50 = np.asarray(
            record.get(
                "timeseries_center_smooth_mm",
                record.get("timeseries_p50_smooth_mm", record.get("timeseries_p50_mm", [])),
            ),
            dtype=np.float32,
        )
        p75 = np.asarray(record.get("timeseries_p75_mm", []), dtype=np.float32)
        if p50.size != len(date_ticks):
            continue
        ax_ts.fill_between(date_ticks, p25, p75, color=color, alpha=0.14, linewidth=0.0, zorder=1)
        ax_ts.plot(
            date_ticks,
            p50,
            color=color,
            lw=2.4,
            solid_capstyle="round",
            label=f"{zone_id}  ({record['area_km2']:.2f} km²)",
            zorder=2,
        )
        ax_ts.scatter(
            date_ticks[-1],
            float(p50[-1]),
            s=18,
            color=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax_ts.text(
            date_ticks[-1],
            float(p50[-1]),
            zone_id,
            color=color,
            fontsize=6.8,
            fontweight="bold",
            va="center",
            ha="left",
        )
    ax_ts.axhline(0.0, color="0.55", lw=0.6, ls="--", zorder=0)
    ax_ts.set_title("Regional Cumulative Displacement", fontsize=8.2, loc="left", fontweight="bold", pad=4)
    _subfig_label(ax_ts, "b", x=0.02, y=0.96)
    ax_ts.set_ylabel("Cumulative displacement [mm]", fontsize=7.5)
    ax_ts.set_xlabel("Date", fontsize=7.5)
    ax_ts.grid(alpha=0.14, lw=0.45, axis="both")
    ax_ts.tick_params(labelsize=6.7)
    ax_ts.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=7))
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax_ts.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    if line_records:
        ymin = min(float(np.nanmin(np.asarray(rec.get("timeseries_p25_mm", []), dtype=np.float32))) for rec in line_records)
        ymax = max(float(np.nanmax(np.asarray(rec.get("timeseries_p75_mm", []), dtype=np.float32))) for rec in line_records)
        yrange = max(ymax - ymin, 1.0)
        ax_ts.set_ylim(ymin - 0.08 * yrange, ymax + 0.12 * yrange)
    if line_records:
        ax_ts.legend(
            loc="upper left",
            fontsize=6.4,
            framealpha=0.90,
            borderpad=0.28,
            handlelength=2.1,
            labelspacing=0.35,
        )
    ax_ts.text(
        0.985,
        0.02,
        f"selected curves = {len(line_records)}\n"
        f"criterion = region score",
        transform=ax_ts.transAxes,
        fontsize=5.0,
        ha="right",
        va="bottom",
        color="0.40",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="0.88", lw=0.35, alpha=0.85),
    )
    fig.suptitle("Detected Deformation Zones And Regional Cumulative Displacement", fontsize=9, x=0.06, ha="left")
    fig_dir = export_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / "velocity_map_zones.png"
    fig.subplots_adjust(left=0.055, right=0.975, bottom=0.12, top=0.90, wspace=0.18)
    _save_figure(fig, out_png)
    plt.close(fig)
    return str(out_png), str(out_png.with_suffix(".pdf"))


def _summarize_support_graph_candidate(
    candidate_id: str,
    support_indices: np.ndarray,
    ctx: dict[str, Any],
    *,
    buffer_radius_deg: float,
    buffer_radius_km: float,
) -> dict[str, Any] | None:
    idx = np.asarray(support_indices, dtype=np.int32)
    if idx.size == 0:
        return None
    zone_series = np.asarray(ctx["series"][idx], dtype=np.float32)
    profile = _series_profile(zone_series)
    if profile is None:
        return None

    area_km2 = float(np.nansum(ctx["pixel_area_km2"][idx]))
    aoi_area_km2 = float(max(ctx["aoi_area_km2"], 1.0e-6))
    area_fraction = float(area_km2 / aoi_area_km2)
    point_count = int(idx.size)
    center_trace = np.asarray(profile["timeseries_center_mm"], dtype=np.float32)
    center_demeaned = center_trace - float(np.nanmean(center_trace))
    center_norm = float(np.linalg.norm(center_demeaned))
    if center_norm <= 1.0e-6:
        center_unit = np.zeros_like(center_demeaned, dtype=np.float32)
    else:
        center_unit = (center_demeaned / center_norm).astype(np.float32)
    corr_to_center = np.sum(ctx["series_unit"][idx] * center_unit[None, :], axis=1).astype(np.float32)
    medoid_local = int(np.argmax(corr_to_center))
    medoid_idx = int(idx[medoid_local])
    internal_trace_corr = float(np.nanmedian(np.sum(ctx["series_unit"][idx] * ctx["series_unit"][medoid_idx][None, :], axis=1)))
    temporal_coherence = float(np.nanmedian(ctx["temporal_coherence"][idx]))
    activity_level = float(np.nanmedian(ctx["peak_abs_disp_mm"][idx]))

    in_region = np.zeros(ctx["n_support_points"], dtype=bool)
    in_region[idx] = True
    boundary_idx = np.unique(ctx["neighbor_idx"][idx].reshape(-1)) if ctx["neighbor_idx"].size else np.empty(0, dtype=np.int32)
    boundary_idx = boundary_idx[(boundary_idx >= 0) & ~in_region[boundary_idx]]
    inside_level = 0.60 * float(np.nanmedian(np.abs(ctx["net_disp_mm"][idx]))) + 0.40 * float(np.nanmedian(ctx["peak_abs_disp_mm"][idx]))
    if boundary_idx.size > 0:
        outside_level = 0.60 * float(np.nanmedian(np.abs(ctx["net_disp_mm"][boundary_idx]))) + 0.40 * float(np.nanmedian(ctx["peak_abs_disp_mm"][boundary_idx]))
    else:
        outside_level = 0.0
    boundary_contrast = float(np.clip((inside_level - outside_level) / max(inside_level, 1.0), 0.0, 1.5))

    lon_pts = np.asarray(ctx["longitude"][idx], dtype=np.float32)
    lat_pts = np.asarray(ctx["latitude"][idx], dtype=np.float32)
    x_pts = np.asarray(ctx["x_km"][idx], dtype=np.float32)
    y_pts = np.asarray(ctx["y_km"][idx], dtype=np.float32)
    tmp_mask = np.ones(idx.size, dtype=bool)
    geom = _zone_geom_from_points(tmp_mask, lon_pts, lat_pts, buffer_radius=buffer_radius_deg)
    geom_proj = _zone_geom_from_points(
        tmp_mask,
        x_pts,
        y_pts,
        buffer_radius=buffer_radius_km,
        ratio=SUPPORT_GRAPH_COMPACTNESS_RATIO,
    )
    compactness = 0.0
    if geom_proj is not None and not geom_proj.is_empty:
        compactness = float(np.clip(area_km2 / max(float(getattr(geom_proj, "area", area_km2)), area_km2, 1.0e-6), 0.0, 1.0))
    region_score = float(
        np.sqrt(max(area_fraction, 1.0e-8))
        * max(internal_trace_corr, 0.0)
        * max(temporal_coherence, 0.0)
        * max(boundary_contrast, 0.05)
        * np.log1p(max(activity_level, 0.0))
    )
    zone_velocity = np.asarray(ctx["velocity"][idx], dtype=np.float32)
    strict_count = int(np.sum(ctx["strict_flag"][idx]))
    relaxed_count = int(np.sum(ctx["relaxed_flag"][idx]))
    bbox_wsen = [
        float(np.nanmin(lon_pts)),
        float(np.nanmin(lat_pts)),
        float(np.nanmax(lon_pts)),
        float(np.nanmax(lat_pts)),
    ]
    centroid_lonlat = [float(np.nanmean(lon_pts)), float(np.nanmean(lat_pts))]
    return {
        "candidate_id": candidate_id,
        "support_indices": idx,
        "support_point_count": point_count,
        "core_point_count": int(np.sum(np.asarray(ctx["activity"][idx], dtype=np.float32) >= float(np.nanpercentile(ctx["activity"], 75)))),
        "area_km2": area_km2,
        "aoi_area_fraction": area_fraction,
        "internal_trace_corr": internal_trace_corr,
        "temporal_coherence": temporal_coherence,
        "boundary_contrast": boundary_contrast,
        "compactness": compactness,
        "activity_level": activity_level,
        "region_score": region_score,
        "zone_type": _classify_zone_type(zone_velocity),
        "bbox_wsen": bbox_wsen,
        "centroid_lonlat": centroid_lonlat,
        "mean_velocity_mm_yr": float(np.nanmean(zone_velocity)),
        "median_velocity_mm_yr": float(np.nanmedian(zone_velocity)),
        "p10_velocity_mm_yr": float(np.nanpercentile(zone_velocity, 10)),
        "p90_velocity_mm_yr": float(np.nanpercentile(zone_velocity, 90)),
        "max_abs_velocity_mm_yr": float(np.nanmax(np.abs(zone_velocity))),
        "probability_mean": float(np.nanmean(ctx["saliency"][idx])),
        "strict_point_count": strict_count,
        "relaxed_point_count": relaxed_count,
        "forecast_point_count": point_count,
        "median_abs_velocity_mm_yr": float(np.nanmedian(np.abs(zone_velocity))),
        "geometry": geom,
        "geometry_proj": geom_proj,
        **profile,
    }


def _candidate_center_trace_corr(a: dict[str, Any], b: dict[str, Any]) -> float:
    ta = np.asarray(a.get("timeseries_center_mm", []), dtype=np.float32)
    tb = np.asarray(b.get("timeseries_center_mm", []), dtype=np.float32)
    if ta.size == 0 or tb.size == 0 or ta.size != tb.size:
        return 0.0
    if np.allclose(ta, ta[0]) or np.allclose(tb, tb[0]):
        return 0.0
    corr = np.corrcoef(ta, tb)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def _candidate_centroid_distance_km(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax = float(a["centroid_lonlat"][0])
    ay = float(a["centroid_lonlat"][1])
    bx = float(b["centroid_lonlat"][0])
    by = float(b["centroid_lonlat"][1])
    lat_c = 0.5 * (ay + by)
    dx = (ax - bx) * 111.320 * np.cos(np.radians(lat_c))
    dy = (ay - by) * 110.540
    return float(np.hypot(dx, dy))


def _merge_support_graph_candidates(
    retained_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    *,
    ctx: dict[str, Any],
    buffer_radius_deg: float,
    buffer_radius_km: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not retained_rows or not rejected_rows:
        return retained_rows, rejected_rows, {
            "merge_trace_corr_min": SUPPORT_GRAPH_MERGE_TRACE_CORR_MIN,
            "merge_distance_km": SUPPORT_GRAPH_MERGE_DISTANCE_KM,
            "merge_score_ratio_min": SUPPORT_GRAPH_MERGE_SCORE_RATIO_MIN,
            "merged_candidate_ids": [],
        }

    merged_ids: list[str] = []
    remaining_rejected = [dict(item) for item in rejected_rows]
    merged_retained: list[dict[str, Any]] = []
    next_generated_id = 1
    for anchor in retained_rows:
        anchor_row = dict(anchor)
        anchor_ids = [str(anchor_row.get("candidate_id", f"M{next_generated_id}"))]
        changed = True
        while changed:
            changed = False
            new_remaining: list[dict[str, Any]] = []
            for candidate in remaining_rejected:
                same_type = str(candidate.get("zone_type", "")) == str(anchor_row.get("zone_type", ""))
                score_ok = float(candidate.get("region_score", 0.0)) >= max(
                    SUPPORT_GRAPH_MERGE_SCORE_FLOOR,
                    SUPPORT_GRAPH_MERGE_SCORE_RATIO_MIN * float(anchor_row.get("region_score", 0.0)),
                )
                corr = _candidate_center_trace_corr(anchor_row, candidate)
                dist_km = _candidate_centroid_distance_km(anchor_row, candidate)
                if same_type and score_ok and corr >= SUPPORT_GRAPH_MERGE_TRACE_CORR_MIN and dist_km <= SUPPORT_GRAPH_MERGE_DISTANCE_KM:
                    union_idx = np.unique(
                        np.concatenate(
                            [
                                np.asarray(anchor_row["support_indices"], dtype=np.int32),
                                np.asarray(candidate["support_indices"], dtype=np.int32),
                            ]
                        )
                    )
                    merged = _summarize_support_graph_candidate(
                        f"M{next_generated_id}",
                        union_idx,
                        ctx,
                        buffer_radius_deg=buffer_radius_deg,
                        buffer_radius_km=buffer_radius_km,
                    )
                    if merged is not None:
                        merged["retention_mode"] = "merged_adjacent_candidates"
                        merged["merged_candidate_ids"] = anchor_ids + [str(candidate.get("candidate_id", f"R{next_generated_id}"))]
                        anchor_ids = list(merged["merged_candidate_ids"])
                        anchor_row = merged
                        merged_ids.extend(anchor_ids[-1:])
                        changed = True
                        next_generated_id += 1
                        continue
                new_remaining.append(candidate)
            remaining_rejected = new_remaining
        if "retention_mode" not in anchor_row:
            anchor_row["retention_mode"] = str(anchor.get("retention_mode", "regional_main"))
        if anchor_ids != [str(anchor.get("candidate_id", f"M{next_generated_id}"))]:
            anchor_row["merged_candidate_ids"] = anchor_ids
        merged_retained.append(anchor_row)

    return merged_retained, remaining_rejected, {
        "merge_trace_corr_min": SUPPORT_GRAPH_MERGE_TRACE_CORR_MIN,
        "merge_distance_km": SUPPORT_GRAPH_MERGE_DISTANCE_KM,
        "merge_score_ratio_min": SUPPORT_GRAPH_MERGE_SCORE_RATIO_MIN,
        "merged_candidate_ids": merged_ids,
    }


def _build_raster_growth_context(
    *,
    rel0_cube: np.ndarray,
    velocity: np.ndarray,
    valid_mask: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> dict[str, Any]:
    finite_mask = valid_mask & np.all(np.isfinite(rel0_cube), axis=-1)
    rows, cols = np.where(finite_mask)
    series = rel0_cube[finite_mask].astype(np.float32)
    demeaned = (series - np.nanmean(series, axis=1, keepdims=True)).astype(np.float32)
    norms = np.linalg.norm(demeaned, axis=1).astype(np.float32)
    net_disp = (series[:, -1] - series[:, 0]).astype(np.float32)
    peak_abs = np.nanmax(np.abs(series), axis=1).astype(np.float32)
    diffs = np.diff(series, axis=1).astype(np.float32)
    meaningful = np.abs(diffs) > 0.25
    path_length = np.sum(np.abs(diffs) * meaningful, axis=1).astype(np.float32)
    pos_steps = np.sum((diffs > 0) & meaningful, axis=1).astype(np.float32)
    neg_steps = np.sum((diffs < 0) & meaningful, axis=1).astype(np.float32)
    total_steps = np.maximum(pos_steps + neg_steps, 1.0)
    sign_consistency = np.divide(
        np.maximum(pos_steps, neg_steps),
        total_steps,
        out=np.zeros_like(total_steps, dtype=np.float32),
        where=total_steps > 0,
    ).astype(np.float32)
    net_to_path_ratio = np.divide(
        np.abs(net_disp),
        np.maximum(path_length, 1.0e-3),
        out=np.zeros_like(net_disp, dtype=np.float32),
        where=path_length > 0,
    ).astype(np.float32)
    temporal_coherence = np.clip(0.5 * net_to_path_ratio + 0.5 * sign_consistency, 0.0, 1.0).astype(np.float32)
    x_km_full, y_km_full, _ = _project_lonlat_km(longitude, latitude)
    return {
        "finite_mask": finite_mask,
        "rows": rows.astype(np.int32),
        "cols": cols.astype(np.int32),
        "series": series,
        "demeaned": demeaned,
        "norms": norms,
        "net_disp": net_disp,
        "peak_abs": peak_abs,
        "temporal_coherence": temporal_coherence,
        "velocity": np.asarray(velocity, dtype=np.float32),
        "x_km_full": x_km_full.astype(np.float32),
        "y_km_full": y_km_full.astype(np.float32),
    }


def _expand_zone_on_raster(
    row: dict[str, Any],
    *,
    ctx: dict[str, Any],
    growth_ctx: dict[str, Any],
    rel0_cube: np.ndarray,
    velocity: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    pixel_area_km2: np.ndarray,
    strict_flag: np.ndarray,
    relaxed_flag: np.ndarray,
    buffer_radius_deg: float,
    buffer_radius_km: float,
    occupied_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    seed_mask = np.zeros_like(velocity, dtype=bool)
    support_indices = np.asarray(row["support_indices"], dtype=np.int32)
    seed_mask[ctx["rows"][support_indices], ctx["cols"][support_indices]] = True
    center = np.asarray(row.get("timeseries_center_mm", []), dtype=np.float32)
    if center.size == 0:
        return seed_mask, row
    center_demeaned = center - float(np.nanmean(center))
    center_norm = float(np.linalg.norm(center_demeaned))
    if center_norm <= 1.0e-6:
        return seed_mask, row
    center_unit = (center_demeaned / center_norm).astype(np.float32)

    corr = np.sum(growth_ctx["demeaned"] * center_unit[None, :], axis=1)
    corr = np.divide(
        corr,
        np.maximum(growth_ctx["norms"], 1.0e-6),
        out=np.zeros_like(corr, dtype=np.float32),
        where=growth_ctx["norms"] > 1.0e-6,
    ).astype(np.float32)
    corr_map = np.full_like(velocity, np.nan, dtype=np.float32)
    corr_map[growth_ctx["rows"], growth_ctx["cols"]] = corr

    net_map = np.full_like(velocity, np.nan, dtype=np.float32)
    net_map[growth_ctx["rows"], growth_ctx["cols"]] = growth_ctx["net_disp"]
    peak_map = np.full_like(velocity, np.nan, dtype=np.float32)
    peak_map[growth_ctx["rows"], growth_ctx["cols"]] = growth_ctx["peak_abs"]
    coherence_map = np.full_like(velocity, np.nan, dtype=np.float32)
    coherence_map[growth_ctx["rows"], growth_ctx["cols"]] = growth_ctx["temporal_coherence"]

    zone_type = str(row.get("zone_type", "mixed"))
    zone_net_abs = abs(float(row.get("temporal_net_disp_mm", 0.0)))
    zone_peak = float(row.get("activity_level", 0.0))
    zone_vel = float(row.get("median_abs_velocity_mm_yr", 0.0))
    corr_min = float(max(ZONE_RASTER_GROW_CORR_MIN, float(row.get("internal_trace_corr", 0.90)) - 0.08))
    coh_min = float(max(0.45, float(row.get("temporal_coherence", 0.55)) - 0.08))
    peak_min = float(max(ZONE_RASTER_GROW_PEAK_MIN_MM, 0.35 * max(zone_peak, 0.0)))
    net_min = float(max(ZONE_RASTER_GROW_NET_MIN_MM, 0.40 * max(zone_net_abs, 0.0)))
    vel_min = float(max(ZONE_RASTER_GROW_VEL_MIN_MM_YR, 0.50 * max(zone_vel, 0.0)))

    finite_mask = growth_ctx["finite_mask"] & ~occupied_mask
    if zone_type == "subsidence":
        sign_mask = (net_map <= -net_min) | (velocity <= -vel_min)
        bridge_mask = (net_map <= -(0.75 * net_min)) | (velocity <= -(0.80 * vel_min))
    elif zone_type == "uplift":
        sign_mask = (net_map >= net_min) | (velocity >= vel_min)
        bridge_mask = (net_map >= 0.75 * net_min) | (velocity >= 0.80 * vel_min)
    else:
        sign_mask = np.abs(net_map) >= net_min
        bridge_mask = np.abs(net_map) >= 0.75 * net_min

    active = finite_mask & sign_mask & (peak_map >= peak_min) & (corr_map >= corr_min) & (coherence_map >= coh_min)
    bridge = finite_mask & bridge_mask & (peak_map >= 0.70 * peak_min) & (corr_map >= max(0.86, corr_min - 0.05)) & (coherence_map >= max(0.40, coh_min - 0.05))
    structure = np.ones((3, 3), dtype=bool)
    grown = binary_propagation(seed_mask, structure=structure, mask=(active | seed_mask))
    grown = binary_propagation(grown, structure=structure, mask=(bridge | grown))
    grown = binary_closing(grown, structure=structure, iterations=2)
    grown = binary_fill_holes(grown)
    grown &= ~occupied_mask
    if not np.any(grown):
        grown = seed_mask & ~occupied_mask

    grown_coords = np.column_stack([growth_ctx["x_km_full"][grown], growth_ctx["y_km_full"][grown]])
    if grown_coords.size > 0:
        absorb_radius_km = 1.5
        strong_corr_min = max(0.95, corr_min + 0.03)
        if zone_type == "subsidence" and float(row.get("area_km2", 0.0)) < 1.0:
            absorb_radius_km = 6.0
            strong_corr_min = max(0.97, corr_min + 0.05)
        strong = finite_mask & sign_mask & (peak_map >= 0.80 * peak_min) & (corr_map >= strong_corr_min) & (coherence_map >= coh_min)
        strong &= ~grown
        if np.any(strong):
            tree = cKDTree(grown_coords)
            cand_coords = np.column_stack([growth_ctx["x_km_full"][strong], growth_ctx["y_km_full"][strong]])
            dists, _ = tree.query(cand_coords, k=1)
            supplement = np.zeros_like(grown, dtype=bool)
            supplement_idx = np.flatnonzero(strong)
            supplement_flat = supplement.reshape(-1)
            supplement_flat[supplement_idx[dists <= absorb_radius_km]] = True
            grown |= supplement

    series_mask = grown & growth_ctx["finite_mask"]
    zone_series = rel0_cube[series_mask]
    profile = _series_profile(zone_series) if zone_series.ndim == 2 and zone_series.shape[0] >= MIN_ZONE_TIMESERIES_POINTS else None
    if profile is None:
        profile = {k: row[k] for k in row.keys() if k.startswith("timeseries_") or k.startswith("temporal_")}

    zone_velocity = velocity[grown]
    area_km2 = float(np.nansum(pixel_area_km2[grown]))
    geom = _zone_geom_from_points(grown, longitude, latitude, buffer_radius=buffer_radius_deg)
    geom_proj = _zone_geom_from_points(grown, growth_ctx["x_km_full"], growth_ctx["y_km_full"], buffer_radius=buffer_radius_km, ratio=SUPPORT_GRAPH_COMPACTNESS_RATIO)
    compactness = float(row.get("compactness", 0.0))
    if geom_proj is not None and not geom_proj.is_empty:
        compactness = float(np.clip(area_km2 / max(float(getattr(geom_proj, "area", area_km2)), area_km2, 1.0e-6), 0.0, 1.0))
    updated = dict(row)
    updated.update(
        {
            "area_km2": area_km2,
            "pixel_count": int(np.sum(grown)),
            "bbox_wsen": [
                float(np.nanmin(longitude[grown])),
                float(np.nanmin(latitude[grown])),
                float(np.nanmax(longitude[grown])),
                float(np.nanmax(latitude[grown])),
            ],
            "centroid_lonlat": [float(np.nanmean(longitude[grown])), float(np.nanmean(latitude[grown]))],
            "mean_velocity_mm_yr": float(np.nanmean(zone_velocity)),
            "median_velocity_mm_yr": float(np.nanmedian(zone_velocity)),
            "p10_velocity_mm_yr": float(np.nanpercentile(zone_velocity, 10)),
            "p90_velocity_mm_yr": float(np.nanpercentile(zone_velocity, 90)),
            "max_abs_velocity_mm_yr": float(np.nanmax(np.abs(zone_velocity))),
            "strict_point_count": int(np.sum(strict_flag[grown])),
            "relaxed_point_count": int(np.sum(relaxed_flag[grown])),
            "forecast_point_count": int(np.sum((strict_flag | relaxed_flag)[grown])),
            "aoi_area_fraction": float(area_km2 / max(float(ctx.get("aoi_area_km2", 1.0)), 1.0e-6)),
            "median_abs_velocity_mm_yr": float(np.nanmedian(np.abs(zone_velocity))),
            "compactness": compactness,
            "geometry": geom,
            "geometry_proj": geom_proj,
            **profile,
        }
    )
    return grown, updated


def _retain_support_graph_candidates(
    candidate_rows: list[dict[str, Any]],
    *,
    n_support_points: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    default_min_support_points = int(max(256, np.ceil(SUPPORT_GRAPH_POINT_RATIO_MIN * max(n_support_points, 1))))
    default_salient_min_points = int(max(SUPPORT_GRAPH_SALIENT_MIN_POINTS, np.ceil(SUPPORT_GRAPH_SALIENT_POINT_RATIO_MIN * max(n_support_points, 1))))
    if not candidate_rows:
        return [], [], {
            "largest_candidate_area_fraction": 0.0,
            "best_region_score": 0.0,
            "min_area_fraction": SUPPORT_GRAPH_AREA_FRACTION_MIN,
            "min_support_points": default_min_support_points,
            "salient_min_points": default_salient_min_points,
            "min_internal_trace_corr": SUPPORT_GRAPH_INTERNAL_TRACE_CORR_MIN,
            "min_temporal_coherence": SUPPORT_GRAPH_TEMPORAL_COHERENCE_MIN,
            "min_compactness": SUPPORT_GRAPH_COMPACTNESS_MIN,
            "min_region_score": 0.0,
            "salient_region_score": 0.0,
            "salient_temporal_coherence": SUPPORT_GRAPH_SALIENT_TEMPORAL_COHERENCE_MIN,
            "salient_boundary_contrast": SUPPORT_GRAPH_SALIENT_BOUNDARY_CONTRAST_MIN,
            "salient_activity_threshold": 0.0,
            "fallback_triggered": False,
            "fallback_reason": "",
        }

    ordered = sorted(candidate_rows, key=lambda item: float(item["region_score"]), reverse=True)
    largest_candidate_area_fraction = float(max(item["aoi_area_fraction"] for item in ordered))
    best_region_score = float(max(item["region_score"] for item in ordered))
    min_area_fraction = float(max(SUPPORT_GRAPH_AREA_FRACTION_MIN, 0.25 * largest_candidate_area_fraction))
    min_support_points = int(max(256, np.ceil(SUPPORT_GRAPH_POINT_RATIO_MIN * max(n_support_points, 1))))
    salient_min_points = int(max(SUPPORT_GRAPH_SALIENT_MIN_POINTS, np.ceil(SUPPORT_GRAPH_SALIENT_POINT_RATIO_MIN * max(n_support_points, 1))))
    min_region_score = float(SUPPORT_GRAPH_REGION_SCORE_RATIO_MIN * best_region_score)
    salient_score_ratio = float(SUPPORT_GRAPH_SALIENT_REGION_SCORE_RATIO * best_region_score)
    salient_activity_threshold = float(np.nanpercentile([item["activity_level"] for item in ordered], SUPPORT_GRAPH_SALIENT_ACTIVITY_PERCENTILE))
    retained: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for item in ordered:
        reasons: list[str] = []
        if float(item["aoi_area_fraction"]) < min_area_fraction:
            reasons.append("area_fraction_below_threshold")
        if int(item["support_point_count"]) < min_support_points:
            reasons.append("support_point_count_below_threshold")
        if float(item["internal_trace_corr"]) < SUPPORT_GRAPH_INTERNAL_TRACE_CORR_MIN:
            reasons.append("internal_trace_corr_below_threshold")
        if float(item["temporal_coherence"]) < SUPPORT_GRAPH_TEMPORAL_COHERENCE_MIN:
            reasons.append("temporal_coherence_below_threshold")
        if float(item["compactness"]) < SUPPORT_GRAPH_COMPACTNESS_MIN:
            reasons.append("compactness_below_threshold")
        if float(item["region_score"]) < min_region_score:
            reasons.append("region_score_below_threshold")
        salient_exception = (
            float(item["region_score"]) >= salient_score_ratio
            and int(item["support_point_count"]) >= salient_min_points
            and float(item["internal_trace_corr"]) >= SUPPORT_GRAPH_INTERNAL_TRACE_CORR_MIN
            and float(item["temporal_coherence"]) >= SUPPORT_GRAPH_SALIENT_TEMPORAL_COHERENCE_MIN
            and float(item["boundary_contrast"]) >= SUPPORT_GRAPH_SALIENT_BOUNDARY_CONTRAST_MIN
            and float(item["activity_level"]) >= salient_activity_threshold
        )
        if salient_exception:
            retained.append({**item, "retention_mode": "salient_exception"})
        elif reasons:
            rejected.append({**item, "reasons": reasons})
        else:
            retained.append({**item, "retention_mode": "regional_main"})

    fallback_triggered = False
    fallback_reason = ""
    if not retained and ordered:
        best = ordered[0]
        if (
            int(best["support_point_count"]) >= min_support_points
            and float(best["internal_trace_corr"]) >= 0.55
            and float(best["temporal_coherence"]) >= 0.50
        ):
            retained = [{**best, "retention_mode": "fallback_best"}]
            fallback_triggered = True
            fallback_reason = "retain_best_candidate_due_to_empty_threshold_pass"

    return retained, rejected, {
        "largest_candidate_area_fraction": largest_candidate_area_fraction,
        "best_region_score": best_region_score,
        "min_area_fraction": min_area_fraction,
        "min_support_points": min_support_points,
        "salient_min_points": salient_min_points,
        "min_internal_trace_corr": SUPPORT_GRAPH_INTERNAL_TRACE_CORR_MIN,
        "min_temporal_coherence": SUPPORT_GRAPH_TEMPORAL_COHERENCE_MIN,
        "min_compactness": SUPPORT_GRAPH_COMPACTNESS_MIN,
        "min_region_score": min_region_score,
        "salient_region_score": salient_score_ratio,
        "salient_temporal_coherence": SUPPORT_GRAPH_SALIENT_TEMPORAL_COHERENCE_MIN,
        "salient_boundary_contrast": SUPPORT_GRAPH_SALIENT_BOUNDARY_CONTRAST_MIN,
        "salient_activity_threshold": salient_activity_threshold,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
    }


def _detect_deformation_zones_support_graph(
    mintpy_dir: str | Path,
    *,
    qc_report_dir: str | Path,
    export_dir: str | Path,
    detector_mode: str,
    detection_domain: str,
    zone_semantics: str,
    output_geometry: str,
    forecast_point_scope: str,
) -> dict[str, Any]:
    mintpy_dir = Path(mintpy_dir).resolve()
    qc_report_dir = Path(qc_report_dir).resolve()
    export_dir = Path(export_dir).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = export_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    velocity_path = _resolve_velocity_raster(export_dir)
    if velocity_path.name != "velocity.tif" or velocity_path.parent != export_dir:
        logger.info(f"[zone] velocity raster resolved to {velocity_path}")
    velocity, ref_profile = _read_tif(velocity_path)
    ps_score, _ = _read_tif(qc_report_dir / "ps_score.tif")
    strict_mask, _ = _read_tif(qc_report_dir / "mask_ps_strict.tif")
    relaxed_mask, _ = _read_tif(qc_report_dir / "mask_ps_relaxed.tif")
    tcoh, _ = _read_tif(qc_report_dir / "tcoh_component.tif")
    metrics = _load_metrics(qc_report_dir)

    data = _load_mintpy_data(mintpy_dir)
    latitude = np.asarray(data.get("lat"), dtype=np.float32)
    longitude = np.asarray(data.get("lon"), dtype=np.float32)
    if latitude.shape != velocity.shape or longitude.shape != velocity.shape:
        raise RuntimeError(f"geometryRadar 与 {velocity_path.name} 尺寸不一致，无法检测形变区。")
    dates, rel0_cube = _load_rel0_timeseries(mintpy_dir)
    valid_mask = np.isfinite(latitude) & np.isfinite(longitude) & np.isfinite(velocity) & (latitude > 0.1)
    pixel_area_km2 = _pixel_area_km2(ref_profile, latitude)
    zone_support_valid = valid_mask & (strict_mask.astype(bool) | relaxed_mask.astype(bool))
    buffer_radius_deg = _estimate_zone_buffer_radius(longitude, latitude, zone_support_valid)

    ctx = _build_support_graph_context(
        rel0_cube=rel0_cube,
        dates=dates,
        latitude=latitude,
        longitude=longitude,
        pixel_area_km2=pixel_area_km2,
        strict_flag=strict_mask.astype(bool),
        relaxed_flag=relaxed_mask.astype(bool),
        tcoh=np.asarray(tcoh, dtype=np.float32),
        ps_score=np.asarray(ps_score, dtype=np.float32),
        valid_pair_ratio=np.asarray(metrics["valid_pair_ratio"], dtype=np.float32),
        maincc_ratio=np.asarray(metrics["mainCC_ratio"], dtype=np.float32),
        velocity=np.asarray(velocity, dtype=np.float32),
        valid_mask=valid_mask,
    )
    buffer_radius_km = float(max(0.75 * float(ctx.get("median_edge_km", 0.0)), 0.02))
    candidate_components, graph_summary = _build_support_graph_candidates(ctx)

    candidate_rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidate_components, start=1):
        row = _summarize_support_graph_candidate(
            f"C{idx}",
            np.asarray(candidate["support_indices"], dtype=np.int32),
            ctx,
            buffer_radius_deg=buffer_radius_deg,
            buffer_radius_km=buffer_radius_km,
        )
        if row is not None:
            candidate_rows.append(row)
    retained_rows, rejected_candidates, retention_summary = _retain_support_graph_candidates(
        candidate_rows,
        n_support_points=int(ctx.get("n_support_points", 0)),
    )
    retained_rows, rejected_candidates, merge_summary = _merge_support_graph_candidates(
        retained_rows,
        rejected_candidates,
        ctx=ctx,
        buffer_radius_deg=buffer_radius_deg,
        buffer_radius_km=buffer_radius_km,
    )
    growth_ctx = _build_raster_growth_context(
        rel0_cube=rel0_cube,
        velocity=np.asarray(velocity, dtype=np.float32),
        valid_mask=valid_mask,
        longitude=longitude,
        latitude=latitude,
    )

    zone_id_raster = np.zeros(velocity.shape, dtype=np.int32)
    zone_records_map: dict[int, dict[str, Any]] = {}
    occupied_mask = np.zeros_like(zone_id_raster, dtype=bool)
    for zid, row in enumerate(retained_rows, start=1):
        expanded_mask, row = _expand_zone_on_raster(
            row,
            ctx=ctx,
            growth_ctx=growth_ctx,
            rel0_cube=rel0_cube,
            velocity=np.asarray(velocity, dtype=np.float32),
            longitude=longitude,
            latitude=latitude,
            pixel_area_km2=pixel_area_km2,
            strict_flag=strict_mask.astype(bool),
            relaxed_flag=relaxed_mask.astype(bool),
            buffer_radius_deg=buffer_radius_deg,
            buffer_radius_km=buffer_radius_km,
            occupied_mask=occupied_mask,
        )
        zone_id_raster[expanded_mask] = int(zid)
        occupied_mask |= expanded_mask
        support_indices = np.asarray(row["support_indices"], dtype=np.int32)
        zone_records_map[zid] = {
            "zone_id": f"Z{zid}",
            "zone_type": row["zone_type"],
            "area_km2": float(row["area_km2"]),
            "pixel_count": int(row["pixel_count"]),
            "bbox_wsen": row["bbox_wsen"],
            "centroid_lonlat": row["centroid_lonlat"],
            "mean_velocity_mm_yr": float(row["mean_velocity_mm_yr"]),
            "median_velocity_mm_yr": float(row["median_velocity_mm_yr"]),
            "p10_velocity_mm_yr": float(row["p10_velocity_mm_yr"]),
            "p90_velocity_mm_yr": float(row["p90_velocity_mm_yr"]),
            "max_abs_velocity_mm_yr": float(row["max_abs_velocity_mm_yr"]),
            "probability_mean": float(row["probability_mean"]),
            "strict_point_count": int(row["strict_point_count"]),
            "relaxed_point_count": int(row["relaxed_point_count"]),
            "forecast_point_count": int(row["forecast_point_count"]),
            "support_point_count": int(row["support_point_count"]),
            "aoi_area_fraction": float(row["aoi_area_fraction"]),
            "median_abs_velocity_mm_yr": float(row["median_abs_velocity_mm_yr"]),
            "internal_trace_corr": float(row["internal_trace_corr"]),
            "temporal_coherence": float(row["temporal_coherence"]),
            "boundary_contrast": float(row["boundary_contrast"]),
            "compactness": float(row["compactness"]),
            "activity_level": float(row["activity_level"]),
            "region_score": float(row["region_score"]),
            "candidate_id": str(row["candidate_id"]),
            "retention_mode": str(row.get("retention_mode", "regional_main")),
            "merged_candidate_ids": [str(v) for v in row.get("merged_candidate_ids", [])],
            "geometry": row["geometry"],
            "geometry_proj": row["geometry_proj"],
            "support_indices": support_indices,
            **{k: v for k, v in row.items() if k.startswith("timeseries_") or k.startswith("temporal_")},
        }

    zone_records, gdf = _shape_records(
        zone_id_raster,
        ref_profile,
        zone_records_map,
        longitude=longitude,
        latitude=latitude,
    )
    vector_paths = _write_zone_vectors(features_out=zone_records, gdf=gdf, export_dir=export_dir)

    probability_raster = np.full(velocity.shape, np.nan, dtype=np.float32)
    if int(ctx.get("n_support_points", 0)) > 0:
        probability_raster[ctx["rows"], ctx["cols"]] = np.asarray(ctx["saliency"], dtype=np.float32)
    probability_path = _write_tif(
        export_dir / "deformation_zone_probability.tif",
        probability_raster.astype(np.float32),
        ref_profile,
        dtype="float32",
        nodata=np.nan,
    )
    zone_mask_path = _write_tif(
        export_dir / "deformation_zone_mask.tif",
        (zone_id_raster > 0).astype(np.uint8),
        ref_profile,
        dtype="uint8",
        nodata=0,
    )
    zone_id_path = _write_tif(
        export_dir / "deformation_zone_id.tif",
        zone_id_raster.astype(np.int32),
        ref_profile,
        dtype="int32",
        nodata=0,
    )

    csv_path = export_dir / "deformation_zones.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "zone_id",
                "zone_type",
                "area_km2",
                "pixel_count",
                "bbox_wsen",
                "centroid_lonlat",
                "mean_velocity_mm_yr",
                "median_velocity_mm_yr",
                "p10_velocity_mm_yr",
                "p90_velocity_mm_yr",
                "max_abs_velocity_mm_yr",
                "probability_mean",
                "strict_point_count",
                "relaxed_point_count",
                "forecast_point_count",
                "aoi_area_fraction",
                "internal_trace_corr",
                "temporal_coherence",
                "boundary_contrast",
                "compactness",
                "activity_level",
                "region_score",
            ]
        )
        for record in zone_records_map.values():
            writer.writerow(
                [
                    record["zone_id"],
                    record["zone_type"],
                    f"{record['area_km2']:.6f}",
                    int(record["pixel_count"]),
                    json.dumps(record["bbox_wsen"], ensure_ascii=False),
                    json.dumps(record["centroid_lonlat"], ensure_ascii=False),
                    f"{record['mean_velocity_mm_yr']:.6f}",
                    f"{record['median_velocity_mm_yr']:.6f}",
                    f"{record['p10_velocity_mm_yr']:.6f}",
                    f"{record['p90_velocity_mm_yr']:.6f}",
                    f"{record['max_abs_velocity_mm_yr']:.6f}",
                    f"{record['probability_mean']:.6f}",
                    int(record["strict_point_count"]),
                    int(record["relaxed_point_count"]),
                    int(record["forecast_point_count"]),
                    f"{record['aoi_area_fraction']:.8f}",
                    f"{record['internal_trace_corr']:.6f}",
                    f"{record['temporal_coherence']:.6f}",
                    f"{record['boundary_contrast']:.6f}",
                    f"{record['compactness']:.6f}",
                    f"{record['activity_level']:.6f}",
                    f"{record['region_score']:.6f}",
                ]
            )

    timeseries_csv_path = export_dir / "deformation_zone_timeseries.csv"
    with timeseries_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "zone_id",
                "date",
                "p25_mm",
                "p50_mm",
                "center_trimmed_mean_mm",
                "center_trimmed_mean_smooth_mm",
                "p75_mm",
                "n_points",
                "temporal_net_disp_mm",
                "temporal_continuity_score",
            ]
        )
        for record in zone_records_map.values():
            p25 = np.asarray(record["timeseries_p25_mm"], dtype=np.float32)
            p50 = np.asarray(record["timeseries_p50_mm"], dtype=np.float32)
            center = np.asarray(record["timeseries_center_mm"], dtype=np.float32)
            center_smooth = np.asarray(record["timeseries_center_smooth_mm"], dtype=np.float32)
            p75 = np.asarray(record["timeseries_p75_mm"], dtype=np.float32)
            counts = np.asarray(record["timeseries_point_count_by_date"], dtype=np.int32)
            for idx, date in enumerate(dates):
                writer.writerow(
                    [
                        record["zone_id"],
                        date,
                        float(p25[idx]),
                        float(p50[idx]),
                        float(center[idx]),
                        float(center_smooth[idx]),
                        float(p75[idx]),
                        int(counts[idx]),
                        float(record["temporal_net_disp_mm"]),
                        float(record["temporal_continuity_score"]),
                    ]
                )

    fig_png, fig_pdf = _render_zone_figure(
        mintpy_dir=mintpy_dir,
        export_dir=export_dir,
        velocity=velocity.astype(np.float32),
        latitude=latitude,
        longitude=longitude,
        dates=dates,
        zone_records=[zone_records_map[int(str(feature["properties"]["zone_id"]).lstrip("Z"))] for feature in zone_records],
        gdf=gdf,
    )

    status = "ok" if zone_records_map else "no_zone_detected"
    summary = {
        "detector_mode": detector_mode,
        "detector_backbone": "support_graph_v1",
        "detection_domain": detection_domain,
        "zone_semantics": zone_semantics,
        "output_geometry": output_geometry,
        "forecast_point_scope": forecast_point_scope,
        "status": status,
        "support_source": "strict_relaxed_union",
        "n_support_points": int(ctx.get("n_support_points", 0)),
        "graph_k": int(graph_summary.get("graph_k", 0)),
        "graph_median_edge_km": float(graph_summary.get("graph_median_edge_km", 0.0)),
        "candidate_region_count": int(len(candidate_rows)),
        "retained_region_count": int(len(zone_records_map)),
        "rejected_candidates": [
            {
                "candidate_id": str(item["candidate_id"]),
                "reasons": list(item.get("reasons", [])),
                "area_fraction": float(item["aoi_area_fraction"]),
                "support_point_count": int(item["support_point_count"]),
                "region_score": float(item["region_score"]),
                "zone_type": str(item.get("zone_type", "")),
            }
            for item in rejected_candidates
        ],
        "region_score_terms": {
            "formula": "sqrt(area_fraction) * internal_trace_corr * temporal_coherence * boundary_contrast * log1p(activity_level)",
            "min_area_fraction": float(retention_summary.get("min_area_fraction", SUPPORT_GRAPH_AREA_FRACTION_MIN)),
            "min_support_points": int(retention_summary.get("min_support_points", 0)),
            "salient_min_points": int(retention_summary.get("salient_min_points", 0)),
            "min_internal_trace_corr": float(retention_summary.get("min_internal_trace_corr", SUPPORT_GRAPH_INTERNAL_TRACE_CORR_MIN)),
            "min_temporal_coherence": float(retention_summary.get("min_temporal_coherence", SUPPORT_GRAPH_TEMPORAL_COHERENCE_MIN)),
            "min_compactness": float(retention_summary.get("min_compactness", SUPPORT_GRAPH_COMPACTNESS_MIN)),
            "min_region_score": float(retention_summary.get("min_region_score", 0.0)),
            "salient_region_score": float(retention_summary.get("salient_region_score", 0.0)),
            "salient_temporal_coherence": float(retention_summary.get("salient_temporal_coherence", SUPPORT_GRAPH_SALIENT_TEMPORAL_COHERENCE_MIN)),
            "salient_boundary_contrast": float(retention_summary.get("salient_boundary_contrast", SUPPORT_GRAPH_SALIENT_BOUNDARY_CONTRAST_MIN)),
            "salient_activity_threshold": float(retention_summary.get("salient_activity_threshold", 0.0)),
        },
        "support_graph_summary": graph_summary,
        "retention_summary": retention_summary,
        "merge_summary": merge_summary,
        "n_detected_zones": int(len(zone_records_map)),
        "aoi_area_km2": float(ctx.get("aoi_area_km2", 0.0)),
        "total_zone_area_km2": float(sum(float(item["area_km2"]) for item in zone_records_map.values())),
        "zone_ids": [item["zone_id"] for item in zone_records_map.values()],
        "paths": {
            "probability_tif": str(probability_path),
            "zone_mask_tif": str(zone_mask_path),
            "zone_id_tif": str(zone_id_path),
            "zone_csv": str(csv_path),
            "zone_timeseries_csv": str(timeseries_csv_path),
            "zone_geojson": vector_paths["geojson"],
            "zone_shp": vector_paths["shp"],
            "zone_kmz": vector_paths["kmz"],
            "velocity_map_zones_png": fig_png,
            "velocity_map_zones_pdf": fig_pdf,
        },
        "zones": [_record_public_view(item) for item in zone_records_map.values()],
    }
    summary_path = export_dir / "deformation_zone_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"deformation zones detected: {summary['n_detected_zones']} -> {summary_path}")
    return {
        "deformation_zone_probability_tif": str(probability_path),
        "deformation_zone_mask_tif": str(zone_mask_path),
        "deformation_zone_id_tif": str(zone_id_path),
        "deformation_zones_geojson": vector_paths["geojson"],
        "deformation_zones_shp": vector_paths["shp"],
        "deformation_zones_kmz": vector_paths["kmz"],
        "deformation_zone_summary_json": str(summary_path),
        "deformation_zones_csv": str(csv_path),
        "deformation_zone_timeseries_csv": str(timeseries_csv_path),
        "velocity_map_zones_png": fig_png,
        "velocity_map_zones_pdf": fig_pdf,
        "status": status,
        "n_detected_zones": int(summary["n_detected_zones"]),
    }


def detect_deformation_zones(
    mintpy_dir: str | Path,
    *,
    qc_report_dir: str | Path,
    export_dir: str | Path,
    detector_mode: str = "support_graph_v1",
    detection_domain: str = "hybrid",
    zone_semantics: str = "anomalous_deformation",
    output_geometry: str = "polygon",
    forecast_point_scope: str = "all_high_confidence",
) -> dict[str, Any]:
    if detector_mode == "support_graph_v1":
        if detection_domain != "hybrid":
            raise ValueError(f"Unsupported detection_domain: {detection_domain}")
        return _detect_deformation_zones_support_graph(
            mintpy_dir,
            qc_report_dir=qc_report_dir,
            export_dir=export_dir,
            detector_mode=detector_mode,
            detection_domain=detection_domain,
            zone_semantics=zone_semantics,
            output_geometry=output_geometry,
            forecast_point_scope=forecast_point_scope,
        )

    mintpy_dir = Path(mintpy_dir).resolve()
    qc_report_dir = Path(qc_report_dir).resolve()
    export_dir = Path(export_dir).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = export_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if detector_mode != "weak_ml_v1":
        raise ValueError(f"Unsupported detector_mode: {detector_mode}")
    if detection_domain != "hybrid":
        raise ValueError(f"Unsupported detection_domain: {detection_domain}")

    velocity_path = _resolve_velocity_raster(export_dir)
    if velocity_path.name != "velocity.tif" or velocity_path.parent != export_dir:
        logger.info(f"[zone] velocity raster resolved to {velocity_path}")
    velocity, ref_profile = _read_tif(velocity_path)
    ps_score, _ = _read_tif(qc_report_dir / "ps_score.tif")
    strict_mask, _ = _read_tif(qc_report_dir / "mask_ps_strict.tif")
    relaxed_mask, _ = _read_tif(qc_report_dir / "mask_ps_relaxed.tif")
    tcoh, _ = _read_tif(qc_report_dir / "tcoh_component.tif")
    metrics = _load_metrics(qc_report_dir)

    data = _load_mintpy_data(mintpy_dir)
    latitude = np.asarray(data.get("lat"), dtype=np.float32)
    longitude = np.asarray(data.get("lon"), dtype=np.float32)
    if latitude.shape != velocity.shape or longitude.shape != velocity.shape:
        raise RuntimeError(f"geometryRadar 与 {velocity_path.name} 尺寸不一致，无法检测形变区。")
    dates, rel0_cube = _load_rel0_timeseries(mintpy_dir)
    anomaly_exposure = _load_anomaly_exposure(mintpy_dir, qc_report_dir, velocity.shape)
    strict_flag = strict_mask.astype(bool)
    relaxed_flag = relaxed_mask.astype(bool)

    features, valid_mask = _build_feature_stack(
        velocity=np.asarray(velocity, dtype=np.float32),
        tcoh=np.asarray(tcoh, dtype=np.float32),
        ps_score=np.asarray(ps_score, dtype=np.float32),
        valid_pair_ratio=np.asarray(metrics["valid_pair_ratio"], dtype=np.float32),
        maincc_ratio=np.asarray(metrics["mainCC_ratio"], dtype=np.float32),
        jump_risk=np.asarray(metrics["jump_risk"], dtype=np.float32),
        anomaly_exposure=np.asarray(anomaly_exposure, dtype=np.float32),
        strict_flag=strict_flag.astype(np.float32),
        relaxed_flag=relaxed_flag.astype(np.float32),
    )
    valid_mask &= np.isfinite(latitude) & np.isfinite(longitude) & (latitude > 0.1)

    positive_seed, negative_seed, thresholds = _build_weak_supervision_seeds(features, valid_mask)
    probability, training_summary = _train_probability_model(features, valid_mask, positive_seed, negative_seed)
    probability_smooth = _local_mean_window(np.nan_to_num(probability, nan=0.0).astype(np.float32), valid_mask, size=7)
    probability_fused = (0.68 * np.nan_to_num(probability, nan=0.0) + 0.32 * np.nan_to_num(probability_smooth, nan=0.0)).astype(np.float32)
    probability_fused[~valid_mask] = np.nan
    pixel_area_km2 = _pixel_area_km2(ref_profile, latitude)
    aoi_base_mask = np.isfinite(latitude) & np.isfinite(longitude) & (latitude > 0.1) & np.isfinite(velocity)
    aoi_area_km2 = float(np.nansum(pixel_area_km2[aoi_base_mask]))
    min_zone_area_dynamic = float(max(MIN_ZONE_AREA_KM2, aoi_area_km2 * MIN_ZONE_AREA_FRACTION_AOI))
    zone_support_mask = (strict_flag | relaxed_flag) & valid_mask

    zone_mask, segmentation_summary = _build_principal_component_mask(
        features,
        valid_mask,
        probability_fused,
        rel0_cube=rel0_cube,
        support_flag=zone_support_mask,
        pixel_area_km2=pixel_area_km2,
        min_zone_area_dynamic=min_zone_area_dynamic,
    )
    if not np.any(zone_mask):
        region_grow_mask, region_grow_summary = _build_region_grow_mask(features, valid_mask, probability_fused)
        if np.any(region_grow_mask):
            zone_mask = region_grow_mask
            segmentation_summary = region_grow_summary
            segmentation_summary["fallback_triggered"] = True
            segmentation_summary["fallback_reason"] = "principal_component_empty_use_region_grow"
        else:
            segmentation_summary["fallback_triggered"] = True
            segmentation_summary["fallback_reason"] = "principal_component_and_region_grow_empty"
            zone_mask = np.isfinite(probability_fused) & (probability_fused >= PROBABILITY_THRESHOLD) & valid_mask

    structure = np.ones((3, 3), dtype=bool)
    zone_mask = binary_closing(zone_mask, structure=structure, iterations=2)
    zone_mask = binary_opening(zone_mask, structure=structure, iterations=1)
    zone_mask = binary_fill_holes(zone_mask)
    zone_mask &= valid_mask

    zone_id_raster = np.zeros(zone_mask.shape, dtype=np.int32)
    median_abs_velocity_threshold = float(max(MIN_ZONE_MEDIAN_ABS_VELOCITY_MM_YR, 0.30 * thresholds["positive_abs_velocity_threshold_mm_yr"]))
    median_abs_anomaly_threshold = float(max(3.5, 0.85 * thresholds.get("positive_local_anomaly_threshold_mm_yr", 3.5)))
    labeled, n_comp = _connected_components(zone_mask)
    zone_records_map: dict[int, dict[str, Any]] = {}
    next_id = 1
    for comp_id in range(1, int(n_comp) + 1):
        comp_mask = labeled == comp_id
        pixel_count = int(np.sum(comp_mask))
        if pixel_count < MIN_ZONE_PIXELS:
            continue
        area_km2 = float(np.nansum(pixel_area_km2[comp_mask]))
        if area_km2 < min_zone_area_dynamic:
            continue
        strict_count = int(np.sum(strict_flag[comp_mask]))
        relaxed_count = int(np.sum(relaxed_flag[comp_mask]))
        forecast_point_count = int(np.sum((strict_flag | relaxed_flag)[comp_mask] & valid_mask[comp_mask]))
        if forecast_point_count < MIN_ZONE_SUPPORT_PIXELS:
            continue
        rr, cc = np.where(comp_mask)
        bbox_pixel_count = int((rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1))
        bbox_fill_ratio = float(pixel_count / max(bbox_pixel_count, 1))
        if bbox_fill_ratio < MIN_ZONE_BBOX_FILL_RATIO:
            continue
        support_ratio = float(forecast_point_count / max(pixel_count, 1))
        if support_ratio < MIN_ZONE_SUPPORT_RATIO:
            continue
        zone_id_raster[comp_mask] = next_id
        zone_velocity = velocity[comp_mask]
        dominant_sign_ratio = float(max(np.mean(zone_velocity > 0), np.mean(zone_velocity < 0)))
        if dominant_sign_ratio < MIN_ZONE_SIGN_COHERENCE:
            zone_id_raster[comp_mask] = 0
            continue
        zone_median_abs_velocity = float(np.nanmedian(np.abs(zone_velocity)))
        if zone_median_abs_velocity < median_abs_velocity_threshold:
            zone_id_raster[comp_mask] = 0
            continue
        zone_median_abs_anomaly = float(np.nanmedian(features["abs_local_velocity_anomaly"][comp_mask]))
        required_zone_anomaly = float(
            max(
                1.25,
                median_abs_anomaly_threshold * min(1.0, min_zone_area_dynamic / max(area_km2, 1.0e-6)),
            )
        )
        if str(segmentation_summary.get("segmentation_mode", "")) == "principal_component_v3":
            required_zone_anomaly = float(min(required_zone_anomaly, 1.0))
        if zone_median_abs_anomaly < required_zone_anomaly:
            zone_id_raster[comp_mask] = 0
            continue
        zone_probability = probability_fused[comp_mask]
        zone_lon = longitude[comp_mask]
        zone_lat = latitude[comp_mask]
        temporal_profile = _zone_timeseries_profile(rel0_cube, comp_mask, zone_support_mask)
        if temporal_profile is None:
            zone_id_raster[comp_mask] = 0
            continue
        if abs(float(temporal_profile["temporal_net_disp_mm"])) < MIN_ZONE_TEMPORAL_NET_DISP_MM:
            zone_id_raster[comp_mask] = 0
            continue
        if float(temporal_profile["temporal_continuity_score"]) < MIN_ZONE_TEMPORAL_CONTINUITY:
            zone_id_raster[comp_mask] = 0
            continue
        bbox_wsen = [
            float(np.nanmin(zone_lon)),
            float(np.nanmin(zone_lat)),
            float(np.nanmax(zone_lon)),
            float(np.nanmax(zone_lat)),
        ]
        centroid_lonlat = [float(np.nanmean(zone_lon)), float(np.nanmean(zone_lat))]
        zone_records_map[next_id] = {
            "zone_id": f"Z{next_id}",
            "zone_type": _classify_zone_type(zone_velocity),
            "area_km2": area_km2,
            "pixel_count": pixel_count,
            "bbox_wsen": bbox_wsen,
            "centroid_lonlat": centroid_lonlat,
            "mean_velocity_mm_yr": float(np.nanmean(zone_velocity)),
            "median_velocity_mm_yr": float(np.nanmedian(zone_velocity)),
            "p10_velocity_mm_yr": float(np.nanpercentile(zone_velocity, 10)),
            "p90_velocity_mm_yr": float(np.nanpercentile(zone_velocity, 90)),
            "max_abs_velocity_mm_yr": float(np.nanmax(np.abs(zone_velocity))),
            "probability_mean": float(np.nanmean(zone_probability)),
            "strict_point_count": strict_count,
            "relaxed_point_count": relaxed_count,
            "forecast_point_count": forecast_point_count,
            "aoi_area_fraction": float(area_km2 / max(aoi_area_km2, 1e-6)),
            "bbox_fill_ratio": bbox_fill_ratio,
            "support_ratio": support_ratio,
            "dominant_sign_ratio": dominant_sign_ratio,
            "median_abs_velocity_mm_yr": zone_median_abs_velocity,
            "median_abs_local_anomaly_mm_yr": zone_median_abs_anomaly,
            "required_local_anomaly_mm_yr": required_zone_anomaly,
            **temporal_profile,
        }
        next_id += 1

    zone_records, gdf = _shape_records(
        zone_id_raster,
        ref_profile,
        zone_records_map,
        longitude=longitude,
        latitude=latitude,
    )
    vector_paths = _write_zone_vectors(features_out=zone_records, gdf=gdf, export_dir=export_dir)

    probability_path = _write_tif(
        export_dir / "deformation_zone_probability.tif",
        np.where(valid_mask, probability_fused, np.nan).astype(np.float32),
        ref_profile,
        dtype="float32",
        nodata=np.nan,
    )
    zone_mask_path = _write_tif(
        export_dir / "deformation_zone_mask.tif",
        zone_id_raster > 0,
        ref_profile,
        dtype="uint8",
        nodata=0,
    )
    zone_id_path = _write_tif(
        export_dir / "deformation_zone_id.tif",
        zone_id_raster,
        ref_profile,
        dtype="int32",
        nodata=0,
    )

    csv_path = export_dir / "deformation_zones.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "zone_id",
                "zone_type",
                "area_km2",
                "pixel_count",
                "bbox_wsen",
                "centroid_lonlat",
                "mean_velocity_mm_yr",
                "median_velocity_mm_yr",
                "p10_velocity_mm_yr",
                "p90_velocity_mm_yr",
                "max_abs_velocity_mm_yr",
                "probability_mean",
                "strict_point_count",
                "relaxed_point_count",
                "forecast_point_count",
                "aoi_area_fraction",
                "bbox_fill_ratio",
                "support_ratio",
                "dominant_sign_ratio",
                "median_abs_velocity_mm_yr",
                "temporal_net_disp_mm",
                "temporal_peak_to_peak_mm",
                "temporal_continuity_score",
            ]
        )
        for record in zone_records_map.values():
            writer.writerow(
                [
                    record["zone_id"],
                    record["zone_type"],
                    f"{record['area_km2']:.6f}",
                    int(record["pixel_count"]),
                    json.dumps(record["bbox_wsen"], ensure_ascii=False),
                    json.dumps(record["centroid_lonlat"], ensure_ascii=False),
                    f"{record['mean_velocity_mm_yr']:.6f}",
                    f"{record['median_velocity_mm_yr']:.6f}",
                    f"{record['p10_velocity_mm_yr']:.6f}",
                    f"{record['p90_velocity_mm_yr']:.6f}",
                    f"{record['max_abs_velocity_mm_yr']:.6f}",
                    f"{record['probability_mean']:.6f}",
                    int(record["strict_point_count"]),
                    int(record["relaxed_point_count"]),
                    int(record["forecast_point_count"]),
                    f"{record['aoi_area_fraction']:.8f}",
                    f"{record['bbox_fill_ratio']:.6f}",
                    f"{record['support_ratio']:.6f}",
                    f"{record['dominant_sign_ratio']:.6f}",
                    f"{record['median_abs_velocity_mm_yr']:.6f}",
                    f"{record['temporal_net_disp_mm']:.6f}",
                    f"{record['temporal_peak_to_peak_mm']:.6f}",
                    f"{record['temporal_continuity_score']:.6f}",
                ]
            )

    timeseries_csv_path = export_dir / "deformation_zone_timeseries.csv"
    with timeseries_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "zone_id",
                "date",
                "p25_mm",
                "p50_mm",
                "p50_smooth_mm",
                "p75_mm",
                "n_points",
                "temporal_net_disp_mm",
                "temporal_continuity_score",
            ]
        )
        for record in zone_records_map.values():
            p25 = np.asarray(record["timeseries_p25_mm"], dtype=np.float32)
            p50 = np.asarray(record["timeseries_p50_mm"], dtype=np.float32)
            p50s = np.asarray(record["timeseries_p50_smooth_mm"], dtype=np.float32)
            p75 = np.asarray(record["timeseries_p75_mm"], dtype=np.float32)
            counts = np.asarray(record["timeseries_point_count_by_date"], dtype=np.int32)
            for idx, date in enumerate(dates):
                writer.writerow(
                    [
                        record["zone_id"],
                        date,
                        float(p25[idx]),
                        float(p50[idx]),
                        float(p50s[idx]),
                        float(p75[idx]),
                        int(counts[idx]),
                        float(record["temporal_net_disp_mm"]),
                        float(record["temporal_continuity_score"]),
                    ]
                )

    fig_png, fig_pdf = _render_zone_figure(
        mintpy_dir=mintpy_dir,
        export_dir=export_dir,
        velocity=velocity.astype(np.float32),
        latitude=latitude,
        longitude=longitude,
        dates=dates,
        zone_records=[zone_records_map[int(str(feature["properties"]["zone_id"]).lstrip("Z"))] for feature in zone_records],
        gdf=gdf,
    )

    status = "ok" if zone_records_map else "no_zone_detected"
    summary = {
        "detector_mode": detector_mode,
        "detection_domain": detection_domain,
        "zone_semantics": zone_semantics,
        "output_geometry": output_geometry,
        "forecast_point_scope": forecast_point_scope,
        "status": status,
        "probability_threshold": float(PROBABILITY_THRESHOLD),
        "min_zone_pixels": int(MIN_ZONE_PIXELS),
        "min_zone_area_km2": float(min_zone_area_dynamic),
        "min_zone_area_fraction_aoi": float(MIN_ZONE_AREA_FRACTION_AOI),
        "min_zone_support_pixels": int(MIN_ZONE_SUPPORT_PIXELS),
        "min_zone_bbox_fill_ratio": float(MIN_ZONE_BBOX_FILL_RATIO),
        "min_zone_support_ratio": float(MIN_ZONE_SUPPORT_RATIO),
        "min_zone_sign_coherence": float(MIN_ZONE_SIGN_COHERENCE),
        "min_zone_median_abs_velocity_mm_yr": float(median_abs_velocity_threshold),
        "min_zone_temporal_net_disp_mm": float(MIN_ZONE_TEMPORAL_NET_DISP_MM),
        "min_zone_temporal_continuity": float(MIN_ZONE_TEMPORAL_CONTINUITY),
        "n_detected_zones": int(len(zone_records_map)),
        "aoi_area_km2": aoi_area_km2,
        "total_zone_area_km2": float(sum(float(item["area_km2"]) for item in zone_records_map.values())),
        "zone_ids": [item["zone_id"] for item in zone_records_map.values()],
        "seed_thresholds": thresholds,
        "seed_summary": training_summary,
        "segmentation_summary": segmentation_summary,
        "positive_seed_count": int(np.sum(positive_seed)),
        "negative_seed_count": int(np.sum(negative_seed)),
        "valid_pixel_count": int(np.sum(valid_mask)),
        "paths": {
            "probability_tif": str(probability_path),
            "zone_mask_tif": str(zone_mask_path),
            "zone_id_tif": str(zone_id_path),
            "zone_csv": str(csv_path),
            "zone_timeseries_csv": str(timeseries_csv_path),
            "zone_geojson": vector_paths["geojson"],
            "zone_shp": vector_paths["shp"],
            "zone_kmz": vector_paths["kmz"],
            "velocity_map_zones_png": fig_png,
            "velocity_map_zones_pdf": fig_pdf,
        },
        "zones": [_record_public_view(item) for item in zone_records_map.values()],
    }
    summary_path = export_dir / "deformation_zone_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"deformation zones detected: {summary['n_detected_zones']} -> {summary_path}")
    return {
        "deformation_zone_probability_tif": str(probability_path),
        "deformation_zone_mask_tif": str(zone_mask_path),
        "deformation_zone_id_tif": str(zone_id_path),
        "deformation_zones_geojson": vector_paths["geojson"],
        "deformation_zones_shp": vector_paths["shp"],
        "deformation_zones_kmz": vector_paths["kmz"],
        "deformation_zone_summary_json": str(summary_path),
        "deformation_zones_csv": str(csv_path),
        "deformation_zone_timeseries_csv": str(timeseries_csv_path),
        "velocity_map_zones_png": fig_png,
        "velocity_map_zones_pdf": fig_pdf,
        "status": status,
        "n_detected_zones": int(summary["n_detected_zones"]),
    }


__all__ = ["detect_deformation_zones"]
