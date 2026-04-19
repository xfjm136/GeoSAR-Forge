"""
DePSI-like QC utilities for the existing ISCE2 -> Dolphin -> MintPy chain.

This module does NOT depend on DePSI software. Instead, it borrows a few
high-confidence PSI ideas and applies them to the current chain:
1. PS-oriented confidence scoring from multiple indicators.
2. Simple time-model consistency checks on high-confidence points.
3. Reference-point candidate ranking with neighborhood and edge constraints.
4. Pair/date feedback derived from strict-PS residual behavior.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from scipy.ndimage import convolve

from . import config as cfg
from .config import logger
from .mintpy_runner import DOLPHIN_STRIDE_X, DOLPHIN_STRIDE_Y


DEPSI_LIKE_DEFAULTS = {
    "strict_ps_threshold": 0.75,
    "relaxed_ps_threshold": 0.60,
    "jump_threshold_base_mm": 28.0,
    "jump_threshold_quantile": 0.95,
    "jump_threshold_scale": 1.0,
    "abnormal_date_abs_frac_threshold": 0.20,
    "abnormal_date_abs_residual_mm": 20.0,
    "abnormal_date_quantile": 0.90,
    "ref_candidate_top_n": 20,
    "ref_candidate_patch_radius": 2,
    "ref_network_min_distance_m": 400.0,
    "ref_min_tcoh": 0.70,
    "ref_min_valid_pair_ratio": 0.80,
    "ref_min_maincc_ratio": 0.80,
    "ref_min_dist_to_edge_px": 100.0,
    "ref_min_strict_neighbor_count": 3.0,
    "ref_max_local_velocity_gradient_quantile": 0.90,
    "ref_velocity_neutrality_base_mm_yr": 5.0,
    "ref_velocity_neutrality_abs_quantile": 0.25,
    "ref_velocity_neutrality_iqr_quantile": 0.70,
    "reference_bias_top_k": 20,
    "strict_sparse_ratio_threshold": 0.005,
    "strict_sparse_candidate_threshold": 2000,
    "strict_too_loose_ratio_threshold": 0.20,
    "strict_relaxed_too_loose_ratio_threshold": 0.40,
    "max_reference_fallback_rank": 3,
}

PAIR_Q_THRESHOLDS = {
    "keep": 0.75,
    "downweight": 0.50,
}

ANOMALY_SCOPE_THRESHOLDS = {
    "global_max": 0.20,
    "mixed_max": 0.40,
}

FEEDBACK_PENALTIES = {
    "global": 0.10,
    "mixed": 0.05,
    "local": 0.025,
    "low_consistency_extra": 0.10,
    "max_penalty": 0.20,
}

MODEL_NAME_TO_CODE = {
    "linear": 0,
    "linear_annual": 1,
    "piecewise_linear": 2,
}
MODEL_CODE_TO_NAME = {v: k for k, v in MODEL_NAME_TO_CODE.items()}


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize(values: np.ndarray, *, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    vals = arr[finite]
    lo = float(np.nanpercentile(vals, 5)) if lower is None else float(lower)
    hi = float(np.nanpercentile(vals, 95)) if upper is None else float(upper)
    if math.isclose(lo, hi):
        out[finite] = 0.0
    else:
        out[finite] = (vals - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _ensure_dir(path: Path | None) -> Path:
    path = Path(path or (cfg.WORK_DIR / "mainchain_qc"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_tif(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
    return arr, profile


def _read_subset_from_vrt(vrt_path: Path, row_off: int, col_off: int, rows: int, cols: int) -> np.ndarray:
    with rasterio.open(vrt_path) as src:
        row_off = max(0, int(row_off))
        col_off = max(0, int(col_off))
        rows = min(int(rows), src.height - row_off)
        cols = min(int(cols), src.width - col_off)
        window = ((row_off, row_off + rows), (col_off, col_off + cols))
        return src.read(1, window=window)


def _write_like_tif(path: Path, array: np.ndarray, ref_path: Path, *, dtype: str = "float32", nodata: float | None = np.nan) -> Path:
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
    profile.update(count=1, dtype=dtype, compress="deflate", tiled=True)
    if nodata is not None:
        profile.update(nodata=nodata)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
    return path


def _date_pairs_from_dolphin(dolphin_dir: Path) -> list[tuple[str, str]]:
    pairs = []
    for path in sorted((dolphin_dir / "unwrapped").glob("*.unw.tif")):
        stem = path.stem.replace(".unw", "")
        if "_" not in stem:
            continue
        d1, d2 = stem.split("_", 1)
        pairs.append((d1, d2))
    return pairs


def _scene_dates_from_pairs(pairs: list[tuple[str, str]]) -> list[str]:
    return sorted({d for pair in pairs for d in pair})


def _find_water_mask(mintpy_dir: Path) -> np.ndarray | None:
    candidates = [
        mintpy_dir / "waterMask.h5",
        mintpy_dir / "inputs" / "waterMask.h5",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            key = "waterMask" if "waterMask" in f else next(iter(f.keys()))
            arr = np.asarray(f[key][:]).astype(bool)
        return arr
    return None


def _load_geometry(
    mintpy_dir: Path,
    geom_source_dir: Path | None = None,
    output_shape: tuple[int, int] | None = None,
    crop_offset: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    mintpy_dir = Path(mintpy_dir)
    geom_path = mintpy_dir / "inputs" / "geometryRadar.h5"
    if geom_path.exists():
        out: dict[str, np.ndarray] = {}
        with h5py.File(geom_path, "r") as f:
            for key in ["latitude", "longitude", "height"]:
                if key in f:
                    out[key] = np.asarray(f[key][:], dtype=np.float32)
        if "latitude" in out and "longitude" in out:
            return out

    geom_source_dir = Path(geom_source_dir or (cfg.ISCE_WORK_DIR / "merged" / "geom_reference"))
    lat_path = geom_source_dir / "lat.rdr"
    lon_path = geom_source_dir / "lon.rdr"
    hgt_path = geom_source_dir / "hgt.rdr"
    if not lat_path.exists() or not lon_path.exists():
        raise FileNotFoundError(
            "几何文件不存在: 既未找到 MintPy inputs/geometryRadar.h5，"
            f"也未找到 ISCE 几何源 {lat_path} / {lon_path}"
        )

    out = {}
    crop_offset = crop_offset or getattr(cfg, "_AOI_CROP_OFFSET", None)
    if output_shape is not None and crop_offset is not None:
        row_off, col_off = [int(v) for v in crop_offset]
        height, width = [int(v) for v in output_shape]
        full_rows = height * DOLPHIN_STRIDE_Y
        full_cols = width * DOLPHIN_STRIDE_X
        lat_full = _read_subset_from_vrt(
            geom_source_dir / "lat.rdr.full.vrt",
            row_off=row_off,
            col_off=col_off,
            rows=full_rows,
            cols=full_cols,
        )
        lon_full = _read_subset_from_vrt(
            geom_source_dir / "lon.rdr.full.vrt",
            row_off=row_off,
            col_off=col_off,
            rows=full_rows,
            cols=full_cols,
        )
        out["latitude"] = lat_full[::DOLPHIN_STRIDE_Y, ::DOLPHIN_STRIDE_X][:height, :width].astype(np.float32)
        out["longitude"] = lon_full[::DOLPHIN_STRIDE_Y, ::DOLPHIN_STRIDE_X][:height, :width].astype(np.float32)
        if hgt_path.exists():
            hgt_full = _read_subset_from_vrt(
                geom_source_dir / "hgt.rdr.full.vrt",
                row_off=row_off,
                col_off=col_off,
                rows=full_rows,
                cols=full_cols,
            )
            out["height"] = hgt_full[::DOLPHIN_STRIDE_Y, ::DOLPHIN_STRIDE_X][:height, :width].astype(np.float32)
        return out

    with rasterio.open(lat_path) as ds:
        out["latitude"] = ds.read(1).astype(np.float32)
    with rasterio.open(lon_path) as ds:
        out["longitude"] = ds.read(1).astype(np.float32)
    if hgt_path.exists():
        with rasterio.open(hgt_path) as ds:
            out["height"] = ds.read(1).astype(np.float32)
    return out


def _load_reference_timeseries_files(dolphin_dir: Path) -> tuple[list[str], list[Path]]:
    ts_dir = dolphin_dir / "timeseries"
    files = sorted(ts_dir.glob("*.tif"))
    date_files: list[tuple[str, str, Path]] = []
    for path in files:
        name = path.stem
        if name.startswith("residuals_") or name in {"velocity", "conncomp_intersection", "unw_inversion_residuals"}:
            continue
        parts = name.split("_")
        if len(parts) != 2:
            continue
        d1, d2 = parts
        if len(d1) == 8 and len(d2) == 8 and d1.isdigit() and d2.isdigit():
            date_files.append((d1, d2, path))
    if not date_files:
        raise FileNotFoundError(f"未找到 Dolphin 累积时序: {ts_dir}")

    ref_date = min(d1 for d1, _, _ in date_files)
    filtered = [(d1, d2, p) for d1, d2, p in date_files if d1 == ref_date]
    filtered = sorted(filtered, key=lambda item: item[1])
    dates = [ref_date] + [d2 for _, d2, _ in filtered]
    paths = [p for _, _, p in filtered]
    return dates, paths


def _load_candidate_series(dolphin_dir: Path, rows: np.ndarray, cols: np.ndarray) -> tuple[list[str], np.ndarray, Path]:
    dates, paths = _load_reference_timeseries_files(dolphin_dir)
    stack = np.zeros((len(dates), len(rows)), dtype=np.float32)
    for i, path in enumerate(paths, start=1):
        with rasterio.open(path) as ds:
            arr = ds.read(1)
        stack[i, :] = arr[rows, cols]
    return dates, stack, paths[0]


def _phase_to_los_mm(phase_radian: np.ndarray, wavelength: float) -> np.ndarray:
    return phase_radian * (wavelength / (4.0 * np.pi)) * 1000.0


def _fit_design_matrix(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pinv = np.linalg.pinv(x)
    coef = pinv @ y
    pred = x @ coef
    resid = y - pred
    sse = np.sum(resid ** 2, axis=0)
    return coef, pred, sse


def _bic_from_sse(sse: np.ndarray, n_obs: int, k_param: int) -> np.ndarray:
    eps = 1e-6
    return n_obs * np.log(np.maximum(sse / max(n_obs, 1), eps)) + k_param * np.log(max(n_obs, 1))


def _fit_candidate_models(series_mm: np.ndarray, day_offsets: np.ndarray) -> dict[str, np.ndarray]:
    y = np.asarray(series_mm, dtype=np.float64)
    t = np.asarray(day_offsets, dtype=np.float64)
    n_obs, n_pts = y.shape
    ones = np.ones_like(t)

    x_linear = np.column_stack([ones, t])
    coef_lin, pred_lin, sse_lin = _fit_design_matrix(y, x_linear)
    bic_lin = _bic_from_sse(sse_lin, n_obs, x_linear.shape[1])

    omega = 2.0 * np.pi / 365.25
    x_ann = np.column_stack([ones, t, np.sin(omega * t), np.cos(omega * t)])
    coef_ann, pred_ann, sse_ann = _fit_design_matrix(y, x_ann)
    bic_ann = _bic_from_sse(sse_ann, n_obs, x_ann.shape[1])

    split_candidates = list(range(3, max(4, n_obs - 3)))
    bic_piece = np.full(n_pts, np.inf, dtype=np.float64)
    pred_piece = np.zeros_like(y, dtype=np.float64)
    split_idx_best = np.full(n_pts, -1, dtype=np.int16)
    for split_idx in split_candidates:
        x_piece = np.column_stack([ones, t, np.maximum(t - t[split_idx], 0.0)])
        _, pred_tmp, sse_tmp = _fit_design_matrix(y, x_piece)
        bic_tmp = _bic_from_sse(sse_tmp, n_obs, x_piece.shape[1])
        better = bic_tmp < bic_piece
        bic_piece[better] = bic_tmp[better]
        pred_piece[:, better] = pred_tmp[:, better]
        split_idx_best[better] = split_idx

    bic_stack = np.vstack([bic_lin, bic_ann, bic_piece])
    best_code = np.argmin(bic_stack, axis=0).astype(np.int16)
    bic_sorted = np.sort(bic_stack, axis=0)
    delta_bic = bic_sorted[1] - bic_sorted[0]

    pred_best = np.where(best_code[None, :] == 0, pred_lin, pred_ann)
    pred_best = np.where(best_code[None, :] == 2, pred_piece, pred_best)
    resid_best = y - pred_best
    model_rms = np.sqrt(np.mean(resid_best**2, axis=0)).astype(np.float32)

    slope_full = coef_lin[1] * 365.25
    sensitivity = np.zeros(n_pts, dtype=np.float64)
    for leave_idx in range(1, n_obs):
        keep = np.arange(n_obs) != leave_idx
        x_sub = np.column_stack([np.ones(np.sum(keep)), t[keep]])
        coef_sub, _, _ = _fit_design_matrix(y[keep, :], x_sub)
        slope_sub = coef_sub[1] * 365.25
        sensitivity = np.maximum(sensitivity, np.abs(slope_sub - slope_full))

    return {
        "best_model_code": best_code,
        "best_model_name": np.array([MODEL_CODE_TO_NAME[int(v)] for v in best_code], dtype=object),
        "model_rms": model_rms,
        "delta_bic": delta_bic.astype(np.float32),
        "leave_one_pair_out_sensitivity": sensitivity.astype(np.float32),
        "pred_best": pred_best.astype(np.float32),
        "split_idx_best": split_idx_best,
    }


def _compute_pair_actual_and_residuals(
    dolphin_dir: Path,
    rows: np.ndarray,
    cols: np.ndarray,
    date_to_idx: dict[str, int],
    pred_best: np.ndarray,
    wavelength: float,
) -> dict[str, Any]:
    pairs = _date_pairs_from_dolphin(dolphin_dir)
    n_pairs = len(pairs)
    n_pts = len(rows)
    abs_residual = np.full((n_pairs, n_pts), np.nan, dtype=np.float32)
    valid_mask = np.zeros((n_pairs, n_pts), dtype=bool)
    main_mask = np.zeros((n_pairs, n_pts), dtype=bool)

    for i, (d1, d2) in enumerate(pairs):
        pair_name = f"{d1}_{d2}"
        with rasterio.open(dolphin_dir / "unwrapped" / f"{pair_name}.unw.tif") as ds:
            unw = ds.read(1)
        with rasterio.open(dolphin_dir / "unwrapped" / f"{pair_name}.unw.conncomp.tif") as ds:
            cc = ds.read(1)

        cc_vals = cc[rows, cols]
        valid_cc = (cc_vals > 0) & (cc_vals < 65535)
        actual = _phase_to_los_mm(unw[rows, cols].astype(np.float32), wavelength)
        valid = valid_cc & np.isfinite(actual)
        valid_mask[i, :] = valid
        if np.any(valid_cc):
            labels, counts = np.unique(cc_vals[valid_cc], return_counts=True)
            main_label = labels[np.argmax(counts)]
            main_mask[i, :] = valid & (cc_vals == main_label)

        pred_pair = pred_best[date_to_idx[d2], :] - pred_best[date_to_idx[d1], :]
        resid = np.abs(actual - pred_pair)
        abs_residual[i, valid] = resid[valid].astype(np.float32)

    return {
        "pairs": pairs,
        "abs_residual": abs_residual,
        "valid_mask": valid_mask,
        "main_mask": main_mask,
    }


def _count_strict_neighbors(mask: np.ndarray, size: int = 5) -> np.ndarray:
    kernel = np.ones((size, size), dtype=np.int16)
    counts = convolve(mask.astype(np.int16), kernel, mode="constant", cval=0)
    return counts - mask.astype(np.int16)


def _distance_to_edge(shape: tuple[int, int], rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    h, w = shape
    return np.minimum.reduce([rows, cols, h - 1 - rows, w - 1 - cols]).astype(np.float32)


def _compute_velocity_gradient(velocity: np.ndarray) -> np.ndarray:
    vy, vx = np.gradient(np.nan_to_num(velocity.astype(np.float32), nan=0.0))
    return np.sqrt(vx**2 + vy**2).astype(np.float32)


def _gacos_safe_flags(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    atmo_method: str | None,
    current_bounds: dict[str, float] | None,
) -> np.ndarray:
    if atmo_method != "gacos" or not current_bounds:
        return np.ones_like(lat_vals, dtype=np.int16)
    safe = (
        (lat_vals >= float(current_bounds["S"]))
        & (lat_vals <= float(current_bounds["N"]))
        & (lon_vals >= float(current_bounds["W"]))
        & (lon_vals <= float(current_bounds["E"]))
    )
    return safe.astype(np.int16)


def _build_sparse_reference_network(
    rows: np.ndarray,
    cols: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    quality_vals: np.ndarray,
    min_distance_m: float,
) -> np.ndarray:
    if len(rows) == 0:
        return np.empty(0, dtype=np.int32)
    if min_distance_m <= 0:
        return np.arange(len(rows), dtype=np.int32)

    order = np.argsort(-np.asarray(quality_vals, dtype=np.float64))
    selected: list[int] = []
    sel_lat: list[float] = []
    sel_lon: list[float] = []
    min_sq = float(min_distance_m) ** 2

    for idx in order:
        lat0 = float(lat_vals[idx])
        lon0 = float(lon_vals[idx])
        if not np.isfinite(lat0) or not np.isfinite(lon0):
            continue
        if sel_lat:
            lat_arr = np.asarray(sel_lat, dtype=np.float64)
            lon_arr = np.asarray(sel_lon, dtype=np.float64)
            dx = (lon_arr - lon0) * (111320.0 * math.cos(math.radians(lat0)))
            dy = (lat_arr - lat0) * 110540.0
            if np.min(dx * dx + dy * dy) < min_sq:
                continue
        selected.append(int(idx))
        sel_lat.append(lat0)
        sel_lon.append(lon0)
    return np.asarray(selected, dtype=np.int32)


def _compute_network_neighbor_counts(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    neighbor_radius_m: float,
) -> np.ndarray:
    n = len(lat_vals)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    counts = np.zeros(n, dtype=np.int32)
    radius_sq = float(neighbor_radius_m) ** 2
    for i in range(n):
        lat0 = float(lat_vals[i])
        lon0 = float(lon_vals[i])
        dx = (np.asarray(lon_vals, dtype=np.float64) - lon0) * (111320.0 * math.cos(math.radians(lat0)))
        dy = (np.asarray(lat_vals, dtype=np.float64) - lat0) * 110540.0
        dist_sq = dx * dx + dy * dy
        counts[i] = int(np.sum((dist_sq <= radius_sq) & (dist_sq > 0.0)))
    return counts


def _compute_reference_patch_stats(
    row: int,
    col: int,
    radius: int,
    strict_mask: np.ndarray,
    ps_score: np.ndarray,
    tcoh: np.ndarray,
    maincc_ratio: np.ndarray,
    model_rms: np.ndarray,
    jump_risk: np.ndarray,
    valid_pair_ratio: np.ndarray,
    velocity_proxy: np.ndarray | None = None,
) -> dict[str, float]:
    r0 = max(0, int(row) - int(radius))
    r1 = min(strict_mask.shape[0], int(row) + int(radius) + 1)
    c0 = max(0, int(col) - int(radius))
    c1 = min(strict_mask.shape[1], int(col) + int(radius) + 1)
    patch_mask = strict_mask[r0:r1, c0:c1]
    if not np.any(patch_mask):
        return {
            "reference_patch_count": 0.0,
            "reference_patch_ps_score_median": float("nan"),
            "reference_patch_tcoh_median": float("nan"),
            "reference_patch_mainCC_ratio_median": float("nan"),
            "reference_patch_model_rms_median": float("nan"),
            "reference_patch_jump_risk_median": float("nan"),
            "reference_patch_valid_pair_ratio_median": float("nan"),
            "reference_patch_velocity_median": float("nan"),
            "reference_patch_velocity_abs_median": float("nan"),
            "reference_patch_velocity_iqr": float("nan"),
        }
    patch_index = patch_mask.astype(bool)
    if velocity_proxy is not None:
        patch_vel = velocity_proxy[r0:r1, c0:c1][patch_index]
        finite_vel = patch_vel[np.isfinite(patch_vel)]
        if finite_vel.size:
            patch_vel_median = float(np.nanmedian(finite_vel))
            patch_vel_abs_median = float(np.nanmedian(np.abs(finite_vel)))
            patch_vel_iqr = float(np.nanpercentile(finite_vel, 75) - np.nanpercentile(finite_vel, 25))
        else:
            patch_vel_median = float("nan")
            patch_vel_abs_median = float("nan")
            patch_vel_iqr = float("nan")
    else:
        patch_vel_median = float("nan")
        patch_vel_abs_median = float("nan")
        patch_vel_iqr = float("nan")
    return {
        "reference_patch_count": float(np.sum(patch_index)),
        "reference_patch_ps_score_median": float(np.nanmedian(ps_score[r0:r1, c0:c1][patch_index])),
        "reference_patch_tcoh_median": float(np.nanmedian(tcoh[r0:r1, c0:c1][patch_index])),
        "reference_patch_mainCC_ratio_median": float(np.nanmedian(maincc_ratio[r0:r1, c0:c1][patch_index])),
        "reference_patch_model_rms_median": float(np.nanmedian(model_rms[r0:r1, c0:c1][patch_index])),
        "reference_patch_jump_risk_median": float(np.nanmedian(jump_risk[r0:r1, c0:c1][patch_index])),
        "reference_patch_valid_pair_ratio_median": float(np.nanmedian(valid_pair_ratio[r0:r1, c0:c1][patch_index])),
        "reference_patch_velocity_median": patch_vel_median,
        "reference_patch_velocity_abs_median": patch_vel_abs_median,
        "reference_patch_velocity_iqr": patch_vel_iqr,
    }


def _select_reference_candidates(
    strict_mask: np.ndarray,
    ps_score: np.ndarray,
    tcoh: np.ndarray,
    maincc_ratio: np.ndarray,
    model_rms: np.ndarray,
    jump_risk: np.ndarray,
    valid_pair_ratio: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    velocity_gradient: np.ndarray,
    gacos_safe_flag: np.ndarray,
    velocity_proxy: np.ndarray,
    report_dir: Path,
    top_n: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg_like = dict(DEPSI_LIKE_DEFAULTS)
    cfg_like.update(config or {})
    rows, cols = np.where(strict_mask)
    if len(rows) == 0:
        out_path = report_dir / "ref_candidates.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "row", "col", "lon", "lat", "ref_score", "ps_score", "tcoh",
                "mainCC_ratio", "model_rms", "jump_risk", "dist_to_edge",
                "strict_neighbor_count", "local_velocity_gradient", "gacos_safe_flag",
                "network_rank", "reference_patch_radius", "reference_patch_count",
                "reference_patch_ps_score_median", "reference_patch_tcoh_median",
                "reference_patch_mainCC_ratio_median", "reference_patch_model_rms_median",
                "reference_patch_jump_risk_median", "reference_patch_valid_pair_ratio_median",
                "reference_patch_velocity_median", "reference_patch_velocity_abs_median",
                "reference_patch_velocity_iqr", "network_neighbor_count",
            ])
        network_csv = report_dir / "ref_primary_network.csv"
        with network_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "network_rank", "row", "col", "lon", "lat", "network_quality", "ref_score",
                "ps_score", "tcoh", "mainCC_ratio", "valid_pair_ratio", "jump_risk",
                "dist_to_edge", "strict_neighbor_count", "local_velocity_gradient",
                "reference_patch_velocity_median", "reference_patch_velocity_abs_median",
                "reference_patch_velocity_iqr", "network_neighbor_count",
            ])
        return {
            "ref_candidates_csv": str(out_path),
            "count": 0,
            "ref_primary_network_csv": str(network_csv),
            "ref_primary_network_tif": str(report_dir / "ref_primary_network.tif"),
            "primary_network_count": 0,
            "selection_fallback_reason": "empty_strict_mask",
        }

    strict_neighbors = _count_strict_neighbors(strict_mask, size=5)
    dist_edge = _distance_to_edge(strict_mask.shape, rows, cols)
    neighbor_count = strict_neighbors[rows, cols].astype(np.float32)
    neigh_norm = _normalize(neighbor_count)
    grad_vals = velocity_gradient[rows, cols].astype(np.float32)

    grad_threshold = (
        float(np.nanquantile(grad_vals, float(cfg_like["ref_max_local_velocity_gradient_quantile"])))
        if np.any(np.isfinite(grad_vals))
        else np.inf
    )
    hard_ok = (
        (dist_edge >= float(cfg_like["ref_min_dist_to_edge_px"]))
        & (neighbor_count >= float(cfg_like["ref_min_strict_neighbor_count"]))
        & (grad_vals <= grad_threshold)
        & (gacos_safe_flag[rows, cols] > 0)
        & (tcoh[rows, cols] >= float(cfg_like["ref_min_tcoh"]))
        & (valid_pair_ratio[rows, cols] >= float(cfg_like["ref_min_valid_pair_ratio"]))
        & (maincc_ratio[rows, cols] >= float(cfg_like["ref_min_maincc_ratio"]))
    )
    fallback_reason = ""
    if not np.any(hard_ok):
        hard_ok = (
            (dist_edge >= float(cfg_like["ref_min_dist_to_edge_px"]))
            & (neighbor_count >= float(cfg_like["ref_min_strict_neighbor_count"]))
            & (grad_vals <= grad_threshold)
            & (gacos_safe_flag[rows, cols] > 0)
        )
        fallback_reason = "relaxed_reference_hard_constraints"
    if not np.any(hard_ok):
        hard_ok = np.ones_like(dist_edge, dtype=bool)
        fallback_reason = "fallback_all_strict_candidates"

    ref_score = (
        0.35 * np.clip(tcoh[rows, cols], 0.0, 1.0)
        + 0.25 * np.clip(maincc_ratio[rows, cols], 0.0, 1.0)
        + 0.20 * (1.0 - _normalize(model_rms[rows, cols]))
        + 0.10 * np.clip(valid_pair_ratio[rows, cols], 0.0, 1.0)
        + 0.10 * neigh_norm
    )
    network_quality = (
        0.70 * ref_score
        + 0.20 * np.clip(ps_score[rows, cols], 0.0, 1.0)
        + 0.10 * (1.0 - np.clip(jump_risk[rows, cols], 0.0, 1.0))
    ).astype(np.float32)

    candidate_rows = rows[hard_ok]
    candidate_cols = cols[hard_ok]
    candidate_lat = lat[candidate_rows, candidate_cols].astype(np.float32)
    candidate_lon = lon[candidate_rows, candidate_cols].astype(np.float32)
    candidate_quality = network_quality[hard_ok]
    selected_network_idx = _build_sparse_reference_network(
        candidate_rows,
        candidate_cols,
        candidate_lat,
        candidate_lon,
        candidate_quality,
        float(cfg_like["ref_network_min_distance_m"]),
    )
    if selected_network_idx.size == 0:
        selected_network_idx = np.arange(len(candidate_rows), dtype=np.int32)
        if not fallback_reason:
            fallback_reason = "fallback_dense_reference_candidates"

    net_rows = candidate_rows[selected_network_idx]
    net_cols = candidate_cols[selected_network_idx]
    net_quality = candidate_quality[selected_network_idx]
    net_ref_score = ref_score[hard_ok][selected_network_idx]
    net_ps_score = ps_score[net_rows, net_cols]
    net_tcoh = tcoh[net_rows, net_cols]
    net_maincc = maincc_ratio[net_rows, net_cols]
    net_valid_pair = valid_pair_ratio[net_rows, net_cols]
    net_jump = jump_risk[net_rows, net_cols]
    net_dist_edge = dist_edge[hard_ok][selected_network_idx]
    net_neighbor = neighbor_count[hard_ok][selected_network_idx]
    net_grad = grad_vals[hard_ok][selected_network_idx]
    patch_radius = int(cfg_like["ref_candidate_patch_radius"])
    network_patch_stats = [
        _compute_reference_patch_stats(
            int(r),
            int(c),
            patch_radius,
            strict_mask,
            ps_score,
            tcoh,
            maincc_ratio,
            model_rms,
            jump_risk,
            valid_pair_ratio,
            velocity_proxy,
        )
        for r, c in zip(net_rows, net_cols)
    ]

    network_neighbor_count = _compute_network_neighbor_counts(
        candidate_lat[selected_network_idx],
        candidate_lon[selected_network_idx],
        max(float(cfg_like["ref_network_min_distance_m"]) * 2.0, 1.0),
    )

    network_csv = report_dir / "ref_primary_network.csv"
    with network_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "network_rank", "row", "col", "lon", "lat", "network_quality", "ref_score",
            "ps_score", "tcoh", "mainCC_ratio", "valid_pair_ratio", "jump_risk",
            "dist_to_edge", "strict_neighbor_count", "local_velocity_gradient",
            "network_neighbor_count",
        ])
        net_order = np.argsort(-net_quality)
        for network_rank, idx in enumerate(net_order, start=1):
            writer.writerow([
                network_rank,
                int(net_rows[idx]),
                int(net_cols[idx]),
                float(lon[int(net_rows[idx]), int(net_cols[idx])]),
                float(lat[int(net_rows[idx]), int(net_cols[idx])]),
                float(net_quality[idx]),
                float(net_ref_score[idx]),
                float(net_ps_score[idx]),
                float(net_tcoh[idx]),
                float(net_maincc[idx]),
                float(net_valid_pair[idx]),
                float(net_jump[idx]),
                float(net_dist_edge[idx]),
                float(net_neighbor[idx]),
                float(net_grad[idx]),
                float(network_patch_stats[idx]["reference_patch_velocity_median"]),
                float(network_patch_stats[idx]["reference_patch_velocity_abs_median"]),
                float(network_patch_stats[idx]["reference_patch_velocity_iqr"]),
                int(network_neighbor_count[idx]),
            ])

    network_mask = np.zeros(strict_mask.shape, dtype=np.uint8)
    network_mask[net_rows, net_cols] = 1
    _write_like_tif(report_dir / "ref_primary_network.tif", network_mask, report_dir / "mask_ps_strict.tif", dtype="uint8", nodata=0)

    patch_count = np.asarray([item["reference_patch_count"] for item in network_patch_stats], dtype=np.float32)
    patch_ps_median = np.asarray([item["reference_patch_ps_score_median"] for item in network_patch_stats], dtype=np.float32)
    patch_model_median = np.asarray([item["reference_patch_model_rms_median"] for item in network_patch_stats], dtype=np.float32)
    patch_jump_median = np.asarray([item["reference_patch_jump_risk_median"] for item in network_patch_stats], dtype=np.float32)
    patch_vel_abs_median = np.asarray([item["reference_patch_velocity_abs_median"] for item in network_patch_stats], dtype=np.float32)
    patch_vel_iqr = np.asarray([item["reference_patch_velocity_iqr"] for item in network_patch_stats], dtype=np.float32)
    candidate_keep = np.ones(len(net_rows), dtype=bool)
    if len(candidate_keep) > 3:
        candidate_keep &= (network_neighbor_count > 0)
    if not np.any(candidate_keep):
        candidate_keep = np.ones(len(net_rows), dtype=bool)

    finite_abs = patch_vel_abs_median[np.isfinite(patch_vel_abs_median)]
    if finite_abs.size:
        neutrality_threshold = max(
            float(cfg_like["ref_velocity_neutrality_base_mm_yr"]),
            float(np.nanquantile(finite_abs, float(cfg_like["ref_velocity_neutrality_abs_quantile"]))),
        )
    else:
        neutrality_threshold = float(cfg_like["ref_velocity_neutrality_base_mm_yr"])
    finite_iqr = patch_vel_iqr[np.isfinite(patch_vel_iqr)]
    if finite_iqr.size:
        velocity_iqr_threshold = float(
            np.nanquantile(finite_iqr, float(cfg_like["ref_velocity_neutrality_iqr_quantile"]))
        )
    else:
        velocity_iqr_threshold = float("inf")

    neutral_keep = (
        np.isfinite(patch_vel_abs_median)
        & np.isfinite(patch_vel_iqr)
        & (patch_vel_abs_median <= neutrality_threshold)
        & (patch_vel_iqr <= velocity_iqr_threshold)
    )
    combined_keep = candidate_keep & neutral_keep
    if np.any(combined_keep):
        candidate_keep = combined_keep
    elif not fallback_reason:
        fallback_reason = "no_velocity_neutral_candidate"

    patch_vel_abs_sort = np.where(np.isfinite(patch_vel_abs_median), patch_vel_abs_median, np.inf)
    patch_vel_iqr_sort = np.where(np.isfinite(patch_vel_iqr), patch_vel_iqr, np.inf)
    patch_model_sort = np.where(np.isfinite(patch_model_median), patch_model_median, np.inf)
    patch_jump_sort = np.where(np.isfinite(patch_jump_median), patch_jump_median, np.inf)

    candidate_order = np.lexsort((
        patch_jump_sort,
        patch_model_sort,
        net_grad,
        -net_neighbor,
        -network_neighbor_count,
        -net_dist_edge,
        -patch_ps_median,
        patch_vel_iqr_sort,
        patch_vel_abs_sort,
    ))
    candidate_order = candidate_order[np.isin(candidate_order, np.where(candidate_keep)[0])][:top_n]
    sel_rows = net_rows[candidate_order]
    sel_cols = net_cols[candidate_order]

    out_path = report_dir / "ref_candidates.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "row", "col", "lon", "lat", "ref_score", "ps_score", "tcoh",
            "mainCC_ratio", "model_rms", "jump_risk", "dist_to_edge",
            "strict_neighbor_count", "local_velocity_gradient", "gacos_safe_flag",
            "network_rank", "reference_patch_radius", "reference_patch_count",
            "reference_patch_ps_score_median", "reference_patch_tcoh_median",
            "reference_patch_mainCC_ratio_median", "reference_patch_model_rms_median",
            "reference_patch_jump_risk_median", "reference_patch_valid_pair_ratio_median",
            "reference_patch_velocity_median", "reference_patch_velocity_abs_median",
            "reference_patch_velocity_iqr", "network_neighbor_count",
        ])
        ranked_network_order = {int(idx): rank for rank, idx in enumerate(np.argsort(-net_quality), start=1)}
        for rank, idx in enumerate(candidate_order, start=1):
            r = int(net_rows[idx])
            c = int(net_cols[idx])
            patch = network_patch_stats[idx]
            writer.writerow([
                rank,
                r,
                c,
                float(lon[r, c]),
                float(lat[r, c]),
                float(net_ref_score[idx]),
                float(net_ps_score[idx]),
                float(net_tcoh[idx]),
                float(net_maincc[idx]),
                float(model_rms[r, c]),
                float(net_jump[idx]),
                float(net_dist_edge[idx]),
                float(net_neighbor[idx]),
                float(net_grad[idx]),
                int(gacos_safe_flag[r, c]),
                int(ranked_network_order.get(int(idx), 0)),
                patch_radius,
                float(patch["reference_patch_count"]),
                float(patch["reference_patch_ps_score_median"]),
                float(patch["reference_patch_tcoh_median"]),
                float(patch["reference_patch_mainCC_ratio_median"]),
                float(patch["reference_patch_model_rms_median"]),
                float(patch["reference_patch_jump_risk_median"]),
                float(patch["reference_patch_valid_pair_ratio_median"]),
                float(patch["reference_patch_velocity_median"]),
                float(patch["reference_patch_velocity_abs_median"]),
                float(patch["reference_patch_velocity_iqr"]),
                int(network_neighbor_count[idx]),
            ])
    return {
        "ref_candidates_csv": str(out_path),
        "count": int(len(sel_rows)),
        "ref_primary_network_csv": str(network_csv),
        "ref_primary_network_tif": str(report_dir / "ref_primary_network.tif"),
        "primary_network_count": int(len(net_rows)),
        "neutrality_threshold_mm_yr": float(neutrality_threshold),
        "velocity_iqr_threshold_mm_yr": float(velocity_iqr_threshold),
        "selection_fallback_reason": fallback_reason or "",
    }


def _update_pair_qc_round1(
    pair_qc_csv: Path,
    pair_metrics: dict[str, dict[str, float]],
    summary_json: Path | None = None,
) -> dict[str, Any]:
    with pair_qc_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"status": "empty_pair_qc"}

    fieldnames = list(rows[0].keys())
    extra_fields = [
        "base_risk",
        "base_action",
        "base_pair_weight",
        "bad_ratio",
        "main_cc_coverage",
        "unwrap_jump_index",
        "strict_ps_consistency",
        "gacos_overlap_ratio",
        "q_pair_round1",
        "action_round1",
        "pair_weight_round1",
        "feedback_penalty",
        "q_pair_round2",
        "action_round2",
        "pair_weight_round2",
        "final_selected_round",
    ]
    for name in extra_fields:
        if name not in fieldnames:
            fieldnames.append(name)

    counts = {"keep": 0, "downweight": 0, "drop": 0}
    for row in rows:
        pair_key = f"{row['date1']}_{row['date2']}"
        metrics = pair_metrics.get(pair_key, {})
        row["base_risk"] = row.get("base_risk") or row.get("risk", "")
        row["base_action"] = row.get("base_action") or row.get("action", "keep")
        row["base_pair_weight"] = row.get("base_pair_weight") or row.get("pair_weight", "1.0")
        row["bad_ratio"] = float(metrics.get("bad_ratio", 1.0))
        row["main_cc_coverage"] = float(metrics.get("main_cc_coverage", 0.0))
        row["unwrap_jump_index"] = float(metrics.get("unwrap_jump_index", 1.0))
        row["strict_ps_consistency"] = float(metrics.get("strict_ps_consistency", 0.0))
        row["gacos_overlap_ratio"] = float(metrics.get("gacos_overlap_ratio", 1.0))
        q1 = float(metrics.get("q_pair_round1", 0.0))
        row["q_pair_round1"] = q1
        if q1 >= PAIR_Q_THRESHOLDS["keep"]:
            action = "keep"
        elif q1 >= PAIR_Q_THRESHOLDS["downweight"]:
            action = "downweight"
        else:
            action = "drop"
        weight = 1.0 if action == "keep" else 0.5 if action == "downweight" else 0.0
        row["action_round1"] = action
        row["pair_weight_round1"] = weight
        row["feedback_penalty"] = row.get("feedback_penalty") or 0.0
        row["q_pair_round2"] = row.get("q_pair_round2") or ""
        row["action_round2"] = row.get("action_round2") or ""
        row["pair_weight_round2"] = row.get("pair_weight_round2") or ""
        row["action"] = action
        row["pair_weight"] = weight
        row["final_selected_round"] = row.get("final_selected_round") or "round1"
        counts[action] += 1

    with pair_qc_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if summary_json and Path(summary_json).exists():
        summary = _read_json(Path(summary_json))
        summary["depsi_like_round1_action_counts"] = counts
        _write_json(Path(summary_json), summary)

    return {
        "pair_qc_csv": str(pair_qc_csv),
        "round1_action_counts": counts,
    }


def run_depsi_like_qc(
    dolphin_dir: Path | None = None,
    mintpy_dir: Path | None = None,
    geom_source_dir: Path | None = None,
    report_dir: Path | None = None,
    pair_qc_csv: Path | None = None,
    pair_qc_summary_json: Path | None = None,
    gacos_coverage_report: Path | None = None,
    atmo_method: str | None = None,
    config: dict[str, Any] | None = None,
    wavelength: float | None = None,
    crop_offset: tuple[int, int] | None = None,
) -> dict[str, Any]:
    dolphin_dir = Path(dolphin_dir or cfg.DOLPHIN_DIR)
    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    geom_source_dir = Path(geom_source_dir or (cfg.ISCE_WORK_DIR / "merged" / "geom_reference"))
    report_dir = _ensure_dir(report_dir)
    pair_qc_csv = Path(pair_qc_csv or (report_dir / "pair_qc.csv"))
    pair_qc_summary_json = Path(pair_qc_summary_json or (report_dir / "pair_qc_summary.json"))

    cfg_like = dict(DEPSI_LIKE_DEFAULTS)
    cfg_like.update(config or {})

    tcoh_path = max(
        (dolphin_dir / "interferograms").glob("temporal_coherence_average_*.tif"),
        key=lambda p: p.stat().st_mtime,
    )
    ps_mask_path = dolphin_dir / "interferograms" / "ps_mask_looked.tif"
    amp_disp_path = dolphin_dir / "interferograms" / "amp_dispersion_looked.tif"
    velocity_path = dolphin_dir / "timeseries" / "velocity.tif"

    tcoh, _ = _read_tif(tcoh_path)
    ps_mask, _ = _read_tif(ps_mask_path)
    amp_disp, _ = _read_tif(amp_disp_path)
    velocity, _ = _read_tif(velocity_path)
    geom = _load_geometry(
        mintpy_dir,
        geom_source_dir,
        output_shape=tcoh.shape,
        crop_offset=crop_offset,
    )
    water_mask = _find_water_mask(mintpy_dir)

    valid_geom = np.isfinite(geom["latitude"]) & np.isfinite(geom["longitude"]) & (geom["latitude"] > 0.1)
    candidate_mask = (ps_mask > 0) & valid_geom
    if water_mask is not None and water_mask.shape == candidate_mask.shape:
        candidate_mask &= water_mask

    rows, cols = np.where(candidate_mask)
    if len(rows) == 0:
        raise RuntimeError("DePSI-like QC 未找到可用 PS 候选点。")

    scene_dates, ts_stack_mm, ref_tif = _load_candidate_series(dolphin_dir, rows, cols)
    day_offsets = np.array(
        [(np.datetime64(d) - np.datetime64(scene_dates[0])).astype(int) for d in scene_dates],
        dtype=np.float64,
    )
    wave = float(wavelength or 0.05546576)

    model_fit = _fit_candidate_models(ts_stack_mm, day_offsets)
    date_to_idx = {d: i for i, d in enumerate(scene_dates)}
    pair_data = _compute_pair_actual_and_residuals(
        dolphin_dir=dolphin_dir,
        rows=rows,
        cols=cols,
        date_to_idx=date_to_idx,
        pred_best=model_fit["pred_best"],
        wavelength=wave,
    )

    abs_res = pair_data["abs_residual"]
    finite_res = abs_res[np.isfinite(abs_res)]
    res_p90 = float(np.nanpercentile(finite_res, 90)) if finite_res.size else 0.0
    res_p95 = float(np.nanpercentile(finite_res, 95)) if finite_res.size else 0.0
    jump_threshold = max(
        float(cfg_like["jump_threshold_base_mm"]),
        float(res_p95) * float(cfg_like["jump_threshold_scale"]),
    )

    valid_pair_ratio = pair_data["valid_mask"].mean(axis=0).astype(np.float32)
    main_cc_ratio = pair_data["main_mask"].mean(axis=0).astype(np.float32)
    jump_hits = np.where(pair_data["valid_mask"], abs_res > jump_threshold, False)
    jump_count = jump_hits.sum(axis=0).astype(np.float32)
    valid_count = pair_data["valid_mask"].sum(axis=0).astype(np.float32)
    jump_risk = np.divide(jump_count, np.maximum(valid_count, 1.0), dtype=np.float32)

    model_rms = model_fit["model_rms"]
    model_rms_norm = _normalize(model_rms)
    tcoh_vals = np.clip(tcoh[rows, cols].astype(np.float32), 0.0, 1.0)
    maincc_component = np.clip(main_cc_ratio, 0.0, 1.0)
    model_component = 1.0 - model_rms_norm.astype(np.float32)
    jump_component = 1.0 - np.clip(jump_risk, 0.0, 1.0)
    ps_score_vals = (
        0.30 * tcoh_vals
        + 0.20 * np.clip(valid_pair_ratio, 0.0, 1.0)
        + 0.20 * maincc_component
        + 0.15 * model_component
        + 0.15 * jump_component
    ).astype(np.float32)

    shape = tcoh.shape
    ps_score = np.full(shape, np.nan, dtype=np.float32)
    valid_pair_ratio_grid = np.full(shape, np.nan, dtype=np.float32)
    maincc_ratio_grid = np.full(shape, np.nan, dtype=np.float32)
    model_rms_grid = np.full(shape, np.nan, dtype=np.float32)
    jump_risk_grid = np.full(shape, np.nan, dtype=np.float32)
    delta_bic_grid = np.full(shape, np.nan, dtype=np.float32)
    sensitivity_grid = np.full(shape, np.nan, dtype=np.float32)
    best_model_code_grid = np.full(shape, -1, dtype=np.int16)
    ps_score[rows, cols] = ps_score_vals
    valid_pair_ratio_grid[rows, cols] = valid_pair_ratio
    maincc_ratio_grid[rows, cols] = main_cc_ratio
    model_rms_grid[rows, cols] = model_rms
    jump_risk_grid[rows, cols] = jump_risk
    delta_bic_grid[rows, cols] = model_fit["delta_bic"]
    sensitivity_grid[rows, cols] = model_fit["leave_one_pair_out_sensitivity"]
    best_model_code_grid[rows, cols] = model_fit["best_model_code"]

    strict_mask = np.isfinite(ps_score) & (ps_score >= float(cfg_like["strict_ps_threshold"]))
    relaxed_mask = np.isfinite(ps_score) & (ps_score >= float(cfg_like["relaxed_ps_threshold"])) & (~strict_mask)

    total_valid = int(np.sum(valid_geom))
    strict_count = int(np.sum(strict_mask))
    relaxed_count = int(np.sum(relaxed_mask))
    strict_ratio = float(strict_count / max(total_valid, 1))
    relaxed_ratio = float(relaxed_count / max(total_valid, 1))

    warning_flag = False
    warning_reasons: list[str] = []
    if strict_ratio < float(cfg_like["strict_sparse_ratio_threshold"]) or strict_count < int(cfg_like["strict_sparse_candidate_threshold"]):
        warning_flag = True
        warning_reasons.append("too_sparse")
    if strict_ratio > float(cfg_like["strict_too_loose_ratio_threshold"]):
        warning_flag = True
        warning_reasons.append("strict_too_loose")
    if strict_ratio + relaxed_ratio > float(cfg_like["strict_relaxed_too_loose_ratio_threshold"]):
        warning_flag = True
        warning_reasons.append("strict_relaxed_too_loose")

    _write_like_tif(report_dir / "ps_score.tif", ps_score, tcoh_path, dtype="float32", nodata=np.nan)
    _write_like_tif(report_dir / "mask_ps_strict.tif", strict_mask.astype(np.uint8), tcoh_path, dtype="uint8", nodata=0)
    _write_like_tif(report_dir / "mask_ps_relaxed.tif", relaxed_mask.astype(np.uint8), tcoh_path, dtype="uint8", nodata=0)
    _write_like_tif(report_dir / "tcoh_component.tif", np.where(candidate_mask, tcoh, np.nan), tcoh_path, dtype="float32", nodata=np.nan)
    _write_like_tif(report_dir / "mainCC_component.tif", maincc_ratio_grid, tcoh_path, dtype="float32", nodata=np.nan)
    _write_like_tif(report_dir / "model_component.tif", np.where(np.isfinite(model_rms_grid), 1.0 - _normalize(model_rms_grid), np.nan), tcoh_path, dtype="float32", nodata=np.nan)
    _write_like_tif(report_dir / "jump_component.tif", np.where(np.isfinite(jump_risk_grid), 1.0 - np.clip(jump_risk_grid, 0.0, 1.0), np.nan), tcoh_path, dtype="float32", nodata=np.nan)

    with h5py.File(report_dir / "ps_model_metrics.h5", "w") as f:
        f.create_dataset("best_model_code", data=best_model_code_grid, compression="gzip")
        f.create_dataset("model_rms", data=model_rms_grid, compression="gzip")
        f.create_dataset("delta_bic", data=delta_bic_grid, compression="gzip")
        f.create_dataset("leave_one_pair_out_sensitivity", data=sensitivity_grid, compression="gzip")
        f.create_dataset("valid_pair_ratio", data=valid_pair_ratio_grid, compression="gzip")
        f.create_dataset("mainCC_ratio", data=maincc_ratio_grid, compression="gzip")
        f.create_dataset("jump_risk", data=jump_risk_grid, compression="gzip")
        f.attrs["scene_dates"] = json.dumps(scene_dates, ensure_ascii=False)

    gacos_current_bounds = None
    gacos_per_date = None
    if gacos_coverage_report and Path(gacos_coverage_report).exists():
        gacos_report = _read_json(Path(gacos_coverage_report))
        gacos_current_bounds = gacos_report.get("current_gacos_bounds")
        candidate_results = gacos_report.get("candidate_results") or []
        if candidate_results:
            gacos_per_date = candidate_results[0].get("per_date")

    gacos_safe_flag_grid = np.ones(shape, dtype=np.int16)
    gacos_safe_vals = _gacos_safe_flags(
        geom["latitude"][rows, cols],
        geom["longitude"][rows, cols],
        atmo_method,
        gacos_current_bounds,
    )
    gacos_safe_flag_grid[rows, cols] = gacos_safe_vals

    ref_summary = _select_reference_candidates(
        strict_mask=strict_mask,
        ps_score=ps_score,
        tcoh=tcoh,
        maincc_ratio=maincc_ratio_grid,
        model_rms=model_rms_grid,
        jump_risk=jump_risk_grid,
        valid_pair_ratio=valid_pair_ratio_grid,
        lat=geom["latitude"],
        lon=geom["longitude"],
        velocity_gradient=_compute_velocity_gradient(velocity),
        gacos_safe_flag=gacos_safe_flag_grid,
        velocity_proxy=velocity,
        report_dir=report_dir,
        top_n=int(cfg_like["ref_candidate_top_n"]),
        config=cfg_like,
    )

    strict_indices = np.where(strict_mask[rows, cols])[0]
    pair_metrics_raw: dict[str, dict[str, float]] = {}
    pair_median_abs = []
    pairs = pair_data["pairs"]
    for i, (d1, d2) in enumerate(pairs):
        pair_key = f"{d1}_{d2}"
        if len(strict_indices) == 0:
            pair_metrics_raw[pair_key] = {
                "bad_ratio": 1.0,
                "main_cc_coverage": 0.0,
                "unwrap_jump_index": 1.0,
                "strict_ps_consistency_raw": 0.0,
                "gacos_overlap_ratio": 1.0,
            }
            pair_median_abs.append(np.nan)
            continue
        valid_strict = pair_data["valid_mask"][i, strict_indices]
        main_strict = pair_data["main_mask"][i, strict_indices]
        abs_strict = pair_data["abs_residual"][i, strict_indices]
        bad_ratio = float(1.0 - np.mean(valid_strict))
        main_cov = float(np.mean(main_strict))
        if np.any(valid_strict):
            unwrap_jump_index = float(np.mean(abs_strict[valid_strict] > jump_threshold))
            med_abs = float(np.nanmedian(abs_strict[valid_strict]))
        else:
            unwrap_jump_index = 1.0
            med_abs = np.nan
        if atmo_method == "gacos" and gacos_per_date:
            d1_cov = float((gacos_per_date.get(d1) or {}).get("valid_coverage_ratio", 1.0))
            d2_cov = float((gacos_per_date.get(d2) or {}).get("valid_coverage_ratio", 1.0))
            gacos_overlap = min(d1_cov, d2_cov)
        else:
            gacos_overlap = 1.0
        pair_metrics_raw[pair_key] = {
            "bad_ratio": bad_ratio,
            "main_cc_coverage": main_cov,
            "unwrap_jump_index": unwrap_jump_index,
            "strict_ps_consistency_raw": med_abs,
            "gacos_overlap_ratio": gacos_overlap,
        }
        pair_median_abs.append(med_abs)

    pair_median_abs_arr = np.asarray(pair_median_abs, dtype=np.float64)
    med_norm = _normalize(pair_median_abs_arr)
    for i, pair_key in enumerate([f"{d1}_{d2}" for d1, d2 in pairs]):
        consistency = 1.0 - float(med_norm[i]) if np.isfinite(pair_median_abs_arr[i]) else 0.0
        pair_metrics_raw[pair_key]["strict_ps_consistency"] = consistency
        q1 = (
            0.30 * (1.0 - pair_metrics_raw[pair_key]["bad_ratio"])
            + 0.25 * pair_metrics_raw[pair_key]["main_cc_coverage"]
            + 0.20 * (1.0 - pair_metrics_raw[pair_key]["unwrap_jump_index"])
            + 0.15 * pair_metrics_raw[pair_key]["strict_ps_consistency"]
            + 0.10 * pair_metrics_raw[pair_key]["gacos_overlap_ratio"]
        )
        pair_metrics_raw[pair_key]["q_pair_round1"] = float(np.clip(q1, 0.0, 1.0))

    pair_round1_summary = _update_pair_qc_round1(pair_qc_csv, pair_metrics_raw, pair_qc_summary_json)

    ps_score_summary = {
        "report_dir": str(report_dir),
        "ps_score_tif": str(report_dir / "ps_score.tif"),
        "mask_ps_strict_tif": str(report_dir / "mask_ps_strict.tif"),
        "mask_ps_relaxed_tif": str(report_dir / "mask_ps_relaxed.tif"),
        "ref_primary_network_tif": ref_summary.get("ref_primary_network_tif"),
        "ref_primary_network_csv": ref_summary.get("ref_primary_network_csv"),
        "tcoh_component_tif": str(report_dir / "tcoh_component.tif"),
        "mainCC_component_tif": str(report_dir / "mainCC_component.tif"),
        "model_component_tif": str(report_dir / "model_component.tif"),
        "jump_component_tif": str(report_dir / "jump_component.tif"),
        "ps_model_metrics_h5": str(report_dir / "ps_model_metrics.h5"),
        "ref_candidates_csv": ref_summary["ref_candidates_csv"],
        "candidate_count": int(len(rows)),
        "strict_candidate_count": strict_count,
        "primary_network_count": int(ref_summary.get("primary_network_count", 0)),
        "strict_coverage_ratio": strict_ratio,
        "relaxed_coverage_ratio": relaxed_ratio,
        "coverage_warning_flag": bool(warning_flag),
        "coverage_warning_reason": warning_reasons,
        "selection_fallback_reason": ref_summary.get("selection_fallback_reason", ""),
        "strict_ps_threshold": float(cfg_like["strict_ps_threshold"]),
        "relaxed_ps_threshold": float(cfg_like["relaxed_ps_threshold"]),
        "jump_threshold_base_mm": float(cfg_like["jump_threshold_base_mm"]),
        "jump_threshold_quantile": float(cfg_like["jump_threshold_quantile"]),
        "jump_threshold_scale": float(cfg_like["jump_threshold_scale"]),
        "jump_threshold_applied_mm": float(jump_threshold),
        "abnormal_date_abs_frac_threshold": float(cfg_like["abnormal_date_abs_frac_threshold"]),
        "abnormal_date_abs_residual_mm": float(cfg_like["abnormal_date_abs_residual_mm"]),
        "abnormal_date_quantile": float(cfg_like["abnormal_date_quantile"]),
        "ref_candidate_top_n": int(cfg_like["ref_candidate_top_n"]),
        "ref_candidate_patch_radius": int(cfg_like["ref_candidate_patch_radius"]),
        "ref_network_min_distance_m": float(cfg_like["ref_network_min_distance_m"]),
        "ref_min_tcoh": float(cfg_like["ref_min_tcoh"]),
        "ref_min_valid_pair_ratio": float(cfg_like["ref_min_valid_pair_ratio"]),
        "ref_min_maincc_ratio": float(cfg_like["ref_min_maincc_ratio"]),
        "strict_ps_residual_p90_mm": float(res_p90),
        "strict_ps_residual_p95_mm": float(res_p95),
        "reference_neutrality_threshold_mm_yr": float(ref_summary.get("neutrality_threshold_mm_yr", float("nan"))),
        "reference_velocity_iqr_threshold_mm_yr": float(ref_summary.get("velocity_iqr_threshold_mm_yr", float("nan"))),
        "pair_round1_action_counts": pair_round1_summary["round1_action_counts"],
    }
    _write_json(report_dir / "ps_score_summary.json", ps_score_summary)
    return ps_score_summary


def _pick_final_timeseries(mintpy_dir: Path) -> Path | None:
    candidates = [
        mintpy_dir / "timeseries_SET_GACOS_ramp_demErr.h5",
        mintpy_dir / "timeseries_SET_ERA5_ramp_demErr.h5",
        mintpy_dir / "timeseries_SET_GACOS_ramp.h5",
        mintpy_dir / "timeseries_SET_ERA5_ramp.h5",
        mintpy_dir / "timeseries_SET_GACOS.h5",
        mintpy_dir / "timeseries_SET_ERA5.h5",
        mintpy_dir / "timeseries_SET.h5",
        mintpy_dir / "timeseries.h5",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_mintpy_timeseries(mintpy_dir: Path) -> tuple[list[str], np.ndarray]:
    ts_path = _pick_final_timeseries(mintpy_dir)
    if ts_path is None:
        raise FileNotFoundError(f"未找到 MintPy 时序结果: {mintpy_dir}")
    with h5py.File(ts_path, "r") as f:
        dates = [d.decode("utf-8") for d in f["date"][:]]
        ts = np.asarray(f["timeseries"][:], dtype=np.float32) * 1000.0
    return dates, ts


def _load_ref_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        item = dict(row)
        for key in ["rank", "row", "col", "gacos_safe_flag", "network_rank", "reference_patch_radius", "network_neighbor_count"]:
            if key in item and item[key] != "":
                item[key] = int(float(item[key]))
        for key in [
            "ref_score",
            "ps_score",
            "tcoh",
            "mainCC_ratio",
            "model_rms",
            "jump_risk",
            "dist_to_edge",
            "strict_neighbor_count",
            "local_velocity_gradient",
            "lon",
            "lat",
            "reference_patch_count",
            "reference_patch_ps_score_median",
            "reference_patch_tcoh_median",
            "reference_patch_mainCC_ratio_median",
            "reference_patch_model_rms_median",
            "reference_patch_jump_risk_median",
            "reference_patch_valid_pair_ratio_median",
            "reference_patch_velocity_median",
            "reference_patch_velocity_abs_median",
            "reference_patch_velocity_iqr",
        ]:
            if key in item and item[key] != "":
                item[key] = float(item[key])
        out.append(item)
    return out


def _finalize_pair_qc_selection(pair_qc_csv: Path, selected_round: str) -> dict[str, Any]:
    with pair_qc_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"status": "empty"}
    fieldnames = list(rows[0].keys())
    counts = {"keep": 0, "downweight": 0, "drop": 0}
    for row in rows:
        if selected_round == "round2" and row.get("action_round2"):
            row["action"] = row["action_round2"]
            row["pair_weight"] = row["pair_weight_round2"]
        else:
            row["action"] = row.get("action_round1") or row.get("action") or "keep"
            row["pair_weight"] = row.get("pair_weight_round1") or row.get("pair_weight") or "1.0"
        row["final_selected_round"] = selected_round
        counts[row["action"]] += 1
    with pair_qc_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return {"status": "ok", "selected_round": selected_round, "action_counts": counts}


def lock_reference_candidate(
    ifgram_file: Path,
    ref_candidates_csv: Path,
    candidate_rank: int = 1,
) -> dict[str, Any]:
    candidates = _load_ref_candidates(ref_candidates_csv)
    if not candidates:
        return {"status": "skipped_no_candidates"}
    selected = None
    for item in candidates:
        if int(item["rank"]) == int(candidate_rank):
            selected = item
            break
    if selected is None:
        selected = candidates[0]
    with h5py.File(ifgram_file, "r+") as f:
        f.attrs["REF_Y"] = str(int(selected["row"]))
        f.attrs["REF_X"] = str(int(selected["col"]))
        f.attrs["qc_preserve_reference_candidate"] = "true"
        f.attrs["qc_selected_reference_rank"] = str(int(selected["rank"]))
        f.attrs["qc_selected_reference_network_rank"] = str(int(selected.get("network_rank", selected["rank"])))
        f.attrs["qc_selected_reference_ref_score"] = str(float(selected["ref_score"]))
        f.attrs["qc_reference_patch_radius"] = str(int(selected.get("reference_patch_radius", DEPSI_LIKE_DEFAULTS["ref_candidate_patch_radius"])))
        f.attrs["qc_reference_patch_count"] = str(float(selected.get("reference_patch_count", 0.0)))
        f.attrs["qc_reference_patch_ps_score_median"] = str(float(selected.get("reference_patch_ps_score_median", float("nan"))))
        f.attrs["qc_reference_patch_model_rms_median"] = str(float(selected.get("reference_patch_model_rms_median", float("nan"))))
        f.attrs["qc_reference_patch_jump_risk_median"] = str(float(selected.get("reference_patch_jump_risk_median", float("nan"))))
        f.attrs["qc_reference_patch_velocity_median"] = str(float(selected.get("reference_patch_velocity_median", float("nan"))))
        f.attrs["qc_reference_patch_velocity_abs_median"] = str(float(selected.get("reference_patch_velocity_abs_median", float("nan"))))
        f.attrs["qc_reference_patch_velocity_iqr"] = str(float(selected.get("reference_patch_velocity_iqr", float("nan"))))
    return {
        "status": "ok",
        "candidate_rank": int(selected["rank"]),
        "ref_y": int(selected["row"]),
        "ref_x": int(selected["col"]),
        "ref_score": float(selected["ref_score"]),
        "network_rank": int(selected.get("network_rank", selected["rank"])),
        "reference_patch_radius": int(selected.get("reference_patch_radius", DEPSI_LIKE_DEFAULTS["ref_candidate_patch_radius"])),
        "reference_patch_count": float(selected.get("reference_patch_count", 0.0)),
        "reference_patch_velocity_median": float(selected.get("reference_patch_velocity_median", float("nan"))),
        "reference_patch_velocity_abs_median": float(selected.get("reference_patch_velocity_abs_median", float("nan"))),
        "reference_patch_velocity_iqr": float(selected.get("reference_patch_velocity_iqr", float("nan"))),
    }


def _load_mintpy_velocity(mintpy_dir: Path) -> np.ndarray:
    with h5py.File(Path(mintpy_dir) / "velocity.h5", "r") as f:
        return np.asarray(f["velocity"][:], dtype=np.float32) * 1000.0


def _compute_patch_velocity_metrics(
    velocity_mm_yr: np.ndarray,
    strict_mask: np.ndarray,
    row: int,
    col: int,
    radius: int,
) -> dict[str, float]:
    r0 = max(0, int(row) - int(radius))
    r1 = min(strict_mask.shape[0], int(row) + int(radius) + 1)
    c0 = max(0, int(col) - int(radius))
    c1 = min(strict_mask.shape[1], int(col) + int(radius) + 1)
    patch_mask = strict_mask[r0:r1, c0:c1].astype(bool)
    patch_vals = velocity_mm_yr[r0:r1, c0:c1]
    valid = patch_mask & np.isfinite(patch_vals)
    if not np.any(valid):
        valid = np.isfinite(patch_vals)
    finite_vals = patch_vals[valid]
    if finite_vals.size == 0:
        return {
            "patch_velocity_median_mm_yr": float("nan"),
            "patch_velocity_abs_median_mm_yr": float("nan"),
            "patch_velocity_iqr_mm_yr": float("nan"),
            "patch_point_count": 0,
            "patch_radius": int(radius),
        }
    return {
        "patch_velocity_median_mm_yr": float(np.nanmedian(finite_vals)),
        "patch_velocity_abs_median_mm_yr": float(np.nanmedian(np.abs(finite_vals))),
        "patch_velocity_iqr_mm_yr": float(np.nanpercentile(finite_vals, 75) - np.nanpercentile(finite_vals, 25)),
        "patch_point_count": int(finite_vals.size),
        "patch_radius": int(radius),
    }


def audit_reference_bias(
    mintpy_dir: Path,
    qc_report_dir: Path,
    *,
    selected_reference_rank: int | None = None,
    top_k: int | None = None,
    output_name: str = "reference_bias_audit.json",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    qc_report_dir = Path(qc_report_dir)
    cfg_like = dict(DEPSI_LIKE_DEFAULTS)
    cfg_like.update(config or {})
    strict_path = qc_report_dir / "mask_ps_strict.tif"
    ref_csv = qc_report_dir / "ref_candidates.csv"
    if not strict_path.exists() or not ref_csv.exists():
        return {"status": "skipped_missing_inputs"}

    strict_mask, _ = _read_tif(strict_path)
    strict_mask = strict_mask > 0
    velocity_mm_yr = _load_mintpy_velocity(Path(mintpy_dir))
    candidates = _load_ref_candidates(ref_csv)
    if not candidates:
        summary = {"status": "skipped_no_candidates"}
        _write_json(qc_report_dir / output_name, summary)
        return summary

    top_k = int(top_k or cfg_like["reference_bias_top_k"])
    candidate_records = []
    for item in candidates[:top_k]:
        patch = _compute_patch_velocity_metrics(
            velocity_mm_yr,
            strict_mask,
            int(item["row"]),
            int(item["col"]),
            int(item.get("reference_patch_radius", cfg_like["ref_candidate_patch_radius"])),
        )
        candidate_records.append({
            "rank": int(item["rank"]),
            "row": int(item["row"]),
            "col": int(item["col"]),
            "ref_score": float(item["ref_score"]),
            "patch_velocity_median_mm_yr": patch["patch_velocity_median_mm_yr"],
            "patch_velocity_abs_median_mm_yr": patch["patch_velocity_abs_median_mm_yr"],
            "patch_velocity_iqr_mm_yr": patch["patch_velocity_iqr_mm_yr"],
            "patch_point_count": int(patch["patch_point_count"]),
        })

    finite_abs = np.asarray(
        [item["patch_velocity_abs_median_mm_yr"] for item in candidate_records if np.isfinite(item["patch_velocity_abs_median_mm_yr"])],
        dtype=np.float64,
    )
    threshold = max(
        float(cfg_like["ref_velocity_neutrality_base_mm_yr"]),
        float(np.nanquantile(finite_abs, float(cfg_like["ref_velocity_neutrality_abs_quantile"]))) if finite_abs.size else float(cfg_like["ref_velocity_neutrality_base_mm_yr"]),
    )
    finite_iqr = np.asarray(
        [item["patch_velocity_iqr_mm_yr"] for item in candidate_records if np.isfinite(item["patch_velocity_iqr_mm_yr"])],
        dtype=np.float64,
    )
    iqr_threshold = (
        float(np.nanquantile(finite_iqr, float(cfg_like["ref_velocity_neutrality_iqr_quantile"])))
        if finite_iqr.size else float("inf")
    )

    strict_scene_vals = velocity_mm_yr[strict_mask & np.isfinite(velocity_mm_yr)]
    strict_scene_median = float(np.nanmedian(strict_scene_vals)) if strict_scene_vals.size else float("nan")

    selected = None
    if selected_reference_rank is not None:
        selected = next((item for item in candidate_records if int(item["rank"]) == int(selected_reference_rank)), None)
    if selected is None:
        selected = candidate_records[0]

    reference_bias_detected = bool(
        np.isfinite(selected["patch_velocity_abs_median_mm_yr"])
        and selected["patch_velocity_abs_median_mm_yr"] > threshold
    )
    neutral_candidates = [
        item for item in candidate_records
        if np.isfinite(item["patch_velocity_abs_median_mm_yr"])
        and np.isfinite(item["patch_velocity_iqr_mm_yr"])
        and item["patch_velocity_abs_median_mm_yr"] <= threshold
        and item["patch_velocity_iqr_mm_yr"] <= iqr_threshold
    ]
    ranked_pool = neutral_candidates if neutral_candidates else candidate_records
    recommended = min(
        ranked_pool,
        key=lambda item: (
            float("inf") if not np.isfinite(item["patch_velocity_abs_median_mm_yr"]) else item["patch_velocity_abs_median_mm_yr"],
            float("inf") if not np.isfinite(item["patch_velocity_iqr_mm_yr"]) else item["patch_velocity_iqr_mm_yr"],
            int(item["rank"]),
        ),
    )

    summary = {
        "status": "ok",
        "selected_rank_pass1": int(selected["rank"]) if selected_reference_rank is not None else None,
        "selected_patch_velocity_median_mm_yr": float(selected["patch_velocity_median_mm_yr"]),
        "selected_patch_velocity_abs_median_mm_yr": float(selected["patch_velocity_abs_median_mm_yr"]),
        "selected_patch_velocity_iqr_mm_yr": float(selected["patch_velocity_iqr_mm_yr"]),
        "topk_neutrality_threshold_mm_yr": float(threshold),
        "topk_velocity_iqr_threshold_mm_yr": float(iqr_threshold),
        "strict_ps_scene_median_velocity_mm_yr": float(strict_scene_median),
        "reference_bias_detected": reference_bias_detected,
        "recommended_rank_for_pass2": int(recommended["rank"]) if reference_bias_detected else None,
        "candidate_records": candidate_records,
    }
    _write_json(qc_report_dir / output_name, summary)
    return summary


def evaluate_mintpy_pass(
    mintpy_dir: Path,
    qc_report_dir: Path,
    *,
    pass_label: str,
    selected_reference_rank: int | None = None,
) -> dict[str, Any]:
    qc_report_dir = Path(qc_report_dir)
    strict_path = qc_report_dir / "mask_ps_strict.tif"
    summary_path = qc_report_dir / "ps_score_summary.json"
    ref_csv = qc_report_dir / "ref_candidates.csv"
    dates, ts = _load_mintpy_timeseries(Path(mintpy_dir))
    velocity_mm_yr = _load_mintpy_velocity(Path(mintpy_dir))
    strict_mask, _ = _read_tif(strict_path)
    strict_mask = strict_mask > 0
    rows, cols = np.where(strict_mask)
    if len(rows) == 0:
        return {"status": "skipped_empty_strict_mask", "pass_label": pass_label}

    series_mm = ts[:, rows, cols]
    model_fit = _fit_candidate_models(series_mm, np.array([(np.datetime64(d) - np.datetime64(dates[0])).astype(int) for d in dates], dtype=np.float64))
    summary = _read_json(summary_path)
    jump_threshold = float(summary.get("jump_threshold_applied_mm", DEPSI_LIKE_DEFAULTS["jump_threshold_base_mm"]))

    actual_adj = np.diff(series_mm, axis=0)
    pred_adj = np.diff(model_fit["pred_best"], axis=0)
    adj_resid = np.abs(actual_adj - pred_adj)
    adjacent_jump_ratio = float(np.mean(adj_resid > jump_threshold))
    median_model_rms = float(np.nanmedian(model_fit["model_rms"]))
    retained_count = int(np.sum(np.all(np.isfinite(series_mm), axis=0)))
    strict_scene_vals = velocity_mm_yr[strict_mask & np.isfinite(velocity_mm_yr)]
    strict_scene_median_velocity = float(np.nanmedian(strict_scene_vals)) if strict_scene_vals.size else float("nan")

    reference_abnormal = False
    reference_metrics = None
    suggested_next_rank = None
    if selected_reference_rank is not None and ref_csv.exists():
        candidates = _load_ref_candidates(ref_csv)
        selected = next((c for c in candidates if int(c["rank"]) == int(selected_reference_rank)), None)
        if selected is not None:
            row = int(selected["row"])
            col = int(selected["col"])
            patch_radius = int(selected.get("reference_patch_radius", DEPSI_LIKE_DEFAULTS["ref_candidate_patch_radius"]))
            patch_hits = np.where(
                (rows >= row - patch_radius)
                & (rows <= row + patch_radius)
                & (cols >= col - patch_radius)
                & (cols <= col + patch_radius)
            )[0]
            if patch_hits.size == 0:
                patch_hits = np.where((rows == row) & (cols == col))[0]
            if patch_hits.size:
                point_resid = np.abs(series_mm[:, patch_hits] - model_fit["pred_best"][:, patch_hits])
                point_adj = np.abs(actual_adj[:, patch_hits] - pred_adj[:, patch_hits])
                global_point_median = float(np.nanmedian(np.abs(series_mm - model_fit["pred_best"])))
                patch_velocity_metrics = _compute_patch_velocity_metrics(
                    velocity_mm_yr,
                    strict_mask,
                    row,
                    col,
                    patch_radius,
                )
                reference_metrics = {
                    "mean_abs_residual_mm": float(np.nanmean(point_resid)),
                    "adjacent_jump_ratio": float(np.mean(point_adj > jump_threshold)),
                    "global_median_abs_residual_mm": global_point_median,
                    "patch_point_count": int(max(patch_hits.size, patch_velocity_metrics["patch_point_count"])),
                    "patch_radius": int(patch_radius),
                    "patch_velocity_median_mm_yr": float(patch_velocity_metrics["patch_velocity_median_mm_yr"]),
                    "patch_velocity_abs_median_mm_yr": float(patch_velocity_metrics["patch_velocity_abs_median_mm_yr"]),
                    "patch_velocity_iqr_mm_yr": float(patch_velocity_metrics["patch_velocity_iqr_mm_yr"]),
                }
                reference_abnormal = (
                    reference_metrics["mean_abs_residual_mm"] > max(global_point_median * 1.5, 1.0)
                    or reference_metrics["adjacent_jump_ratio"] > max(adjacent_jump_ratio * 1.5, 0.15)
                )
                if reference_abnormal and selected_reference_rank < int(DEPSI_LIKE_DEFAULTS["max_reference_fallback_rank"]):
                    suggested_next_rank = int(selected_reference_rank) + 1

    out = {
        "status": "ok",
        "pass_label": pass_label,
        "adjacent_jump_ratio": adjacent_jump_ratio,
        "median_model_rms": median_model_rms,
        "retained_count": retained_count,
        "strict_ps_scene_median_velocity_mm_yr": strict_scene_median_velocity,
        "selected_reference_rank": selected_reference_rank,
        "reference_candidate_abnormal": reference_abnormal,
        "reference_candidate_metrics": reference_metrics,
        "suggested_next_reference_rank": suggested_next_rank,
    }
    _write_json(qc_report_dir / f"{pass_label}_evaluation.json", out)
    return out


def compute_date_feedback(
    mintpy_dir: Path,
    qc_report_dir: Path,
    pair_qc_csv: Path,
    *,
    selected_reference_rank: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    qc_report_dir = Path(qc_report_dir)
    pair_qc_csv = Path(pair_qc_csv)
    cfg_like = dict(DEPSI_LIKE_DEFAULTS)
    cfg_like.update(config or {})

    dates, ts = _load_mintpy_timeseries(Path(mintpy_dir))
    strict_mask, _ = _read_tif(qc_report_dir / "mask_ps_strict.tif")
    strict_mask = strict_mask > 0
    rows, cols = np.where(strict_mask)
    if len(rows) == 0:
        return {"status": "skipped_empty_strict_mask"}

    series_mm = ts[:, rows, cols]
    model_fit = _fit_candidate_models(
        series_mm,
        np.array([(np.datetime64(d) - np.datetime64(dates[0])).astype(int) for d in dates], dtype=np.float64),
    )
    summary = _read_json(qc_report_dir / "ps_score_summary.json")
    jump_threshold = float(summary.get("jump_threshold_applied_mm", cfg_like["jump_threshold_base_mm"]))

    abs_res = np.abs(series_mm - model_fit["pred_best"])
    date_median = np.nanmedian(abs_res, axis=1)
    date_frac = np.mean(abs_res > jump_threshold, axis=1)
    date_signed_bias = np.nanmedian(series_mm - model_fit["pred_best"], axis=1)

    med_q = float(np.nanquantile(date_median, float(cfg_like["abnormal_date_quantile"])))
    frac_q = float(np.nanquantile(date_frac, float(cfg_like["abnormal_date_quantile"])))

    h, w = strict_mask.shape
    row_bins = np.minimum((rows.astype(np.float64) / max(h, 1) * 10).astype(int), 9)
    col_bins = np.minimum((cols.astype(np.float64) / max(w, 1) * 10).astype(int), 9)

    records = []
    flagged_dates: dict[str, dict[str, Any]] = {}
    for i, date in enumerate(dates):
        anomalies = abs_res[i, :] > jump_threshold
        n_anom = int(np.sum(anomalies))
        if n_anom > 0:
            grid_counts = np.zeros((10, 10), dtype=np.int32)
            for rb, cb in zip(row_bins[anomalies], col_bins[anomalies]):
                grid_counts[int(rb), int(cb)] += 1
            concentration = float(grid_counts.max() / max(n_anom, 1))
        else:
            concentration = 0.0

        if concentration <= ANOMALY_SCOPE_THRESHOLDS["global_max"]:
            scope = "global"
        elif concentration <= ANOMALY_SCOPE_THRESHOLDS["mixed_max"]:
            scope = "mixed"
        else:
            scope = "local"

        flag_abs = bool(
            (float(date_frac[i]) >= float(cfg_like["abnormal_date_abs_frac_threshold"]))
            or (float(date_median[i]) >= float(cfg_like["abnormal_date_abs_residual_mm"]))
        )
        record = {
            "date": date,
            "median_abs_residual_mm": float(date_median[i]),
            "frac_gt_threshold": float(date_frac[i]),
            "signed_bias_mm": float(date_signed_bias[i]),
            "spatial_concentration_index": concentration,
            "anomaly_scope": scope,
            "flag_by_absolute_rule": flag_abs,
            "flag_by_quantile_rule": False,
            "final_abnormal_flag": flag_abs,
        }
        records.append(record)

    median_sorted = sorted(records, key=lambda item: item["median_abs_residual_mm"], reverse=True)
    frac_sorted = sorted(records, key=lambda item: item["frac_gt_threshold"], reverse=True)
    rank_med = {item["date"]: i + 1 for i, item in enumerate(median_sorted)}
    rank_frac = {item["date"]: i + 1 for i, item in enumerate(frac_sorted)}
    top_k = max(1, int(math.ceil(len(records) * (1.0 - float(cfg_like["abnormal_date_quantile"])))))
    for record in records:
        record["rank_by_median_abs"] = int(rank_med[record["date"]])
        record["rank_by_frac_gt"] = int(rank_frac[record["date"]])
        record["flag_by_quantile_rule"] = bool(
            record["rank_by_median_abs"] <= top_k
            or record["rank_by_frac_gt"] <= top_k
        )
        record["final_abnormal_flag"] = bool(
            record["flag_by_absolute_rule"] or record["flag_by_quantile_rule"]
        )
        if record["final_abnormal_flag"]:
            flagged_dates[record["date"]] = record

    date_qc_csv = qc_report_dir / "date_qc.csv"
    with date_qc_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    with pair_qc_csv.open("r", encoding="utf-8") as f:
        pair_rows = list(csv.DictReader(f))
    if pair_rows:
        fieldnames = list(pair_rows[0].keys())
        for name in ["feedback_penalty", "q_pair_round2", "action_round2", "pair_weight_round2", "final_selected_round"]:
            if name not in fieldnames:
                fieldnames.append(name)
        for row in pair_rows:
            penalty = 0.0
            for key in (row["date1"], row["date2"]):
                rec = flagged_dates.get(key)
                if not rec:
                    continue
                penalty += FEEDBACK_PENALTIES[rec["anomaly_scope"]]
            if float(row.get("strict_ps_consistency") or 0.0) < 0.50:
                penalty += FEEDBACK_PENALTIES["low_consistency_extra"]
            penalty = min(penalty, FEEDBACK_PENALTIES["max_penalty"])
            q1 = float(row.get("q_pair_round1") or 0.0)
            q2 = max(0.0, q1 - penalty)
            if q2 >= PAIR_Q_THRESHOLDS["keep"]:
                action = "keep"
            elif q2 >= PAIR_Q_THRESHOLDS["downweight"]:
                action = "downweight"
            else:
                action = "drop"
            weight = 1.0 if action == "keep" else 0.5 if action == "downweight" else 0.0
            row["feedback_penalty"] = float(penalty)
            row["q_pair_round2"] = float(q2)
            row["action_round2"] = action
            row["pair_weight_round2"] = float(weight)
            row["action"] = action
            row["pair_weight"] = float(weight)
            row["final_selected_round"] = "round2"

        with pair_qc_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pair_rows)

    pass_eval = evaluate_mintpy_pass(
        mintpy_dir=Path(mintpy_dir),
        qc_report_dir=qc_report_dir,
        pass_label="pass1",
        selected_reference_rank=selected_reference_rank,
    )

    summary = {
        "status": "ok",
        "date_qc_csv": str(date_qc_csv),
        "abnormal_date_count": int(sum(1 for item in records if item["final_abnormal_flag"])),
        "abnormal_dates": [item["date"] for item in records if item["final_abnormal_flag"]],
        "median_abs_quantile_threshold_mm": med_q,
        "frac_quantile_threshold": frac_q,
        "absolute_frac_threshold": float(cfg_like["abnormal_date_abs_frac_threshold"]),
        "absolute_residual_threshold_mm": float(cfg_like["abnormal_date_abs_residual_mm"]),
        "selected_reference_rank": selected_reference_rank,
        "suggested_next_reference_rank": pass_eval.get("suggested_next_reference_rank"),
        "reference_candidate_abnormal": pass_eval.get("reference_candidate_abnormal"),
    }
    _write_json(qc_report_dir / "mintpy_feedback_summary.json", summary)
    return summary


def snapshot_mintpy_outputs(mintpy_dir: Path, snapshot_dir: Path) -> dict[str, Any]:
    mintpy_dir = Path(mintpy_dir)
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(mintpy_dir.glob("*.h5")):
        shutil.copy2(path, snapshot_dir / path.name)
        copied.append(path.name)
    return {"snapshot_dir": str(snapshot_dir), "files": copied}


def clear_mintpy_outputs_for_rerun(mintpy_dir: Path) -> dict[str, Any]:
    mintpy_dir = Path(mintpy_dir)
    removed = []
    keep = {"waterMask.h5"}
    for path in sorted(mintpy_dir.glob("*.h5")):
        if path.name in keep:
            continue
        path.unlink()
        removed.append(path.name)
    return {"removed": removed}


def restore_mintpy_snapshot(mintpy_dir: Path, snapshot_dir: Path) -> dict[str, Any]:
    mintpy_dir = Path(mintpy_dir)
    snapshot_dir = Path(snapshot_dir)
    clear_mintpy_outputs_for_rerun(mintpy_dir)
    restored = []
    for path in sorted(snapshot_dir.glob("*.h5")):
        shutil.copy2(path, mintpy_dir / path.name)
        restored.append(path.name)
    return {"restored": restored}


def _load_mintpy_export_data(
    mintpy_dir: Path,
    geom_source_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    geom = _load_geometry(mintpy_dir, geom_source_dir)
    with h5py.File(mintpy_dir / "velocity.h5", "r") as f:
        vel = np.asarray(f["velocity"][:], dtype=np.float32) * 1000.0
        if "velocityStd" in f:
            vstd = np.asarray(f["velocityStd"][:], dtype=np.float32) * 1000.0
        else:
            vstd = np.full_like(vel, np.nan, dtype=np.float32)
    with h5py.File(mintpy_dir / "temporalCoherence.h5", "r") as f:
        tcoh = np.asarray(f["temporalCoherence"][:], dtype=np.float32)
    dates, ts = _load_mintpy_timeseries(mintpy_dir)
    return vel, vstd, dates, ts, geom["latitude"], geom["longitude"], tcoh


def export_depsi_like_outputs(
    mintpy_dir: Path | None = None,
    qc_report_dir: Path | None = None,
    export_dir: Path | None = None,
) -> dict[str, Any]:
    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    qc_report_dir = Path(qc_report_dir or (cfg.WORK_DIR / "mainchain_qc"))
    export_dir = Path(export_dir or cfg.EXPORT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)

    strict_mask, _ = _read_tif(qc_report_dir / "mask_ps_strict.tif")
    strict_mask = strict_mask > 0
    ps_score, _ = _read_tif(qc_report_dir / "ps_score.tif")
    vel, vstd, dates, ts, lat, lon, tcoh = _load_mintpy_export_data(
        mintpy_dir,
        cfg.ISCE_WORK_DIR / "merged" / "geom_reference",
    )

    with rasterio.open(qc_report_dir / "mask_ps_strict.tif") as ref:
        profile = ref.profile.copy()
    profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", tiled=True)

    out_tif = export_dir / "velocity_high_confidence.tif"
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(np.where(strict_mask, vel, np.nan).astype(np.float32), 1)

    # ── DePSI 三合一面板图：(a) High-conf velocity  (b) PS score  (c) Strict PS mask ──
    # 直接复用主链速率图的格网和 hillshade（保证风格完全一致）
    valid = strict_mask & np.isfinite(vel) & np.isfinite(lat) & np.isfinite(lon) & (lat > 0.1)
    valid_score = np.isfinite(ps_score) & np.isfinite(lat) & np.isfinite(lon) & (lat > 0.1)

    from .viz import (_load_mintpy_data, _prepare_geo_canvas,
                      _draw_hillshade_background, _draw_water_overlay,
                      _add_north_arrow, _add_water_legend,
                      _subfig_label, _format_degree_axis, _save_figure,
                      _COL2, _WATER_COLOR, _symmetric_vlim)
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    # 加载几何数据 → 构建和速率图一致的地理画布
    viz_data = _load_mintpy_data(mintpy_dir)
    lat_g, lon_g, hgt_g = viz_data["lat"], viz_data["lon"], viz_data["height"]
    if lat_g is None:
        logger.warning("无几何数据，跳过 DePSI 面板图")
        out_map = out_ps = out_mask = export_dir / "depsi_quality_panel.png"
    else:
        geo_valid = np.isfinite(lat_g) & (lat_g > 0.1) & np.isfinite(lon_g)
        aoi = cfg._AOI_BBOX
        if aoi is not None:
            S, N, W, E = aoi
            geo_valid &= (lat_g >= S) & (lat_g <= N) & (lon_g >= W) & (lon_g <= E)

        canvas = _prepare_geo_canvas(
            {"lat": lat_g, "lon": lon_g, "height": hgt_g},
            geo_valid,
            target_cols=760,
        )
        strict_ratio = float(valid.sum()) / max(int(geo_valid.sum()), 1)
        water_present = canvas["water_grid"] is not None and np.any(canvas["water_grid"])

        def _prep_axis(ax, title, lbl):
            ax.set_facecolor("white")
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.42)
            _draw_hillshade_background(ax, canvas["hs_grid"], None, canvas["extent"], alpha=0.26)
            _draw_water_overlay(ax, canvas["water_grid"], canvas["extent"])
            ax.set_xlim(canvas["lon_min"], canvas["lon_max"])
            ax.set_ylim(canvas["lat_min"], canvas["lat_max"])
            ax.set_aspect(1.0 / np.cos(np.radians(canvas["lat_c"])))
            ax.set_title(title, fontsize=7.5, loc='left', fontweight='bold', pad=3)
            _subfig_label(ax, lbl, x=0.03, y=0.96)

        aspect = (
            (canvas["lat_max"] - canvas["lat_min"]) /
            max((canvas["lon_max"] - canvas["lon_min"]) * np.cos(np.radians(canvas["lat_c"])), 0.001)
        )
        fig_h = float(np.clip(3.55 + aspect * 1.05, 4.4, 5.9))
        fig = plt.figure(figsize=(8.0, fig_h))
        gs = GridSpec(
            2, 4, figure=fig,
            width_ratios=[1.42, 0.045, 0.95, 0.045],
            height_ratios=[1.0, 1.0],
            left=0.06, right=0.97, bottom=0.10, top=0.93,
            wspace=0.12, hspace=0.10,
        )

        # (a) High-conf velocity — 左侧主图
        ax_a = fig.add_subplot(gs[:, 0])
        cax_a = fig.add_subplot(gs[:, 1])
        _prep_axis(ax_a, "High-Confidence Velocity", "a")
        if np.any(valid):
            vabs = _symmetric_vlim(vel[valid], pct=95)
            norm_a = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
            cmap_a = plt.cm.get_cmap("RdBu_r")
            ax_a.scatter(
                lon[valid], lat[valid], c=vel[valid],
                cmap=cmap_a, norm=norm_a,
                s=4.0, linewidths=0.0, alpha=0.92,
                rasterized=True, zorder=4,
            )
            sm_a = plt.cm.ScalarMappable(cmap=cmap_a, norm=norm_a)
            sm_a.set_array([])
            cb_a = fig.colorbar(sm_a, cax=cax_a, extend='both')
            cb_a.set_label("mm yr$^{-1}$", fontsize=6)
            cb_a.ax.tick_params(labelsize=5)
            cb_a.outline.set_linewidth(0.3)
        ax_a.set_ylabel("Latitude (°N)", fontsize=7)
        ax_a.set_xlabel("Longitude (°E)", fontsize=7)
        _format_degree_axis(ax_a, 3)
        _add_north_arrow(ax_a, x=0.94, y=0.90, size=0.07)
        if water_present:
            _add_water_legend(ax_a, loc='lower left', fontsize=5.4)
        ax_a.text(0.985, 0.985,
                  f'N = {int(valid.sum()):,}\nstrict ratio = {strict_ratio:.3%}',
                  transform=ax_a.transAxes, fontsize=5, ha='right', va='top',
                  color='0.4',
                  bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='0.88', lw=0.35, alpha=0.86))

        # (b) PS score — 右上辅图
        ax_b = fig.add_subplot(gs[0, 2])
        cax_b = fig.add_subplot(gs[0, 3])
        _prep_axis(ax_b, "PS Score", "b")
        if np.any(valid_score):
            norm_b = Normalize(vmin=0, vmax=1)
            cmap_b = plt.cm.get_cmap("inferno")
            ax_b.scatter(
                lon[valid_score], lat[valid_score], c=ps_score[valid_score],
                cmap=cmap_b, norm=norm_b,
                s=2.0, linewidths=0.0, alpha=0.76,
                rasterized=True, zorder=4,
            )
            sm_b = plt.cm.ScalarMappable(cmap=cmap_b, norm=norm_b)
            sm_b.set_array([])
            cb_b = fig.colorbar(sm_b, cax=cax_b)
            cb_b.set_label("PS score", fontsize=6)
            cb_b.ax.tick_params(labelsize=5)
            cb_b.outline.set_linewidth(0.3)
            median_ps = float(np.nanmedian(ps_score[valid_score]))
            ax_b.text(0.98, 0.98, f'median = {median_ps:.2f}',
                      transform=ax_b.transAxes, fontsize=4.9, ha='right', va='top',
                      color='0.38',
                      bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='0.88', lw=0.35, alpha=0.84))
        ax_b.set_xlabel("")
        _format_degree_axis(ax_b, 3)
        ax_b.set_yticklabels([])
        ax_b.set_xticklabels([])

        # (c) Strict PS mask — 右下辅图
        ax_c = fig.add_subplot(gs[1, 2])
        ax_c.set_facecolor("white")
        for sp in ax_c.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.42)
        _draw_hillshade_background(ax_c, canvas["hs_grid"], None, canvas["extent"], alpha=0.26)
        _draw_water_overlay(ax_c, canvas["water_grid"], canvas["extent"])
        if np.any(valid):
            ax_c.scatter(
                lon[valid], lat[valid],
                s=1.8, c='#33a3dc', linewidths=0.0,
                alpha=0.60, rasterized=True, zorder=4,
            )
        ax_c.set_xlim(canvas["lon_min"], canvas["lon_max"])
        ax_c.set_ylim(canvas["lat_min"], canvas["lat_max"])
        ax_c.set_aspect(1.0 / np.cos(np.radians(canvas["lat_c"])))
        ax_c.set_title("Strict PS Coverage", fontsize=7.5, loc='left', fontweight='bold', pad=3)
        _subfig_label(ax_c, 'c', x=0.03, y=0.96)
        ax_c.set_xlabel("Longitude (°E)", fontsize=7)
        ax_c.set_yticklabels([])
        _format_degree_axis(ax_c, 3)
        ax_c.text(0.98, 0.98, f'strict points = {int(valid.sum()):,}',
                  transform=ax_c.transAxes, fontsize=4.9, ha='right', va='top',
                  color='0.38',
                  bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='0.88', lw=0.35, alpha=0.84))
        ax_c.legend(handles=[Patch(facecolor=[0.19, 0.63, 0.83], edgecolor='#0e425f',
                                   lw=0.5, label='Strict PS')], loc='lower left',
                    fontsize=5.2, framealpha=0.88, borderpad=0.2, handlelength=0.9)

        out_depsi = export_dir / "depsi_quality_panel.png"
        _save_figure(fig, out_depsi)
        plt.close(fig)
        out_map = out_ps = out_mask = out_depsi

    # vector / csv
    rows, cols = np.where(valid)
    csv_path = export_dir / "velocity_points_high_confidence.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["UID", "longitude", "latitude", "tcoh", "LOS_mm_yr", "velocityStd_mm_yr"])
        for uid, (r, c) in enumerate(zip(rows, cols), start=1):
            writer.writerow([uid, round(float(lon[r, c]), 6), round(float(lat[r, c]), 6), round(float(tcoh[r, c]), 3), round(float(vel[r, c]), 3), round(float(vstd[r, c]), 3)])

    shp_path = None
    kmz_path = None
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {
                "vel_mm_yr": vel[rows, cols],
                "coherence": tcoh[rows, cols],
                "ps_score": ps_score[rows, cols],
            },
            geometry=[Point(lon[r, c], lat[r, c]) for r, c in zip(rows, cols)],
            crs="EPSG:4326",
        )
        shp_path = export_dir / "velocity_points_high_confidence.shp"
        gdf.to_file(shp_path)
    except Exception as e:
        logger.warning(f"高可信 SHP 导出失败: {e}")

    try:
        import simplekml

        kml = simplekml.Kml()
        vmax = max(np.nanpercentile(np.abs(vel[rows, cols]), 95), 1.0) if len(rows) else 1.0
        for r, c in zip(rows[:50000], cols[:50000]):
            norm = np.clip(vel[r, c] / vmax, -1.0, 1.0)
            if norm < 0:
                rc, gc, bc = 255, int(255 * (1 + norm)), int(255 * (1 + norm))
            else:
                rc, gc, bc = int(255 * (1 - norm)), int(255 * (1 - norm)), 255
            pnt = kml.newpoint(coords=[(float(lon[r, c]), float(lat[r, c]))], description=f"Vel: {vel[r, c]:.1f} mm/yr")
            pnt.style.iconstyle.color = simplekml.Color.rgb(rc, gc, bc)
            pnt.style.iconstyle.scale = 0.4
        kmz_path = export_dir / "velocity_points_high_confidence.kmz"
        kml.savekmz(str(kmz_path))
    except Exception as e:
        logger.warning(f"高可信 KMZ 导出失败: {e}")

    # time-series samples
    candidate_info = []
    for r, c in zip(rows, cols):
        series = ts[:, r, c]
        candidate_info.append({
            "row": int(r),
            "col": int(c),
            "lon": float(lon[r, c]),
            "lat": float(lat[r, c]),
            "vel": float(vel[r, c]),
            "ps_score": float(ps_score[r, c]),
            "stability": float(np.nanstd(np.diff(series))),
        })
    stable = sorted(candidate_info, key=lambda item: item["stability"])[:4]
    deform = sorted(candidate_info, key=lambda item: abs(item["vel"]), reverse=True)[:4]
    edge = sorted(candidate_info, key=lambda item: item["ps_score"])[:4]
    chosen = []
    seen = set()
    for group in [stable, deform, edge]:
        for item in group:
            key = (item["row"], item["col"])
            if key in seen:
                continue
            seen.add(key)
            chosen.append(item)
            if len(chosen) >= 12:
                break
        if len(chosen) >= 12:
            break

    sample_csv = export_dir / "timeseries_samples_high_confidence.csv"
    with sample_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "row", "col", "longitude", "latitude", "LOS_mm_yr", "ps_score", *dates])
        for idx, item in enumerate(chosen, start=1):
            r, c = item["row"], item["col"]
            writer.writerow([idx, r, c, round(item["lon"], 6), round(item["lat"], 6), round(item["vel"], 3), round(item["ps_score"], 3), *[round(float(v), 3) for v in ts[:, r, c]]])

    sample_png = export_dir / "timeseries_samples_high_confidence.png"
    if chosen:
        n = len(chosen)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 2.8 * nrows), squeeze=False)
        x = np.arange(len(dates))
        for ax, item, sid in zip(axes.ravel(), chosen, range(1, n + 1)):
            r, c = item["row"], item["col"]
            series = ts[:, r, c]
            ax.set_facecolor("white")
            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)
            ax.plot(x, series, color="#2980B9", lw=1.0, marker='o',
                    ms=2.5, mew=0.3, mec='white', alpha=0.85, zorder=3)
            # 线性拟合趋势线
            valid_s = np.isfinite(series)
            if valid_s.sum() > 3:
                coeff = np.polyfit(x[valid_s], series[valid_s], 1)
                ax.plot(x, np.polyval(coeff, x), color='#E74C3C',
                        lw=0.7, ls='--', alpha=0.6, zorder=2)
            ax.axhline(0, color='0.6', lw=0.5, ls='--', zorder=1)
            ax.set_title(f"S{sid}  v={item['vel']:.1f} mm/yr  ps={item['ps_score']:.2f}",
                         fontsize=7, loc='left', fontweight='bold', pad=2)
            ax.set_xticks(x[:: max(1, len(x) // 5)])
            ax.set_xticklabels([dates[i] for i in x[:: max(1, len(x) // 5)]],
                               rotation=30, ha="right", fontsize=6)
            ax.tick_params(axis='y', labelsize=6.5)
            ax.grid(alpha=0.12, lw=0.3)
            ax.set_ylabel("LOS (mm)", fontsize=6.5)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.suptitle("High-Confidence PS Time Series Samples", fontsize=9,
                     fontweight='bold', y=1.01)
        fig.tight_layout()
        fig.savefig(sample_png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(sample_png.with_suffix('.pdf'), bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return {
        "velocity_high_confidence_tif": str(out_tif),
        "depsi_quality_panel_png": str(out_map),
        "velocity_map_high_confidence_png": str(out_map),
        "ps_score_map_png": str(out_ps),
        "strict_ps_mask_png": str(out_mask),
        "velocity_points_high_confidence_csv": str(csv_path),
        "velocity_points_high_confidence_shp": str(shp_path) if shp_path else None,
        "velocity_points_high_confidence_kmz": str(kmz_path) if kmz_path else None,
        "timeseries_samples_high_confidence_csv": str(sample_csv),
        "timeseries_samples_high_confidence_png": str(sample_png) if chosen else None,
        "strict_point_count": int(len(rows)),
    }
