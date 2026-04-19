"""
Main-chain QC utilities executed between Dolphin and MintPy.

This module keeps the original InSAR chain intact while adding:
1. Pair-level quality control from Dolphin outputs.
2. GACOS / geometry / padding sanity checks before MintPy.
3. QC-adjusted ifgramStack generation for MintPy consumption.
4. Optional Dolphin ablation helpers.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from . import config as cfg
from .atmosphere import (
    compute_recommended_gacos_bounds,
    convert_gacos_tif_to_ztd,
    prepare_external_atmo_inputs,
)
from .config import logger, save_project_progress
from .depsi_like_qc import (
    _finalize_pair_qc_selection,
    audit_reference_bias,
    clear_mintpy_outputs_for_rerun,
    compute_date_feedback,
    evaluate_mintpy_pass,
    export_depsi_like_outputs,
    lock_reference_candidate,
    restore_mintpy_snapshot,
    run_depsi_like_qc,
    snapshot_mintpy_outputs,
)
from .mintpy_runner import (
    DOLPHIN_STRIDE_X,
    DOLPHIN_STRIDE_Y,
    _extract_s1_metadata,
    build_mintpy_hdf5,
    run_mintpy,
)


PAIR_QC_COLUMNS = [
    "date1",
    "date2",
    "coh_mean",
    "valid_conncomp_frac",
    "n_conncomp",
    "closure_risk_proxy",
    "pair_p95_abs_los_mm",
    "pair_frac_abs_los_gt_50mm",
    "risk",
    "action",
    "pair_weight",
]

PAIR_WEIGHT = {
    "keep": 1.0,
    "downweight": 0.5,
    "drop": 0.0,
}

ACTION_CODE = {
    "keep": 0,
    "downweight": 1,
    "drop": 2,
}

DEFAULT_PADDING_CANDIDATES = (20, 10, 0)
DEFAULT_GACOS_WARN = 0.95
DEFAULT_GACOS_OK = 0.99
DEFAULT_GACOS_MANUAL_WAIT = True


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


def _ensure_dir(path: Path | None) -> Path:
    path = Path(path or (cfg.WORK_DIR / "mainchain_qc"))
    path.mkdir(parents=True, exist_ok=True)
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


def _scene_dates_from_pairs(pairs: Iterable[tuple[str, str]]) -> list[str]:
    return sorted({d for pair in pairs for d in pair})


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    vmin = float(arr[finite].min())
    vmax = float(arr[finite].max())
    if math.isclose(vmin, vmax):
        out = np.zeros_like(arr)
        out[~finite] = 0.0
        return out
    out = (arr - vmin) / (vmax - vmin)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0)


def _compute_extreme_ratio(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    scale = max(mad * 1.4826, 1e-6)
    return float(np.mean(np.abs(finite - med) > (5.0 * scale)))


def _read_gacos_rsc(rsc_path: Path) -> dict[str, float]:
    info: dict[str, float] = {}
    for line in rsc_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            key = parts[0]
            try:
                info[key] = float(parts[1])
            except ValueError:
                continue
    required = ["WIDTH", "FILE_LENGTH", "X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP"]
    missing = [k for k in required if k not in info]
    if missing:
        raise ValueError(f"GACOS RSC 缺少字段: {missing}")
    return info


def _load_gacos_grid(ztd_path: Path, rsc_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    info = _read_gacos_rsc(rsc_path)
    width = int(info["WIDTH"])
    length = int(info["FILE_LENGTH"])
    data = np.fromfile(ztd_path, dtype=np.float32)
    if data.size != width * length:
        raise ValueError(
            f"GACOS 文件尺寸不匹配: {ztd_path} -> {data.size}, expected {width * length}"
        )
    data = data.reshape(length, width)
    x0 = info["X_FIRST"]
    y0 = info["Y_FIRST"]
    dx = info["X_STEP"]
    dy = info["Y_STEP"]
    lon = x0 + np.arange(width, dtype=np.float64) * dx
    lat = y0 + np.arange(length, dtype=np.float64) * dy
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[::-1, :]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        data = data[:, ::-1]
    return lat, lon, data


def _bounds_from_grid(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    mask = valid_mask
    if mask is None:
        mask = np.isfinite(lat_grid) & np.isfinite(lon_grid) & (lat_grid > 0.1)
    if not np.any(mask):
        raise RuntimeError("无法从当前网格计算地理边界")
    return {
        "S": float(np.nanmin(lat_grid[mask])),
        "N": float(np.nanmax(lat_grid[mask])),
        "W": float(np.nanmin(lon_grid[mask])),
        "E": float(np.nanmax(lon_grid[mask])),
    }


def _expand_bounds(
    bounds: dict[str, float],
    lat_margin_deg: float = 0.005,
    lon_margin_deg: float = 0.005,
) -> dict[str, float]:
    return {
        "S": max(-90.0, float(bounds["S"]) - float(lat_margin_deg)),
        "N": min(90.0, float(bounds["N"]) + float(lat_margin_deg)),
        "W": max(-180.0, float(bounds["W"]) - float(lon_margin_deg)),
        "E": min(180.0, float(bounds["E"]) + float(lon_margin_deg)),
    }


def _read_any_gacos_bounds(gacos_dir: Path) -> dict[str, float] | None:
    rsc_files = sorted(gacos_dir.glob("*.ztd.rsc"))
    if not rsc_files:
        return None
    info = _read_gacos_rsc(rsc_files[0])
    width = int(info["WIDTH"])
    length = int(info["FILE_LENGTH"])
    x0 = float(info["X_FIRST"])
    y0 = float(info["Y_FIRST"])
    dx = float(info["X_STEP"])
    dy = float(info["Y_STEP"])
    lon_1 = x0
    lon_2 = x0 + (width - 1) * dx
    lat_1 = y0
    lat_2 = y0 + (length - 1) * dy
    return {
        "S": min(lat_1, lat_2),
        "N": max(lat_1, lat_2),
        "W": min(lon_1, lon_2),
        "E": max(lon_1, lon_2),
    }


def _coverage_gap_degrees(
    required_bounds: dict[str, float],
    current_bounds: dict[str, float] | None,
) -> dict[str, float] | None:
    if not current_bounds:
        return None
    return {
        "south_gap_deg": max(0.0, float(current_bounds["S"]) - float(required_bounds["S"])),
        "north_gap_deg": max(0.0, float(required_bounds["N"]) - float(current_bounds["N"])),
        "west_gap_deg": max(0.0, float(current_bounds["W"]) - float(required_bounds["W"])),
        "east_gap_deg": max(0.0, float(required_bounds["E"]) - float(current_bounds["E"])),
    }


def _format_bounds_text(bounds: dict[str, float] | None) -> str:
    if not bounds:
        return "N/A"
    return (
        f"N={float(bounds['N']):.4f}, S={float(bounds['S']):.4f}, "
        f"W={float(bounds['W']):.4f}, E={float(bounds['E']):.4f}"
    )


def _format_center_line_utc(geom_source_dir: Path) -> str | None:
    try:
        meta = _extract_s1_metadata(geom_source_dir)
        center_line_utc = meta.get("CENTER_LINE_UTC")
        if center_line_utc is None:
            return None
        total_seconds = int(round(float(center_line_utc)))
        total_seconds %= 24 * 3600
        hh = total_seconds // 3600
        mm = (total_seconds % 3600) // 60
        ss = total_seconds % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    except Exception:
        return None


def _print_gacos_download_guidance(
    scene_dates: Iterable[str],
    recommended_bounds: dict[str, float] | None,
    required_bounds: dict[str, float] | None = None,
    utc_time_hms: str | None = None,
    current_bounds: dict[str, float] | None = None,
    missing_dates: Iterable[str] | None = None,
    gacos_dir: Path | None = None,
    title: str = "GACOS 下载建议",
) -> None:
    print(f"\n[INFO] {title}")
    if current_bounds:
        print(f"  当前 GACOS 覆盖: {_format_bounds_text(current_bounds)}")
    if required_bounds:
        print(f"  最低必要范围: {_format_bounds_text(required_bounds)}")
    if recommended_bounds:
        print(f"  稳妥推荐范围: {_format_bounds_text(recommended_bounds)}")
    dates = list(scene_dates)
    if utc_time_hms:
        print(f"  Sentinel-1 获取 UTC 时间: {utc_time_hms}")
    print(f"  目标日期数: {len(dates)}")
    print("  目标日期列表: 见 notebook 单独竖排输出，或查看下载说明文件")
    if missing_dates:
        missing = list(missing_dates)
        print(f"  缺失日期数: {len(missing)}")
        print("  缺失日期: " + ", ".join(missing))
    if gacos_dir:
        print(f"  下载完成后请放入: {gacos_dir}")


def _summarize_scene_qc(
    pair_qc_csv: Path,
    report_dir: Path,
) -> dict[str, Any]:
    if not pair_qc_csv.exists():
        raise FileNotFoundError(f"pair_qc.csv 不存在: {pair_qc_csv}")

    with pair_qc_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    scene_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        risk = float(row["risk"])
        weight = float(row["pair_weight"])
        action = row["action"]
        for date_key in ("date1", "date2"):
            date = row[date_key]
            item = scene_map.setdefault(
                date,
                {
                    "date": date,
                    "pair_count": 0,
                    "mean_pair_risk": 0.0,
                    "max_pair_risk": 0.0,
                    "mean_pair_weight": 0.0,
                    "keep_pairs": 0,
                    "downweight_pairs": 0,
                    "drop_pairs": 0,
                },
            )
            item["pair_count"] += 1
            item["mean_pair_risk"] += risk
            item["mean_pair_weight"] += weight
            item["max_pair_risk"] = max(float(item["max_pair_risk"]), risk)
            if action == "keep":
                item["keep_pairs"] += 1
            elif action == "downweight":
                item["downweight_pairs"] += 1
            elif action == "drop":
                item["drop_pairs"] += 1

    scene_rows = []
    for date in sorted(scene_map):
        item = scene_map[date]
        n = max(int(item["pair_count"]), 1)
        item["mean_pair_risk"] = float(item["mean_pair_risk"] / n)
        item["mean_pair_weight"] = float(item["mean_pair_weight"] / n)
        scene_rows.append(item)

    csv_path = report_dir / "scene_qc.csv"
    fieldnames = [
        "date",
        "pair_count",
        "mean_pair_risk",
        "max_pair_risk",
        "mean_pair_weight",
        "keep_pairs",
        "downweight_pairs",
        "drop_pairs",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scene_rows:
            writer.writerow(row)

    summary = {
        "scene_qc_csv": str(csv_path),
        "n_scenes": len(scene_rows),
        "scene_rows": scene_rows,
    }
    summary_path = _write_json(report_dir / "scene_qc_summary.json", summary)
    summary["scene_qc_summary_json"] = str(summary_path)
    return summary


def _write_gacos_download_guide(
    output_path: Path,
    scene_dates: Iterable[str],
    recommended_bounds: dict[str, float] | None,
    required_bounds: dict[str, float] | None = None,
    utc_time_hms: str | None = None,
    current_bounds: dict[str, float] | None = None,
    missing_dates: Iterable[str] | None = None,
    gacos_dir: Path | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "GACOS 下载建议",
        "=" * 60,
        "",
    ]
    if current_bounds:
        lines.extend([
            "当前 GACOS 覆盖:",
            f"  {_format_bounds_text(current_bounds)}",
            "",
        ])
    if utc_time_hms:
        lines.extend([
            f"Sentinel-1 获取 UTC 时间: {utc_time_hms}",
            "",
        ])
    if required_bounds:
        lines.extend([
            "最低必要范围 (required_download_bounds):",
            f"  North: {float(required_bounds['N']):.4f}",
            f"  South: {float(required_bounds['S']):.4f}",
            f"  West:  {float(required_bounds['W']):.4f}",
            f"  East:  {float(required_bounds['E']):.4f}",
            "",
        ])
    if recommended_bounds:
        lines.extend([
            "稳妥推荐范围 (recommended_download_bounds):",
            f"  North: {float(recommended_bounds['N']):.4f}",
            f"  South: {float(recommended_bounds['S']):.4f}",
            f"  West:  {float(recommended_bounds['W']):.4f}",
            f"  East:  {float(recommended_bounds['E']):.4f}",
            "",
        ])
    dates = list(scene_dates or [])
    lines.append(f"目标日期列表 ({len(dates)} 个):")
    lines.append("-" * 20)
    lines.extend(dates)
    lines.append("-" * 20)
    if missing_dates:
        missing = list(missing_dates)
        lines.extend([
            "",
            f"缺失日期 ({len(missing)} 个):",
            ", ".join(missing),
        ])
    if gacos_dir:
        lines.extend([
            "",
            f"下载完成后请放入: {gacos_dir}",
        ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _read_subset_from_vrt(vrt_path: Path, row_off: int, col_off: int, rows: int, cols: int) -> np.ndarray:
    import rasterio

    with rasterio.open(vrt_path) as src:
        row_off = max(0, int(row_off))
        col_off = max(0, int(col_off))
        rows = min(int(rows), src.height - row_off)
        cols = min(int(cols), src.width - col_off)
        window = ((row_off, row_off + rows), (col_off, col_off + cols))
        return src.read(1, window=window)


def _downsample_current_crop(
    geom_source_dir: Path,
    output_shape: tuple[int, int],
    crop_offset: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    row_off, col_off = crop_offset
    height, width = output_shape
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
    lat = lat_full[::DOLPHIN_STRIDE_Y, ::DOLPHIN_STRIDE_X][:height, :width]
    lon = lon_full[::DOLPHIN_STRIDE_Y, ::DOLPHIN_STRIDE_X][:height, :width]
    return lat.astype(np.float32), lon.astype(np.float32)


def _get_current_dolphin_shape(dolphin_dir: Path) -> tuple[int, int]:
    import rasterio

    files = sorted((dolphin_dir / "unwrapped").glob("*.unw.tif"))
    if not files:
        raise FileNotFoundError(f"未找到 Dolphin 解缠结果: {dolphin_dir / 'unwrapped'}")
    with rasterio.open(files[0]) as src:
        return src.height, src.width


def _compute_candidate_windows(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    aoi_bbox: Iterable[float],
    padding_candidates: Iterable[int],
) -> list[dict[str, Any]]:
    S, N, W, E = list(aoi_bbox)
    valid_mask = (
        np.isfinite(lat_grid)
        & np.isfinite(lon_grid)
        & (lat_grid >= S)
        & (lat_grid <= N)
        & (lon_grid >= W)
        & (lon_grid <= E)
        & (lat_grid > 0.1)
    )
    if not valid_mask.any():
        raise RuntimeError("无法在 Dolphin 当前网格中定位 AOI，无法生成 padding 候选窗口")

    rows, cols = np.where(valid_mask)
    y_min = int(rows.min())
    y_max = int(rows.max())
    x_min = int(cols.min())
    x_max = int(cols.max())
    height, width = lat_grid.shape

    results: list[dict[str, Any]] = []
    for pad in padding_candidates:
        pad_y = int(math.ceil(float(pad) / float(DOLPHIN_STRIDE_Y)))
        pad_x = int(math.ceil(float(pad) / float(DOLPHIN_STRIDE_X)))
        y0 = max(0, y_min - pad_y)
        y1 = min(height, y_max + pad_y + 1)
        x0 = max(0, x_min - pad_x)
        x1 = min(width, x_max + pad_x + 1)
        lat_sub = lat_grid[y0:y1, x0:x1]
        lon_sub = lon_grid[y0:y1, x0:x1]
        valid_sub = np.isfinite(lat_sub) & np.isfinite(lon_sub) & (lat_sub > 0.1)
        required_bounds = _bounds_from_grid(lat_sub, lon_sub, valid_sub)
        results.append(
            {
                "padding_pixels": int(pad),
                "padding_dolphin_pixels": {"y": pad_y, "x": pad_x},
                "subset_window": [int(y0), int(y1), int(x0), int(x1)],
                "shape": [int(y1 - y0), int(x1 - x0)],
                "required_download_bounds": required_bounds,
                "recommended_download_bounds": _expand_bounds(required_bounds),
            }
        )
    return results


def _interpolate_gacos_stats(
    ztd_path: Path,
    rsc_path: Path,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> dict[str, float]:
    lat_axis, lon_axis, data = _load_gacos_grid(ztd_path, rsc_path)
    interp = RegularGridInterpolator(
        (lat_axis, lon_axis),
        data,
        bounds_error=False,
        fill_value=np.nan,
    )
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    interp_values = interp(points).reshape(lat_grid.shape)
    finite = np.isfinite(interp_values)
    valid_ratio = float(np.mean(finite))

    zero_ratio = 1.0
    extreme_ratio = 1.0
    if finite.any():
        finite_values = interp_values[finite]
        zero_ratio = float(np.mean(np.isclose(finite_values, 0.0)))
        extreme_ratio = _compute_extreme_ratio(finite_values)
    return {
        "valid_coverage_ratio": valid_ratio,
        "zero_value_ratio": zero_ratio,
        "extreme_value_ratio": extreme_ratio,
    }


def _gacos_files_complete(gacos_dir: Path, scene_dates: Iterable[str]) -> tuple[bool, list[str]]:
    missing = []
    for date in scene_dates:
        if not (gacos_dir / f"{date}.ztd").exists() or not (gacos_dir / f"{date}.ztd.rsc").exists():
            missing.append(date)
    return len(missing) == 0, missing


def _validate_gacos_files(gacos_dir: Path, scene_dates: Iterable[str]) -> dict[str, Any]:
    converted_count = int(convert_gacos_tif_to_ztd(gacos_dir))
    scene_dates = list(scene_dates)
    per_date: dict[str, Any] = {}
    missing_dates: list[str] = []
    invalid_dates: list[str] = []
    valid_dates: list[str] = []

    for date in scene_dates:
        ztd_path = gacos_dir / f"{date}.ztd"
        rsc_path = gacos_dir / f"{date}.ztd.rsc"
        entry = {
            "has_ztd": ztd_path.exists(),
            "has_rsc": rsc_path.exists(),
            "readable": False,
            "shape": None,
            "bounds": None,
            "error": None,
        }
        if not entry["has_ztd"] or not entry["has_rsc"]:
            missing_dates.append(date)
            entry["error"] = "missing_ztd_or_rsc"
            per_date[date] = entry
            continue
        try:
            lat_axis, lon_axis, data = _load_gacos_grid(ztd_path, rsc_path)
            entry["readable"] = True
            entry["shape"] = [int(data.shape[0]), int(data.shape[1])]
            entry["bounds"] = {
                "S": float(min(lat_axis[0], lat_axis[-1])),
                "N": float(max(lat_axis[0], lat_axis[-1])),
                "W": float(min(lon_axis[0], lon_axis[-1])),
                "E": float(max(lon_axis[0], lon_axis[-1])),
            }
            valid_dates.append(date)
        except Exception as exc:
            invalid_dates.append(date)
            entry["error"] = str(exc)
        per_date[date] = entry

    usable = (len(missing_dates) == 0) and (len(invalid_dates) == 0)
    return {
        "gacos_dir": str(gacos_dir),
        "converted_tif_count": converted_count,
        "n_dates": len(scene_dates),
        "date_complete": len(missing_dates) == 0,
        "usable": usable,
        "missing_dates": missing_dates,
        "invalid_dates": invalid_dates,
        "valid_dates": valid_dates,
        "current_gacos_bounds": _read_any_gacos_bounds(gacos_dir),
        "per_date": per_date,
    }


def _prompt_wait_for_gacos(
    *,
    issue: str,
    guide_path: Path,
    validation_report_path: Path,
    gacos_dir: Path,
    allow_era5: bool,
) -> str:
    print("\nGACOS 需要人工下载/补齐后才能继续。")
    print(f"当前问题: {issue}")
    print(f"目标目录: {gacos_dir}")
    print(f"下载指引: {guide_path}")
    print(f"校验报告: {validation_report_path}")
    print("\n可选操作：")
    print("  [回车] 下载完成并放入目录后，立即重检")
    print("  [h] 回退到 height_correlation")
    if allow_era5:
        print("  [e] 回退到 ERA5")
    else:
        print("  [e] 回退到 ERA5（当前未检测到本地 ERA5，后续可能仍需准备）")
    print("  [a] 终止")
    choice = input("请选择 [回车/h/e/a] (默认回车重检): ").strip().lower()
    if choice == "h":
        return "height_correlation"
    if choice == "e":
        return "era5"
    if choice == "a":
        return "abort"
    return "retry"


def _choose_interactive_gacos_action(candidate_results: list[dict[str, Any]]) -> str:
    print("\nGACOS 覆盖不足。可选处理：")
    print("  [1] 缩 padding（优先，默认）")
    print("      影响：MintPy 输入范围收缩到更靠近严格 AOI，边缘受影响。")
    print("  [2] 回退 ERA5")
    print("      影响：继续保留当前空间范围，但需已有 ERA5 数据。")
    print("  [3] 终止")
    print("      影响：不继续进入 MintPy。")
    print("\n各 padding 候选覆盖率：")
    for item in candidate_results:
        print(
            f"  padding={item['padding_pixels']:>2d} px  "
            f"coverage(min)={item.get('min_valid_coverage_ratio', float('nan')):.3f}  "
            f"window={item['subset_window']}"
        )
    preferred = candidate_results[0] if candidate_results else {}
    if preferred.get("recommended_download_bounds"):
        print("\n如果要继续使用 GACOS，建议重新下载范围:")
        print(f"  {_format_bounds_text(preferred.get('recommended_download_bounds'))}")
    choice = input("请选择 [1/2/3] (默认 1): ").strip()
    if choice == "2":
        return "era5"
    if choice == "3":
        return "abort"
    return "padding"


def _choose_interactive_low_coverage_fallback(
    era5_available: bool,
    recommended_bounds: dict[str, float] | None = None,
) -> str:
    print("\n缩 padding 在 20/10/0 下仍无法让 GACOS 覆盖达到要求。")
    if recommended_bounds:
        print("如需继续使用 GACOS，建议重新下载以下范围:")
        print(f"  {_format_bounds_text(recommended_bounds)}")
    print("可选回退：")
    print("  [1] 改用 height_correlation（默认）")
    print("      影响：不依赖外部天气数据，优先保证流程可继续。")
    if era5_available:
        print("  [2] 回退 ERA5")
        print("      影响：改用再分析气象场进行对流层校正。")
    else:
        print("  [2] 回退 ERA5（当前目录下未检测到现成 ERA5 数据，后续可能仍需准备）")
    print("  [3] 终止")
    choice = input("请选择 [1/2/3] (默认 1): ").strip()
    if choice == "2":
        return "era5"
    if choice == "3":
        return "abort"
    return "height_correlation"


def _pick_selected_candidate(
    candidate_results: list[dict[str, Any]],
    interactive: bool,
    min_warn: float,
    min_ok: float,
    fallback_mode: str = "prompt",
    era5_available: bool = False,
) -> tuple[dict[str, Any], str, str]:
    current = candidate_results[0]
    current_cov = float(current.get("min_valid_coverage_ratio", 0.0))
    if current_cov >= min_ok:
        return current, "gacos", "ok"
    if current_cov >= min_warn:
        return current, "gacos", "warn_continue"

    if interactive:
        action = _choose_interactive_gacos_action(candidate_results)
    else:
        action = "padding"

    if action == "abort":
        raise RuntimeError("GACOS 覆盖不足，用户选择终止 MintPy。")

    if action == "era5":
        return current, "era5", "fallback_era5"

    for item in candidate_results:
        if float(item.get("min_valid_coverage_ratio", 0.0)) >= min_warn:
            status = "auto_shrink_padding"
            if interactive:
                status = "interactive_shrink_padding"
            return item, "gacos", status

    fallback_mode = (fallback_mode or "prompt").strip().lower()
    if fallback_mode in {"height_correlation", "phase_dem", "dem"}:
        return current, "height_correlation", "fallback_height_correlation"
    if fallback_mode == "era5":
        return current, "era5", "fallback_era5"
    if fallback_mode == "abort":
        raise RuntimeError("GACOS 覆盖在 padding=20/10/0 下均 < 95%，按配置终止。")

    if interactive:
        recommended_bounds = None
        if candidate_results:
            recommended_bounds = candidate_results[0].get("recommended_download_bounds")
        secondary = _choose_interactive_low_coverage_fallback(
            era5_available=era5_available,
            recommended_bounds=recommended_bounds,
        )
        if secondary == "abort":
            raise RuntimeError("GACOS 覆盖在 padding=20/10/0 下均 < 95%，用户选择终止。")
        if secondary == "era5":
            return current, "era5", "interactive_fallback_era5"
        return current, "height_correlation", "interactive_fallback_height_correlation"

    raise RuntimeError("GACOS 覆盖在 padding=20/10/0 下均 < 95%，默认停止。")


def _load_pair_window_metric_arrays(
    dolphin_dir: Path,
    pair_name: str,
    subset_window: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import rasterio

    y0, y1, x0, x1 = subset_window or (0, None, 0, None)
    y0 = int(y0)
    x0 = int(x0)

    def _read(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            yy1 = src.height if y1 is None else int(min(y1, src.height))
            xx1 = src.width if x1 is None else int(min(x1, src.width))
            return src.read(1, window=((y0, yy1), (x0, xx1)))

    unw = _read(dolphin_dir / "unwrapped" / f"{pair_name}.unw.tif").astype(np.float32)
    coh = _read(dolphin_dir / "interferograms" / f"{pair_name}.int.cor.tif").astype(np.float32)
    cc = _read(dolphin_dir / "unwrapped" / f"{pair_name}.unw.conncomp.tif")
    return unw, coh, cc


def _phase_to_los_mm(phase_radian: np.ndarray, wavelength: float) -> np.ndarray:
    return phase_radian * (wavelength / (4.0 * np.pi)) * 1000.0


def compute_pair_qc(
    dolphin_dir: Path | None = None,
    geom_source_dir: Path | None = None,
    report_dir: Path | None = None,
    subset_window: Iterable[int] | None = None,
    wavelength: float | None = None,
) -> dict[str, Any]:
    dolphin_dir = Path(dolphin_dir or cfg.DOLPHIN_DIR)
    geom_source_dir = Path(geom_source_dir or (cfg.ISCE_WORK_DIR / "merged" / "geom_reference"))
    report_dir = _ensure_dir(report_dir)

    pairs = _date_pairs_from_dolphin(dolphin_dir)
    if not pairs:
        raise FileNotFoundError(f"未找到 Dolphin pair 输出: {dolphin_dir}")

    pair_names = [f"{d1}_{d2}" for d1, d2 in pairs]
    subset = tuple(int(v) for v in subset_window) if subset_window is not None else None
    wave = float(wavelength or _extract_s1_metadata(geom_source_dir).get("WAVELENGTH", "0.05546576"))

    rows: list[dict[str, Any]] = []
    coh_values = []
    valid_values = []
    ncc_values = []
    p95_values = []
    frac50_values = []

    for d1, d2 in pairs:
        pair_name = f"{d1}_{d2}"
        unw, coh, cc = _load_pair_window_metric_arrays(dolphin_dir, pair_name, subset)
        valid_cc = (cc > 0) & (cc < 65535)
        valid_mask = valid_cc & np.isfinite(unw)

        coh_finite = coh[np.isfinite(coh)]
        coh_nonzero = coh_finite[coh_finite > 0]
        coh_mean = float(np.nanmean(coh_nonzero)) if coh_nonzero.size else 0.0

        valid_conncomp_frac = float(np.mean(valid_cc))
        unique_conn = np.unique(cc[valid_cc]) if valid_cc.any() else np.array([], dtype=np.int64)
        n_conncomp = int(unique_conn.size)

        closure_risk_proxy = 1.0
        if valid_cc.any():
            vals, cnt = np.unique(cc[valid_cc], return_counts=True)
            main_fraction = float(cnt.max() / cnt.sum()) if cnt.sum() else 0.0
            closure_risk_proxy = float(1.0 - main_fraction)

        pair_p95_abs_los_mm = 0.0
        pair_frac_abs_los_gt_50mm = 0.0
        if valid_mask.any():
            los_mm = _phase_to_los_mm(unw[valid_mask], wave)
            abs_los = np.abs(los_mm)
            pair_p95_abs_los_mm = float(np.nanpercentile(abs_los, 95))
            pair_frac_abs_los_gt_50mm = float(np.mean(abs_los > 50.0))

        row = {
            "date1": d1,
            "date2": d2,
            "coh_mean": coh_mean,
            "valid_conncomp_frac": valid_conncomp_frac,
            "n_conncomp": n_conncomp,
            "closure_risk_proxy": closure_risk_proxy,
            "pair_p95_abs_los_mm": pair_p95_abs_los_mm,
            "pair_frac_abs_los_gt_50mm": pair_frac_abs_los_gt_50mm,
        }
        rows.append(row)
        coh_values.append(coh_mean)
        valid_values.append(valid_conncomp_frac)
        ncc_values.append(n_conncomp)
        p95_values.append(pair_p95_abs_los_mm)
        frac50_values.append(pair_frac_abs_los_gt_50mm)

    coh_norm = np.clip(np.asarray(coh_values, dtype=np.float64), 0.0, 1.0)
    ncc_norm = _normalize(np.asarray(ncc_values, dtype=np.float64))
    p95_norm = _normalize(np.asarray(p95_values, dtype=np.float64))
    frac50_norm = np.clip(np.asarray(frac50_values, dtype=np.float64), 0.0, 1.0)

    for i, row in enumerate(rows):
        risk = (
            0.25 * (1.0 - coh_norm[i])
            + 0.20 * (1.0 - float(row["valid_conncomp_frac"]))
            + 0.15 * ncc_norm[i]
            + 0.20 * p95_norm[i]
            + 0.20 * frac50_norm[i]
        )
        action = "downweight"
        if (
            row["coh_mean"] >= 0.55
            and row["valid_conncomp_frac"] >= 0.85
            and row["pair_frac_abs_los_gt_50mm"] <= 0.05
        ):
            action = "keep"
        if (
            row["coh_mean"] < 0.45
            or row["valid_conncomp_frac"] < 0.65
            or row["pair_frac_abs_los_gt_50mm"] > 0.15
        ):
            action = "drop"

        row["risk"] = float(risk)
        row["action"] = action
        row["pair_weight"] = float(PAIR_WEIGHT[action])

    csv_path = report_dir / "pair_qc.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_QC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "report_dir": str(report_dir),
        "pair_qc_csv": str(csv_path),
        "n_pairs": len(rows),
        "subset_window": list(subset) if subset else None,
        "wavelength_m": wave,
        "action_counts": {
            key: int(sum(1 for row in rows if row["action"] == key))
            for key in ["keep", "downweight", "drop"]
        },
        "risk_mean": float(np.mean([row["risk"] for row in rows])) if rows else None,
        "risk_p90": float(np.percentile([row["risk"] for row in rows], 90)) if rows else None,
        "worst_pairs": sorted(
            [
                {
                    "date1": row["date1"],
                    "date2": row["date2"],
                    "risk": row["risk"],
                    "action": row["action"],
                }
                for row in rows
            ],
            key=lambda item: item["risk"],
            reverse=True,
        )[:10],
    }
    summary_path = _write_json(report_dir / "pair_qc_summary.json", summary)
    summary["pair_qc_summary_json"] = str(summary_path)
    return summary


def _maybe_compute_gacos_coverage(
    atmo_config: dict[str, Any] | None,
    gacos_dir: Path,
    scene_dates: list[str],
    candidate_results: list[dict[str, Any]],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    validation_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strict_aoi_bounds = {
        "S": float(cfg._AOI_BBOX[0]),
        "N": float(cfg._AOI_BBOX[1]),
        "W": float(cfg._AOI_BBOX[2]),
        "E": float(cfg._AOI_BBOX[3]),
    } if cfg._AOI_BBOX is not None else None
    report: dict[str, Any] = {
        "method": "skipped_not_gacos",
        "date_complete": None,
        "missing_dates": [],
        "invalid_dates": [],
        "gacos_file_validation_ok": None,
        "candidate_results": candidate_results,
        "scene_dates": scene_dates,
        "strict_aoi_bounds": strict_aoi_bounds,
        "recommended_download_bounds_from_aoi": (
            compute_recommended_gacos_bounds(
                {
                    "N": strict_aoi_bounds["N"],
                    "S": strict_aoi_bounds["S"],
                    "W": strict_aoi_bounds["W"],
                    "E": strict_aoi_bounds["E"],
                }
            )
            if strict_aoi_bounds else None
        ),
    }
    if not atmo_config or atmo_config.get("method") != "gacos":
        return report

    report["method"] = "gacos"
    if validation_report:
        report["date_complete"] = bool(validation_report.get("date_complete"))
        report["missing_dates"] = list(validation_report.get("missing_dates", []))
        report["invalid_dates"] = list(validation_report.get("invalid_dates", []))
        report["gacos_file_validation_ok"] = bool(validation_report.get("usable"))
        report["current_gacos_bounds"] = validation_report.get("current_gacos_bounds")
    else:
        complete, missing = _gacos_files_complete(gacos_dir, scene_dates)
        report["date_complete"] = bool(complete)
        report["missing_dates"] = missing
        report["current_gacos_bounds"] = _read_any_gacos_bounds(gacos_dir)
        report["gacos_file_validation_ok"] = bool(complete)
    if candidate_results:
        default_candidate = candidate_results[0]
        report["default_candidate_required_download_bounds"] = default_candidate.get("required_download_bounds")
        report["default_candidate_recommended_download_bounds"] = default_candidate.get("recommended_download_bounds")
    if not report.get("date_complete", False):
        return report
    if validation_report and not validation_report.get("usable", False):
        return report

    for item in candidate_results:
        y0, y1, x0, x1 = item["subset_window"]
        lat_sub = lat_grid[y0:y1, x0:x1]
        lon_sub = lon_grid[y0:y1, x0:x1]
        per_date = {}
        min_valid = 1.0
        mean_valid = []
        for date in scene_dates:
            stats = _interpolate_gacos_stats(
                gacos_dir / f"{date}.ztd",
                gacos_dir / f"{date}.ztd.rsc",
                lat_sub,
                lon_sub,
            )
            per_date[date] = stats
            min_valid = min(min_valid, float(stats["valid_coverage_ratio"]))
            mean_valid.append(float(stats["valid_coverage_ratio"]))
        item["per_date"] = per_date
        item["min_valid_coverage_ratio"] = float(min_valid)
        item["mean_valid_coverage_ratio"] = float(np.mean(mean_valid)) if mean_valid else 0.0
        item["coverage_gap_degrees"] = _coverage_gap_degrees(
            item.get("recommended_download_bounds"),
            report.get("current_gacos_bounds"),
        )

    if candidate_results:
        default_candidate = candidate_results[0]
        best_candidate = max(
            candidate_results,
            key=lambda item: float(item.get("min_valid_coverage_ratio", 0.0)),
        )
        report["best_candidate_padding_pixels"] = int(best_candidate["padding_pixels"])
        report["best_candidate_recommended_download_bounds"] = best_candidate.get("recommended_download_bounds")
    return report


def run_pre_mintpy_qc(
    dolphin_dir: Path | None = None,
    geom_source_dir: Path | None = None,
    mintpy_dir: Path | None = None,
    gacos_dir: Path | None = None,
    atmo_config: dict[str, Any] | None = None,
    report_dir: Path | None = None,
    interactive: bool = True,
    gacos_min_coverage_warn: float = DEFAULT_GACOS_WARN,
    gacos_min_coverage_ok: float = DEFAULT_GACOS_OK,
    padding_candidates: Iterable[int] = DEFAULT_PADDING_CANDIDATES,
    low_coverage_fallback: str = "prompt",
    manual_wait: bool = DEFAULT_GACOS_MANUAL_WAIT,
    depsi_like_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dolphin_dir = Path(dolphin_dir or cfg.DOLPHIN_DIR)
    geom_source_dir = Path(geom_source_dir or (cfg.ISCE_WORK_DIR / "merged" / "geom_reference"))
    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    gacos_dir = Path(gacos_dir or cfg.GACOS_DIR)
    report_dir = _ensure_dir(report_dir)

    if cfg._AOI_BBOX is None or cfg._AOI_CROP_OFFSET is None:
        raise RuntimeError("缺少 AOI_BBOX 或 AOI_CROP_OFFSET，无法进行主链 QC")

    shape = _get_current_dolphin_shape(dolphin_dir)
    lat_grid, lon_grid = _downsample_current_crop(
        geom_source_dir=geom_source_dir,
        output_shape=shape,
        crop_offset=tuple(cfg._AOI_CROP_OFFSET),
    )
    candidate_results = _compute_candidate_windows(
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        aoi_bbox=cfg._AOI_BBOX,
        padding_candidates=padding_candidates,
    )

    scene_dates = _scene_dates_from_pairs(_date_pairs_from_dolphin(dolphin_dir))
    acquisition_utc_hms = _format_center_line_utc(geom_source_dir)
    aoi_bounds = {
        "N": float(cfg._AOI_BBOX[1]),
        "S": float(cfg._AOI_BBOX[0]),
        "W": float(cfg._AOI_BBOX[2]),
        "E": float(cfg._AOI_BBOX[3]),
    }
    atmo_prepare_summary = prepare_external_atmo_inputs(
        atmo_config=atmo_config,
        scenes_or_dates=scene_dates,
        bounds=aoi_bounds,
    )
    _write_json(report_dir / "atmo_prepare_summary.json", atmo_prepare_summary)

    selected_candidate = candidate_results[0]
    effective_atmo_method = atmo_config.get("method") if atmo_config else None
    coverage_status = "not_applicable"
    validation_report_path = report_dir / "gacos_file_validation_report.json"
    guide_path = report_dir / "gacos_download_guide.txt"
    while True:
        if atmo_config and atmo_config.get("method") == "gacos":
            gacos_validation = _validate_gacos_files(gacos_dir, scene_dates)
        else:
            gacos_validation = {
                "gacos_dir": str(gacos_dir),
                "skipped": True,
                "reason": "method_not_gacos",
                "date_complete": None,
                "usable": None,
                "missing_dates": [],
                "invalid_dates": [],
                "valid_dates": [],
                "current_gacos_bounds": _read_any_gacos_bounds(gacos_dir),
                "per_date": {},
            }
        _write_json(validation_report_path, gacos_validation)
        gacos_report = _maybe_compute_gacos_coverage(
            atmo_config=atmo_config,
            gacos_dir=gacos_dir,
            scene_dates=scene_dates,
            candidate_results=candidate_results,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            validation_report=gacos_validation,
        )
        _write_json(report_dir / "gacos_coverage_report.json", gacos_report)

        if gacos_report.get("method") != "gacos":
            break

        if not gacos_validation.get("date_complete", False) or not gacos_validation.get("usable", False):
            guide_path = _write_gacos_download_guide(
                guide_path,
                scene_dates=gacos_report.get("scene_dates", scene_dates),
                required_bounds=gacos_report.get("default_candidate_required_download_bounds"),
                recommended_bounds=gacos_report.get("default_candidate_recommended_download_bounds"),
                utc_time_hms=acquisition_utc_hms,
                current_bounds=gacos_validation.get("current_gacos_bounds"),
                missing_dates=gacos_validation.get("missing_dates", []),
                gacos_dir=gacos_dir,
            )
            issue_parts = []
            if gacos_validation.get("missing_dates"):
                issue_parts.append(f"缺失日期 {len(gacos_validation.get('missing_dates', []))} 个")
            if gacos_validation.get("invalid_dates"):
                issue_parts.append(f"无效文件 {len(gacos_validation.get('invalid_dates', []))} 个")
            issue_text = "，".join(issue_parts) or "GACOS 文件不符合要求"
            _print_gacos_download_guidance(
                scene_dates=gacos_report.get("scene_dates", scene_dates),
                required_bounds=gacos_report.get("default_candidate_required_download_bounds"),
                recommended_bounds=gacos_report.get("default_candidate_recommended_download_bounds"),
                utc_time_hms=acquisition_utc_hms,
                current_bounds=gacos_validation.get("current_gacos_bounds"),
                missing_dates=gacos_validation.get("missing_dates", []),
                gacos_dir=gacos_dir,
                title="GACOS 文件未就绪或校验未通过",
            )
            print("  说明: 只有当 .ztd/.ztd.rsc 日期齐全、RSC 字段完整、尺寸与二进制内容匹配后，才会继续做覆盖检查。")
            print(f"  校验报告: {validation_report_path}")
            print(f"  下载说明文件: {guide_path}")
            if interactive and manual_wait:
                era5_available = bool(list(cfg.ERA5_DIR.glob('ERA5_*.grb')))
                action = _prompt_wait_for_gacos(
                    issue=issue_text,
                    guide_path=guide_path,
                    validation_report_path=validation_report_path,
                    gacos_dir=gacos_dir,
                    allow_era5=era5_available,
                )
                if action == "retry":
                    continue
                if action == "abort":
                    raise RuntimeError("用户在 GACOS 人工下载等待点选择终止。")
                effective_atmo_method = action
                coverage_status = f"manual_wait_fallback_{action}"
                break
            raise RuntimeError(
                f"GACOS 文件未就绪或校验未通过。请检查 {validation_report_path} 与 {guide_path}"
            )

        max_candidate_cov = max(
            float(item.get("min_valid_coverage_ratio", 0.0)) for item in candidate_results
        ) if candidate_results else 0.0
        if max_candidate_cov < float(gacos_min_coverage_warn) and interactive and manual_wait:
            guide_path = _write_gacos_download_guide(
                guide_path,
                scene_dates=gacos_report.get("scene_dates", scene_dates),
                required_bounds=gacos_report.get("default_candidate_required_download_bounds"),
                recommended_bounds=gacos_report.get("best_candidate_recommended_download_bounds")
                or gacos_report.get("default_candidate_recommended_download_bounds"),
                utc_time_hms=acquisition_utc_hms,
                current_bounds=gacos_report.get("current_gacos_bounds"),
                gacos_dir=gacos_dir,
            )
            best_cov = max_candidate_cov
            _print_gacos_download_guidance(
                scene_dates=gacos_report.get("scene_dates", scene_dates),
                required_bounds=gacos_report.get("default_candidate_required_download_bounds"),
                recommended_bounds=gacos_report.get("best_candidate_recommended_download_bounds")
                or gacos_report.get("default_candidate_recommended_download_bounds"),
                utc_time_hms=acquisition_utc_hms,
                current_bounds=gacos_report.get("current_gacos_bounds"),
                gacos_dir=gacos_dir,
                title="当前 GACOS 空间覆盖不足",
            )
            print(
                f"  说明: 当前所有 padding 候选下的最优覆盖率仅为 {best_cov:.3f}，"
                "建议按下载指引补齐更大范围的 GACOS 后再继续。"
            )
            print(f"  校验报告: {validation_report_path}")
            print(f"  下载说明文件: {guide_path}")
            era5_available = bool(list(cfg.ERA5_DIR.glob('ERA5_*.grb')))
            action = _prompt_wait_for_gacos(
                issue=f"空间覆盖不足（最优覆盖率 {best_cov:.3f} < {float(gacos_min_coverage_warn):.2f}）",
                guide_path=guide_path,
                validation_report_path=validation_report_path,
                gacos_dir=gacos_dir,
                allow_era5=era5_available,
            )
            if action == "retry":
                continue
            if action == "abort":
                raise RuntimeError("用户在 GACOS 空间覆盖等待点选择终止。")
            effective_atmo_method = action
            coverage_status = f"manual_wait_fallback_{action}"
            break

        era5_available = bool(list(cfg.ERA5_DIR.glob("ERA5_*.grb")))
        selected_candidate, effective_atmo_method, coverage_status = _pick_selected_candidate(
            candidate_results=candidate_results,
            interactive=interactive,
            min_warn=gacos_min_coverage_warn,
            min_ok=gacos_min_coverage_ok,
            fallback_mode=low_coverage_fallback,
            era5_available=era5_available,
        )
        if float(selected_candidate.get("min_valid_coverage_ratio", 0.0)) < float(gacos_min_coverage_warn):
            guide_path = _write_gacos_download_guide(
                guide_path,
                scene_dates=gacos_report.get("scene_dates", scene_dates),
                required_bounds=selected_candidate.get("required_download_bounds"),
                recommended_bounds=selected_candidate.get("recommended_download_bounds"),
                utc_time_hms=acquisition_utc_hms,
                current_bounds=gacos_report.get("current_gacos_bounds"),
                gacos_dir=gacos_dir,
            )
            _print_gacos_download_guidance(
                scene_dates=gacos_report.get("scene_dates", scene_dates),
                required_bounds=selected_candidate.get("required_download_bounds"),
                recommended_bounds=selected_candidate.get("recommended_download_bounds"),
                utc_time_hms=acquisition_utc_hms,
                current_bounds=gacos_report.get("current_gacos_bounds"),
                gacos_dir=gacos_dir,
                title="当前 GACOS 空间覆盖不足",
            )
            print("  说明: 该范围按当前 Dolphin 网格与 MintPy 默认 padding 候选反推，"
                  "比前面的严格 AOI 更适合作为 GACOS 提交范围。")
            print(f"  下载说明文件: {guide_path}")
        break

    effective_atmo_config = dict(atmo_config or {})
    if effective_atmo_method:
        effective_atmo_config["method"] = effective_atmo_method
        if effective_atmo_method == "era5" and not effective_atmo_config.get("model"):
            effective_atmo_config["model"] = "ERA5"
            effective_atmo_config["dir"] = str(cfg.ERA5_DIR)
        elif effective_atmo_method == "gacos":
            if not effective_atmo_config.get("dir"):
                effective_atmo_config["dir"] = str(gacos_dir)
            effective_atmo_config.pop("model", None)
        else:
            effective_atmo_config.pop("model", None)
            effective_atmo_config.pop("dir", None)

    mintpy_input_build = build_mintpy_hdf5(
        dolphin_dir=dolphin_dir,
        geom_source_dir=geom_source_dir,
        mintpy_dir=mintpy_dir,
        subset_window=selected_candidate["subset_window"],
        ifgram_filename="ifgramStack_original.h5",
        force_rebuild=False,
    )

    pair_qc_summary = compute_pair_qc(
        dolphin_dir=dolphin_dir,
        geom_source_dir=geom_source_dir,
        report_dir=report_dir,
        subset_window=selected_candidate["subset_window"],
    )
    depsi_like_summary = run_depsi_like_qc(
        dolphin_dir=dolphin_dir,
        mintpy_dir=mintpy_dir,
        geom_source_dir=geom_source_dir,
        report_dir=report_dir,
        pair_qc_csv=Path(pair_qc_summary["pair_qc_csv"]),
        pair_qc_summary_json=Path(pair_qc_summary["pair_qc_summary_json"]),
        gacos_coverage_report=report_dir / "gacos_coverage_report.json",
        atmo_method=effective_atmo_config.get("method"),
        config=depsi_like_config,
        crop_offset=tuple(int(v) for v in cfg._AOI_CROP_OFFSET),
    )
    scene_qc_summary = _summarize_scene_qc(
        pair_qc_csv=Path(pair_qc_summary["pair_qc_csv"]),
        report_dir=report_dir,
    )
    pair_qc_matrix_summary = plot_pair_qc_matrix(
        pair_qc_csv=Path(pair_qc_summary["pair_qc_csv"]),
        output_path=report_dir / "pair_qc_matrix.png",
    )

    pre_qc = {
        "report_dir": str(report_dir),
        "mintpy_dir": str(mintpy_dir),
        "dolphin_dir": str(dolphin_dir),
        "geom_source_dir": str(geom_source_dir),
        "mintpy_input_build": mintpy_input_build,
        "mintpy_original_ifgram_path": mintpy_input_build.get("ifgram_path"),
        "mintpy_geometry_path": mintpy_input_build.get("geometry_path"),
        "gacos_coverage_report": str(report_dir / "gacos_coverage_report.json"),
        "gacos_file_validation_report": str(validation_report_path),
        "atmo_prepare_summary": str(report_dir / "atmo_prepare_summary.json"),
        "pair_qc_csv": pair_qc_summary["pair_qc_csv"],
        "pair_qc_summary_json": pair_qc_summary["pair_qc_summary_json"],
        "scene_qc_csv": scene_qc_summary["scene_qc_csv"],
        "scene_qc_summary_json": scene_qc_summary["scene_qc_summary_json"],
        "pair_qc_matrix_png": pair_qc_matrix_summary["pair_qc_matrix_png"],
        "ps_score_tif": depsi_like_summary.get("ps_score_tif"),
        "mask_ps_strict_tif": depsi_like_summary.get("mask_ps_strict_tif"),
        "mask_ps_relaxed_tif": depsi_like_summary.get("mask_ps_relaxed_tif"),
        "ref_primary_network_tif": depsi_like_summary.get("ref_primary_network_tif"),
        "ref_primary_network_csv": depsi_like_summary.get("ref_primary_network_csv"),
        "ref_candidates_csv": depsi_like_summary.get("ref_candidates_csv"),
        "ps_score_summary_json": str(report_dir / "ps_score_summary.json"),
        "depsi_like_qc_summary": depsi_like_summary,
        "scene_dates": scene_dates,
        "selected_padding_pixels": int(selected_candidate["padding_pixels"]),
        "selected_subset_window": list(selected_candidate["subset_window"]),
        "selected_subset_shape": list(selected_candidate["shape"]),
        "coverage_status": coverage_status,
        "low_coverage_fallback": str(low_coverage_fallback),
        "effective_atmo_config": effective_atmo_config,
        "atmo_prepare": atmo_prepare_summary,
        "gacos_current_bounds": gacos_report.get("current_gacos_bounds"),
        "gacos_file_validation": gacos_validation,
        "gacos_redownload_required_bounds": selected_candidate.get("required_download_bounds"),
        "gacos_redownload_recommended_bounds": selected_candidate.get("recommended_download_bounds"),
        "gacos_download_guide": str(report_dir / "gacos_download_guide.txt"),
        "acquisition_utc_hms": acquisition_utc_hms,
        "candidate_results": candidate_results,
    }
    pre_qc_path = _write_json(report_dir / "pre_mintpy_qc.json", pre_qc)
    pre_qc["pre_mintpy_qc_json"] = str(pre_qc_path)

    save_project_progress(
        "pre_mintpy_qc_done",
        mintpy_padding_pixels=int(selected_candidate["padding_pixels"]),
        mintpy_subset_window=list(selected_candidate["subset_window"]),
        mintpy_effective_atmo=effective_atmo_config.get("method"),
    )
    return pre_qc


def _verify_drop_ifgram_semantics(stack_path: Path) -> dict[str, Any]:
    from mintpy.objects import ifgramStack

    obj = ifgramStack(str(stack_path))
    obj.open(print_msg=False)
    kept_count = len(obj.get_date12_list(dropIfgram=True))
    raw_count = len(obj.get_date12_list(dropIfgram=False))
    flag_sum = int(np.sum(obj.dropIfgram))
    semantics_ok = kept_count == flag_sum and raw_count >= kept_count
    return {
        "true_means_kept": bool(semantics_ok),
        "kept_count": kept_count,
        "all_count": raw_count,
        "flag_sum": flag_sum,
    }


def create_qc_ifgramstack(
    source_ifgramstack: Path | None = None,
    pair_qc_csv: Path | None = None,
    output_ifgramstack: Path | None = None,
    metadata_report: Path | None = None,
) -> dict[str, Any]:
    source_ifgramstack = Path(source_ifgramstack or (cfg.MINTPY_DIR / "inputs" / "ifgramStack_original.h5"))
    pair_qc_csv = Path(pair_qc_csv or (cfg.WORK_DIR / "mainchain_qc" / "pair_qc.csv"))
    output_ifgramstack = Path(output_ifgramstack or (cfg.MINTPY_DIR / "inputs" / "ifgramStack_qc.h5"))
    output_ifgramstack.parent.mkdir(parents=True, exist_ok=True)

    if not source_ifgramstack.exists():
        raise FileNotFoundError(f"原始 ifgramStack 不存在: {source_ifgramstack}")
    if not pair_qc_csv.exists():
        raise FileNotFoundError(f"pair_qc.csv 不存在: {pair_qc_csv}")

    semantics = _verify_drop_ifgram_semantics(source_ifgramstack)
    if not semantics["true_means_kept"]:
        raise RuntimeError(
            "当前 MintPy 环境下无法确认 dropIfgram=True 表示 kept，停止写入 QC 结果。"
        )

    rows = []
    with pair_qc_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    date12_to_action = {}
    date12_to_weight = {}
    for row in rows:
        date12 = f"{row['date1']}_{row['date2']}"
        date12_to_action[date12] = row["action"]
        date12_to_weight[date12] = float(row["pair_weight"])

    shutil.copy2(source_ifgramstack, output_ifgramstack)
    with h5py.File(output_ifgramstack, "r+") as f:
        date_pairs = [
            f"{d1.decode('utf-8')}_{d2.decode('utf-8')}"
            for d1, d2 in f["date"][:]
        ]

        keep_flags = np.ones(len(date_pairs), dtype=np.bool_)
        pair_weight = np.ones(len(date_pairs), dtype=np.float32)
        action_code = np.zeros(len(date_pairs), dtype=np.int16)
        coherence_scale = np.ones(len(date_pairs), dtype=np.float32)

        downweighted_pairs: list[str] = []
        dropped_pairs: list[str] = []
        for i, date12 in enumerate(date_pairs):
            action = date12_to_action.get(date12, "keep")
            weight = float(date12_to_weight.get(date12, PAIR_WEIGHT[action]))
            pair_weight[i] = weight
            action_code[i] = ACTION_CODE[action]
            if action == "drop":
                keep_flags[i] = False
                dropped_pairs.append(date12)
            elif action == "downweight":
                downweighted_pairs.append(date12)
                coherence_scale[i] = 0.5

        f["dropIfgram"][:] = keep_flags

        if "coherence" in f:
            coh = f["coherence"]
            for idx, factor in enumerate(coherence_scale):
                if not math.isclose(float(factor), 1.0):
                    coh[idx, :, :] = coh[idx, :, :] * factor

        for dset_name in ["qc_pair_weight", "qc_action_code", "qc_coherence_scale_factor"]:
            if dset_name in f:
                del f[dset_name]
        f.create_dataset("qc_pair_weight", data=pair_weight)
        f.create_dataset("qc_action_code", data=action_code)
        f.create_dataset("qc_coherence_scale_factor", data=coherence_scale)

        # The active reference point in the source stack may no longer belong to
        # the common connected component after pair dropping/downweighting.
        # Clear stale reference metadata so MintPy re-selects it in the
        # reference_point step using the QC-adjusted network.
        removed_ref_attrs = []
        for key in ["REF_Y", "REF_X", "REF_LAT", "REF_LON"]:
            if key in f.attrs:
                removed_ref_attrs.append(key)
                del f.attrs[key]

        f.attrs["qc_source_ifgramStack"] = str(source_ifgramstack)
        f.attrs["qc_original_coherence_preserved"] = "true"
        f.attrs["qc_drop_ifgram_true_means_kept"] = "true"
        f.attrs["qc_downweighted_pairs"] = json.dumps(downweighted_pairs, ensure_ascii=False)
        f.attrs["qc_drop_pairs"] = json.dumps(dropped_pairs, ensure_ascii=False)
        f.attrs["qc_removed_reference_attrs"] = json.dumps(removed_ref_attrs, ensure_ascii=False)

    report = {
        "source_ifgramStack": str(source_ifgramstack),
        "qc_ifgramStack": str(output_ifgramstack),
        "pair_qc_csv": str(pair_qc_csv),
        "dropIfgram_semantics": semantics,
        "n_downweighted": int(sum(1 for row in rows if row["action"] == "downweight")),
        "n_dropped": int(sum(1 for row in rows if row["action"] == "drop")),
        "n_kept": int(sum(1 for row in rows if row["action"] == "keep")),
        "qc_original_coherence_preserved": True,
        "qc_removed_reference_attrs": ["REF_Y", "REF_X", "REF_LAT", "REF_LON"],
    }
    if metadata_report:
        _write_json(Path(metadata_report), report)
    return report


def activate_qc_ifgramstack(
    mintpy_dir: Path | None = None,
    qc_ifgramstack: Path | None = None,
) -> dict[str, Any]:
    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    qc_ifgramstack = Path(qc_ifgramstack or (mintpy_dir / "inputs" / "ifgramStack_qc.h5"))
    active_path = mintpy_dir / "inputs" / "ifgramStack.h5"
    if not qc_ifgramstack.exists():
        raise FileNotFoundError(f"QC ifgramStack 不存在: {qc_ifgramstack}")

    backup_path = mintpy_dir / "inputs" / "ifgramStack_pre_qc_backup.h5"
    if active_path.exists() or active_path.is_symlink():
        if active_path.is_symlink():
            active_path.unlink()
        else:
            if not backup_path.exists():
                shutil.move(active_path, backup_path)
            else:
                active_path.unlink()

    # Use a physical copy instead of a symlink for the active ifgramStack.
    # MintPy updates this file in-place (dropIfgram / bridging datasets / attrs),
    # and a copy avoids surprising side effects on the QC template artifact.
    shutil.copy2(qc_ifgramstack, active_path)
    mode = "copy"

    return {
        "active_ifgram_path": str(active_path),
        "qc_ifgramStack": str(qc_ifgramstack),
        "activation_mode": mode,
        "backup_path": str(backup_path) if backup_path.exists() else None,
    }


def _find_existing_watermask(mintpy_dir: Path) -> Path | None:
    candidates = [
        mintpy_dir / "waterMask.h5",
        mintpy_dir / "inputs" / "waterMask.h5",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_geometry_sanity_check(
    mintpy_dir: Path | None = None,
    geom_source_dir: Path | None = None,
    subset_window: Iterable[int] | None = None,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    geom_source_dir = Path(geom_source_dir or (cfg.ISCE_WORK_DIR / "merged" / "geom_reference"))
    report_dir = _ensure_dir(report_dir)

    ifgram_path = mintpy_dir / "inputs" / "ifgramStack.h5"
    geom_path = mintpy_dir / "inputs" / "geometryRadar.h5"
    water_path = _find_existing_watermask(mintpy_dir)

    if not ifgram_path.exists():
        raise FileNotFoundError(f"ifgramStack.h5 不存在: {ifgram_path}")
    if not geom_path.exists():
        raise FileNotFoundError(f"geometryRadar.h5 不存在: {geom_path}")

    report: dict[str, Any] = {
        "ifgramStack": str(ifgram_path),
        "geometryRadar": str(geom_path),
        "waterMask": str(water_path) if water_path else None,
        "shape_consistent": True,
        "shape_details": {},
        "latlon_points": [],
        "reference_response": {"status": "skipped_missing_mintpy_outputs"},
    }

    with h5py.File(ifgram_path, "r") as f_ifg, h5py.File(geom_path, "r") as f_geo:
        ifg_shape = tuple(int(v) for v in f_ifg["unwrapPhase"].shape[-2:])
        geom_shape = tuple(int(v) for v in f_geo["latitude"].shape)
        report["shape_details"]["ifgramStack"] = list(ifg_shape)
        report["shape_details"]["geometryRadar"] = list(geom_shape)
        if ifg_shape != geom_shape:
            report["shape_consistent"] = False

        if water_path and water_path.exists():
            with h5py.File(water_path, "r") as f_wm:
                wm = f_wm["waterMask"][:]
            report["shape_details"]["waterMask"] = list(wm.shape)
            if tuple(int(v) for v in wm.shape) != geom_shape:
                report["shape_consistent"] = False

        if not report["shape_consistent"]:
            _write_json(report_dir / "geometry_sanity_report.json", report)
            raise RuntimeError("geometry/ifgramStack/waterMask 尺寸不一致，停止 MintPy。")

        if cfg._AOI_CROP_OFFSET is not None:
            full_lat, full_lon = _downsample_current_crop(
                geom_source_dir=geom_source_dir,
                output_shape=ifg_shape,
                crop_offset=tuple(cfg._AOI_CROP_OFFSET),
            )
            if subset_window is not None:
                y0, y1, x0, x1 = [int(v) for v in subset_window]
                expected_lat = full_lat[y0:y1, x0:x1]
                expected_lon = full_lon[y0:y1, x0:x1]
            else:
                expected_lat = full_lat
                expected_lon = full_lon

            actual_lat = f_geo["latitude"][:]
            actual_lon = f_geo["longitude"][:]
            h, w = actual_lat.shape
            points = [
                ("UL", 0, 0),
                ("UR", 0, max(0, w - 1)),
                ("LL", max(0, h - 1), 0),
                ("LR", max(0, h - 1), max(0, w - 1)),
                ("C", h // 2, w // 2),
            ]
            for label, yy, xx in points:
                item = {
                    "label": label,
                    "yx": [int(yy), int(xx)],
                    "actual_lat": float(actual_lat[yy, xx]),
                    "actual_lon": float(actual_lon[yy, xx]),
                    "expected_lat": float(expected_lat[yy, xx]),
                    "expected_lon": float(expected_lon[yy, xx]),
                    "lat_abs_diff": float(abs(actual_lat[yy, xx] - expected_lat[yy, xx])),
                    "lon_abs_diff": float(abs(actual_lon[yy, xx] - expected_lon[yy, xx])),
                }
                report["latlon_points"].append(item)

            max_lat_diff = max(item["lat_abs_diff"] for item in report["latlon_points"])
            max_lon_diff = max(item["lon_abs_diff"] for item in report["latlon_points"])
            report["latlon_consistent"] = bool(max_lat_diff < 1e-4 and max_lon_diff < 1e-4)
            if not report["latlon_consistent"]:
                _write_json(report_dir / "geometry_sanity_report.json", report)
                raise RuntimeError("geometryRadar 经纬度与 AOI 偏移采样不一致，停止 MintPy。")

        ref_y = f_ifg.attrs.get("REF_Y")
        ref_x = f_ifg.attrs.get("REF_X")
        ts_candidates = [
            ("raw", mintpy_dir / "timeseries.h5"),
            ("gacos_only", mintpy_dir / "timeseries_SET_GACOS.h5"),
            ("final", mintpy_dir / "timeseries_SET_GACOS_ramp_demErr.h5"),
        ]
        if ref_y is not None and ref_x is not None and all(path.exists() for _, path in ts_candidates):
            ref_y = int(ref_y)
            ref_x = int(ref_x)
            response = {}
            for label, path in ts_candidates:
                with h5py.File(path, "r") as f_ts:
                    response[label] = {
                        "mean_mm": float(np.nanmean(f_ts["timeseries"][:, ref_y, ref_x]) * 1000.0),
                        "std_mm": float(np.nanstd(f_ts["timeseries"][:, ref_y, ref_x]) * 1000.0),
                    }
            report["reference_response"] = {"status": "ok", "ref_yx": [ref_y, ref_x], "series": response}

    report_path = _write_json(report_dir / "geometry_sanity_report.json", report)
    report["geometry_sanity_report_json"] = str(report_path)
    return report


def plot_pair_qc_matrix(
    pair_qc_csv: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    pair_qc_csv = Path(pair_qc_csv or (cfg.WORK_DIR / "mainchain_qc" / "pair_qc.csv"))
    output_path = Path(output_path or (cfg.WORK_DIR / "mainchain_qc" / "pair_qc_matrix.png"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not pair_qc_csv.exists():
        raise FileNotFoundError(f"pair_qc.csv 不存在: {pair_qc_csv}")

    with pair_qc_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    dates = sorted({row["date1"] for row in rows} | {row["date2"] for row in rows})
    idx = {date: i for i, date in enumerate(dates)}
    matrix = np.full((len(dates), len(dates)), np.nan, dtype=np.float32)
    risk_mat = np.full((len(dates), len(dates)), np.nan, dtype=np.float32)
    for row in rows:
        i = idx[row["date1"]]
        j = idx[row["date2"]]
        matrix[i, j] = ACTION_CODE[row["action"]]
        risk_mat[i, j] = float(row["risk"])

    cmap = plt.matplotlib.colors.ListedColormap(["#6abf69", "#f0c95d", "#db5f57"])
    norm = plt.matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title("Pair QC Matrix (keep / downweight / drop)")
    ax.set_xlabel("date2")
    ax.set_ylabel("date1")
    ax.set_xticks(range(len(dates)))
    ax.set_yticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=90, fontsize=7)
    ax.set_yticklabels(dates, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["keep", "downweight", "drop"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "pair_qc_matrix_png": str(output_path),
        "n_dates": len(dates),
        "n_pairs": len(rows),
    }


def prepare_mintpy_qc_inputs(
    mintpy_dir: Path | None = None,
    geom_source_dir: Path | None = None,
    pair_qc_csv: Path | None = None,
    subset_window: Iterable[int] | None = None,
    report_dir: Path | None = None,
    source_ifgram_name: str = "ifgramStack_original.h5",
    qc_ifgram_name: str = "ifgramStack_qc.h5",
    activate_qc: bool = True,
) -> dict[str, Any]:
    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    report_dir = _ensure_dir(report_dir)
    source_ifgram = mintpy_dir / "inputs" / source_ifgram_name
    qc_ifgram = mintpy_dir / "inputs" / qc_ifgram_name

    qc_report = create_qc_ifgramstack(
        source_ifgramstack=source_ifgram,
        pair_qc_csv=pair_qc_csv,
        output_ifgramstack=qc_ifgram,
        metadata_report=report_dir / "ifgramStack_qc_metadata.json",
    )
    activation = {}
    if activate_qc:
        activation = activate_qc_ifgramstack(mintpy_dir=mintpy_dir, qc_ifgramstack=qc_ifgram)
    geometry = run_geometry_sanity_check(
        mintpy_dir=mintpy_dir,
        geom_source_dir=geom_source_dir,
        subset_window=subset_window,
        report_dir=report_dir,
    )
    matrix = plot_pair_qc_matrix(pair_qc_csv=pair_qc_csv, output_path=report_dir / "pair_qc_matrix.png")

    summary = {
        "qc_ifgram_report": qc_report,
        "activation": activation,
        "geometry": geometry,
        "pair_qc_matrix": matrix,
    }
    _write_json(report_dir / "pre_mintpy_qc.json", {**_read_json(report_dir / "pre_mintpy_qc.json"), **summary})
    return summary


def run_mintpy_feedback_roundtrip(
    template_path: Path | None = None,
    mintpy_dir: Path | None = None,
    geom_source_dir: Path | None = None,
    log_file: Path | None = None,
    report_dir: Path | None = None,
    depsi_like_config: dict[str, Any] | None = None,
    source_ifgram_name: str = "ifgramStack_original.h5",
    qc_ifgram_name: str = "ifgramStack_qc.h5",
) -> dict[str, Any]:
    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    geom_source_dir = Path(geom_source_dir or (cfg.ISCE_WORK_DIR / "merged" / "geom_reference"))
    report_dir = _ensure_dir(report_dir)
    template_path = Path(template_path or cfg.TEMPLATE_PATH)
    log_file = Path(log_file or cfg.LOG_FILE)

    pre_qc = _read_json(report_dir / "pre_mintpy_qc.json")
    pair_qc_csv = Path(pre_qc.get("pair_qc_csv") or (report_dir / "pair_qc.csv"))
    pair_qc_summary_json = Path(pre_qc.get("pair_qc_summary_json") or (report_dir / "pair_qc_summary.json"))
    ref_candidates_csv = Path(pre_qc.get("ref_candidates_csv") or (report_dir / "ref_candidates.csv"))
    subset_window = pre_qc.get("selected_subset_window")
    subset_window = tuple(int(v) for v in subset_window) if subset_window else None

    if not pair_qc_csv.exists():
        raise FileNotFoundError(f"pair_qc.csv 不存在: {pair_qc_csv}")

    pass1_inputs = prepare_mintpy_qc_inputs(
        mintpy_dir=mintpy_dir,
        geom_source_dir=geom_source_dir,
        pair_qc_csv=pair_qc_csv,
        subset_window=subset_window,
        report_dir=report_dir,
        source_ifgram_name=source_ifgram_name,
        qc_ifgram_name=qc_ifgram_name,
        activate_qc=True,
    )
    active_ifgram_path = Path(
        pass1_inputs.get("activation", {}).get("active_ifgram_path")
        or (mintpy_dir / "inputs" / "ifgramStack.h5")
    )
    lock1 = lock_reference_candidate(active_ifgram_path, ref_candidates_csv, candidate_rank=1)
    selected_rank1 = lock1.get("candidate_rank") if lock1.get("status") == "ok" else None

    run_mintpy(template_path, mintpy_dir, log_file, ifgram_stack_path=active_ifgram_path)
    pass1_audit = audit_pair_qc_after_mintpy(
        mintpy_dir=mintpy_dir,
        pair_qc_csv=pair_qc_csv,
        summary_json=pair_qc_summary_json,
    )
    pass1_eval = evaluate_mintpy_pass(
        mintpy_dir=mintpy_dir,
        qc_report_dir=report_dir,
        pass_label="pass1",
        selected_reference_rank=selected_rank1,
    )
    pass1_bias_audit = audit_reference_bias(
        mintpy_dir=mintpy_dir,
        qc_report_dir=report_dir,
        selected_reference_rank=selected_rank1,
        top_k=int((depsi_like_config or {}).get("reference_bias_top_k", 20)),
        output_name="reference_bias_audit.json",
        config=depsi_like_config,
    )

    snapshot_dir = report_dir / "mintpy_pass1_snapshot"
    pass1_snapshot = snapshot_mintpy_outputs(mintpy_dir, snapshot_dir)
    feedback_summary = compute_date_feedback(
        mintpy_dir=mintpy_dir,
        qc_report_dir=report_dir,
        pair_qc_csv=pair_qc_csv,
        selected_reference_rank=selected_rank1,
        config=depsi_like_config,
    )

    selected_rank2 = selected_rank1 or 1
    pass2_reference_selection_reason = "keep_pass1_reference"
    if feedback_summary.get("suggested_next_reference_rank"):
        selected_rank2 = int(feedback_summary["suggested_next_reference_rank"])
        pass2_reference_selection_reason = "reference_abnormal_fallback"
    elif pass1_bias_audit.get("reference_bias_detected") and pass1_bias_audit.get("recommended_rank_for_pass2"):
        selected_rank2 = int(pass1_bias_audit["recommended_rank_for_pass2"])
        pass2_reference_selection_reason = "reference_bias_audit"
    clear_summary = clear_mintpy_outputs_for_rerun(mintpy_dir)

    pass2_inputs = prepare_mintpy_qc_inputs(
        mintpy_dir=mintpy_dir,
        geom_source_dir=geom_source_dir,
        pair_qc_csv=pair_qc_csv,
        subset_window=subset_window,
        report_dir=report_dir,
        source_ifgram_name=source_ifgram_name,
        qc_ifgram_name=qc_ifgram_name,
        activate_qc=True,
    )
    active_ifgram_path2 = Path(
        pass2_inputs.get("activation", {}).get("active_ifgram_path")
        or (mintpy_dir / "inputs" / "ifgramStack.h5")
    )
    lock2 = lock_reference_candidate(active_ifgram_path2, ref_candidates_csv, candidate_rank=int(selected_rank2))
    selected_rank2 = lock2.get("candidate_rank") if lock2.get("status") == "ok" else selected_rank2

    run_mintpy(template_path, mintpy_dir, log_file, ifgram_stack_path=active_ifgram_path2)
    pass2_audit = audit_pair_qc_after_mintpy(
        mintpy_dir=mintpy_dir,
        pair_qc_csv=pair_qc_csv,
        summary_json=pair_qc_summary_json,
    )
    pass2_eval = evaluate_mintpy_pass(
        mintpy_dir=mintpy_dir,
        qc_report_dir=report_dir,
        pass_label="pass2",
        selected_reference_rank=int(selected_rank2) if selected_rank2 is not None else None,
    )
    pass2_bias_audit = audit_reference_bias(
        mintpy_dir=mintpy_dir,
        qc_report_dir=report_dir,
        selected_reference_rank=int(selected_rank2) if selected_rank2 is not None else None,
        top_k=int((depsi_like_config or {}).get("reference_bias_top_k", 20)),
        output_name="reference_bias_audit_pass2.json",
        config=depsi_like_config,
    )

    pass1_adj = float(pass1_eval.get("adjacent_jump_ratio", np.inf))
    pass2_adj = float(pass2_eval.get("adjacent_jump_ratio", np.inf))
    pass1_rms = float(pass1_eval.get("median_model_rms", np.inf))
    pass2_rms = float(pass2_eval.get("median_model_rms", np.inf))
    pass1_count = int(pass1_eval.get("retained_count", 0))
    pass2_count = int(pass2_eval.get("retained_count", 0))
    pass1_scene_median_vel = abs(float(pass1_eval.get("strict_ps_scene_median_velocity_mm_yr", np.inf)))
    pass2_scene_median_vel = abs(float(pass2_eval.get("strict_ps_scene_median_velocity_mm_yr", np.inf)))
    pass1_ref_abs_vel = abs(float((pass1_eval.get("reference_candidate_metrics") or {}).get("patch_velocity_abs_median_mm_yr", np.inf)))
    pass2_ref_abs_vel = abs(float((pass2_eval.get("reference_candidate_metrics") or {}).get("patch_velocity_abs_median_mm_yr", np.inf)))

    rejection_reasons = []
    if not (pass2_adj <= pass1_adj):
        rejection_reasons.append("adjacent_jump_ratio_not_improved")
    if not (pass2_rms <= 1.05 * pass1_rms):
        rejection_reasons.append("median_model_rms_degraded")
    if not (pass2_count >= 0.85 * pass1_count):
        rejection_reasons.append("retained_count_collapsed")
    if not (pass2_scene_median_vel <= pass1_scene_median_vel):
        rejection_reasons.append("strict_ps_scene_median_velocity_not_improved")
    if not (pass2_ref_abs_vel <= pass1_ref_abs_vel):
        rejection_reasons.append("reference_bias_not_improved")

    accepted_pass = "pass2" if not rejection_reasons else "pass1"
    if accepted_pass == "pass2":
        final_pair_selection = _finalize_pair_qc_selection(pair_qc_csv, "round2")
        pass2_acceptance = {
            "accepted": True,
            "accepted_pass": accepted_pass,
            "pass2_rejected_reason": [],
        }
    else:
        restore_summary = restore_mintpy_snapshot(mintpy_dir, snapshot_dir)
        final_pair_selection = _finalize_pair_qc_selection(pair_qc_csv, "round1")
        prepare_mintpy_qc_inputs(
            mintpy_dir=mintpy_dir,
            geom_source_dir=geom_source_dir,
            pair_qc_csv=pair_qc_csv,
            subset_window=subset_window,
            report_dir=report_dir,
            source_ifgram_name=source_ifgram_name,
            qc_ifgram_name=qc_ifgram_name,
            activate_qc=True,
        )
        pass2_acceptance = {
            "accepted": False,
            "accepted_pass": accepted_pass,
            "pass2_rejected_reason": rejection_reasons,
            "restored_pass1_snapshot": restore_summary,
        }

    pair_summary = _read_json(pair_qc_summary_json)
    pair_summary["mintpy_feedback_roundtrip"] = {
        "accepted_pass": accepted_pass,
        "pass1_selected_reference_rank": selected_rank1,
        "pass2_selected_reference_rank": selected_rank2,
        "pass2_reference_selection_reason": pass2_reference_selection_reason,
        "pass2_rejected_reason": rejection_reasons,
        "final_pair_selection": final_pair_selection,
    }
    _write_json(pair_qc_summary_json, pair_summary)

    summary = {
        "status": "ok",
        "mintpy_dir": str(mintpy_dir),
        "report_dir": str(report_dir),
        "pair_qc_csv": str(pair_qc_csv),
        "pair_qc_summary_json": str(pair_qc_summary_json),
        "pass1_inputs": pass1_inputs,
        "pass1_reference_lock": lock1,
        "pass1_audit": pass1_audit,
        "pass1_evaluation": pass1_eval,
        "pass1_reference_bias_audit": pass1_bias_audit,
        "pass1_snapshot": pass1_snapshot,
        "feedback_summary": feedback_summary,
        "clear_for_pass2": clear_summary,
        "pass2_inputs": pass2_inputs,
        "pass2_reference_lock": lock2,
        "pass2_audit": pass2_audit,
        "pass2_evaluation": pass2_eval,
        "pass2_reference_bias_audit": pass2_bias_audit,
        "pass2_reference_selection_reason": pass2_reference_selection_reason,
        "pass2_acceptance": pass2_acceptance,
        "final_pair_selection": final_pair_selection,
        "accepted_pass": accepted_pass,
    }
    summary_path = _write_json(report_dir / "mintpy_feedback_roundtrip.json", summary)
    summary["mintpy_feedback_roundtrip_json"] = str(summary_path)
    return summary


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


def audit_pair_qc_after_mintpy(
    mintpy_dir: Path | None = None,
    pair_qc_csv: Path | None = None,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    pair_qc_csv = Path(pair_qc_csv or (cfg.WORK_DIR / "mainchain_qc" / "pair_qc.csv"))
    summary_json = Path(summary_json or (cfg.WORK_DIR / "mainchain_qc" / "pair_qc_summary.json"))

    final_ts = _pick_final_timeseries(mintpy_dir)
    if final_ts is None or not pair_qc_csv.exists():
        return {"status": "skipped_missing_final_timeseries_or_pair_qc"}

    with pair_qc_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with h5py.File(final_ts, "r") as f_ts:
        dates = [d.decode("utf-8") for d in f_ts["date"][:]]
        date_to_idx = {d: i for i, d in enumerate(dates)}
        ts = f_ts["timeseries"]
        for row in rows:
            d1 = row["date1"]
            d2 = row["date2"]
            if d1 not in date_to_idx or d2 not in date_to_idx:
                row["final_p95_abs_inc_mm"] = ""
                row["final_frac_abs_inc_gt_50mm"] = ""
                continue
            inc = (ts[date_to_idx[d2], :, :] - ts[date_to_idx[d1], :, :]) * 1000.0
            valid = np.isfinite(inc)
            if not valid.any():
                row["final_p95_abs_inc_mm"] = ""
                row["final_frac_abs_inc_gt_50mm"] = ""
                continue
            abs_inc = np.abs(inc[valid])
            row["final_p95_abs_inc_mm"] = float(np.percentile(abs_inc, 95))
            row["final_frac_abs_inc_gt_50mm"] = float(np.mean(abs_inc > 50.0))

    fieldnames = list(rows[0].keys()) if rows else PAIR_QC_COLUMNS
    with pair_qc_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = _read_json(summary_json)
    summary["audit_final_timeseries"] = str(final_ts)
    summary["audit_completed"] = True
    summary["final_gt50_pairs"] = sorted(
        [
            {
                "date1": row["date1"],
                "date2": row["date2"],
                "final_frac_abs_inc_gt_50mm": float(row["final_frac_abs_inc_gt_50mm"]),
            }
            for row in rows
            if row.get("final_frac_abs_inc_gt_50mm") not in ("", None)
        ],
        key=lambda item: item["final_frac_abs_inc_gt_50mm"],
        reverse=True,
    )[:10]
    _write_json(summary_json, summary)
    return {
        "status": "ok",
        "pair_qc_csv": str(pair_qc_csv),
        "pair_qc_summary_json": str(summary_json),
        "audit_final_timeseries": str(final_ts),
    }


def _collect_mintpy_metrics(mintpy_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "coherenceSpatialAvg": None,
        "valid_conncomp_frac": None,
        "n_conncomp": None,
        "velocityStd": None,
        "adjacent_epoch_jump_gt_28mm_ratio": None,
        "residual_ramp_metric": None,
    }

    coh_txt = mintpy_dir / "coherenceSpatialAvg.txt"
    if coh_txt.exists():
        vals = []
        for line in coh_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            for part in reversed(parts):
                try:
                    vals.append(float(part))
                    break
                except ValueError:
                    continue
        if vals:
            metrics["coherenceSpatialAvg"] = float(np.nanmean(vals))

    mask_conn = mintpy_dir / "maskConnComp.h5"
    if mask_conn.exists():
        with h5py.File(mask_conn, "r") as f:
            key = next(iter(f.keys()))
            data = f[key][:]
        metrics["valid_conncomp_frac"] = float(np.mean(data.astype(bool)))

    tri_file = mintpy_dir / "numTriNonzeroIntAmbiguity.h5"
    if tri_file.exists():
        with h5py.File(tri_file, "r") as f:
            key = next(iter(f.keys()))
            data = f[key][:]
        metrics["n_conncomp"] = int(np.nanmax(data))

    vel_std = mintpy_dir / "velocityStd.h5"
    if vel_std.exists():
        with h5py.File(vel_std, "r") as f:
            key = next(iter(f.keys()))
            data = f[key][:]
        finite = data[np.isfinite(data)]
        if finite.size:
            metrics["velocityStd"] = float(np.nanmedian(finite) * 1000.0)

    ts_final = _pick_final_timeseries(mintpy_dir)
    if ts_final:
        with h5py.File(ts_final, "r") as f:
            ts = f["timeseries"]
            if ts.shape[0] >= 2:
                gt_ratios = []
                for i in range(1, ts.shape[0]):
                    inc = np.abs((ts[i, :, :] - ts[i - 1, :, :]) * 1000.0)
                    valid = np.isfinite(inc)
                    gt_ratios.append(float(np.mean(inc[valid] > 28.0)) if valid.any() else 0.0)
                metrics["adjacent_epoch_jump_gt_28mm_ratio"] = float(np.mean(gt_ratios))

    rms_file = mintpy_dir / "rms_timeseriesResidual_ramp.txt"
    if rms_file.exists():
        vals = []
        for line in rms_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                vals.append(float(parts[-1]))
            except ValueError:
                continue
        if vals:
            metrics["residual_ramp_metric"] = float(np.nanmean(vals))
    return metrics


def run_dolphin_ablation(
    enabled: bool = False,
    slc_pattern: str | None = None,
    work_root: Path | None = None,
    geom_source_dir: Path | None = None,
    atmo_config: dict[str, Any] | None = None,
    n_workers: int | None = None,
) -> dict[str, Any]:
    work_root = Path(work_root or (cfg.WORK_DIR / "dolphin_ablation"))
    work_root.mkdir(parents=True, exist_ok=True)

    summary_path = work_root / "dolphin_ablation_summary.csv"
    report_path = work_root / "dolphin_ablation_report.json"

    variants = [
        {"name": "g1_default", "half_window": (11, 5), "strides": (6, 3)},
        {"name": "g2_window_15x7", "half_window": (15, 7), "strides": (6, 3)},
        {"name": "g3_stride_8x4", "half_window": (11, 5), "strides": (8, 4)},
    ]

    if not enabled:
        report = {
            "status": "disabled",
            "variants": variants,
            "summary_csv": str(summary_path),
            "report_json": str(report_path),
        }
        _write_json(report_path, report)
        return report

    from .dolphin_runner import build_dolphin_config, run_dolphin
    from .mintpy_runner import build_mintpy_hdf5, generate_mintpy_template, run_mintpy

    rows = []
    for variant in variants:
        name = variant["name"]
        dolphin_dir = work_root / name / "dolphin_work"
        mintpy_dir = work_root / name / "mintpy"
        tpl_path = mintpy_dir / "custom_template.txt"

        cfg_path = build_dolphin_config(
            slc_pattern=slc_pattern,
            work_dir=dolphin_dir,
            n_workers=n_workers,
            half_window=variant["half_window"],
            strides=variant["strides"],
            max_bandwidth=3,
            unwrap_method="snaphu",
        )
        run_dolphin(cfg_path, log_file=work_root / name / "dolphin_ablation.log", n_workers=n_workers)
        build_mintpy_hdf5(
            dolphin_dir=dolphin_dir,
            geom_source_dir=geom_source_dir,
            mintpy_dir=mintpy_dir,
            ifgram_filename="ifgramStack.h5",
        )
        tpl = generate_mintpy_template(
            dolphin_dir=dolphin_dir,
            dem_file=cfg.DEM_FILE,
            era5_dir=cfg.ERA5_DIR,
            gacos_dir=cfg.GACOS_DIR,
            output_path=tpl_path,
            atmo_config=atmo_config,
        )
        run_mintpy(tpl, mintpy_dir, work_root / name / "mintpy_ablation.log")
        metrics = _collect_mintpy_metrics(mintpy_dir)
        row = {"name": name, **variant, **metrics}
        rows.append(row)

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report = {
        "status": "ok",
        "variants": rows,
        "summary_csv": str(summary_path),
        "report_json": str(report_path),
    }
    _write_json(report_path, report)
    return report
