"""Visualization for MintPy downstream forecast products."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from .dataset import load_forecast_context
from insar_utils.viz import _CMAP_DISP, _compute_hillshade, _load_mintpy_data


def _load_forecast_h5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        def _decode(path_str: str, default: str = "") -> str:
            if path_str not in f:
                return default
            value = f[path_str][()]
            return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)

        def _decode_array(path_str: str) -> list[str]:
            if path_str not in f:
                return []
            return [x.decode("utf-8") for x in f[path_str][:]]

        return {
            "future_dates": _decode_array("meta/future_dates"),
            "history_dates": _decode_array("meta/history_dates"),
            "forecast_mode_requested": _decode("meta/forecast_mode_requested", "generic"),
            "forecast_mode_actual": _decode("meta/forecast_mode_actual", "generic"),
            "selection_mode": _decode("meta/selection_mode", ""),
            "zone_detection_status": _decode("meta/zone_detection_status", ""),
            "zone_filter_mode": _decode("meta/zone_filter_mode", ""),
            "zone_mask_path": _decode("meta/zone_mask_path", ""),
            "confidence_mode": _decode("meta/confidence_mode", ""),
            "uncertainty_mode": _decode("meta/uncertainty_mode", ""),
            "active_model": _decode("meta/active_model", ""),
            "cumulative_reference_date": _decode("meta/cumulative_reference_date", ""),
            "row": f["points/row"][:].astype(np.int32),
            "col": f["points/col"][:].astype(np.int32),
            "lon": f["points/longitude"][:].astype(np.float32),
            "lat": f["points/latitude"][:].astype(np.float32),
            "strict_or_relaxed": [x.decode("utf-8") for x in f["points/strict_or_relaxed"][:]],
            "c_pred": f["predictions/c_pred"][:].astype(np.float32),
            "interval_width_raw": f["predictions/interval_width_raw"][:].astype(np.float32),
            "interval_width_calibrated": f["predictions/interval_width_calibrated"][:].astype(np.float32),
            "pred_rel0_p10": f["predictions/pred_rel0_p10"][:].astype(np.float32),
            "pred_rel0_p50": f["predictions/pred_rel0_p50"][:].astype(np.float32),
            "pred_rel0_p90": f["predictions/pred_rel0_p90"][:].astype(np.float32),
            "pred_rel0_p10_raw": f["predictions/pred_rel0_p10_raw"][:].astype(np.float32),
            "pred_rel0_p50_raw": f["predictions/pred_rel0_p50_raw"][:].astype(np.float32),
            "pred_rel0_p90_raw": f["predictions/pred_rel0_p90_raw"][:].astype(np.float32),
            "neighbor_attention_mean": f["explainability/neighbor_attention_mean"][:].astype(np.float32) if "explainability/neighbor_attention_mean" in f else np.empty((0, 0), dtype=np.float32),
            "neighbor_attention_topk_index": f["explainability/neighbor_attention_topk_index"][:].astype(np.int32) if "explainability/neighbor_attention_topk_index" in f else np.empty((0, 0), dtype=np.int32),
            "neighbor_attention_topk_weight": f["explainability/neighbor_attention_topk_weight"][:].astype(np.float32) if "explainability/neighbor_attention_topk_weight" in f else np.empty((0, 0), dtype=np.float32),
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_project_aoi_bounds(mintpy_dir: Path) -> tuple[float, float, float, float] | None:
    project_json = mintpy_dir.parent / "project.json"
    if not project_json.exists():
        return None
    try:
        payload = json.loads(project_json.read_text(encoding="utf-8"))
        bbox = payload.get("aoi_bbox")
        if bbox and len(bbox) == 4:
            s, n, w, e = [float(v) for v in bbox]
            return s, n, w, e
    except Exception:
        return None
    return None


def _smooth_hillshade_grid(hgt: np.ndarray, lon_valid: np.ndarray, lat_valid: np.ndarray, glon: np.ndarray, glat: np.ndarray) -> np.ndarray | None:
    if hgt.size < 1000:
        return None
    pts = np.column_stack([lon_valid, lat_valid])
    dem_linear = griddata(pts, hgt, (glon, glat), method="linear")
    dem_nearest = griddata(pts, hgt, (glon, glat), method="nearest")
    if dem_nearest is None:
        return None
    dem_grid = dem_nearest if dem_linear is None else np.where(np.isfinite(dem_linear), dem_linear, dem_nearest)
    dem_grid = gaussian_filter(np.nan_to_num(dem_grid, nan=np.nanmedian(dem_grid)), sigma=1.0)
    cellsize = abs(glon[0, 1] - glon[0, 0]) if glon.shape[1] > 1 else 0.0005
    return _compute_hillshade(dem_grid, cellsize)


def _prepare_map_canvas(mintpy_dir: Path, lon_points: np.ndarray, lat_points: np.ndarray) -> dict[str, Any]:
    data = _load_mintpy_data(mintpy_dir)
    lat = data.get("lat")
    lon = data.get("lon")
    hgt = data.get("height")
    if lat is None or lon is None:
        raise RuntimeError("MintPy geometry 不完整，无法生成统一风格地图。")
    point_ok = np.isfinite(lon_points) & np.isfinite(lat_points)
    if not np.any(point_ok):
        raise RuntimeError("预测点经纬度无效，无法生成统一风格地图。")
    aoi_bounds = _load_project_aoi_bounds(mintpy_dir)
    if aoi_bounds is not None:
        lat_min, lat_max, lon_min, lon_max = aoi_bounds
        lat_pad = max((lat_max - lat_min) * 0.01, 0.001)
        lon_pad = max((lon_max - lon_min) * 0.01, 0.001)
    else:
        lon_m = lon_points[point_ok]
        lat_m = lat_points[point_ok]
        lon_pad = max((float(np.nanmax(lon_m)) - float(np.nanmin(lon_m))) * 0.03, 0.002)
        lat_pad = max((float(np.nanmax(lat_m)) - float(np.nanmin(lat_m))) * 0.03, 0.002)
        lat_min = float(np.nanmin(lat_m) - lat_pad)
        lat_max = float(np.nanmax(lat_m) + lat_pad)
        lon_min = float(np.nanmin(lon_m) - lon_pad)
        lon_max = float(np.nanmax(lon_m) + lon_pad)
    res = 0.0005
    glon_1d = np.arange(lon_min, lon_max, res)
    glat_1d = np.arange(lat_max, lat_min, -res)
    if glon_1d.size < 2:
        glon_1d = np.linspace(lon_min, lon_max, 2)
    if glat_1d.size < 2:
        glat_1d = np.linspace(lat_max, lat_min, 2)
    glon, glat = np.meshgrid(glon_1d, glat_1d)
    extent = [lon_min, lon_max, lat_min, lat_max]
    hs_grid = None
    if hgt is not None:
        geo_ok = (
            np.isfinite(hgt)
            & np.isfinite(lat)
            & np.isfinite(lon)
            & (lat > 0.1)
            & (lon > 0.1)
            & (lat >= lat_min - lat_pad)
            & (lat <= lat_max + lat_pad)
            & (lon >= lon_min - lon_pad)
            & (lon <= lon_max + lon_pad)
        )
        if int(np.sum(geo_ok)) > 1000:
            hs_grid = _smooth_hillshade_grid(hgt[geo_ok], lon[geo_ok], lat[geo_ok], glon, glat)
    return {
        "extent": extent,
        "hillshade": hs_grid,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }


def _map_figure_size(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> tuple[float, float]:
    fig_w = 6.5
    fig_h = fig_w * (lat_max - lat_min) / max(lon_max - lon_min, 1e-6) / np.cos(np.radians((lat_min + lat_max) / 2)) + 0.8
    return fig_w, fig_h


def _draw_map_background(ax, hillshade: np.ndarray | None, extent: list[float]) -> None:
    ax.set_facecolor("white")
    if hillshade is None:
        return
    hs_rgba = plt.cm.gray(np.clip(hillshade, 0.05, 0.95))
    ax.imshow(hs_rgba, extent=extent, aspect="auto", interpolation="bilinear", origin="upper", alpha=0.24)


def _draw_map_axes(ax, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> None:
    import matplotlib.ticker as mticker

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude [°E]", fontsize=8)
    ax.set_ylabel("Latitude [°N]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))


def _draw_scalebar(ax, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> None:
    deg10 = 10 / (111.0 * np.cos(np.radians((lat_min + lat_max) / 2)))
    x0 = lon_min + 0.05 * (lon_max - lon_min)
    y0 = lat_min + 0.04 * (lat_max - lat_min)
    ax.plot([x0, x0 + deg10], [y0, y0], "k-", lw=1.5, solid_capstyle="butt")
    ax.text(x0 + deg10 / 2, y0 + 0.002, "10 km", ha="center", va="bottom", fontsize=6.5, fontweight="bold")


def _mode_label(payload: dict[str, Any]) -> str:
    requested = payload["forecast_mode_requested"]
    actual = payload["forecast_mode_actual"]
    if requested != actual:
        return f"requested={requested} | actual={actual} | fallback"
    return f"mode={actual}"


def _plot_calibration_curve(calibration_path: Path, output_path: Path) -> Path | None:
    payload = _load_json(calibration_path)
    if not payload:
        return None
    per_h = payload.get("per_horizon") or payload.get("models", {}).get(payload.get("active_model", ""), {}).get("per_horizon", [])
    if not per_h:
        return None
    horizons = [int(item["horizon"]) for item in per_h]
    raw = [float(item.get("raw_coverage", np.nan)) for item in per_h]
    cal = [float(item.get("calibrated_coverage", np.nan)) for item in per_h]
    target = float(payload.get("target_coverage", 0.80))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(horizons, raw, marker="o", lw=1.4, label="Raw coverage")
    ax.plot(horizons, cal, marker="s", lw=1.4, label="Calibrated coverage")
    ax.axhline(target, color="0.35", ls="--", lw=1.0, label=f"Target={target:.2f}")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Coverage")
    ax.set_title("Forecast Calibration Curve", fontsize=9, loc="left")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _plot_feature_group_importance(explainability_path: Path, output_path: Path) -> Path | None:
    payload = _load_json(explainability_path)
    if not payload:
        return None
    generic_channel = payload.get("generic", {}).get("channel_group_importance", {})
    generic_static = payload.get("generic", {}).get("static_feature_permutation_importance", {})
    hazard_channel = payload.get("hazard", {}).get("channel_group_importance", {})

    if not generic_channel and not hazard_channel and not generic_static:
        return None

    panels: list[tuple[str, dict[str, Any], str]] = []
    if generic_channel:
        panels.append(("Generic Channel Groups", generic_channel, "#2C7BB6"))
    if hazard_channel:
        panels.append(("Hazard Channel Groups", hazard_channel, "#C0392B"))
    elif generic_static:
        panels.append(("Generic Static Features", generic_static, "#1F9D8A"))

    if not panels:
        return None

    ncols = len(panels)
    fig_w = 6.2 if ncols == 1 else 10.8
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 4.2))
    if ncols == 1:
        axes = [axes]

    for ax, (title, importance_dict, color) in zip(axes, panels):
        group_imp = importance_dict
        names = list(group_imp.keys())
        vals = [float(group_imp[n]) for n in names]
        order = np.argsort(vals)[::-1]
        ax.barh(
            np.asarray(names)[order],
            np.asarray(vals)[order],
            color=color,
        )
        ax.invert_yaxis()
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("RMSE increase [mm]")
        ax.grid(alpha=0.18, axis="x")
        ax.tick_params(labelsize=7)

    fig.suptitle("Forecast Feature Group Importance", fontsize=9, x=0.06, ha="left")
    if not hazard_channel and generic_static:
        fig.text(
            0.06,
            0.92,
            "Hazard unavailable: right panel shows generic static feature importance",
            fontsize=7,
            color="0.45",
            ha="left",
        )
    elif not hazard_channel:
        fig.text(
            0.06,
            0.92,
            "Unavailable: hazard",
            fontsize=7,
            color="0.45",
            ha="left",
        )
    fig.tight_layout(rect=[0, 0, 1, 0.94 if not hazard_channel else 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _plot_neighbor_influence_map(payload: dict[str, Any], export_dir: Path, canvas: dict[str, Any]) -> Path | None:
    weights = np.asarray(payload["neighbor_attention_topk_weight"], dtype=np.float32)
    if weights.size == 0:
        return None
    influence = np.nanmean(weights, axis=1).astype(np.float32)
    lon = payload["lon"]
    lat = payload["lat"]
    extent = canvas["extent"]
    hs_grid = canvas["hillshade"]
    lon_min = float(canvas["lon_min"])
    lon_max = float(canvas["lon_max"])
    lat_min = float(canvas["lat_min"])
    lat_max = float(canvas["lat_max"])
    fig_w, fig_h = _map_figure_size(lat_min, lat_max, lon_min, lon_max)
    fig, ax = plt.subplots(figsize=(fig_w + 0.4, fig_h))
    _draw_map_background(ax, hs_grid, extent)
    sc = ax.scatter(lon, lat, c=influence, s=4.4, cmap="OrRd", linewidths=0, alpha=0.84)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Mean top-k neighbor attention", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    _draw_map_axes(ax, lon_min, lon_max, lat_min, lat_max)
    _draw_scalebar(ax, lon_min, lon_max, lat_min, lat_max)
    ax.set_title("Forecast Neighbor Influence Map", fontsize=8, loc="left")
    out = export_dir / "forecast_neighbor_influence_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def generate_forecast_figures(
    mintpy_dir: str | Path,
    *,
    qc_report_dir: str | Path,
    forecast_path: str | Path,
    export_dir: str | Path,
    zone_mask_path: str | Path | None = None,
    forecast_point_scope: str = "all_high_confidence",
) -> dict[str, str]:
    mintpy_dir = Path(mintpy_dir).resolve()
    qc_report_dir = Path(qc_report_dir).resolve()
    forecast_path = Path(forecast_path).resolve()
    export_dir = Path(export_dir).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_forecast_h5(forecast_path)
    calibration_path = forecast_path.parent / "forecast_calibration.json"
    explainability_path = forecast_path.parent / "forecast_explainability.json"
    ctx = load_forecast_context(
        mintpy_dir,
        qc_report_dir=qc_report_dir,
        zone_mask_path=zone_mask_path or payload.get("zone_mask_path") or None,
        forecast_point_scope=forecast_point_scope,
    )
    lon = payload["lon"]
    lat = payload["lat"]
    pred_rel0_p10 = payload["pred_rel0_p10"]
    pred_rel0_p50 = payload["pred_rel0_p50"]
    pred_rel0_p90 = payload["pred_rel0_p90"]
    c_pred = payload["c_pred"]
    future_dates = payload["future_dates"]

    outputs: dict[str, str] = {}
    canvas = _prepare_map_canvas(mintpy_dir, lon, lat)
    extent = canvas["extent"]
    hs_grid = canvas["hillshade"]
    lon_min = float(canvas["lon_min"])
    lon_max = float(canvas["lon_max"])
    lat_min = float(canvas["lat_min"])
    lat_max = float(canvas["lat_max"])
    fig_w, fig_h = _map_figure_size(lat_min, lat_max, lon_min, lon_max)
    mode_label = _mode_label(payload)

    fig = plt.figure(figsize=(fig_w + 0.55, fig_h))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.05], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    _draw_map_background(ax, hs_grid, extent)
    sc = ax.scatter(lon, lat, c=c_pred, s=4.2, cmap="Blues", vmin=0.0, vmax=1.0, linewidths=0, alpha=0.82)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Calibrated confidence", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    _draw_map_axes(ax, lon_min, lon_max, lat_min, lat_max)
    _draw_scalebar(ax, lon_min, lon_max, lat_min, lat_max)
    ax.set_title("Forecast Confidence", fontsize=8, loc="left", pad=4)
    ax.text(
        0.99,
        0.01,
        f"{mode_label} | {payload['active_model']} | {payload['uncertainty_mode']} | {payload.get('zone_filter_mode', '')}",
        transform=ax.transAxes,
        fontsize=6,
        ha="right",
        va="bottom",
        color="0.4",
    )
    out = export_dir / "forecast_confidence_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    outputs["forecast_confidence_map"] = str(out)

    horizon = pred_rel0_p50.shape[1]
    panel_w = max(fig_w * 0.88, 4.2)
    fig = plt.figure(figsize=(panel_w * horizon + 0.75, fig_h))
    gs = GridSpec(1, horizon + 1, figure=fig, width_ratios=[1.0] * horizon + [0.075], wspace=0.10)
    axes = [fig.add_subplot(gs[0, h]) for h in range(horizon)]
    cax = fig.add_subplot(gs[0, horizon])
    vmax = max(float(np.nanpercentile(np.abs(pred_rel0_p50), 95)), 1.0)
    sm = plt.cm.ScalarMappable(cmap=_CMAP_DISP, norm=Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    for h in range(horizon):
        ax = axes[h]
        _draw_map_background(ax, hs_grid, extent)
        ax.scatter(lon, lat, c=pred_rel0_p50[:, h], s=4.0, cmap=_CMAP_DISP, vmin=-vmax, vmax=vmax, linewidths=0, alpha=0.84)
        ax.set_title(f"N{h + 1}", fontsize=7, pad=2)
        _draw_map_axes(ax, lon_min, lon_max, lat_min, lat_max)
        if h == 0:
            _draw_scalebar(ax, lon_min, lon_max, lat_min, lat_max)
        else:
            ax.tick_params(axis="y", labelleft=False)
    cbar = fig.colorbar(sm, cax=cax, extend="both")
    cbar.set_label("Predicted LOS displacement [mm]", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"Forecast Horizon Panel ({mode_label})", y=0.98, fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.92, bottom=0.08)
    out = export_dir / "forecast_horizon_panel.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    outputs["forecast_horizon_panel"] = str(out)

    pick = np.argsort(-c_pred)[: min(6, len(c_pred))]
    if pick.size:
        n = len(pick)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 3.2 * nrows), squeeze=False)
        x_hist = np.arange(len(ctx.dates))
        x_fut = np.arange(len(ctx.dates), len(ctx.dates) + horizon)
        for ax, idx, sid in zip(axes.ravel(), pick, range(1, n + 1)):
            point_idx = int(payload["row"][idx] * 0 + idx)  # keep local stable numbering for display only
            hist_idx = int(np.argmin(np.square(ctx.points["longitude"] - lon[idx]) + np.square(ctx.points["latitude"] - lat[idx])))
            hist = ctx.rel0_mm[hist_idx]
            ax.set_facecolor("white")
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            ax.plot(x_hist, hist, color="#2980B9", lw=1.0, marker="o", ms=2.5, mew=0.3, mec="white", label="Observed", zorder=3)
            last_hist_val = float(hist[-1]) if np.isfinite(hist[-1]) else 0.0
            ax.plot([x_hist[-1], x_fut[0]], [last_hist_val, float(pred_rel0_p50[idx, 0])], color="#C0392B", lw=1.0, ls="--", alpha=0.6, zorder=3)
            ax.plot(x_fut, pred_rel0_p50[idx], color="#C0392B", lw=1.0, marker="s", ms=3.0, mew=0.3, mec="white", label="Forecast (p50)", zorder=4)
            fill_x = np.concatenate([[x_hist[-1]], x_fut])
            fill_lo = np.concatenate([[last_hist_val], pred_rel0_p10[idx]])
            fill_hi = np.concatenate([[last_hist_val], pred_rel0_p90[idx]])
            ax.fill_between(fill_x, fill_lo, fill_hi, color="#F6C667", alpha=0.30, label="p10-p90", zorder=2)
            ax.axhline(0, color="0.6", lw=0.5, ls="--", zorder=1)
            tick_idx = np.concatenate([x_hist[:: max(1, len(x_hist) // 5)], x_fut])
            tick_labels = [ctx.dates[i] if i < len(ctx.dates) else future_dates[i - len(ctx.dates)] for i in tick_idx]
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=6)
            ax.tick_params(axis="y", labelsize=6.5)
            ax.set_title(f"S{sid}  c={c_pred[idx]:.2f}", fontsize=7, loc="left", fontweight="bold", pad=2)
            ax.grid(alpha=0.12, lw=0.3)
            ax.set_ylabel("LOS (mm)", fontsize=6.5)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7)
        fig.suptitle(f"Forecast Example Time Series ({mode_label})", y=0.99, fontsize=9, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = export_dir / "forecast_examples.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        outputs["forecast_examples"] = str(out)

    cal_out = _plot_calibration_curve(calibration_path, export_dir / "forecast_calibration_curve.png")
    if cal_out is not None:
        outputs["forecast_calibration_curve"] = str(cal_out)

    feat_out = _plot_feature_group_importance(explainability_path, export_dir / "forecast_feature_group_importance.png")
    if feat_out is not None:
        outputs["forecast_feature_group_importance"] = str(feat_out)

    neighbor_out = _plot_neighbor_influence_map(payload, export_dir, canvas)
    if neighbor_out is not None:
        outputs["forecast_neighbor_influence_map"] = str(neighbor_out)
    return outputs


__all__ = ["generate_forecast_figures"]
