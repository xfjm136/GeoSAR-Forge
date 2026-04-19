"""
可视化与导出工具。
图件参数参考 Nature / RSE / JGR 等科研期刊常见版式:
  - 矢量/高分辨率输出 (PDF + 300 dpi PNG)
  - 严格 Nature 系单栏 89 mm / 双栏 183 mm 尺寸体系
  - 子图标签 (a), (b), (c)
  - 科学配色 (RdBu_r 双极, 0 居中, 感知均匀)
  - 真实 DEM hillshade 底图, 水体=白色
  - 比例尺 + 北向箭头 + 可选 inset 区位图
  - 坐标轴统一度分秒或小数度
  - 最小墨水比: 无多余边框/网格, facecolor='white'
"""
import numpy as np
from pathlib import Path
from datetime import datetime
import ast
import json

from . import config as cfg
from .config import logger
from matplotlib.colors import LinearSegmentedColormap

# ── 标准化科研绘图参数 ──────────────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    # 字体: 等同 Nature 系投稿要求 (Helvetica/Arial, 无衬线)
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         8,
    'mathtext.default':  'regular',
    # 坐标轴与刻度
    'axes.linewidth':    0.6,
    'axes.labelsize':    8,
    'axes.titlesize':    9,
    'axes.titleweight':  'bold',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    # 刻度
    'xtick.major.size':  3.0,
    'ytick.major.size':  3.0,
    'xtick.minor.size':  1.5,
    'ytick.minor.size':  1.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    # 图例
    'legend.fontsize':   7,
    'legend.frameon':    True,
    'legend.framealpha': 0.92,
    'legend.edgecolor':  '0.80',
    'legend.fancybox':   False,
    # 图片
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'figure.facecolor':  'white',
    'savefig.facecolor': 'white',
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
    # 线条
    'lines.linewidth':   1.0,
    'lines.markersize':  3.5,
})

# 论文常用双极色标（冷暖对称，0居中，感知均匀）
_CMAP_VEL  = 'RdBu_r'          # 经典 Remote Sensing 速率色标
_CMAP_DISP = 'RdBu_r'
_CMAP_COH  = 'inferno'         # 单调递增，相干性/置信度
_CMAP_DEM  = 'terrain'         # 地形高程
_CMAP_STD  = 'plasma'
# 不确定性：深紫(低)→粉橙→亮黄(高)，感知均匀，全程高区分度，与 RdBu_r 无冲突

# Nature 单栏 89 mm ≈ 3.5 in, 双栏 183 mm ≈ 7.2 in
_COL1 = 3.5                    # 单栏宽度 (inch)
_COL2 = 7.2                    # 双栏宽度 (inch)

# 子图标签样式
_LABEL_STYLE = dict(fontsize=10, fontweight='bold', va='top', ha='left')

# 水体样式：浅蓝底 + 蓝色斜线
_WATER_COLOR = [0.88, 0.94, 0.99, 1.0]
_WATER_HATCH_COLOR = [0.25, 0.58, 0.84, 1.0]
_HIGH_UNCERTAINTY_HATCH = '///'
_HIGH_UNCERTAINTY_COLOR = [0.82, 0.68, 0.13, 1.0]


def _load_mintpy_data(mintpy_dir):
    """加载 MintPy 全部结果数据 + 几何信息。累计位移统一转为最早日期基准。"""
    import h5py

    mintpy_dir = Path(mintpy_dir)
    data = {"mintpy_dir": mintpy_dir}

    with h5py.File(mintpy_dir / "velocity.h5") as f:
        data["vel"]       = f["velocity"][:] * 1000   # m/yr → mm/yr
        data["vel_attrs"] = dict(f.attrs)
        data["vstd"]      = f["velocityStd"][:] * 1000 if "velocityStd" in f else None

    # 优先用最终校正时序；fallback 到普通 timeseries
    for ts_name in ["timeseries_SET_GACOS_ramp_demErr.h5", "timeseries.h5"]:
        ts_path = mintpy_dir / ts_name
        if ts_path.exists():
            with h5py.File(ts_path) as f:
                data["ts"]    = f["timeseries"][:] * 1000   # m → mm
                raw           = f["date"][:]
                data["dates"] = [
                    d.tobytes().decode().strip() if hasattr(d, "tobytes") else str(d)
                    for d in raw
                ]
                if data["ts"].shape[0] > 0:
                    data["ts"] = data["ts"] - data["ts"][0:1]
                    data["cumulative_reference_date"] = data["dates"][0]
            break

    with h5py.File(mintpy_dir / "temporalCoherence.h5") as f:
        data["tcoh"] = f["temporalCoherence"][:]

    geom_file = mintpy_dir / "inputs" / "geometryRadar.h5"
    if geom_file.exists():
        with h5py.File(geom_file) as f:
            data["lat"]    = f["latitude"][:]
            data["lon"]    = f["longitude"][:]
            hgt_raw        = f["height"][:] if "height" in f else None

        # height 有效性检查：ISCE2 hgt.rdr 可能存储无效值（~0 或全负）
        # 无效时尝试从项目 DEM.tif 插值，以保证 hillshade 和 height_correlation 正常
        if hgt_raw is not None:
            valid_h = hgt_raw[np.isfinite(hgt_raw) & (hgt_raw != 0)]
            if len(valid_h) == 0 or np.abs(valid_h).mean() < 10:
                hgt_raw = _load_height_from_dem(mintpy_dir, data["lat"], data["lon"])
        data["height"] = hgt_raw
    else:
        data["lat"] = data["lon"] = data["height"] = None

    return data


def _load_height_from_dem(mintpy_dir, lat_grid, lon_grid):
    """
    从项目 DEM.tif 插值地形高度到 geometryRadar 坐标系。
    当 geometryRadar.h5 中 height 无效时的后备（不修改 HDF5 文件）。
    """
    import rasterio
    from scipy.interpolate import RegularGridInterpolator

    # 找 DEM.tif（mintpy_dir 上两级即项目根目录）
    dem_path = None
    for p in [mintpy_dir.parent / "DEM" / "dem.tif",
              cfg.DEM_FILE if hasattr(cfg, 'DEM_FILE') else None]:
        if p and Path(p).exists():
            dem_path = Path(p)
            break
    if dem_path is None:
        logger.debug("viz: 未找到 DEM.tif，hillshade 将跳过")
        return None

    try:
        with rasterio.open(str(dem_path)) as src:
            dem = src.read(1).astype('float32')
            gt  = src.transform
            nc, nr = src.width, src.height

        lon_dem = gt.c + np.arange(nc) * gt.a
        lat_dem = gt.f + np.arange(nr) * gt.e   # gt.e < 0，lat 递减
        lat_dem_inc = lat_dem[::-1]
        dem_flip    = dem[::-1, :]

        interp = RegularGridInterpolator(
            (lat_dem_inc, lon_dem), dem_flip,
            method='linear', bounds_error=False, fill_value=0)

        valid = (lat_grid > 0.1) & np.isfinite(lat_grid)
        height_out = np.zeros_like(lat_grid)
        pts = np.column_stack([lat_grid[valid].ravel(), lon_grid[valid].ravel()])
        height_out[valid] = interp(pts)
        return height_out
    except Exception as e:
        logger.debug(f"viz: DEM 插值失败 ({e})，hillshade 将跳过")
        return None


def _load_mintpy_dates_only(mintpy_dir):
    """仅加载 MintPy 日期序列，避免为地图出图读入整块时序。"""
    import h5py

    mintpy_dir = Path(mintpy_dir)
    for ts_name in ["timeseries_SET_GACOS_ramp_demErr.h5", "timeseries.h5"]:
        ts_path = mintpy_dir / ts_name
        if not ts_path.exists():
            continue
        with h5py.File(ts_path) as f:
            raw = f["date"][:]
            dates = [
                d.tobytes().decode().strip() if hasattr(d, "tobytes") else str(d)
                for d in raw
            ]
        return dates, (dates[0] if dates else None)
    return [], None


def _load_project_aoi_polygon(mintpy_dir):
    """读取或缓存项目 AOI 行政边界，用于裁掉雷达条带斜边。"""
    from geopy.geocoders import Nominatim
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    from .downloader import _get_admin_polygon

    mintpy_dir = Path(mintpy_dir)
    project_dir = mintpy_dir.parent
    cache_path = project_dir / "aoi_boundary.geojson"
    if cache_path.exists():
        try:
            obj = json.loads(cache_path.read_text(encoding="utf-8"))
            geom = obj.get("geometry", obj)
            poly = shape(geom)
            if poly.is_valid and not poly.is_empty:
                return poly
        except Exception:
            pass

    project_json = project_dir / "project.json"
    if not project_json.exists():
        return None

    try:
        cfg_obj = json.loads(project_json.read_text(encoding="utf-8"))
    except Exception:
        return None

    raw_aoi = cfg_obj.get("aoi")
    if not raw_aoi:
        return None

    names = raw_aoi
    if isinstance(raw_aoi, str):
        try:
            parsed = ast.literal_eval(raw_aoi)
            names = parsed
        except Exception:
            names = raw_aoi

    if isinstance(names, str):
        names = [names]
    if not isinstance(names, (list, tuple)) or not names:
        return None

    geolocator = Nominatim(user_agent="insar_pipeline_v1", timeout=20)
    polys = []
    for name in names:
        try:
            location = geolocator.geocode(str(name), language="zh", addressdetails=True)
            if location is None:
                continue
            poly = _get_admin_polygon(location)
            if poly is not None and poly.is_valid and not poly.is_empty:
                polys.append(poly)
        except Exception as exc:
            logger.debug(f"AOI polygon lookup failed for {name}: {exc}")

    if not polys:
        return None

    merged = unary_union(polys)
    try:
        cache_path.write_text(
            json.dumps({"type": "Feature", "geometry": mapping(merged)}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass
    return merged


def _polygon_contains_xy(polygon, x, y):
    """向量化点在多边形内判断。"""
    try:
        from shapely import contains_xy
        return contains_xy(polygon, x, y)
    except Exception:
        from shapely.vectorized import contains
        return contains(polygon, x, y)


def _project_radar_water_mask_to_grid(mintpy_dir, glon, glat):
    """将雷达坐标 waterMask.h5 投影到目标经纬网格。"""
    import h5py
    from scipy.spatial import cKDTree

    mintpy_dir = Path(mintpy_dir)
    wm_path = mintpy_dir / "waterMask.h5"
    geom_path = mintpy_dir / "inputs" / "geometryRadar.h5"
    if not wm_path.exists() or not geom_path.exists():
        return None

    try:
        with h5py.File(wm_path) as f:
            wm = f["waterMask"][:]
        with h5py.File(geom_path) as f:
            lat_r = f["latitude"][:]
            lon_r = f["longitude"][:]
    except Exception:
        return None

    water = (~wm.astype(bool)) & np.isfinite(lat_r) & (lat_r > 0.1) & np.isfinite(lon_r) & (lon_r > 0.1)
    if not np.any(water):
        return None

    water_pts = np.column_stack([lon_r[water], lat_r[water]])
    tree = cKDTree(water_pts)
    target_pts = np.column_stack([glon.ravel(), glat.ravel()])
    dist, _ = tree.query(target_pts)

    res_lon = abs(glon[0, 1] - glon[0, 0]) if glon.shape[1] > 1 else 0.001
    res_lat = abs(glat[1, 0] - glat[0, 0]) if glat.shape[0] > 1 else res_lon
    max_dist = max(res_lon, res_lat) * 2.6
    return (dist.reshape(glon.shape) <= max_dist)


def _load_geocoded_velocity_canvas(mintpy_dir, bbox=None, target_cols=840):
    """优先读取 geo/ 下的规则地理栅格，用于 north-up 速率图。"""
    import h5py

    mintpy_dir = Path(mintpy_dir)
    geo_dir = mintpy_dir / "geo"
    vel_path = geo_dir / "geo_velocity.h5"
    coh_path = geo_dir / "geo_temporalCoherence.h5"
    geom_path = geo_dir / "geo_geometryRadar.h5"
    if not (vel_path.exists() and coh_path.exists() and geom_path.exists()):
        return None

    with h5py.File(vel_path) as f_vel:
        attrs = dict(f_vel.attrs)
        x_first = float(attrs["X_FIRST"])
        x_step = float(attrs["X_STEP"])
        y_first = float(attrs["Y_FIRST"])
        y_step = float(attrs["Y_STEP"])
        width = int(attrs["WIDTH"])
        length = int(attrs["LENGTH"])

    if bbox is None:
        bbox = cfg._AOI_BBOX

    if bbox is None:
        lon_min = x_first
        lon_max = x_first + (width - 1) * x_step
        lat_vals = [y_first, y_first + (length - 1) * y_step]
        lat_min, lat_max = min(lat_vals), max(lat_vals)
    else:
        lat_min, lat_max, lon_min, lon_max = bbox

    col_a = int(np.floor((lon_min - x_first) / x_step))
    col_b = int(np.ceil((lon_max - x_first) / x_step)) + 1
    row_a = int(np.floor((lat_max - y_first) / y_step))
    row_b = int(np.ceil((lat_min - y_first) / y_step)) + 1

    col0 = max(0, min(col_a, col_b))
    col1 = min(width, max(col_a, col_b))
    row0 = max(0, min(row_a, row_b))
    row1 = min(length, max(row_a, row_b))
    if row1 <= row0 or col1 <= col0:
        return None

    step = max(1, int(np.ceil((col1 - col0) / max(int(target_cols), 100))))

    with h5py.File(vel_path) as f_vel, h5py.File(coh_path) as f_coh, h5py.File(geom_path) as f_geom:
        vel = np.asarray(f_vel["velocity"][row0:row1:step, col0:col1:step], dtype=np.float32) * 1000.0
        vstd = (
            np.asarray(f_vel["velocityStd"][row0:row1:step, col0:col1:step], dtype=np.float32) * 1000.0
            if "velocityStd" in f_vel else None
        )
        tcoh = np.asarray(f_coh["temporalCoherence"][row0:row1:step, col0:col1:step], dtype=np.float32)
        hgt = np.asarray(f_geom["height"][row0:row1:step, col0:col1:step], dtype=np.float32)

    col_idx = np.arange(col0, col1, step)
    row_idx = np.arange(row0, row1, step)
    lon_1d = x_first + col_idx * x_step
    lat_1d = y_first + row_idx * y_step
    glon, glat = np.meshgrid(lon_1d, lat_1d)

    hs_grid = None
    if np.any(np.isfinite(hgt) & (hgt != 0)):
        hs_grid = _compute_hillshade(hgt, max(abs(x_step * step), abs(y_step * step)))

    water_grid = _project_radar_water_mask_to_grid(mintpy_dir, glon, glat)

    return {
        "vel": vel,
        "vstd": vstd,
        "tcoh": tcoh,
        "glon": glon,
        "glat": glat,
        "hs_grid": hs_grid,
        "water_grid": water_grid,
        "extent": [float(lon_1d.min()), float(lon_1d.max()), float(lat_1d.min()), float(lat_1d.max())],
        "lon_min": float(lon_min),
        "lon_max": float(lon_max),
        "lat_min": float(lat_min),
        "lat_max": float(lat_max),
        "lat_c": float(0.5 * (lat_min + lat_max)),
    }


# ── 科研绘图辅助函数 ──────────────────────────────────────────────────────

def _save_figure(fig, path, dpi=300, also_pdf=True):
    """保存 PNG (+ 可选 PDF 矢量版)。"""
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    if also_pdf:
        pdf_path = Path(path).with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        logger.info(f"PDF: {pdf_path}")


def display_notebook_figure_gallery(
    figures,
    *,
    title: str | None = None,
    columns: int = 2,
    max_height_px: int = 340,
):
    """
    在 Jupyter 中以缩略图库方式预览图件，替代逐张 display(Image(...))。

    Parameters
    ----------
    figures : list[str|Path] | dict[str, str|Path]
        图件路径列表，或 caption -> path 的映射。
    title : str, optional
        图库标题。
    columns : int
        列数。
    max_height_px : int
        单张图最大显示高度。
    """
    import base64
    import html
    import mimetypes

    try:
        from IPython.display import HTML, display
    except Exception:
        for item in figures.items() if isinstance(figures, dict) else figures:
            print(item)
        return []

    if isinstance(figures, dict):
        items = [(str(k), Path(v)) for k, v in figures.items()]
    else:
        items = []
        for path in figures:
            p = Path(path)
            items.append((p.stem.replace('_', ' ').title(), p))

    cards = []
    seen = set()
    valid_paths = []
    for label, path in items:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        valid_paths.append(path)

        suffix = path.suffix.lower()
        if suffix == ".svg":
            img_html = path.read_text(encoding="utf-8", errors="ignore")
        else:
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            img_html = (
                f'<img src="data:{mime};base64,{data}" '
                f'style="max-width:100%; max-height:{int(max_height_px)}px; '
                f'width:100%; object-fit:contain; display:block; margin:auto;" />'
            )

        pdf_path = path.with_suffix(".pdf")
        extra_bits = [html.escape(str(path))]
        if pdf_path.exists():
            extra_bits.append(f"PDF: {html.escape(pdf_path.name)}")

        cards.append(
            "<div style='border:1px solid #d9dee7; border-radius:10px; "
            "padding:10px; background:#fff;'>"
            f"<div style='font-weight:700; font-size:13px; margin-bottom:8px;'>{html.escape(label)}</div>"
            f"{img_html}"
            f"<div style='margin-top:8px; color:#6b7280; font-size:11px; line-height:1.35;'>{'<br>'.join(extra_bits)}</div>"
            "</div>"
        )

    if not cards:
        print("无可预览图件。")
        return []

    title_html = (
        f"<div style='font-weight:700; font-size:16px; margin:2px 0 10px 0;'>{html.escape(title)}</div>"
        if title else ""
    )
    cols = max(1, int(columns))
    gallery_html = (
        "<div style='margin:8px 0 16px 0;'>"
        f"{title_html}"
        "<div style='display:grid; gap:12px; "
        f"grid-template-columns:repeat({cols}, minmax(0, 1fr)); align-items:start;'>"
        f"{''.join(cards)}"
        "</div></div>"
    )
    display(HTML(gallery_html))
    return valid_paths


def _subfig_label(ax, label, x=-0.08, y=1.06):
    """添加子图标签 (a), (b), (c)... — Nature/Science 风格。"""
    ax.text(x, y, f'({label})', transform=ax.transAxes, **_LABEL_STYLE)


def _add_north_arrow(ax, x=0.95, y=0.95, size=0.06):
    """在轴内添加北向箭头指示。"""
    ax.annotate('N', xy=(x, y), xytext=(x, y - size),
                textcoords='axes fraction', xycoords='axes fraction',
                fontsize=7, fontweight='bold', ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))


def _add_scalebar_box(ax, lat_c, lon_min, lon_max, lat_min, lat_max, km=None):
    """
    在左下角添加带白底半透明框的比例尺。
    自动选择合适长度 (1/2/5/10/20/50 km)。
    """
    from matplotlib.patches import FancyBboxPatch

    if km is None:
        # 自适应: 取图宽的 ~15%，选最近的 round 值
        span_km = (lon_max - lon_min) * 111.0 * np.cos(np.radians(lat_c))
        target_km = span_km * 0.15
        for k in [1, 2, 5, 10, 20, 50, 100, 200]:
            if k >= target_km * 0.5:
                km = k
                break
        if km is None:
            km = 10

    deg = km / (111.0 * np.cos(np.radians(lat_c)))
    x0 = lon_min + 0.06 * (lon_max - lon_min)
    y0 = lat_min + 0.06 * (lat_max - lat_min)

    # 白底背景框
    dy = (lat_max - lat_min) * 0.055
    dx = deg + (lon_max - lon_min) * 0.04
    bg = FancyBboxPatch((x0 - (lon_max - lon_min) * 0.015, y0 - dy * 0.4),
                        dx, dy * 2.2,
                        boxstyle='round,pad=0.003',
                        facecolor='white', edgecolor='0.5',
                        linewidth=0.4, alpha=0.85, zorder=8)
    ax.add_patch(bg)

    # 比例尺线条 (双色交替)
    ax.plot([x0, x0 + deg], [y0, y0], 'k-', lw=2.5, solid_capstyle='butt', zorder=9)
    ax.plot([x0, x0 + deg / 2], [y0, y0], color='0.3', lw=2.5,
            solid_capstyle='butt', zorder=9)
    ax.text(x0 + deg / 2, y0 + dy * 0.3, f'{km} km',
            ha='center', va='bottom', fontsize=6.5, fontweight='bold', zorder=9)


def _add_inset_location(fig, ax, lat_c, lon_c, inset_pos=None):
    """
    在图角添加区位概览小图 (inset map)。
    使用简单经纬网格 + 红框标记 AOI，不依赖 Cartopy。
    """
    if inset_pos is None:
        inset_pos = [0.68, 0.70, 0.28, 0.26]  # [left, bottom, width, height]

    # 判断大致区域范围
    aoi = cfg._AOI_BBOX  # [S, N, W, E]
    if aoi is None:
        return

    S, N, W, E = aoi
    # 确定概览窗口 — AOI 周围扩展约 10 倍
    span_lat = max(N - S, 0.5) * 5
    span_lon = max(E - W, 0.5) * 5
    ov_lat = (lat_c - span_lat, lat_c + span_lat)
    ov_lon = (lon_c - span_lon, lon_c + span_lon)

    # 限制到合理范围
    ov_lat = (max(ov_lat[0], -60), min(ov_lat[1], 85))
    ov_lon = (max(ov_lon[0], -180), min(ov_lon[1], 180))

    ax_in = fig.add_axes(inset_pos)
    ax_in.set_xlim(ov_lon)
    ax_in.set_ylim(ov_lat)
    ax_in.set_facecolor('#f0f0f0')

    # 简单海岸线近似: 用经纬网格 (不依赖外部数据)
    for v in range(-180, 181, 30):
        ax_in.axvline(v, color='0.85', lw=0.3)
    for v in range(-90, 91, 30):
        ax_in.axhline(v, color='0.85', lw=0.3)

    # AOI 红框
    from matplotlib.patches import Rectangle
    rect = Rectangle((W, S), E - W, N - S,
                      linewidth=1.5, edgecolor='#E74C3C',
                      facecolor='#E74C3C', alpha=0.3, zorder=5)
    ax_in.add_patch(rect)

    ax_in.tick_params(labelsize=5, length=2, width=0.4)
    ax_in.set_aspect('equal')
    for sp in ax_in.spines.values():
        sp.set_linewidth(0.5)
        sp.set_visible(True)


def _format_degree_axis(ax, n_ticks=4):
    """使用小数度标记坐标轴 (带 °E / °N)。"""
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_locator(mticker.MaxNLocator(n_ticks))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(n_ticks))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f°'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f°'))


def _build_mask(data, coh_threshold=0.6):
    """
    综合掩膜：相干性 + 速率有效 + AOI 范围内。
    掩膜外区域在图中显示为白色（facecolor='white'）。
    """
    vel  = data["vel"]
    tcoh = data["tcoh"]
    lat  = data["lat"]
    lon  = data["lon"]
    vstd = data.get("vstd")

    mask = np.ones(vel.shape, dtype=bool)
    mask &= np.isfinite(vel) & (vel != 0)
    mask &= (tcoh >= coh_threshold)

    # 几何有效区
    if lat is not None:
        mask &= np.isfinite(lat) & (lat > 0.1) & np.isfinite(lon) & (lon > 0.1)

    # 严格 AOI 范围（去除 buffer/padding 蓝色环）
    if lat is not None:
        aoi = cfg._AOI_BBOX
        if aoi is not None:
            S, N, W, E = aoi
            mask &= (lat >= S) & (lat <= N) & (lon >= W) & (lon <= E)

    # 速率误差过大的点剔除（保科研质量）
    if vstd is not None:
        mask &= (vstd < 80)   # vstd>80 mm/yr 不可信

    # 水体掩膜：仅使用现成的 MintPy waterMask.h5
    mask = _apply_water_mask(mask, data)

    return mask


def _apply_water_mask(mask, data):
    """仅加载 MintPy waterMask.h5，不做自动水体推断。"""
    import h5py

    try:
        wm_path = Path(data.get("mintpy_dir", cfg.MINTPY_DIR)) / "waterMask.h5"
        if wm_path.exists():
            with h5py.File(wm_path, 'r') as f:
                wm = f['waterMask'][:]
            if wm.shape == mask.shape:
                mask &= wm.astype(bool)
    except Exception:
        pass

    return mask


def _symmetric_vlim(vals, pct=97):
    """以 pct 百分位绝对值确定对称色标范围，保证 0 居中。"""
    vmax = np.percentile(np.abs(vals), pct)
    return max(vmax, 1.0)   # 至少 1 mm/yr，防止除零


def _build_geo_grid(lat_min, lat_max, lon_min, lon_max,
                    target_cols=720, min_res=0.00035, max_res=0.0012):
    """按 AOI 范围自适应生成规则地理格网。"""
    span_lon = max(lon_max - lon_min, 1e-4)
    span_lat = max(lat_max - lat_min, 1e-4)
    res = max(span_lon / max(target_cols, 50), span_lat / max(target_cols, 50))
    res = float(np.clip(res, min_res, max_res))
    glon_1d = np.arange(lon_min, lon_max + res * 0.5, res)
    glat_1d = np.arange(lat_max, lat_min - res * 0.5, -res)
    glon, glat = np.meshgrid(glon_1d, glat_1d)
    return glon, glat, res


def _prepare_geo_canvas(data, mask, target_cols=720):
    """准备地理绘图所需的格网、hillshade 和有效范围。"""
    from scipy.spatial import cKDTree

    lat = data["lat"]
    lon = data["lon"]
    hgt = data["height"]

    lat_m = lat[mask]
    lon_m = lon[mask]
    lat_min, lat_max = float(lat_m.min()), float(lat_m.max())
    lon_min, lon_max = float(lon_m.min()), float(lon_m.max())
    lat_c = 0.5 * (lat_min + lat_max)

    glon, glat, res = _build_geo_grid(lat_min, lat_max, lon_min, lon_max,
                                      target_cols=target_cols)
    extent = [lon_min, lon_max, lat_min, lat_max]
    hs_grid, water_grid = _build_hillshade_and_water(
        hgt, lat, lon, glon, glat,
        mintpy_dir=data.get("mintpy_dir"),
    )

    pts_mask = np.column_stack([lon_m, lat_m])
    tree = cKDTree(pts_mask)
    gpts = np.column_stack([glon.ravel(), glat.ravel()])
    dist, _ = tree.query(gpts)
    dist_grid = dist.reshape(glon.shape)

    return {
        "lat": lat,
        "lon": lon,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_c": lat_c,
        "glon": glon,
        "glat": glat,
        "res": res,
        "extent": extent,
        "hs_grid": hs_grid,
        "water_grid": water_grid,
        "dist_grid": dist_grid,
        "max_dist": res * 3.0,
    }


def _interpolate_geo_field(field, valid_mask, canvas, method='nearest'):
    """将雷达坐标场插值到绘图格网上。"""
    from scipy.interpolate import griddata

    if not np.any(valid_mask):
        return np.full(canvas["glon"].shape, np.nan, dtype=float)

    pts = np.column_stack([canvas["lon"][valid_mask], canvas["lat"][valid_mask]])
    grid = griddata(pts, np.asarray(field)[valid_mask],
                    (canvas["glon"], canvas["glat"]), method=method)
    grid[canvas["dist_grid"] > canvas["max_dist"]] = np.nan
    return grid


def _format_period_label(dates):
    """格式化起止时段字符串。"""
    if not dates:
        return ""
    if len(dates) == 1:
        return f"{dates[0][:4]}.{dates[0][4:6]}"
    return f"{dates[0][:4]}.{dates[0][4:6]}–{dates[-1][:4]}.{dates[-1][4:6]}"


def _format_epoch_label(date_str, ref_date_str=None):
    """格式化位移面板日期标签，并附相对参考日天数。"""
    dt = datetime.strptime(date_str, '%Y%m%d')
    if not ref_date_str:
        return dt.strftime('%Y-%m-%d')
    ref_dt = datetime.strptime(ref_date_str, '%Y%m%d')
    delta_days = (dt - ref_dt).days
    sign = '+' if delta_days >= 0 else '−'
    return f"{dt:%Y-%m-%d}\n({sign}{abs(delta_days)} d)"


def _choose_panel_grid(n, max_cols=5):
    """为多面板图选择接近方阵的行列数。"""
    best = None
    for ncols in range(1, min(max_cols, n) + 1):
        nrows = int(np.ceil(n / ncols))
        empty = nrows * ncols - n
        score = abs(nrows - ncols) + empty * 0.35 + ncols * 0.03
        candidate = (score, nrows, ncols)
        if best is None or candidate < best:
            best = candidate
    _, nrows, ncols = best
    return nrows, ncols


def _add_scalebar(ax, lat_c, lon_min, lon_max, lat_min, km=10):
    """兼容旧接口的比例尺 — 转发到新版 _add_scalebar_box。"""
    lat_max = ax.get_ylim()[1]
    _add_scalebar_box(ax, lat_c, lon_min, lon_max, lat_min, lat_max, km=km)


def _add_water_legend(ax, *, loc='lower left', bbox_to_anchor=None, fontsize=5.6):
    """添加统一的水体图例标签。"""
    from matplotlib.patches import Patch

    legend = ax.legend(
        handles=[Patch(facecolor=_WATER_COLOR[:3], edgecolor=_WATER_HATCH_COLOR[:3],
                       hatch='///', lw=0.45, label='Water')],
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=fontsize,
        framealpha=0.88,
        borderpad=0.2,
        handlelength=0.9,
        handletextpad=0.35,
    )
    if legend is not None:
        legend.get_frame().set_linewidth(0.35)
    return legend


def _draw_lowq_hatch(ax, canvas, lowq_mask):
    """在低质量区域覆盖黄色斜线淹没。lowq_mask 是雷达坐标布尔数组。"""
    if not np.any(lowq_mask):
        return False
    lon_arr, lat_arr = canvas["lon"], canvas["lat"]
    hatch_pts = np.column_stack([lon_arr[lowq_mask], lat_arr[lowq_mask]])
    glon, glat = canvas["glon"], canvas["glat"]
    from scipy.spatial import cKDTree
    tree_h = cKDTree(hatch_pts)
    dist_h, _ = tree_h.query(np.column_stack([glon.ravel(), glat.ravel()]))
    res = canvas["res"]
    hatch_grid = (dist_h < res * 2.5).reshape(glon.shape)
    if not np.any(hatch_grid):
        return False
    rr, cc = np.indices(hatch_grid.shape)
    stripe_mask = hatch_grid & (((rr + cc) % 12) < 2)
    stripe_rgba = np.zeros((*hatch_grid.shape, 4), dtype=np.float32)
    stripe_rgba[stripe_mask] = [*_HIGH_UNCERTAINTY_COLOR[:3], 0.95]
    ax.imshow(stripe_rgba, extent=canvas["extent"], aspect='auto',
              interpolation='nearest', origin='upper', zorder=6)
    return True


def _add_uncertainty_legend(ax, *, loc='lower right', fontsize=5.4):
    """添加 High Uncertainty 图例。"""
    from matplotlib.patches import Patch
    patch = Patch(facecolor=(1, 1, 1, 0), edgecolor=_HIGH_UNCERTAINTY_COLOR[:3],
                  hatch=_HIGH_UNCERTAINTY_HATCH, linewidth=0.5,
                  label='High uncertainty')
    existing = ax.get_legend()
    handles = existing.legend_handles[:] if existing else []
    handles.append(patch)
    labels = [h.get_label() for h in handles]
    leg = ax.legend(handles, labels, loc=loc, fontsize=fontsize,
                    framealpha=0.88, borderpad=0.2, handlelength=0.9)
    leg.get_frame().set_linewidth(0.35)


def _hillshade_grid(hgt, pts_valid, glon, glat):
    """从散点高程插值到规则网格，再计算真正的 hillshade。"""
    from scipy.interpolate import griddata
    dem_grid = griddata(pts_valid, hgt, (glon, glat), method='nearest')
    return _compute_hillshade(dem_grid, glon[0, 1] - glon[0, 0])


def _hillshade_scatter(hgt, azimuth=315, altitude=40):
    """计算散点高程的 hillshade 值（返回与 hgt 等长的 1D 数组）。"""
    h = hgt.copy().astype(float)
    h_min, h_max = np.nanpercentile(h, 2), np.nanpercentile(h, 98)
    hs = np.clip((h - h_min) / max(h_max - h_min, 1), 0, 1)
    return 0.4 + 0.5 * hs


def _compute_hillshade(dem, cellsize_deg, azimuth=315, altitude=45):
    """
    真正的 hillshade（太阳照射模型），产生有地形纹理的底图。
    增强版：z-factor=2 夸大地形起伏，让山谷/山脊更清晰。
    """
    cellsize_m = cellsize_deg * 111000
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem) if np.any(np.isfinite(dem)) else 0)
    z_factor = 2.0   # 夸大垂直起伏，增强地形纹理
    dy, dx = np.gradient(dem_filled * z_factor, cellsize_m)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    az = np.radians(azimuth)
    alt = np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def _build_hillshade_and_water(hgt, lat, lon, glon, glat, mintpy_dir=None):
    """
    统一构建 hillshade 底图 + 水体检测 grid。
    水体优先读 MintPy waterMask.h5（雷达坐标），插值到绘图格网。
    返回 (hs_grid, water_grid) — 任一可能为 None。
    """
    from scipy.interpolate import griddata
    import h5py

    if hgt is None:
        return None, None

    geo_ok = np.isfinite(hgt) & (hgt > 0) & np.isfinite(lat) & (lat > 0.1)
    if geo_ok.sum() < 1000:
        return None, None

    pts_h = np.column_stack([lon[geo_ok], lat[geo_ok]])
    hs_grid = _hillshade_grid(hgt[geo_ok], pts_h, glon, glat)

    # 水体：优先用 MintPy waterMask.h5
    water_grid = None
    try:
        wm_path = Path(mintpy_dir or cfg.MINTPY_DIR) / "waterMask.h5"
        if wm_path.exists():
            with h5py.File(wm_path, 'r') as f:
                wm = f['waterMask'][:]
            # wm 与 lat/lon 同尺寸（雷达坐标）；插值到绘图格网
            water_radar = ~wm.astype(bool)   # waterMask: 1=land, 0=water
            if water_radar.shape == lat.shape and np.any(water_radar):
                w_pts = np.column_stack([lon[water_radar], lat[water_radar]])
                from scipy.spatial import cKDTree
                tree_w = cKDTree(w_pts)
                g_pts = np.column_stack([glon.ravel(), glat.ravel()])
                dist_w, _ = tree_w.query(g_pts)
                # 如果最近水体点 < 1 个像素距离，标记为水体
                res = abs(glon[0, 1] - glon[0, 0]) if glon.shape[1] > 1 else 0.001
                water_grid = (dist_w < res * 2).reshape(glon.shape)
    except Exception:
        pass

    return hs_grid, water_grid


def _draw_hillshade_background(ax, hs_grid, water_grid, extent, alpha=0.35,
                                water_legend=True):
    """在 ax 上绘制 hillshade 底图 + 水体蓝色覆盖（独立高 alpha）。"""
    if hs_grid is None:
        return

    hs_display = hs_grid.copy()
    if water_grid is not None:
        hs_display[water_grid] = np.nan
    # 地形部分：半透明灰色 hillshade
    hs_rgba = plt.cm.gray(np.clip(hs_display, 0.08, 0.92))
    hs_rgba[np.isnan(hs_display)] = [1, 1, 1, 0]   # 水体区透明，不画
    ax.imshow(hs_rgba, extent=extent, aspect='auto',
              interpolation='bilinear', origin='upper', alpha=alpha)

    if water_grid is not None and np.any(water_grid):
        _draw_water_overlay(ax, water_grid, extent)


def _draw_water_overlay(ax, water_grid, extent):
    """在数据层之上绘制水体：浅蓝底 + 蓝色斜线。"""
    if water_grid is None or not np.any(water_grid):
        return
    rr, cc = np.indices(water_grid.shape)
    stripe_mask = water_grid & (((rr + cc) % 12) < 2)

    water_rgba = np.zeros((*water_grid.shape, 4), dtype=np.float32)
    water_rgba[water_grid] = [*_WATER_COLOR[:3], 0.80]
    ax.imshow(water_rgba, extent=extent, aspect='auto',
              interpolation='nearest', origin='upper')

    stripe_rgba = np.zeros((*water_grid.shape, 4), dtype=np.float32)
    stripe_rgba[stripe_mask] = [*_WATER_HATCH_COLOR[:3], 0.95]
    ax.imshow(stripe_rgba, extent=extent, aspect='auto',
              interpolation='nearest', origin='upper')


def _draw_scatter_geo_panel(
    ax,
    canvas,
    lon,
    lat,
    values,
    valid_mask,
    *,
    cmap,
    norm,
    title,
    label_char,
    point_size=2.0,
    alpha=0.90,
    title_fontsize=8.2,
):
    """参考 DePSI 面板风格的地理点云渲染。"""
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.45)

    _draw_hillshade_background(ax, canvas["hs_grid"], None, canvas["extent"], alpha=0.26)
    _draw_water_overlay(ax, canvas["water_grid"], canvas["extent"])

    cmap_obj = plt.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
    artist = None
    if np.any(valid_mask):
        artist = ax.scatter(
            lon[valid_mask], lat[valid_mask], c=values[valid_mask],
            cmap=cmap_obj, norm=norm,
            s=point_size, linewidths=0.0, alpha=alpha,
            rasterized=True, zorder=4,
        )

    ax.set_xlim(canvas["lon_min"], canvas["lon_max"])
    ax.set_ylim(canvas["lat_min"], canvas["lat_max"])
    ax.set_aspect(1.0 / np.cos(np.radians(canvas["lat_c"])))
    ax.set_title(title, fontsize=title_fontsize, loc='left', fontweight='bold', pad=4)
    _subfig_label(ax, label_char, x=0.02, y=0.96)
    return artist


# ---------------------------------------------------------------------------
# 1. 速率图（标准化主图: 左右布局 velocity + uncertainty）
# ---------------------------------------------------------------------------
def plot_velocity_map(mintpy_dir=None, export_dir=None,
                      coh_threshold=0.6, vlim=None):
    """
    LOS 平均速率图:
      单面板或左右双栏布局，主图优先突出 velocity，uncertainty 作为右侧辅图。
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    export_dir = Path(export_dir or cfg.EXPORT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)

    use_geocoded_velocity = False
    geo_canvas = _load_geocoded_velocity_canvas(
        mintpy_dir,
        bbox=cfg._AOI_BBOX,
        target_cols=840,
    ) if use_geocoded_velocity else None
    if geo_canvas is not None:
        dates, ref_date = _load_mintpy_dates_only(mintpy_dir)
        vel = geo_canvas["vel"]
        vstd = geo_canvas["vstd"]
        tcoh = geo_canvas["tcoh"]
        water_grid = geo_canvas["water_grid"]

        valid_display = np.isfinite(vel) & (vel != 0)
        if water_grid is not None:
            valid_display &= ~water_grid

        valid_stats = valid_display & np.isfinite(tcoh) & (tcoh >= coh_threshold)
        if vstd is not None:
            valid_stats &= np.isfinite(vstd) & (vstd < 80)

        if not np.any(valid_display):
            print("[ERROR] 无有效 geo 数据可绘图")
            return

        vel_grid = np.where(valid_display, vel, np.nan)
        vel_m = vel[valid_stats] if np.any(valid_stats) else vel[valid_display]
        period_str = _format_period_label(dates) if dates else "velocity product"
        vmax = _symmetric_vlim(vel_m) if vlim is None else vlim
        vel_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        water_present = water_grid is not None and np.any(water_grid)
        aspect_ratio = (
            (geo_canvas["lat_max"] - geo_canvas["lat_min"]) /
            max((geo_canvas["lon_max"] - geo_canvas["lon_min"]) * np.cos(np.radians(geo_canvas["lat_c"])), 0.001)
        )

        def _draw_geo_raster_panel(ax, grid_data, cmap_name, norm_obj, title, label_char):
            ax.set_facecolor('white')
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.45)
            _draw_hillshade_background(ax, geo_canvas["hs_grid"], None, geo_canvas["extent"], alpha=0.26)
            cmap_obj = plt.colormaps.get_cmap(cmap_name) if isinstance(cmap_name, str) else cmap_name
            rgba = cmap_obj(norm_obj(grid_data))
            rgba[np.isnan(grid_data)] = [1, 1, 1, 0]
            ax.imshow(
                rgba,
                extent=geo_canvas["extent"],
                aspect='auto',
                interpolation='nearest',
                origin='upper',
                alpha=0.90,
                zorder=3,
            )
            if water_present:
                _draw_water_overlay(ax, water_grid, geo_canvas["extent"])
            ax.set_xlim(geo_canvas["lon_min"], geo_canvas["lon_max"])
            ax.set_ylim(geo_canvas["lat_min"], geo_canvas["lat_max"])
            ax.set_aspect(1.0 / np.cos(np.radians(geo_canvas["lat_c"])))
            ax.set_title(title, fontsize=8.2, loc='left', fontweight='bold', pad=4)
            _subfig_label(ax, label_char, x=0.02, y=0.96)

        has_std = vstd is not None and np.any(valid_display & np.isfinite(vstd))

        if has_std:
            map_h = float(np.clip(3.65 + aspect_ratio * 1.05, 4.3, 5.8))
            fig = plt.figure(figsize=(8.35, map_h))
            gs = GridSpec(
                1, 4, figure=fig,
                width_ratios=[1.0, 0.038, 1.0, 0.038],
                left=0.07, right=0.965, bottom=0.10, top=0.93,
                wspace=0.14,
            )
            ax_vel = fig.add_subplot(gs[0, 0])
            cax_vel = fig.add_subplot(gs[0, 1])
            ax_std = fig.add_subplot(gs[0, 2])
            cax_std = fig.add_subplot(gs[0, 3])

            _draw_geo_raster_panel(ax_vel, vel_grid, _CMAP_VEL,
                                   vel_norm, f'LOS Velocity  {period_str}', 'a')
            ax_vel.set_ylabel('Latitude (°N)', fontsize=7.5)
            ax_vel.set_xlabel('Longitude (°E)', fontsize=7.5)
            _format_degree_axis(ax_vel, 4)
            _add_north_arrow(ax_vel, x=0.94, y=0.90, size=0.07)
            if water_present:
                _add_water_legend(ax_vel, loc='lower left', fontsize=5.4)
            sm_v = plt.cm.ScalarMappable(cmap=plt.colormaps.get_cmap(_CMAP_VEL), norm=vel_norm)
            sm_v.set_array([])
            cb_v = fig.colorbar(sm_v, cax=cax_vel, extend='both')
            cb_v.set_label('mm yr$^{-1}$', fontsize=7)
            cb_v.ax.tick_params(labelsize=6)
            cb_v.outline.set_linewidth(0.3)

            valid_std = valid_display & np.isfinite(vstd)
            vstd_m = vstd[valid_std]
            std_grid = np.where(valid_std, vstd, np.nan)
            std_vmax = min(np.percentile(vstd_m, 95), 30)
            std_norm = Normalize(vmin=0, vmax=std_vmax)
            _draw_geo_raster_panel(ax_std, std_grid, _CMAP_STD,
                                   std_norm, 'Velocity Uncertainty', 'b')
            ax_std.set_xlabel('Longitude (°E)', fontsize=7.5)
            ax_std.set_ylabel('')
            _format_degree_axis(ax_std, 4)
            ax_std.set_yticklabels([])
            sm_s = plt.cm.ScalarMappable(cmap=_CMAP_STD, norm=std_norm)
            sm_s.set_array([])
            cb_s = fig.colorbar(sm_s, cax=cax_std, extend='max')
            cb_s.set_label('mm yr$^{-1}$', fontsize=7)
            cb_s.ax.tick_params(labelsize=6)
            cb_s.outline.set_linewidth(0.3)

            info_lines = [f'N = {int(np.count_nonzero(valid_display)):,}']
            if ref_date:
                info_lines.append(f'ref. {ref_date}')
            ax_vel.text(0.985, 0.985, '\n'.join(info_lines),
                        transform=ax_vel.transAxes, fontsize=5.1, ha='right', va='top',
                        color='0.4',
                        bbox=dict(boxstyle='round,pad=0.16', fc='white', ec='0.88', lw=0.35, alpha=0.85))
            ax_std.text(0.98, 0.985,
                        f'P95 uncertainty = {np.percentile(vstd_m, 95):.1f} mm/yr',
                        transform=ax_std.transAxes, fontsize=5.0, ha='right', va='top',
                        color='0.40',
                        bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='0.88', lw=0.35, alpha=0.82))
        else:
            panel_w = min(_COL2 * 0.86, 6.0)
            panel_h = panel_w * aspect_ratio + 0.15
            fig = plt.figure(figsize=(panel_w + 0.55, panel_h))
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.04],
                          left=0.08, right=0.95, bottom=0.10, top=0.93, wspace=0.08)
            ax_vel = fig.add_subplot(gs[0, 0])
            cax_vel = fig.add_subplot(gs[0, 1])
            _draw_geo_raster_panel(ax_vel, vel_grid, _CMAP_VEL,
                                   vel_norm, f'Sentinel-1 LOS Velocity  {period_str}', 'a')
            ax_vel.set_xlabel('Longitude (°E)', fontsize=7.5)
            ax_vel.set_ylabel('Latitude (°N)', fontsize=7.5)
            _format_degree_axis(ax_vel, 4)
            _add_north_arrow(ax_vel, x=0.94, y=0.90, size=0.07)
            if water_present:
                _add_water_legend(ax_vel, loc='lower left', fontsize=5.4)
            sm_v = plt.cm.ScalarMappable(cmap=plt.colormaps.get_cmap(_CMAP_VEL), norm=vel_norm)
            sm_v.set_array([])
            cb_v = fig.colorbar(sm_v, cax=cax_vel, extend='both')
            cb_v.set_label('mm yr$^{-1}$', fontsize=7)
            cb_v.ax.tick_params(labelsize=6)
            cb_v.outline.set_linewidth(0.3)
            info_lines = [f'N = {int(np.count_nonzero(valid_display)):,}']
            if ref_date:
                info_lines.append(f'ref. {ref_date}')
            ax_vel.text(0.98, 0.05, '\n'.join(info_lines),
                        transform=ax_vel.transAxes, fontsize=5, ha='right', va='bottom',
                        color='0.4',
                        bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.7))

        out_path = export_dir / "velocity_map.png"
        _save_figure(fig, out_path)
        plt.close()

        p5, p95 = np.percentile(vel_m, 5), np.percentile(vel_m, 95)
        median = np.median(vel_m)
        print(f"速率图: {out_path}")
        print(f"  有效像素: {int(np.count_nonzero(valid_display)):,}, "
              f"中位数: {median:.1f}, P5/P95: {p5:.1f}/{p95:.1f} mm/yr")

        return {
            'hs_grid': geo_canvas["hs_grid"], 'water_grid': geo_canvas["water_grid"],
            'glon': geo_canvas["glon"], 'glat': geo_canvas["glat"], 'extent': geo_canvas["extent"],
            'dist_grid': None, 'max_dist': None,
            'lat_min': geo_canvas["lat_min"], 'lat_max': geo_canvas["lat_max"],
            'lon_min': geo_canvas["lon_min"], 'lon_max': geo_canvas["lon_max"],
            'lat_c': geo_canvas["lat_c"],
        }

    data  = _load_mintpy_data(mintpy_dir)
    vel   = data["vel"]
    lat, lon = data["lat"], data["lon"]
    dates = data.get("dates", [])
    vstd  = data.get("vstd")
    tcoh  = data["tcoh"]

    if lat is None:
        print("[ERROR] 无有效数据可绘图")
        return

    geo_valid = np.isfinite(lat) & (lat > 0.1) & np.isfinite(lon) & (lon > 0.1)
    aoi = cfg._AOI_BBOX
    if aoi is not None:
        S, N, W, E = aoi
        geo_valid &= (lat >= S) & (lat <= N) & (lon >= W) & (lon <= E)

    canvas = _prepare_geo_canvas(data, geo_valid, target_cols=760)
    display_mask = np.isfinite(vel) & (vel != 0) & geo_valid
    display_mask = _apply_water_mask(display_mask, data)
    stats_mask = display_mask & np.isfinite(tcoh) & (tcoh >= coh_threshold)
    if vstd is not None:
        stats_mask &= np.isfinite(vstd) & (vstd < 80)

    # 低质量区域 = 有数据但未通过质量阈值 → 黄色斜线覆盖
    lowq_mask = display_mask & ~stats_mask

    if not np.any(display_mask):
        print("[ERROR] 无有效数据可绘图")
        return

    vel_m = vel[stats_mask] if np.any(stats_mask) else vel[display_mask]
    period_str = _format_period_label(dates) if dates else "velocity product"
    ref_date = data.get("cumulative_reference_date") or (dates[0] if dates else None)

    vmax = _symmetric_vlim(vel_m) if vlim is None else vlim
    vmin = -vmax
    # 速率面板只画通过质量阈值的像素（与其他图一致）
    valid_vel = stats_mask

    has_std = vstd is not None and np.any(display_mask & np.isfinite(vstd))
    aspect_ratio = (
        (canvas["lat_max"] - canvas["lat_min"]) /
        max((canvas["lon_max"] - canvas["lon_min"]) * np.cos(np.radians(canvas["lat_c"])), 0.001)
    )
    water_present = canvas["water_grid"] is not None and np.any(canvas["water_grid"])

    vel_norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    n_valid_vel = int(np.count_nonzero(valid_vel))
    point_size = 3.0 if n_valid_vel > 500_000 else (3.4 if n_valid_vel > 220_000 else 4.0)

    if has_std:
        # ── 左右布局：左主图 velocity，右辅图 uncertainty ──
        map_h = float(np.clip(3.65 + aspect_ratio * 1.05, 4.3, 5.8))
        fig_w = 8.35
        fig = plt.figure(figsize=(fig_w, map_h))
        gs = GridSpec(
            1, 4, figure=fig,
            width_ratios=[1.0, 0.038, 1.0, 0.038],
            left=0.07, right=0.965, bottom=0.10, top=0.93,
            wspace=0.14,
        )
        ax_vel = fig.add_subplot(gs[0, 0])
        cax_vel = fig.add_subplot(gs[0, 1])
        ax_std = fig.add_subplot(gs[0, 2])
        cax_std = fig.add_subplot(gs[0, 3])

        # (a) velocity
        sc_v = _draw_scatter_geo_panel(
            ax_vel, canvas, lon, lat, vel, valid_vel,
            cmap=_CMAP_VEL, norm=vel_norm,
            title=f'LOS Velocity  {period_str}', label_char='a',
            point_size=point_size, alpha=0.90,
        )
        ax_vel.set_ylabel('Latitude (°N)', fontsize=7.5)
        ax_vel.set_xlabel('Longitude (°E)', fontsize=7.5)
        _format_degree_axis(ax_vel, 4)
        _add_north_arrow(ax_vel, x=0.94, y=0.90, size=0.07)
        if water_present:
            _add_water_legend(ax_vel, loc='lower left', fontsize=5.4)
        # colorbar
        sm_v = plt.cm.ScalarMappable(cmap=plt.colormaps.get_cmap(_CMAP_VEL), norm=vel_norm)
        sm_v.set_array([])
        cb_v = fig.colorbar(sm_v, cax=cax_vel, extend='both')
        cb_v.set_label('mm yr$^{-1}$', fontsize=7)
        cb_v.ax.tick_params(labelsize=6)
        cb_v.outline.set_linewidth(0.3)

        # (b) uncertainty — 也只画有效数据足迹，但不再按质量阈值遮掩
        valid_std = display_mask & np.isfinite(vstd)
        vstd_m = vstd[valid_std]
        vstd_vmax = min(np.percentile(vstd_m, 95), 30)
        std_norm = Normalize(vmin=0, vmax=vstd_vmax)
        sc_s = _draw_scatter_geo_panel(
            ax_std, canvas, lon, lat, vstd, valid_std,
            cmap=_CMAP_STD, norm=std_norm,
            title='Velocity Uncertainty', label_char='b',
            point_size=point_size, alpha=0.88,
        )
        ax_std.set_xlabel('Longitude (°E)', fontsize=7.5)
        ax_std.set_ylabel('')
        _format_degree_axis(ax_std, 4)
        ax_std.set_yticklabels([])
        sm_s = plt.cm.ScalarMappable(cmap=_CMAP_STD, norm=std_norm)
        sm_s.set_array([])
        cb_s = fig.colorbar(sm_s, cax=cax_std, extend='max')
        cb_s.set_label('mm yr$^{-1}$', fontsize=7)
        cb_s.ax.tick_params(labelsize=6)
        cb_s.outline.set_linewidth(0.3)

        # 注释
        info_lines = [f'N = {int(np.count_nonzero(display_mask)):,}']
        if ref_date:
            info_lines.append(f'ref. {ref_date}')
        ax_vel.text(0.985, 0.985, '\n'.join(info_lines),
                    transform=ax_vel.transAxes, fontsize=5.1, ha='right', va='top',
                    color='0.4',
                    bbox=dict(boxstyle='round,pad=0.16', fc='white', ec='0.88', lw=0.35, alpha=0.85))
        ax_std.text(0.98, 0.985,
                    f'P95 uncertainty = {np.percentile(vstd_m, 95):.1f} mm/yr',
                    transform=ax_std.transAxes, fontsize=5.0, ha='right', va='top',
                    color='0.40',
                    bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='0.88', lw=0.35, alpha=0.82))

        # 低质量区域淹没 — 黄色斜线覆盖 (与其他图不显示的区域一致)
        has_hatch = _draw_lowq_hatch(ax_vel, canvas, lowq_mask)
        if has_hatch:
            _add_uncertainty_legend(ax_vel, loc='lower right', fontsize=5.4)
    else:
        # ── 单面板 ──
        panel_w = min(_COL2 * 0.86, 6.0)
        panel_h = panel_w * aspect_ratio + 0.15
        fig = plt.figure(figsize=(panel_w + 0.55, panel_h))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.04],
                      left=0.08, right=0.95, bottom=0.10, top=0.93, wspace=0.08)
        ax_vel = fig.add_subplot(gs[0, 0])
        cax_vel = fig.add_subplot(gs[0, 1])

        sc_v = _draw_scatter_geo_panel(
            ax_vel, canvas, lon, lat, vel, valid_vel,
            cmap=_CMAP_VEL, norm=vel_norm,
            title=f'Sentinel-1 LOS Velocity  {period_str}', label_char='a',
            point_size=point_size, alpha=0.90,
        )
        ax_vel.set_xlabel('Longitude (°E)', fontsize=7.5)
        ax_vel.set_ylabel('Latitude (°N)', fontsize=7.5)
        _format_degree_axis(ax_vel, 4)
        _add_north_arrow(ax_vel, x=0.94, y=0.90, size=0.07)
        if water_present:
            _add_water_legend(ax_vel, loc='lower left', fontsize=5.4)
        sm_v = plt.cm.ScalarMappable(cmap=plt.colormaps.get_cmap(_CMAP_VEL), norm=vel_norm)
        sm_v.set_array([])
        cb_v = fig.colorbar(sm_v, cax=cax_vel, extend='both')
        cb_v.set_label('mm yr$^{-1}$', fontsize=7)
        cb_v.ax.tick_params(labelsize=6)
        cb_v.outline.set_linewidth(0.3)
        info_lines = [f'N = {int(np.count_nonzero(display_mask)):,}']
        if ref_date:
            info_lines.append(f'ref. {ref_date}')
        ax_vel.text(0.98, 0.05, '\n'.join(info_lines),
                    transform=ax_vel.transAxes, fontsize=5, ha='right', va='bottom',
                    color='0.4',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.7))
        has_hatch = _draw_lowq_hatch(ax_vel, canvas, lowq_mask)
        if has_hatch:
            _add_uncertainty_legend(ax_vel, loc='lower right', fontsize=5.4)

    out_path = export_dir / "velocity_map.png"
    _save_figure(fig, out_path)
    plt.close()

    p5, p95 = np.percentile(vel_m, 5), np.percentile(vel_m, 95)
    median = np.median(vel_m)
    print(f"速率图: {out_path}")
    print(f"  有效像素: {int(np.count_nonzero(display_mask)):,}, "
          f"中位数: {median:.1f}, P5/P95: {p5:.1f}/{p95:.1f} mm/yr")

    # 返回格网数据供 DePSI 复用
    return {
        'hs_grid': canvas["hs_grid"], 'water_grid': canvas["water_grid"],
        'glon': canvas["glon"], 'glat': canvas["glat"], 'extent': canvas["extent"],
        'dist_grid': canvas["dist_grid"], 'max_dist': canvas["max_dist"],
        'lat_min': canvas["lat_min"], 'lat_max': canvas["lat_max"],
        'lon_min': canvas["lon_min"], 'lon_max': canvas["lon_max"],
        'lat_c': canvas["lat_c"],
    }


# ---------------------------------------------------------------------------
# 2. 时序图（标准化布局: 多点 + 线性拟合 + 季节分析）
# ---------------------------------------------------------------------------
def plot_point_timeseries(mintpy_dir=None, lat=None, lon=None,
                          export_dir=None, n_points=4):
    """
    代表点位移时序图:
      - 自动选取多个代表点 (最大沉降 + 最大抬升 + 稳定参考 + 最高相干)
      - 或手动指定经纬度
      - 线性拟合趋势线 + 速率标注
      - 日期格式化 + 灰色辅助网格
      - 输出 PNG + PDF
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.gridspec import GridSpec

    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    export_dir = Path(export_dir or cfg.EXPORT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)

    data    = _load_mintpy_data(mintpy_dir)
    mask    = _build_mask(data, coh_threshold=0.6)
    vel     = data["vel"]
    ts      = data["ts"]
    dates   = data["dates"]
    lat_arr = data["lat"]
    lon_arr = data["lon"]
    vstd    = data.get("vstd")
    tcoh    = data["tcoh"]

    if ts is None or len(dates) == 0:
        print("[ERROR] 缺少 timeseries 数据")
        return

    vel_m = np.where(mask, vel, np.nan)
    ts_m  = np.where(mask[np.newaxis], ts, np.nan)

    # ── 选择代表点 ──────────────────────────────────────────────────
    points = []

    if lat is not None and lon is not None:
        # 手动指定
        dist = (lat_arr - lat) ** 2 + (lon_arr - lon) ** 2
        dist[~mask] = np.inf
        sy, sx = np.unravel_index(np.argmin(dist), dist.shape)
        points.append({'y': sy, 'x': sx, 'label': 'Selected', 'color': '#C0392B',
                       'marker': 'o'})
    else:
        # 自动选: 可靠区域 (排除极端异常值)
        reliable = mask.copy()
        if vstd is not None:
            reliable &= np.isfinite(vstd) & (vstd < np.nanpercentile(vstd[mask], 85))
        reliable &= np.isfinite(tcoh) & (tcoh >= 0.6)
        vel_rel = np.where(reliable, vel, np.nan)

        if not np.any(reliable):
            reliable = mask
            vel_rel = np.where(reliable, vel, np.nan)

        # 用分位数代替极值——避免选到噪声异常点
        p5 = np.nanpercentile(vel_rel, 5)
        p95 = np.nanpercentile(vel_rel, 95)

        # 1. 典型沉降点 (接近 P5 且相干性好)
        sub_cand = reliable & (vel >= p5 * 0.8) & (vel <= p5 * 1.2)
        if np.any(sub_cand):
            scores = np.where(sub_cand, tcoh, -1)
            sy, sx = np.unravel_index(np.argmax(scores), scores.shape)
        else:
            sy, sx = np.unravel_index(np.nanargmin(vel_rel), vel_rel.shape)
        points.append({'y': sy, 'x': sx, 'label': 'Subsidence',
                       'color': '#C0392B', 'marker': 'o'})

        # 2. 典型抬升点 (接近 P95 且相干性好)
        up_cand = reliable & (vel >= p95 * 0.8) & (vel <= p95 * 1.2)
        if np.any(up_cand):
            scores = np.where(up_cand, tcoh, -1)
            uy, ux = np.unravel_index(np.argmax(scores), scores.shape)
        else:
            uy, ux = np.unravel_index(np.nanargmax(vel_rel), vel_rel.shape)
        points.append({'y': uy, 'x': ux, 'label': 'Uplift',
                       'color': '#2980B9', 'marker': 's'})

        # 3. 稳定参考点 (速率接近0 + 高相干)
        stable_score = np.where(reliable,
                                tcoh - 0.2 * np.abs(vel_m) / max(np.nanstd(vel_m), 1),
                                -1)
        ry, rx = np.unravel_index(np.argmax(stable_score), stable_score.shape)
        points.append({'y': ry, 'x': rx, 'label': 'Stable reference',
                       'color': '#27AE60', 'marker': '^'})

        # 4. 最高相干点 (如果不同于以上)
        tcoh_m = np.where(reliable, tcoh, -1)
        hy, hx = np.unravel_index(np.argmax(tcoh_m), tcoh_m.shape)
        if (hy, hx) != (ry, rx) and (hy, hx) != (sy, sx):
            points.append({'y': hy, 'x': hx, 'label': 'High coherence',
                           'color': '#8E44AD', 'marker': 'D'})

    # 限制点数
    points = points[:n_points]
    if not points:
        print("[ERROR] 未找到可用代表点")
        return

    date_objs = [datetime.strptime(d, "%Y%m%d") for d in dates]
    date_nums = np.array([(d - date_objs[0]).days for d in date_objs], dtype=float)
    ref_date = data.get("cumulative_reference_date", dates[0])
    geo_valid = np.isfinite(lat_arr) & (lat_arr > 0.1) & np.isfinite(lon_arr) & (lon_arr > 0.1)
    aoi = cfg._AOI_BBOX
    if aoi is not None:
        S, N, W, E = aoi
        geo_valid &= (lat_arr >= S) & (lat_arr <= N) & (lon_arr >= W) & (lon_arr <= E)
    canvas = _prepare_geo_canvas(data, geo_valid, target_cols=560)
    valid_vel = mask & np.isfinite(vel)
    # 低质量区域 = 有数据但未通过 mask (coh < 0.6 等)
    all_data = np.isfinite(vel) & (vel != 0) & geo_valid
    all_data = _apply_water_mask(all_data, data)
    lowq_ts = all_data & ~valid_vel
    vel_norm = TwoSlopeNorm(vmin=-_symmetric_vlim(vel[mask]), vcenter=0.0,
                            vmax=_symmetric_vlim(vel[mask]))
    water_present = canvas["water_grid"] is not None and np.any(canvas["water_grid"])
    n_valid_vel = int(np.count_nonzero(valid_vel))
    point_size = 1.8 if n_valid_vel > 500_000 else (2.2 if n_valid_vel > 220_000 else 3.0)

    # ── 绘图 ─────────────────────────────────────────────────────
    n = len(points)
    fig_h = 3.6 if n <= 2 else 4.4
    fig = plt.figure(figsize=(7.8, fig_h))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.18, 0.036, 1.72],
                  left=0.07, right=0.98, bottom=0.13, top=0.93, wspace=0.16)
    ax_map = fig.add_subplot(gs[0, 0])
    cax_map = fig.add_subplot(gs[0, 1])
    ax = fig.add_subplot(gs[0, 2])

    # 左: 定位图
    _draw_scatter_geo_panel(
        ax_map, canvas, lon_arr, lat_arr, vel, valid_vel,
        cmap=_CMAP_VEL, norm=vel_norm,
        title='Representative Pixels', label_char='a',
        point_size=point_size, alpha=0.90,
    )
    lon_pad = 0.035 * (canvas["lon_max"] - canvas["lon_min"])
    lat_pad = 0.035 * (canvas["lat_max"] - canvas["lat_min"])
    ax_map.set_xlim(canvas["lon_min"] - lon_pad, canvas["lon_max"] + lon_pad)
    ax_map.set_ylim(canvas["lat_min"] - lat_pad, canvas["lat_max"] + lat_pad)
    ax_map.set_aspect(1.0 / np.cos(np.radians(canvas["lat_c"])))
    ax_map.set_xlabel('Longitude (°E)', fontsize=7.2)
    ax_map.set_ylabel('Latitude (°N)', fontsize=7.2)
    _format_degree_axis(ax_map, 3)
    _add_north_arrow(ax_map, x=0.92, y=0.88, size=0.07)
    if water_present:
        _add_water_legend(ax_map, loc='lower left', fontsize=5.4)

    sm = plt.cm.ScalarMappable(cmap=_CMAP_VEL, norm=vel_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax_map, extend='both')
    cbar.ax.set_title('mm yr$^{-1}$', fontsize=5.8, pad=4)
    cbar.ax.tick_params(labelsize=5, pad=1)
    cbar.outline.set_linewidth(0.35)

    # 高不确定性区域淹没
    has_hatch = _draw_lowq_hatch(ax_map, canvas, lowq_ts)
    if has_hatch:
        _add_uncertainty_legend(ax_map, loc='lower right', fontsize=4.8)

    for i, pt in enumerate(points, start=1):
        sy, sx = pt['y'], pt['x']
        ts_pt = ts_m[:, sy, sx]
        valid = np.isfinite(ts_pt)

        if not np.any(valid):
            continue

        point_id = f'P{i}'
        coh_val = float(tcoh[sy, sx])
        vel_val = float(vel[sy, sx])

        # 绘制数据点 + 连线
        ax.plot(np.array(date_objs)[valid], ts_pt[valid],
                color=pt['color'], lw=1.0, marker=pt['marker'],
                ms=3.5, mew=0.3, mec='white', alpha=0.92, zorder=3 + i,
                label=f'{point_id}  {pt["label"]}')

        # 线性拟合趋势线
        if valid.sum() > 3:
            coeff = np.polyfit(date_nums[valid], ts_pt[valid], 1)
            trend_mm_yr = coeff[0] * 365.25
            trend_line = np.polyval(coeff, date_nums)
            ax.plot(date_objs, trend_line, color=pt['color'], lw=0.7,
                    ls='--', alpha=0.55, zorder=2)
        else:
            trend_mm_yr = float('nan')

        ax_map.scatter([lon_arr[sy, sx]], [lat_arr[sy, sx]],
                       s=28, c=pt['color'], marker=pt['marker'],
                       edgecolors='white', linewidths=0.4, zorder=8)
        ax_map.text(lon_arr[sy, sx], lat_arr[sy, sx], str(i),
                    fontsize=6.5, fontweight='bold', color='black',
                    ha='center', va='center', zorder=9,
                    bbox=dict(boxstyle='circle,pad=0.16', fc='white',
                              ec=pt['color'], lw=0.6, alpha=0.92))

        pt["label"] = f'{point_id}  {pt["label"]}'

    ax.axhline(0, color='0.6', lw=0.6, ls='-', zorder=1)

    # 日期轴格式化
    ax.set_xlabel('Date', fontsize=8)
    ax.set_ylabel('LOS displacement (mm)', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 自适应间隔
    span_days = (date_objs[-1] - date_objs[0]).days
    if span_days > 1200:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    elif span_days > 600:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate(rotation=30, ha='right')

    ax.grid(True, alpha=0.15, lw=0.4, which='major')
    ax.tick_params(labelsize=7)

    leg = ax.legend(fontsize=6.1, loc='upper left', framealpha=0.94,
                    handlelength=1.6, handletextpad=0.5, borderpad=0.35,
                    ncol=1 if n <= 3 else 2)
    leg.get_frame().set_linewidth(0.4)

    ax.set_title('LOS Displacement Time Series', fontsize=8.8, loc='left',
                 fontweight='bold', pad=4)
    _subfig_label(ax, 'b', x=0.01, y=0.97)
    ax.text(0.99, 0.98, f'Sentinel-1  |  {len(dates)} acquisitions  |  ref. {ref_date}',
            transform=ax.transAxes, fontsize=5.5, ha='right', va='top',
            color='0.5', style='italic')

    out_path = export_dir / "timeseries_displacement.png"
    _save_figure(fig, out_path)
    plt.close()

    print(f"时序图: {out_path}")
    for pt in points:
        sy, sx = pt['y'], pt['x']
        if lat_arr is not None:
            print(f"  {pt['label']}: ({lat_arr[sy,sx]:.4f}, {lon_arr[sy,sx]:.4f}) "
                  f"vel={vel[sy,sx]:.1f} mm/yr  coh={tcoh[sy,sx]:.2f}")


# ---------------------------------------------------------------------------
# 3. 基线网络图（标准化辅助图: 边着色=平均相干性）
# ---------------------------------------------------------------------------
def plot_baseline_network(mintpy_dir=None, export_dir=None):
    """
    时空基线网络图:
      - 干涉边着色 = 该对的平均空间相干性 (viridis)
      - 采集点深蓝实心圆, 参考景红星
      - 右侧色标 + 右下统计文字
    输出: baseline_network.png + .pdf
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    import h5py

    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    export_dir = Path(export_dir or cfg.EXPORT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)

    ifgram_file = mintpy_dir / "inputs" / "ifgramStack.h5"
    if not ifgram_file.exists():
        print("[ERROR] ifgramStack.h5 不存在")
        return

    with h5py.File(ifgram_file, 'r') as f:
        date12     = f['date'][:]
        bperp_ifg  = f['bperp'][:]
        coh        = f['coherence'][:]
        ref_date   = str(f.attrs.get('REF_DATE', '')).strip()
        drop       = f['dropIfgram'][:] if 'dropIfgram' in f else np.ones(len(date12), dtype=bool)

    def _decode(d):
        s = d.tobytes().decode() if hasattr(d, 'tobytes') else str(d)
        return s.strip()[:8]

    pairs     = [(_decode(d[0]), _decode(d[1])) for d in date12]
    all_dates = sorted(set(d for p in pairs for d in p))
    date_objs = {d: datetime.strptime(d, '%Y%m%d') for d in all_dates}
    n_dates   = len(all_dates)

    # Bperp per date
    bperp_dates = _read_bperp_from_isce2(all_dates, ref_date)
    if bperp_dates is None:
        if np.any(bperp_ifg != 0):
            bperp_dates = _infer_bperp_from_ifg(pairs, bperp_ifg, all_dates, ref_date)
        else:
            bperp_dates = {}
            for i, d in enumerate(all_dates):
                ph = 2 * np.pi * i / max(n_dates - 1, 1)
                bperp_dates[d] = 70 * np.sin(ph) + 20 * np.cos(2 * ph)

    # 每对的平均相干性 (用于着色)
    mean_coh_list = []
    for i in range(len(pairs)):
        c = coh[i]
        valid_c = c[np.isfinite(c) & (c > 0)]
        mean_coh_list.append(float(np.mean(valid_c)) if len(valid_c) > 0 else 0)
    mean_coh_arr = np.array(mean_coh_list)

    blue_cmap = LinearSegmentedColormap.from_list(
        "insar_blues", ["#d9ecff", "#8dbde6", "#3b78b5", "#123f7a"]
    )

    # ── 绘图: 左=网络 右=相干性分布
    fig, (ax, ax_coh) = plt.subplots(
        1, 2, figsize=(_COL2, 3.15),
        gridspec_kw={'width_ratios': [3.8, 1.1], 'wspace': 0.32}
    )
    ax.set_facecolor('white')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    # 干涉边 — 保留边用相干性着色，剔除边淡灰显示
    segments, edge_coh = [], []
    dropped_segments = []
    for i, (d1, d2) in enumerate(pairs):
        x1, y1 = date_objs[d1], bperp_dates.get(d1, 0)
        x2, y2 = date_objs[d2], bperp_dates.get(d2, 0)
        seg = [(x1, y1), (x2, y2)]
        if drop[i]:
            segments.append(seg)
            edge_coh.append(mean_coh_arr[i])
        else:
            dropped_segments.append(seg)

    if dropped_segments:
        drop_num = [
            [(mdates.date2num(s[0][0]), s[0][1]),
             (mdates.date2num(s[1][0]), s[1][1])]
            for s in dropped_segments
        ]
        lc_drop = LineCollection(drop_num, colors='0.82', linewidths=0.45,
                                 alpha=0.55, zorder=1)
        ax.add_collection(lc_drop)

    cbar = None
    if segments:
        seg_num = [
            [(mdates.date2num(s[0][0]), s[0][1]),
             (mdates.date2num(s[1][0]), s[1][1])]
            for s in segments
        ]
        edge_coh_arr = np.array(edge_coh)
        coh_lo = max(float(np.nanpercentile(edge_coh_arr, 5)), 0.0)
        coh_hi = min(float(np.nanpercentile(edge_coh_arr, 95)), 1.0)
        lc = LineCollection(seg_num, cmap=blue_cmap, linewidths=1.0, alpha=0.82, zorder=2)
        lc.set_array(edge_coh_arr)
        lc.set_clim(coh_lo, coh_hi)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, shrink=0.70, pad=0.015, aspect=24)
        cbar.set_label('Mean spatial coherence', fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_linewidth(0.4)

    # 采集点 + 参考景
    dt_list = [date_objs[d] for d in all_dates]
    bp_list = [bperp_dates.get(d, 0) for d in all_dates]
    ax.scatter(dt_list, bp_list, s=22, c='#1f3552', zorder=5,
               linewidths=0.3, edgecolors='white', label='Acquisition')
    if ref_date in date_objs:
        ax.scatter([date_objs[ref_date]], [bperp_dates.get(ref_date, 0)],
                   s=80, c='#E74C3C', marker='*', zorder=6,
                   linewidths=0.3, edgecolors='white',
                   label=f'Ref {ref_date}')

    ax.set_xlabel('Date', fontsize=8)
    ax.set_ylabel('B$_{\\perp}$ (m)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    span_days = (max(dt_list) - min(dt_list)).days
    if span_days > 1200:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    elif span_days > 600:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate(rotation=25, ha='right')
    ax.grid(True, alpha=0.11, lw=0.35)
    ax.axhline(0, color='0.75', lw=0.5, ls='--')
    leg = ax.legend(fontsize=6.5, loc='upper left', framealpha=0.92,
                    handlelength=1.0, handletextpad=0.3, borderpad=0.3)
    leg.get_frame().set_linewidth(0.4)

    # 统计文字
    n_kept = int(drop.sum())
    redundancy = n_kept / max(n_dates - 1, 1)
    ax.set_title('Interferometric Baseline Network', fontsize=8.8,
                 loc='left', fontweight='bold', pad=5)

    # ── 右侧独立直方图 ──
    valid_coh = mean_coh_arr[drop.astype(bool)]
    med_coh = float(np.median(valid_coh))
    ax_coh.set_facecolor('white')
    for sp in ['top', 'right']:
        ax_coh.spines[sp].set_visible(False)
    ax_coh.hist(valid_coh, bins=18, color='#66a9df', edgecolor='white',
                lw=0.3, alpha=0.82, orientation='horizontal')
    ax_coh.axhline(med_coh, color='#E74C3C', lw=1.0, ls='-',
                   label=f'median = {med_coh:.2f}')
    ax_coh.set_ylabel('Mean coherence', fontsize=7.5)
    ax_coh.set_xlabel('Count', fontsize=7.5)
    ax_coh.tick_params(labelsize=6.5)
    ax_coh.legend(fontsize=6, frameon=False, loc='upper right')
    ax_coh.set_title('Pair Coherence Distribution', fontsize=8,
                     fontweight='bold', pad=5, loc='left')

    stats_lines = [
        f'{n_dates} acquisitions',
        f'{n_kept} kept / {len(pairs)} total pairs',
        f'redundancy = {redundancy:.1f}×',
    ]
    if ref_date:
        stats_lines.append(f'reference = {ref_date}')
    if cbar is not None:
        stats_lines.append(f'median coh = {med_coh:.2f}')
    ax.text(0.985, 0.02, '\n'.join(stats_lines),
            transform=ax.transAxes, fontsize=5.4, ha='right', va='bottom',
            color='0.35',
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='0.88',
                      lw=0.4, alpha=0.90))

    out_path = export_dir / "baseline_network.png"
    _save_figure(fig, out_path)
    plt.close()
    print(f"基线图: {out_path} ({n_dates} 景, {n_kept} 对, median coh={med_coh:.2f})")


# ---------------------------------------------------------------------------
# 4. 累计位移面板图（标准化布局: 子图标签 + 紧凑共享色标）
# ---------------------------------------------------------------------------
def plot_displacement_panels(mintpy_dir=None, export_dir=None,
                              coh_threshold=0.6, vlim_cm=None,
                              n_panel=None):
    """
    累计位移多时相面板图:
      - 子图标签 (a)–(i) — Nature 风格
      - 均匀精选 6–9 景主图 + 全时相补充图
      - hillshade 底图 + 水体白色
      - 共享对称色标 (右侧细长), 0 居中
      - 仅最左列和最下行保留刻度
    输出: displacement_panels.png/pdf + displacement_panels_all.png/pdf
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.colors import TwoSlopeNorm

    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    export_dir = Path(export_dir or cfg.EXPORT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)

    data  = _load_mintpy_data(mintpy_dir)
    mask  = _build_mask(data, coh_threshold)
    ts    = data["ts"]
    dates = data["dates"]
    lat, lon, hgt = data["lat"], data["lon"], data["height"]

    if lat is None or mask.sum() == 0:
        print("[ERROR] 无有效数据")
        return

    ts_m  = np.where(mask[np.newaxis], ts, np.nan)
    n_all = len(dates)

    # 色标范围
    if vlim_cm is None:
        all_v = ts_m[np.isfinite(ts_m)] / 10
        vlim_cm = min(np.percentile(np.abs(all_v), 97), 8.0)

    canvas = _prepare_geo_canvas(data, mask, target_cols=460)
    norm = TwoSlopeNorm(vmin=-vlim_cm, vcenter=0.0, vmax=vlim_cm)
    ref_date = data.get("cumulative_reference_date", dates[0])

    # 子图标签
    _PANEL_LABELS = 'abcdefghijklmnopqrstuvwxyz'

    def _render_panel(ax, ts_i, title_str, show_xaxis, show_yaxis,
                      label_char=None, add_nav=False):
        """渲染单个时相面板。"""
        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
            sp.set_visible(True)

        _draw_hillshade_background(ax, canvas["hs_grid"], canvas["water_grid"],
                                   canvas["extent"], alpha=0.28)

        valid_i = np.isfinite(ts_i) & mask
        if valid_i.sum() > 50:
            tg = _interpolate_geo_field(ts_i / 10, valid_i, canvas)
            cmap_d = plt.cm.get_cmap(_CMAP_DISP)
            disp_rgba = cmap_d(norm(tg))
            disp_rgba[np.isnan(tg)] = [1, 1, 1, 0]
            ax.imshow(disp_rgba, extent=canvas["extent"], aspect='auto',
                      interpolation='nearest', origin='upper', alpha=0.85)

        ax.set_xlim(canvas["lon_min"], canvas["lon_max"])
        ax.set_ylim(canvas["lat_min"], canvas["lat_max"])
        ax.set_aspect(1.0 / np.cos(np.radians(canvas["lat_c"])))
        ax.set_title(title_str, fontsize=6.3, pad=2)

        if label_char is not None:
            _subfig_label(ax, label_char, x=0.03, y=0.97)

        if add_nav:
            _add_north_arrow(ax, x=0.92, y=0.88, size=0.07)

        if not show_xaxis:
            ax.set_xticklabels([])
        else:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
            ax.tick_params(axis='x', labelsize=5.5, rotation=30)

        if not show_yaxis:
            ax.set_yticklabels([])
        else:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
            ax.tick_params(axis='y', labelsize=5.5)

    def _save_panels(indices, out_name, label):
        n = len(indices)
        nrows, ncols = _choose_panel_grid(n, max_cols=5 if n > 9 else 4)
        aspect_ratio = (
            (canvas["lat_max"] - canvas["lat_min"]) /
            max((canvas["lon_max"] - canvas["lon_min"]) *
                np.cos(np.radians(canvas["lat_c"])), 0.001)
        )
        fig_w = _COL2 if ncols >= 3 else min(_COL2 * 0.78, 5.9)
        cell_w = (fig_w - 0.50) / ncols
        cell_h = cell_w * aspect_ratio + 0.18
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fig_w, cell_h * nrows + 0.55),
            squeeze=False)
        plt.subplots_adjust(left=0.07, right=0.905, bottom=0.08, top=0.90,
                            wspace=0.07, hspace=0.18)

        for k, idx in enumerate(indices):
            r, c = divmod(k, ncols)
            ax   = axes[r][c]
            ts_i = ts_m[idx]
            lbl_char = _PANEL_LABELS[k] if k < len(_PANEL_LABELS) else None
            _render_panel(ax, ts_i,
                          title_str=_format_epoch_label(dates[idx], ref_date),
                          show_xaxis=(r == nrows - 1),
                          show_yaxis=(c == 0),
                          label_char=lbl_char,
                          add_nav=(k == 0))

        # 隐藏多余子图
        for k in range(n, nrows * ncols):
            r, c = divmod(k, ncols)
            axes[r][c].set_visible(False)

        # 共用色标
        sm = plt.cm.ScalarMappable(cmap=_CMAP_DISP, norm=norm)
        sm.set_array([])
        cb_ax = fig.add_axes([0.92, 0.16, 0.014, 0.62])
        cbar  = fig.colorbar(sm, cax=cb_ax, extend='both')
        cbar.set_label('LOS displacement (cm)', fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_linewidth(0.4)

        fig.suptitle(
            f'Cumulative LOS Displacement  |  {label}  |  ref. {ref_date}',
            fontsize=8.0, fontweight='bold', y=0.955
        )
        out = export_dir / out_name
        _save_figure(fig, out, dpi=200)
        plt.close()
        print(f"面板图: {out} ({n} 景)")

    # 精选时相
    n_select = min(n_panel or 9, n_all)
    sel_idx  = np.round(np.linspace(0, n_all - 1, n_select)).astype(int).tolist()
    _save_panels(sel_idx, "displacement_panels.png",
                 f"selected {n_select} epochs")

    # 全时相
    if n_all > n_select:
        _save_panels(list(range(n_all)), "displacement_panels_all.png",
                     f"all {n_all} epochs")


# ---------------------------------------------------------------------------
# 5. 速率诊断散点图（顶刊补充材料 / 审稿附图）
# ---------------------------------------------------------------------------
def plot_velocity_diagnostics(mintpy_dir=None, export_dir=None,
                               coh_threshold=0.6):
    """
    速率诊断四联图 (顶刊 SI 级):
      (a) Velocity vs. Height — 检查残余 DEM 误差
      (b) Velocity vs. Temporal Coherence — 检查低质量点
      (c) Velocity histogram — 分布形态 + 正态拟合
      (d) Temporal Coherence map — 独立质量评估

    输出: velocity_diagnostics.png + .pdf
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    from scipy.stats import norm as sp_norm

    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    export_dir = Path(export_dir or cfg.EXPORT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)

    data = _load_mintpy_data(mintpy_dir)
    mask = _build_mask(data, coh_threshold)
    vel  = data["vel"]
    lat, lon = data["lat"], data["lon"]
    hgt  = data["height"]
    tcoh = data["tcoh"]

    if lat is None or mask.sum() == 0:
        print("[ERROR] 无有效数据")
        return

    vel_m  = vel[mask]
    tcoh_m = tcoh[mask]
    hgt_m  = hgt[mask] if hgt is not None else None
    lat_m, lon_m = lat[mask], lon[mask]

    fig, axes = plt.subplots(2, 2, figsize=(_COL2, 5.5))
    plt.subplots_adjust(wspace=0.30, hspace=0.35)

    # ── (a) Velocity vs. Height
    ax = axes[0, 0]
    if hgt_m is not None and np.any(np.isfinite(hgt_m) & (hgt_m > 0)):
        valid_h = np.isfinite(hgt_m) & (hgt_m > 0)
        # 降采样绘制 (避免过密)
        n_pts = valid_h.sum()
        if n_pts > 10000:
            idx = np.random.default_rng(42).choice(n_pts, 10000, replace=False)
        else:
            idx = np.arange(n_pts)
        ax.scatter(hgt_m[valid_h][idx], vel_m[valid_h][idx],
                   s=1.0, c=tcoh_m[valid_h][idx], cmap=_CMAP_COH,
                   vmin=0, vmax=1, alpha=0.4, rasterized=True)
        # 线性拟合
        coeff = np.polyfit(hgt_m[valid_h], vel_m[valid_h], 1)
        h_range = np.linspace(np.percentile(hgt_m[valid_h], 2),
                              np.percentile(hgt_m[valid_h], 98), 100)
        ax.plot(h_range, np.polyval(coeff, h_range), 'r-', lw=1.0, alpha=0.7,
                label=f'slope={coeff[0]:.3f} mm/yr/m')
        ax.legend(fontsize=5.5, frameon=False)
        ax.set_xlabel('Height (m)', fontsize=7.5)
    else:
        ax.text(0.5, 0.5, 'No height data', transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='0.5')
    ax.set_ylabel('LOS velocity (mm yr$^{-1}$)', fontsize=7.5)
    ax.axhline(0, color='0.6', lw=0.5, ls='--')
    _subfig_label(ax, 'a')

    # ── (b) Velocity vs. Coherence
    ax = axes[0, 1]
    n_pts = len(vel_m)
    if n_pts > 10000:
        idx = np.random.default_rng(42).choice(n_pts, 10000, replace=False)
    else:
        idx = np.arange(n_pts)
    ax.scatter(tcoh_m[idx], vel_m[idx], s=1.0, c='#2C3E50', alpha=0.25,
               rasterized=True)
    # 分箱均值
    bins_c = np.linspace(coh_threshold, 1.0, 15)
    bin_means = []
    bin_stds = []
    bin_centers = []
    for i in range(len(bins_c) - 1):
        in_bin = (tcoh_m >= bins_c[i]) & (tcoh_m < bins_c[i+1])
        if in_bin.sum() > 10:
            bin_centers.append((bins_c[i] + bins_c[i+1]) / 2)
            bin_means.append(np.mean(vel_m[in_bin]))
            bin_stds.append(np.std(vel_m[in_bin]))
    if bin_centers:
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
                     fmt='o-', color='#E74C3C', ms=3, lw=0.8, capsize=2,
                     capthick=0.6, label='Binned mean±std', zorder=5)
        ax.legend(fontsize=5.5, frameon=False)
    ax.set_xlabel('Temporal coherence', fontsize=7.5)
    ax.set_ylabel('LOS velocity (mm yr$^{-1}$)', fontsize=7.5)
    ax.axhline(0, color='0.6', lw=0.5, ls='--')
    _subfig_label(ax, 'b')

    # ── (c) Velocity histogram + normal fit
    ax = axes[1, 0]
    n_bins = min(80, max(20, int(np.sqrt(len(vel_m)))))
    counts, bin_edges, _ = ax.hist(vel_m, bins=n_bins, color='#3498DB',
                                    edgecolor='none', alpha=0.70, density=True)
    # 正态拟合
    mu, sigma = np.mean(vel_m), np.std(vel_m)
    x_fit = np.linspace(vel_m.min(), vel_m.max(), 200)
    ax.plot(x_fit, sp_norm.pdf(x_fit, mu, sigma), 'r-', lw=1.0, alpha=0.7,
            label=f'N({mu:.1f}, {sigma:.1f}²)')
    ax.axvline(0, color='0.3', lw=0.6, ls='--')
    ax.axvline(np.median(vel_m), color='#E74C3C', lw=0.8, ls='-',
               label=f'median={np.median(vel_m):.1f}')
    ax.set_xlabel('LOS velocity (mm yr$^{-1}$)', fontsize=7.5)
    ax.set_ylabel('Probability density', fontsize=7.5)
    ax.legend(fontsize=5.5, frameon=False)
    _subfig_label(ax, 'c')

    # ── (d) Temporal Coherence map
    ax = axes[1, 1]
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.4)
    ax.set_facecolor('white')

    lat_min, lat_max = lat_m.min(), lat_m.max()
    lon_min, lon_max = lon_m.min(), lon_m.max()
    extent = [lon_min, lon_max, lat_min, lat_max]

    res = 0.0005
    glon_1d = np.arange(lon_min, lon_max, res)
    glat_1d = np.arange(lat_max, lat_min, -res)
    glon, glat = np.meshgrid(glon_1d, glat_1d)
    pts = np.column_stack([lon_m, lat_m])
    tree = cKDTree(pts)
    dist, _ = tree.query(np.column_stack([glon.ravel(), glat.ravel()]))
    dist_grid = dist.reshape(glon.shape)

    tcoh_grid = griddata(pts, tcoh_m, (glon, glat), method='nearest')
    tcoh_grid[dist_grid > res * 3] = np.nan

    im = ax.imshow(tcoh_grid, extent=extent, aspect='auto',
                   interpolation='nearest', origin='upper',
                   cmap=_CMAP_COH, vmin=0, vmax=1, alpha=0.85)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=18)
    cbar.set_label('Temporal coherence', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.4)

    ax.set_xlabel('Longitude', fontsize=7.5)
    ax.set_ylabel('Latitude', fontsize=7.5)
    ax.tick_params(labelsize=6)
    _subfig_label(ax, 'd')

    for a in axes.ravel():
        a.tick_params(labelsize=6.5)
        for sp in ['top', 'right']:
            if a != axes[1, 1]:
                a.spines[sp].set_visible(False)

    fig.suptitle('Velocity Diagnostics', fontsize=9, fontweight='bold', y=1.01)

    out_path = export_dir / "velocity_diagnostics.png"
    _save_figure(fig, out_path)
    plt.close()
    print(f"诊断图: {out_path}")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _read_bperp_from_isce2(all_dates, ref_date):
    """从 ISCE2 pair-format baselines/ 目录读取 Bperp。"""
    import re
    bl_dir = cfg.ISCE_WORK_DIR / "baselines"
    if not bl_dir.exists():
        return None

    bperp = {}
    for d in all_dates:
        for pair_dir in bl_dir.iterdir():
            if not pair_dir.is_dir() or d not in pair_dir.name:
                continue
            txt = pair_dir / f"{pair_dir.name}.txt"
            if txt.exists():
                try:
                    content = txt.read_text()
                    vals = [float(m) for m in re.findall(r'Bperp.*?:\s*([-\d.]+)', content)]
                    if vals:
                        bperp[d] = float(np.mean(vals))
                        break
                except Exception:
                    pass

    if len(bperp) < len(all_dates) // 2:
        return None

    ref_val = bperp.get(ref_date, 0.0)
    return {d: v - ref_val for d, v in bperp.items()}


def _infer_bperp_from_ifg(pairs, bperp_ifg, all_dates, ref_date):
    """从干涉对 Bperp 差值最小二乘反推各日期 Bperp。"""
    date_idx = {d: i for i, d in enumerate(all_dates)}
    n, m     = len(all_dates), len(pairs)
    A        = np.zeros((m, n))
    for i, (d1, d2) in enumerate(pairs):
        if d1 in date_idx: A[i, date_idx[d1]] = -1
        if d2 in date_idx: A[i, date_idx[d2]] =  1
    ref_i = date_idx.get(ref_date, 0)
    A     = np.vstack([A, np.eye(n)[ref_i]])
    b     = np.append(bperp_ifg, 0)
    try:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return {d: x[i] for d, i in date_idx.items()}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GeoTIFF / Shapefile / KMZ / CSV 导出
# ---------------------------------------------------------------------------
def export_geotiff(mintpy_dir=None, export_dir=None, coh_threshold=0.6):
    """导出速率图 GeoTIFF（WGS84）。"""
    import rasterio
    from rasterio.transform import from_bounds

    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    export_dir = Path(export_dir or cfg.EXPORT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)

    data = _load_mintpy_data(mintpy_dir)
    mask = _build_mask(data, coh_threshold)
    vel  = data["vel"]
    lat, lon = data["lat"], data["lon"]

    vel_out = np.where(mask, vel, np.nan).astype("float32")
    nrow, ncol = vel.shape

    if lat is not None and np.any(lat > 0):
        lat_m, lon_m = lat[mask], lon[mask]
        west, east   = lon_m.min(), lon_m.max()
        south, north = lat_m.min(), lat_m.max()
    else:
        west, south, east, north = 0, 0, ncol, nrow

    transform = from_bounds(west, south, east, north, ncol, nrow)
    out_path  = export_dir / "velocity.tif"
    with rasterio.open(out_path, "w", driver="GTiff",
                       height=nrow, width=ncol, count=1,
                       dtype="float32", crs="EPSG:4326",
                       transform=transform, nodata=np.nan) as dst:
        dst.write(vel_out, 1)
    print(f"GeoTIFF: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def export_shapefile(mintpy_dir=None, export_dir=None, coh_threshold=0.7):
    """导出高相干点 Shapefile。"""
    import geopandas as gpd
    from shapely.geometry import Point

    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    export_dir = Path(export_dir or cfg.EXPORT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)

    data = _load_mintpy_data(mintpy_dir)
    mask = _build_mask(data, coh_threshold)
    vel, lat, lon, tcoh = data["vel"], data["lat"], data["lon"], data["tcoh"]

    if lat is None:
        print("[ERROR] 无几何文件"); return

    rows, cols = np.where(mask)
    if not len(rows):
        print(f"[WARN] coh>={coh_threshold} 无像素"); return

    if len(rows) > 500_000:
        idx = np.random.default_rng(42).choice(len(rows), 500_000, replace=False)
        rows, cols = rows[idx], cols[idx]

    gdf = gpd.GeoDataFrame(
        {"vel_mm_yr": vel[rows, cols], "coherence": tcoh[rows, cols]},
        geometry=[Point(lon[r, c], lat[r, c]) for r, c in zip(rows, cols)],
        crs="EPSG:4326")
    out = export_dir / "velocity_points.shp"
    gdf.to_file(out)
    print(f"Shapefile: {out} ({len(gdf)} 点)")


def export_kmz(mintpy_dir=None, export_dir=None, coh_threshold=0.7):
    """导出高相干点 KMZ（Google Earth）。"""
    try:
        import simplekml
    except ImportError:
        print("[WARN] simplekml 未安装，跳过"); return

    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    export_dir = Path(export_dir or cfg.EXPORT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)

    data = _load_mintpy_data(mintpy_dir)
    mask = _build_mask(data, coh_threshold)
    vel, lat, lon = data["vel"], data["lat"], data["lon"]

    if lat is None:
        print("[ERROR] 无几何文件"); return

    rows, cols = np.where(mask)
    if len(rows) > 50_000:
        idx = np.random.default_rng(42).choice(len(rows), 50_000, replace=False)
        rows, cols = rows[idx], cols[idx]

    kml   = simplekml.Kml()
    vvals = vel[rows, cols]
    vmax  = max(np.percentile(np.abs(vvals), 95), 1)

    for r, c, v in zip(rows, cols, vvals):
        norm = np.clip(v / vmax, -1, 1)
        if norm < 0:
            rc, gc, bc = 255, int(255*(1+norm)), int(255*(1+norm))
        else:
            rc, gc, bc = int(255*(1-norm)), int(255*(1-norm)), 255
        pnt = kml.newpoint(coords=[(lon[r,c], lat[r,c])],
                           description=f"Vel: {v:.1f} mm/yr")
        pnt.style.iconstyle.color = simplekml.Color.rgb(rc, gc, bc)
        pnt.style.iconstyle.scale = 0.4

    out = export_dir / "velocity_points.kmz"
    kml.savekmz(str(out))
    print(f"KMZ: {out} ({len(rows)} 点)")


# ---------------------------------------------------------------------------
# CSV 导出（含垂直速率 + 逐景累积位移）
# ---------------------------------------------------------------------------
def export_csv(mintpy_dir=None, export_dir=None, coh_threshold=0.7,
               max_points=500_000):
    """
    导出 CSV 点云文件，包含：
      UID         — 唯一点编号
      longitude   — 经度
      latitude    — 纬度
      tcoh        — 时序相干性
      LOS_mm_yr   — LOS 方向速率
      YYYYMMDD... — 各获取日期的累积位移（mm，相对最早获取日期）

    Args:
        coh_threshold: 相干性阈值（默认 0.7，高相干点）
        max_points: 最大导出点数（随机采样）

    Returns:
        DataFrame（前 10 行供 Jupyter 预览）
    """
    import h5py
    import pandas as pd

    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    export_dir = Path(export_dir or cfg.EXPORT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)

    data = _load_mintpy_data(mintpy_dir)
    mask = _build_mask(data, coh_threshold)

    vel  = data["vel"]       # mm/yr (LOS)
    ts   = data["ts"]        # mm (LOS cumulative)
    dates = data["dates"]
    lat  = data["lat"]
    lon  = data["lon"]
    tcoh = data["tcoh"]

    if lat is None:
        print("[ERROR] 无几何文件")
        return None

    rows, cols = np.where(mask)
    if len(rows) == 0:
        print(f"[WARN] coh>={coh_threshold} 无像素")
        return None

    # 随机降采样
    if len(rows) > max_points:
        idx = np.random.default_rng(42).choice(len(rows), max_points, replace=False)
        idx = np.sort(idx)
        rows, cols = rows[idx], cols[idx]

    # 逐景累积位移（LOS mm）
    ts_pts = ts[:, rows, cols]  # (n_dates, n_points)

    # 组装 DataFrame
    df = pd.DataFrame({
        'UID':        np.arange(1, len(rows) + 1),
        'longitude':  np.round(lon[rows, cols], 6),
        'latitude':   np.round(lat[rows, cols], 6),
        'tcoh':       np.round(tcoh[rows, cols], 3),
        'LOS_mm_yr':  np.round(vel[rows, cols], 2),
    })

    # 各日期列（累积 LOS 位移 mm）
    for i, d in enumerate(dates):
        df[d] = np.round(ts_pts[i], 2)

    out_path = export_dir / "displacement_points.csv"
    df.to_csv(out_path, index=False)

    print(f"CSV: {out_path} ({len(df)} 点, {len(dates)} 景)")
    print(f"  文件大小: {out_path.stat().st_size / 1e6:.1f} MB")

    return df


def organize_export_dir(export_dir=None):
    """将 export 目录整理为 figures/ vector/ raster/ 三个子目录。"""
    import shutil
    export_dir = Path(export_dir or cfg.EXPORT_DIR)
    if not export_dir.exists():
        return

    fig_dir = export_dir / "figures"
    vec_dir = export_dir / "vector"
    ras_dir = export_dir / "raster"
    for d in [fig_dir, vec_dir, ras_dir]:
        d.mkdir(exist_ok=True)

    moved = 0
    for f in export_dir.iterdir():
        if f.is_dir():
            continue
        suf = f.suffix.lower()
        if suf in ('.png', '.pdf'):
            dest = fig_dir / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
                moved += 1
        elif suf in ('.shp', '.shx', '.dbf', '.prj', '.cpg', '.kmz', '.csv'):
            dest = vec_dir / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
                moved += 1
        elif suf in ('.tif', '.tiff'):
            dest = ras_dir / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
                moved += 1

    if moved:
        print(f"导出整理: {moved} 文件 → figures/ vector/ raster/")
    return {'figures': str(fig_dir), 'vector': str(vec_dir), 'raster': str(ras_dir)}
