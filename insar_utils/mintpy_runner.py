"""
MintPy time-series inversion driver.
Generates custom_template.txt and runs smallbaselineApp.py.
"""
import sys
import os
import subprocess
import time
from pathlib import Path

from tqdm.auto import tqdm

from . import config as cfg
from .config import logger
from .hardware import build_thread_limited_env, recommend_mintpy_settings


# ---------------------------------------------------------------------------
# MintPy Step Definitions (for progress tracking)
# ---------------------------------------------------------------------------
MINTPY_STEPS = [
    "load_data",
    "modify_network",
    "reference_point",
    "quick_overview",
    "correct_unwrap_error",
    "invert_network",
    "correct_LOD",
    "correct_SET",
    "correct_ionosphere",
    "correct_troposphere",
    "deramp",
    "correct_topography",
    "residual_RMS",
    "reference_date",
    "velocity",
    "geocode",
    "google_earth",
    "hdfeos5",
]


# ---------------------------------------------------------------------------
# Generate MintPy Template
# ---------------------------------------------------------------------------
def generate_mintpy_template(
    dolphin_dir=None,
    dem_file=None,
    era5_dir=None,
    gacos_dir=None,
    output_path=None,
    atmo_config=None,
):
    """
    Generate MintPy custom_template.txt with all error correction enabled.

    Args:
        atmo_config: dict from choose_atmo_correction(), e.g.
                     {"method":"era5","model":"ERA5","dir":Path}
                     如果为 None, 自动检测 GACOS/ERA5 目录
    """
    dolphin_dir = Path(dolphin_dir or cfg.DOLPHIN_DIR)
    dem_file = Path(dem_file or cfg.DEM_FILE)
    era5_dir = Path(era5_dir or cfg.ERA5_DIR)
    gacos_dir = Path(gacos_dir or cfg.GACOS_DIR)
    output_path = Path(output_path or cfg.TEMPLATE_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine tropospheric delay settings from atmo_config or auto-detect
    # 先自动转换 GACOS .ztd.tif → .ztd + .ztd.rsc (如果存在)
    from .atmosphere import convert_gacos_tif_to_ztd
    convert_gacos_tif_to_ztd(gacos_dir)

    if atmo_config is None:
        # 自动检测（仅在用户未指定时触发）: GACOS > ERA5 > height_correlation
        # 验证 GACOS 是否覆盖 AOI（超出 AOI 的 buffer 像素用 fill_value=0 处理，科学上可接受）
        gacos_files = list(gacos_dir.glob("*.ztd")) if gacos_dir.exists() else []
        if gacos_files and _gacos_covers_scene(gacos_dir, cfg.ISCE_WORK_DIR):
            atmo_config = {"method": "gacos", "dir": gacos_dir}
        elif gacos_files:
            logger.warning("GACOS 未覆盖 AOI，改用 height_correlation（自动降级仅在 atmo_config=None 时触发）")
            atmo_config = {"method": "height_correlation"}
        elif list(era5_dir.glob("ERA5_*.grb")):
            atmo_config = {"method": "era5", "model": "ERA5", "dir": era5_dir}
        else:
            atmo_config = {"method": "height_correlation"}
    # 用户传入的 atmo_config 直接使用，不做任何修改

    atmo_method = atmo_config.get("method", "no")

    # Detect geometry files
    geom_dir = dolphin_dir / "geometry"
    inc_file = _find_file(geom_dir, ["incidence*", "los*", "inc*"])
    az_file = _find_file(geom_dir, ["azimuth*", "az*"])

    # 计算干涉网络冗余度（决定是否启用 DEM error 校正）
    from glob import glob as _g
    unw_files = _g(str(dolphin_dir / "unwrapped" / "*.unw.tif"))
    n_ifgram = len(unw_files)
    date_set = set()
    for f in unw_files:
        name = Path(f).stem.replace('.unw', '')
        parts = name.split('_')
        if len(parts) == 2:
            date_set.update(parts)
    n_date = len(date_set) if date_set else max(n_ifgram, 1)
    redundancy = n_ifgram / max(n_date - 1, 1)
    logger.info(f"干涉网络: {n_ifgram} 对, {n_date} 景, 冗余度 {redundancy:.1f}x")

    # 生成水体掩膜（基于 DEM 坡度检测平坦大面积水体）
    water_mask_file = output_path.parent / "waterMask.h5"
    _generate_water_mask(dem_file, water_mask_file)

    template_lines = [
        f"########## MintPy Custom Template - Auto Generated ##########",
        f"",
        f"## 1. Load Data (Dolphin outputs, via HyP3 loader)",
        f"mintpy.load.processor       = hyp3",
        f"mintpy.load.unwFile         = {dolphin_dir}/unwrapped/*.unw.tif",
        f"mintpy.load.corFile         = {dolphin_dir}/interferograms/*.int.cor.tif",
        f"mintpy.load.connCompFile    = {dolphin_dir}/unwrapped/*.unw.conncomp.tif",
        f"mintpy.load.demFile         = {dem_file}",
        f"mintpy.load.waterMaskFile   = {water_mask_file}",
    ]

    if inc_file:
        template_lines.append(f"mintpy.load.incAngleFile    = {inc_file}")
    if az_file:
        template_lines.append(f"mintpy.load.azAngleFile     = {az_file}")

    mintpy_settings = recommend_mintpy_settings(requested_workers=cfg.N_WORKERS)
    template_lines.extend([
        f"",
        f"## 2. Reference Point",
        f"mintpy.reference.lalo       = auto",
        f"",
        f"## 3. Network Modification",
        f"mintpy.network.coherenceBased = yes",
        f"mintpy.network.minCoherence   = 0.6",
        f"",
        f"## 4. Unwrapping Error Correction",
        f"## 网络型干涉图提供相位闭合冗余，可自动检测并校正解缠误差",
        f"mintpy.unwrapError.method   = bridging",
        f"",
        f"## 5. Tropospheric Delay Correction",
    ])

    if atmo_method == "gacos":
        gacos_files = list(gacos_dir.glob("*.ztd")) if gacos_dir.exists() else []
        template_lines.extend([
            f"## 方案: GACOS (高分辨率迭代对流层分解)",
            f"mintpy.troposphericDelay.method   = gacos",
            f"mintpy.troposphericDelay.gacosDir = {gacos_dir}",
        ])
        atmo_desc = f"GACOS ({len(gacos_files)} 个文件)"
    elif atmo_method in ("era5", "merra2"):
        model = atmo_config.get("model", "ERA5")
        weather_dir = atmo_config.get("dir", era5_dir)
        template_lines.extend([
            f"## 方案: {model} (全球再分析天气模型)",
            f"mintpy.troposphericDelay.method       = pyaps",
            f"mintpy.troposphericDelay.weatherModel = {model}",
            f"mintpy.troposphericDelay.weatherDir   = {weather_dir}",
        ])
        atmo_desc = model
    elif atmo_method == "height_correlation":
        # height_correlation 依赖 geometryRadar.h5 中正确的地形高程
        # 若 hgt.rdr.full.vrt 值域异常（ISCE2 SLC 模式已知问题），
        # build_mintpy_hdf5 会自动从 DEM.tif 插值修复，方法本身可正常使用
        template_lines.extend([
            f"## 方案: 相位-高程经验拟合 (无需外部数据)",
            f"## φ_atm ≈ a·h + b, 逐景拟合后扣除地形相关大气分量",
            f"## 注: 依赖 geometryRadar.h5/height 字段正确，已由 build_mintpy_hdf5 自动修复",
            f"mintpy.troposphericDelay.method = height_correlation",
        ])
        atmo_desc = "相位-高程拟合 (无需数据)"
    elif atmo_method == "spatial_filter":
        template_lines.extend([
            f"## 方案: 时空滤波 (无需外部数据)",
            f"## 依赖 deramp + 时间高通滤波分离大气信号",
            f"mintpy.troposphericDelay.method = no",
        ])
        atmo_desc = "时空滤波 (deramp + 时间滤波)"
    else:
        template_lines.extend([
            f"## 不校正大气延迟",
            f"mintpy.troposphericDelay.method = no",
        ])
        atmo_desc = "无 (跳过)"

    template_lines.extend([
        f"",
        f"## 6. Solid Earth Tides (SET) Correction",
        f"## 扣除太阳/月球引力导致的地壳周期性潮汐起伏 (垂直可达数十厘米)",
        f"## 使用 pysolid 计算，需要精确的卫星过境时间和位置",
        f"mintpy.solidEarthTides      = yes",
        f"",
        f"## 7. Topographic Residual (DEM Error) Correction",
        f"## 需要干涉网络有足够冗余约束（冗余度>2x）；星型网络下自动禁用以避免过拟合",
        f"mintpy.topographicResidual  = {'yes' if n_ifgram > n_date * 1.5 else 'no'}",
        f"",
        f"## 8. Orbital Ramp Removal",
        f"## 去除轨道误差和长波大气残差",
        f"mintpy.deramp               = linear",
        f"",
        f"## 9. Plate Motion Correction",
        f"## 扣除板块整体运动 (欧亚板块), 获得局部相对形变",
        f"## 使用稳定参考点即可实现相对校正; 绝对校正需引入 GNSS 数据",
        f"mintpy.velocity.platMotion  = yes",
        f"mintpy.velocity.platMotionModel = ITRF14",
        f"",
        f"## 10. Output",
        f"mintpy.save.kmz             = yes",
        f"",
        f"## 11. 并行计算与内存控制",
        f"## dem_error 等步骤内存密集, worker 数需保守",
        f"mintpy.compute.numWorker    = {mintpy_settings['num_worker']}",
        f"mintpy.compute.maxMemory    = {mintpy_settings['max_memory_gb']}",
        f"",
    ])

    content = "\n".join(template_lines)
    output_path.write_text(content)
    print(f"MintPy 模板已保存: {output_path}")
    print(f"  开启校正: SET固体潮 / DEM误差 / 轨道斜面 / 板块运动")
    print(f"  大气校正: {atmo_desc}")

    return output_path


def _gacos_covers_scene(gacos_dir, isce_work_dir):
    """
    检查 GACOS 文件是否覆盖 AOI（严格 AOI bbox，不含处理缓冲区）。

    逻辑：
    - GACOS 由用户提交时以 AOI bbox 为范围下载，因此只需验证 GACOS 是否覆盖
      严格 AOI 而非整个处理场景（场景包含 10% buffer + padding，GACOS 不覆盖这部分是正常的）
    - 对于超出 GACOS 范围的缓冲/padding 像素，MintPy 使用 fill_value=0 (不校正)，
      这是可接受的科学行为（参见 tropo_gacos.py RegularGridInterpolator bounds_error=False）

    Returns:
        True  — GACOS 覆盖严格 AOI（用 _AOI_BBOX 判断），可以使用 GACOS
        False — GACOS 覆盖不足，建议换用其他方法
    """
    try:
        rsc_files = sorted(Path(gacos_dir).glob("*.ztd.rsc"))
        if not rsc_files:
            return False

        content = rsc_files[0].read_text()
        lines = {}
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                lines[parts[0]] = parts[1]

        gacos_x0 = float(lines.get("X_FIRST", 0))
        gacos_y0 = float(lines.get("Y_FIRST", 0))
        gacos_dx = float(lines.get("X_STEP", 0))
        gacos_dy = float(lines.get("Y_STEP", 0))
        gacos_w  = int(lines.get("WIDTH", 0))
        gacos_l  = int(lines.get("FILE_LENGTH", 0))

        gacos_lon_min = gacos_x0
        gacos_lon_max = gacos_x0 + (gacos_w - 1) * gacos_dx
        gacos_lat_min = gacos_y0 + (gacos_l - 1) * gacos_dy
        gacos_lat_max = gacos_y0

        # 优先用严格 AOI bbox（由 pipeline 在 generate_stack 时写入 cfg._AOI_BBOX）
        aoi_bbox = cfg._AOI_BBOX  # [S, N, W, E]
        if aoi_bbox is not None:
            S, N, W, E = aoi_bbox
            # 小容差 0.01° (~1km) 允许坐标舍入
            covers = (gacos_lat_min <= S + 0.01 and
                      gacos_lat_max >= N - 0.01 and
                      gacos_lon_min <= W + 0.01 and
                      gacos_lon_max >= E - 0.01)
            if not covers:
                logger.warning(
                    f"GACOS 未覆盖 AOI: "
                    f"GACOS lat [{gacos_lat_min:.4f},{gacos_lat_max:.4f}] "
                    f"lon [{gacos_lon_min:.4f},{gacos_lon_max:.4f}], "
                    f"AOI lat [{S:.4f},{N:.4f}] lon [{W:.4f},{E:.4f}]"
                )
            return covers

        # 无 AOI bbox 时，退回全场景比较（保守容差 30%）
        geom_ref = Path(isce_work_dir) / "merged" / "geom_reference"
        lat_vrt = geom_ref / "lat.rdr.full.vrt"
        if not lat_vrt.exists():
            return True

        from osgeo import gdal
        import numpy as np
        gdal.UseExceptions()
        ds = gdal.Open(str(lat_vrt))
        lat = ds.ReadAsArray(); ds = None
        ds2 = gdal.Open(str(geom_ref / "lon.rdr.full.vrt"))
        lon = ds2.ReadAsArray(); ds2 = None

        valid = lat > 0.1
        if not valid.any():
            return True

        slat_min = float(lat[valid].min())
        slat_max = float(lat[valid].max())
        slon_min = float(lon[valid].min())
        slon_max = float(lon[valid].max())

        lat_range = slat_max - slat_min
        lon_range = slon_max - slon_min
        margin = 0.30

        return (gacos_lat_min <= slat_min + margin * lat_range and
                gacos_lat_max >= slat_max - margin * lat_range and
                gacos_lon_min <= slon_min + margin * lon_range and
                gacos_lon_max >= slon_max - margin * lon_range)

    except Exception as e:
        logger.debug(f"_gacos_covers_scene 检查失败: {e}")
        return True  # 无法判断，信任 GACOS


def _generate_water_mask(dem_file, out_file):
    """
    从 DEM 自动生成水体掩膜（通用，不依赖特定 AOI 或外部数据）。

    算法：
    1. 在 DEM 地理坐标上计算坡度，检测大面积平坦区域（水体）
    2. 用 geometryRadar.h5 的 lat/lon 将水体掩膜投影到雷达坐标
    3. 保存为与 ifgramStack.h5 尺寸一致的 HDF5（MintPy waterMask 格式）

    输出的 waterMask: 1=有效（陆地），0=无效（水体）
    """
    import numpy as np
    from pathlib import Path

    out_file = Path(out_file)
    if out_file.exists():
        return

    dem_path = Path(dem_file)
    if not dem_path.exists():
        logger.debug(f"DEM 文件不存在 ({dem_path})，跳过水体掩膜生成")
        return

    try:
        import rasterio
        import h5py
        from scipy.ndimage import label
        from scipy.interpolate import RegularGridInterpolator

        # ── 1. 在 DEM 地理坐标上检测水体
        with rasterio.open(str(dem_path)) as src:
            dem = src.read(1).astype('float32')
            gt = src.transform

        pix_y = abs(gt.e) * 111000
        pix_x = gt.a * 111000 * np.cos(np.radians(gt.f + dem.shape[0] * gt.e / 2))
        dy, dx = np.gradient(dem, pix_y, pix_x)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        flat = slope < 1.0
        labels, n_features = label(flat)

        water_mask_geo = np.zeros(dem.shape, dtype=bool)
        if n_features > 0:
            sizes = np.bincount(labels.ravel())
            large_ids = np.where(sizes > 500)[0]
            large_ids = large_ids[large_ids > 0]
            for lid in large_ids:
                water_mask_geo[labels == lid] = True

        water_count_geo = water_mask_geo.sum()
        if water_count_geo == 0:
            logger.info("未检测到水体")
            return

        logger.info(f"检测到水体像素: {water_count_geo} ({100*water_count_geo/dem.size:.1f}%)")

        # ── 2. 投影到雷达坐标（匹配 ifgramStack 尺寸）
        # 读 geometryRadar.h5 获取雷达坐标系的 lat/lon
        geom_file = out_file.parent / "inputs" / "geometryRadar.h5"
        if not geom_file.exists():
            # 没有 geometryRadar.h5 → 无法投影到雷达坐标，直接存地理坐标版本
            # （MintPy 可能无法使用，但至少 viz.py 能用）
            mask_out = (~water_mask_geo).astype('bool')
            with h5py.File(str(out_file), 'w') as f:
                f.create_dataset('waterMask', data=mask_out)
                f.attrs['FILE_TYPE'] = 'waterMask'
                f.attrs['LENGTH'] = str(dem.shape[0])
                f.attrs['WIDTH'] = str(dem.shape[1])
            print(f"  水体掩膜 (geo): {out_file.name} ({water_count_geo} 水体像素)")
            return

        with h5py.File(str(geom_file), 'r') as f:
            lat_radar = f['latitude'][:]
            lon_radar = f['longitude'][:]

        # DEM 格网坐标
        nrow, ncol = dem.shape
        lon_dem = gt.c + np.arange(ncol) * gt.a
        lat_dem = gt.f + np.arange(nrow) * gt.e  # lat 递减

        # 将 water_mask_geo 插值到雷达坐标
        # waterMask 是 bool，用 nearest 插值
        lat_dem_inc = lat_dem[::-1]
        wm_flip = water_mask_geo[::-1, :].astype('float32')

        interp = RegularGridInterpolator(
            (lat_dem_inc, lon_dem), wm_flip,
            method='nearest', bounds_error=False, fill_value=0)

        valid = np.isfinite(lat_radar) & (lat_radar > 0.1)
        pts = np.column_stack([lat_radar[valid].ravel(), lon_radar[valid].ravel()])
        water_radar = np.zeros(lat_radar.shape, dtype=bool)
        water_radar[valid] = interp(pts) > 0.5

        # waterMask: 1=陆地, 0=水体
        mask_out = (~water_radar).astype('bool')
        water_count_radar = water_radar.sum()

        # ── 3. 保存（雷达坐标，尺寸匹配 ifgramStack）
        with h5py.File(str(out_file), 'w') as f:
            f.create_dataset('waterMask', data=mask_out)
            f.attrs['FILE_TYPE'] = 'waterMask'
            # 从 geometryRadar.h5 复制元数据
            with h5py.File(str(geom_file), 'r') as src:
                for k, v in src.attrs.items():
                    f.attrs[k] = v
            f.attrs['FILE_TYPE'] = 'waterMask'

        print(f"  水体掩膜: {out_file.name} ({water_count_radar} 水体像素, "
              f"尺寸 {lat_radar.shape[1]}×{lat_radar.shape[0]} = ifgramStack)")

    except Exception as e:
        logger.warning(f"水体掩膜生成失败: {e}")

    except Exception as e:
        logger.warning(f"水体掩膜生成失败: {e}")


def _find_file(directory, patterns):
    """Find first matching file in directory."""
    if not directory.exists():
        return None
    for pat in patterns:
        matches = list(directory.glob(pat))
        if matches:
            return str(matches[0])
    return None


# ---------------------------------------------------------------------------
# Dolphin→MintPy HDF5 桥接 (MintPy 不原生支持 dolphin processor)
# ---------------------------------------------------------------------------
# Dolphin strides 常量 (与 dolphin_runner.py 中的配置一致)
DOLPHIN_STRIDE_X = 6   # range 方向步长
DOLPHIN_STRIDE_Y = 3   # azimuth 方向步长


def _subset_signature(subset_window):
    if subset_window is None:
        return "full"
    return ",".join(str(int(v)) for v in subset_window)


def build_mintpy_hdf5(dolphin_dir=None, geom_source_dir=None,
                      mintpy_dir=None, subset_window=None,
                      ifgram_filename="ifgramStack.h5", force_rebuild=False):
    """
    从 Dolphin 输出 + ISCE2 几何文件构建 MintPy 所需的 HDF5 输入。
    解决 MintPy 不支持 'dolphin' processor 的兼容性问题。

    生成:
      - inputs/<ifgram_filename> (解缠相位 + 相干性 + 连通分量)
      - inputs/geometryRadar.h5 (高程 + 入射角 + 经纬度)
    """
    import h5py, rasterio
    import numpy as np
    from glob import glob as _glob

    dolphin_dir = Path(dolphin_dir or cfg.DOLPHIN_DIR)
    mintpy_dir = Path(mintpy_dir or cfg.MINTPY_DIR)
    inputs_dir = mintpy_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    geom_source = Path(geom_source_dir) if geom_source_dir else (
        cfg.ISCE_WORK_DIR / "merged" / "geom_reference")

    # ---- ifgramStack.h5 ----
    subset_sig = _subset_signature(subset_window)
    ifgram_file = inputs_dir / ifgram_filename
    subset_window = tuple(int(v) for v in subset_window) if subset_window is not None else None

    rebuild_ifgram = force_rebuild
    if ifgram_file.exists() and not rebuild_ifgram:
        try:
            with h5py.File(str(ifgram_file), "r") as f:
                existing_sig = f.attrs.get("QC_SUBSET_WINDOW", "full")
                rebuild_ifgram = str(existing_sig) != subset_sig
        except Exception:
            rebuild_ifgram = True

    if ifgram_file.exists() and not rebuild_ifgram:
        print(f"  {ifgram_filename} 已存在，跳过")
    else:
        if ifgram_file.exists():
            ifgram_file.unlink()
        unw_files = sorted(_glob(str(dolphin_dir / "unwrapped" / "*.unw.tif")))
        cor_files = sorted(_glob(str(dolphin_dir / "interferograms" / "*.int.cor.tif")))
        cc_files = sorted(_glob(str(dolphin_dir / "unwrapped" / "*.unw.conncomp.tif")))

        if not unw_files:
            raise FileNotFoundError("未找到 Dolphin unwrapped 文件")

        with rasterio.open(unw_files[0]) as src:
            width, height = src.width, src.height
            if subset_window is not None:
                row0, row1, col0, col1 = subset_window
                height = max(0, min(src.height, row1) - row0)
                width = max(0, min(src.width, col1) - col0)
            else:
                row0, row1, col0, col1 = 0, src.height, 0, src.width

        date_pairs = []
        for f in unw_files:
            name = Path(f).stem.replace('.unw', '')
            d1, d2 = name.split('_')
            date_pairs.append((d1, d2))

        n_ifgram = len(unw_files)
        print(f"  构建 ifgramStack.h5: {n_ifgram} 对, {width}x{height}")

        # 提取真实 bperp
        bperp_values = _compute_bperp_for_ifgrams(date_pairs, geom_source)

        with h5py.File(str(ifgram_file), 'w') as f:
            ds_unw = f.create_dataset('unwrapPhase', (n_ifgram, height, width),
                                      dtype='float32', chunks=True)
            ds_cor = f.create_dataset('coherence', (n_ifgram, height, width),
                                      dtype='float32', chunks=True)
            ds_cc = f.create_dataset('connectComponent', (n_ifgram, height, width),
                                     dtype='int16', chunks=True)

            for i, (uf, cf, ccf) in enumerate(zip(unw_files, cor_files, cc_files)):
                window = ((row0, row1), (col0, col1))
                with rasterio.open(uf) as src: ds_unw[i] = src.read(1, window=window)
                with rasterio.open(cf) as src: ds_cor[i] = src.read(1, window=window)
                with rasterio.open(ccf) as src: ds_cc[i] = src.read(1, window=window).astype('int16')

            date12 = np.array([[d1, d2] for d1, d2 in date_pairs], dtype='S8')
            f.create_dataset('date', data=date12)
            f.create_dataset('dropIfgram', data=np.ones(n_ifgram, dtype='bool'))
            f.create_dataset('bperp', data=bperp_values.astype('float32'))

            # Sentinel-1 元数据 (从 ISCE2 XML 动态提取, 后备用默认值)
            s1_meta = _extract_s1_metadata(geom_source)
            s1_meta['WIDTH'] = str(width)
            s1_meta['LENGTH'] = str(height)
            s1_meta['REF_DATE'] = date_pairs[0][0]
            s1_meta['ALOOKS'] = str(DOLPHIN_STRIDE_Y)
            s1_meta['RLOOKS'] = str(DOLPHIN_STRIDE_X)
            s1_meta['QC_SUBSET_WINDOW'] = subset_sig
            for k, v in s1_meta.items():
                f.attrs[k] = v

        print(f"  {ifgram_filename} OK (bperp range: "
              f"{bperp_values.min():.1f} ~ {bperp_values.max():.1f} m)")

    # ---- geometryRadar.h5 ----
    geom_file = inputs_dir / "geometryRadar.h5"
    rebuild_geom = force_rebuild
    if geom_file.exists() and not rebuild_geom:
        try:
            with h5py.File(str(geom_file), "r") as f:
                existing_sig = f.attrs.get("QC_SUBSET_WINDOW", "full")
                rebuild_geom = str(existing_sig) != subset_sig
        except Exception:
            rebuild_geom = True

    if geom_file.exists() and not rebuild_geom:
        print(f"  geometryRadar.h5 已存在，跳过")
    else:
        if geom_file.exists():
            geom_file.unlink()
        # 从 ifgramStack 获取目标尺寸
        with h5py.File(str(ifgram_file)) as f:
            height = int(f.attrs['LENGTH'])
            width = int(f.attrs['WIDTH'])

        print(f"  构建 geometryRadar.h5: {width}x{height}")

        # 读取 AOI 裁剪偏移（由 postprocess_slc 写入 cfg._AOI_CROP_OFFSET）
        # 几何文件必须从与 SLC 相同的偏移处开始采样，否则 lat/lon 将错位
        crop_offset = cfg._AOI_CROP_OFFSET  # (row_off, col_off) 或 None
        if crop_offset is not None:
            row_off, col_off = crop_offset
            if subset_window is not None:
                row_off += subset_window[0] * DOLPHIN_STRIDE_Y
                col_off += subset_window[2] * DOLPHIN_STRIDE_X
            print(f"  几何采样偏移: row={row_off}, col={col_off} (与 SLC AOI 裁剪对齐)")
        else:
            row_off, col_off = 0, 0
            print(f"  警告: 未找到 AOI 裁剪偏移，几何文件可能与 SLC 不对齐")

        with h5py.File(str(geom_file), 'w') as f:
            for name, src_name, band_idx in [
                ('height', 'hgt.rdr.full.vrt', 1),
                ('incidenceAngle', 'los.rdr.full.vrt', 1),
                ('azimuthAngle', 'los.rdr.full.vrt', 2),
                ('latitude', 'lat.rdr.full.vrt', 1),
                ('longitude', 'lon.rdr.full.vrt', 1),
            ]:
                vrt = geom_source / src_name
                if not vrt.exists():
                    continue
                with rasterio.open(str(vrt)) as src:
                    if band_idx > src.count:
                        continue
                    data = src.read(band_idx)

                # 用 Dolphin strides 正确下采样，从 AOI 裁剪偏移处开始
                data_ml = _downsample_by_strides(data, height, width,
                                                 row_off=row_off, col_off=col_off)

                # height 特殊处理：ISCE2 hgt.rdr 可能存储的不是真实海拔
                # （值域接近 0 或全负数），此时改从 DEM.tif 插值获取正确高程
                if name == 'height':
                    valid_h = data_ml[np.isfinite(data_ml) & (data_ml != 0)]
                    if len(valid_h) == 0 or np.abs(valid_h).mean() < 10:
                        logger.warning(
                            f"hgt.rdr 值域异常 (mean={np.nanmean(data_ml):.4f}m)，"
                            f"改从 DEM.tif 插值地形高度")
                        data_ml = _get_height_from_dem(
                            data_ml.shape, geom_source, mintpy_dir)

                f.create_dataset(name, data=data_ml.astype('float32'))

            # shadowMask
            sm_vrt = geom_source / "shadowMask.rdr.full.vrt"
            if sm_vrt.exists():
                with rasterio.open(str(sm_vrt)) as src:
                    data = src.read(1)
                data_ml = _downsample_by_strides(data, height, width,
                                                 row_off=row_off, col_off=col_off)
                f.create_dataset('shadowMask', data=data_ml.astype('bool'))

            # slantRangeDistance — 从 ISCE2 元数据精确计算
            if 'incidenceAngle' in f:
                slant_range = _compute_slant_range(
                    f['incidenceAngle'][:], geom_source, width)
                f.create_dataset('slantRangeDistance',
                                 data=slant_range.astype('float32'))

            # 属性 (与 ifgramStack 保持一致)
            with h5py.File(str(ifgram_file)) as src:
                for k, v in src.attrs.items():
                    f.attrs[k] = v
            f.attrs['FILE_TYPE'] = 'geometry'
            f.attrs['QC_SUBSET_WINDOW'] = subset_sig

        print(f"  geometryRadar.h5 OK")

    # ---- 自动选参考点 ----
    _auto_set_reference_point(ifgram_file, geom_file)

    print("  MintPy HDF5 桥接完成")
    return {
        "mintpy_dir": str(mintpy_dir),
        "inputs_dir": str(inputs_dir),
        "ifgram_path": str(ifgram_file),
        "geometry_path": str(geom_file),
        "subset_window": list(subset_window) if subset_window is not None else None,
        "subset_signature": subset_sig,
        "force_rebuild": bool(force_rebuild),
    }


def _get_height_from_dem(shape, geom_source, mintpy_dir):
    """
    从 DEM.tif 插值地形高度到 radar 坐标（geometryRadar 尺寸）。
    当 ISCE2 hgt.rdr 值域异常时（值接近 0 或全负数）作为后备。
    """
    import numpy as np
    import rasterio
    from scipy.interpolate import RegularGridInterpolator

    # 找 DEM.tif：相对于 mintpy_dir 向上找 DEM/ 目录
    dem_path = None
    for p in [
        mintpy_dir.parent / "DEM" / "dem.tif",
        cfg.DEM_FILE,
    ]:
        if Path(p).exists():
            dem_path = Path(p)
            break

    if dem_path is None:
        logger.warning("未找到 DEM.tif，height 设为 0")
        return np.zeros(shape, dtype='float32')

    # 读 lat/lon（已写入 geom_file 的前几个 dataset，或从 geom_source 读）
    lat_vrt = geom_source / "lat.rdr.full.vrt"
    lon_vrt = geom_source / "lon.rdr.full.vrt"
    if not lat_vrt.exists():
        return np.zeros(shape, dtype='float32')

    with rasterio.open(str(lat_vrt)) as src:
        lat_full = src.read(1)
    with rasterio.open(str(lon_vrt)) as src:
        lon_full = src.read(1)

    # 读 DEM
    with rasterio.open(str(dem_path)) as src:
        dem = src.read(1).astype('float32')
        gt = src.transform
        ncol, nrow = src.width, src.height

    lon_dem = gt.c + np.arange(ncol) * gt.a
    lat_dem = gt.f + np.arange(nrow) * gt.e   # gt.e < 0

    lat_dem_inc = lat_dem[::-1]
    dem_flip = dem[::-1, :]

    interp = RegularGridInterpolator(
        (lat_dem_inc, lon_dem), dem_flip,
        method='linear', bounds_error=False, fill_value=0)

    # 用与 _downsample_by_strides 相同的偏移采样
    row_off = cfg._AOI_CROP_OFFSET[0] if cfg._AOI_CROP_OFFSET else 0
    col_off = cfg._AOI_CROP_OFFSET[1] if cfg._AOI_CROP_OFFSET else 0
    stride_y = DOLPHIN_STRIDE_Y
    stride_x = DOLPHIN_STRIDE_X

    lat_s = lat_full[row_off::stride_y, col_off::stride_x]
    lon_s = lon_full[row_off::stride_y, col_off::stride_x]

    h, w = shape
    lat_crop = lat_s[:h, :w]
    lon_crop = lon_s[:h, :w]

    valid = (lat_crop > 0.1) & np.isfinite(lat_crop)
    height_out = np.zeros(shape, dtype='float32')

    pts = np.column_stack([lat_crop[valid].ravel(), lon_crop[valid].ravel()])
    height_out[valid] = interp(pts)

    return height_out


def _downsample_by_strides(data, target_height, target_width,
                           row_off=0, col_off=0):
    """
    用 Dolphin 的 strides 参数从全分辨率数据中正确下采样。

    关键参数:
        row_off, col_off: SLC 地理裁剪的起始像素偏移（由 postprocess_slc 写入
                          cfg._AOI_CROP_OFFSET）。几何文件的采样必须从同一位置
                          开始，否则 lat/lon 会对应到错误的雷达坐标。

    Dolphin 的 strides={x:6, y:3} 意味着输出像素 (i,j) 对应
    输入像素 (row_off + i*stride_y, col_off + j*stride_x)。
    超出 sliced 范围的边界区域用 NaN 填充（float）或 0（int）。
    """
    import numpy as np

    stride_y = DOLPHIN_STRIDE_Y
    stride_x = DOLPHIN_STRIDE_X

    # 从 AOI 裁剪偏移处开始 stride slicing
    sliced = data[row_off::stride_y, col_off::stride_x]

    out_h = min(sliced.shape[0], target_height)
    out_w = min(sliced.shape[1], target_width)

    if np.issubdtype(data.dtype, np.floating):
        result = np.full((target_height, target_width), np.nan, dtype=data.dtype)
    else:
        result = np.zeros((target_height, target_width), dtype=data.dtype)

    result[:out_h, :out_w] = sliced[:out_h, :out_w]
    return result


def _compute_bperp_for_ifgrams(date_pairs, geom_source):
    """
    从 ISCE2 baselines 目录提取各干涉对的垂直基线 (Bperp)。

    策略：
    1. 从 baselines/<ref>_<sec>/ 星型文本文件中提取各日期的绝对 bperp
    2. 任意日期对 (d1,d2) 的 bperp = bperp_abs[d2] - bperp_abs[d1]
    3. 即使 Dolphin 使用网络型干涉图，只要 ISCE2 的星型基线文件存在即可计算
    """
    import re
    import numpy as np

    isce_work = geom_source.parent.parent  # .../isce2/
    pair_bl_dir = isce_work / "baselines"

    all_dates = sorted(set(d for pair in date_pairs for d in pair))
    bperp_abs = {}  # 各日期的绝对 bperp（相对于 ISCE2 参考景）

    # ── 从星型基线文件提取各日期的绝对 bperp ──
    if pair_bl_dir.exists():
        for pair_dir in sorted(pair_bl_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            txt = pair_dir / f"{pair_dir.name}.txt"
            if not txt.exists():
                continue
            try:
                content = txt.read_text()
                vals = [float(m) for m in
                        re.findall(r'Bperp.*?:\s*([-\d.]+)', content)]
                if not vals:
                    continue
                avg_bp = float(np.mean(vals))

                # 解析日期对 (d1_d2)
                parts = pair_dir.name.split('_')
                if len(parts) == 2:
                    d_ref, d_sec = parts[0], parts[1]
                    # ISCE2 baselines/<ref>_<sec>.txt 中 Bperp 是 sec 相对于 ref
                    # ref 的 bperp = 0（ISCE2 参考景），sec 的 bperp = avg_bp
                    bperp_abs.setdefault(d_ref, 0.0)
                    bperp_abs[d_sec] = avg_bp
            except Exception:
                continue

    if not bperp_abs:
        logger.warning("baselines/ 目录中未找到有效基线文件，bperp 全部设为 0")
        return np.zeros(len(date_pairs), dtype='float32')

    # 确保所有日期都有 bperp（ISCE2 参考景 bperp=0）
    isce_ref = min(bperp_abs.keys())  # ISCE2 参考景通常是最早日期
    for d in all_dates:
        if d not in bperp_abs:
            bperp_abs[d] = 0.0
            logger.debug(f"日期 {d} 无基线信息，设为 0")

    # ── 计算各干涉对的 bperp = abs[d2] - abs[d1] ──
    bperp_ifg = np.zeros(len(date_pairs), dtype='float32')
    for i, (d1, d2) in enumerate(date_pairs):
        bperp_ifg[i] = bperp_abs.get(d2, 0.0) - bperp_abs.get(d1, 0.0)

    logger.info(f"bperp 提取完成: {len(date_pairs)} 对 "
                f"({len(bperp_abs)} 日期), "
                f"范围 {bperp_ifg.min():.1f}~{bperp_ifg.max():.1f} m")

    return bperp_ifg


def _read_bperp_for_date(bl_dir, date_str):
    """从 ISCE2 baselines 目录读取单个日期的平均垂直基线。"""
    date_dir = bl_dir / date_str
    if not date_dir.exists():
        return None

    # 尝试读取各种格式的 baseline 文件
    for fn in sorted(date_dir.iterdir()):
        # 跳过二进制文件（.vrt/.xml/无扩展名的 grid 文件）
        if fn.suffix in ('.vrt', '.xml', '.hdr', '.aux'):
            continue
        try:
            with open(fn, 'r', encoding='utf-8', errors='strict') as f:
                content = f.read()
            # 文本文件中含 "Bperp" 关键字
            if 'Bperp' in content or 'perp' in content.lower():
                for line in content.splitlines():
                    if 'Bperp' in line or 'perp' in line.lower():
                        try:
                            return float(line.split()[-1])
                        except (ValueError, IndexError):
                            continue
            else:
                # 尝试当作纯数值文本
                import numpy as np
                try:
                    vals = np.fromstring(content, sep='\n')
                    if vals.size > 0:
                        return float(np.mean(vals))
                except (ValueError, OverflowError):
                    continue
        except (UnicodeDecodeError, PermissionError, IsADirectoryError):
            # 跳过二进制或无法读取的文件
            continue
        except Exception:
            continue
    return None


def _compute_slant_range(inc_angle, geom_source, width):
    """
    从 ISCE2 元数据计算精确的 slant range distance。

    使用 STARTING_RANGE + range_pixel_size * col_idx，
    而非粗略的 H/cos(inc) 近似。
    """
    import numpy as np

    # 从 ISCE2 XML 提取 starting range 和 range pixel size
    meta = _extract_s1_metadata(geom_source)
    starting_range = float(meta.get('STARTING_RANGE', '800000.0'))
    range_pixel_size = float(meta.get('RANGE_PIXEL_SIZE', '2.329562'))

    height, w = inc_angle.shape

    # 每列的 slant range = starting_range + col * range_pixel_size * RLOOKS
    col_indices = np.arange(w) * DOLPHIN_STRIDE_X  # 多视后的列索引映射回全分辨率
    slant_range_1d = starting_range + col_indices * range_pixel_size
    slant_range = np.broadcast_to(slant_range_1d[np.newaxis, :], (height, w)).copy()

    return slant_range


def _extract_s1_metadata(geom_source_dir):
    """从 ISCE2 XML 文件动态提取 Sentinel-1 SAR 元数据。"""
    import xml.etree.ElementTree as ET

    meta = {
        'PROCESSOR': 'isce', 'FILE_TYPE': 'ifgramStack',
        'WAVELENGTH': '0.05546576', 'RANGE_PIXEL_SIZE': '2.329562',
        'AZIMUTH_PIXEL_SIZE': '13.96', 'EARTH_RADIUS': '6371000.0',
        'HEIGHT': '700000.0', 'STARTING_RANGE': '800000.0',
        'PLATFORM': 'Sen1', 'ORBIT_DIRECTION': 'DESCENDING',
        'ALOOKS': '3', 'RLOOKS': '6', 'ANTENNA_SIDE': '-1',
        'CENTER_LINE_UTC': '83421.0', 'HEADING': '-167.0',
        'UNIT': 'radian',
    }

    # 尝试从 ISCE2 reference XML 中提取实际值
    ref_xml = Path(geom_source_dir).parent.parent / "reference" / "IW1.xml"
    if not ref_xml.exists():
        ref_xml = Path(geom_source_dir).parent.parent / "reference" / "IW2.xml"

    if ref_xml.exists():
        try:
            tree = ET.parse(str(ref_xml))
            root = tree.getroot()
            for prop in root.iter('property'):
                name = prop.get('name', '')
                val_elem = prop.find('value')
                if val_elem is None or val_elem.text is None:
                    continue
                val = val_elem.text.strip()
                if name == 'startingrange':
                    meta['STARTING_RANGE'] = val
                elif name == 'heading':
                    meta['HEADING'] = val
                elif name == 'platformheading':
                    meta['HEADING'] = val
                elif name == 'orbitdirection':
                    meta['ORBIT_DIRECTION'] = val.upper()
            logger.info(f"从 {ref_xml} 提取 SAR 元数据")
        except Exception:
            pass

    return meta


def _auto_set_reference_point(ifgram_file, geom_file):
    """在高相干稳定区域自动选择参考点。"""
    import h5py, numpy as np

    with h5py.File(str(ifgram_file), 'r') as f:
        if 'REF_Y' in f.attrs and 'REF_X' in f.attrs:
            ry, rx = int(f.attrs['REF_Y']), int(f.attrs['REF_X'])
            cor = f['coherence'][0]
            if cor[ry, rx] > 0.3:
                return

    with h5py.File(str(ifgram_file), 'r') as f:
        cor = np.mean(f['coherence'][:], axis=0)

    with h5py.File(str(geom_file), 'r') as f:
        lat = f['latitude'][:] if 'latitude' in f else None

    mask = cor.copy()
    # 排除边界 100 像素
    mask[:100, :] = 0; mask[-100:, :] = 0
    mask[:, :100] = 0; mask[:, -100:] = 0
    # 排除无效几何区 (NaN 或接近 0 的坐标)
    if lat is not None:
        mask[~np.isfinite(lat) | (lat < 0.1)] = 0

    ry, rx = np.unravel_index(np.argmax(mask), mask.shape)

    with h5py.File(str(ifgram_file), 'a') as f:
        f.attrs['REF_Y'] = str(ry)
        f.attrs['REF_X'] = str(rx)
    print(f"  参考点: y={ry}, x={rx}, coh={cor[ry,rx]:.3f}")


def _clear_qc_stale_reference_point(ifgram_file):
    """Clear stale REF_* metadata for QC-adjusted networks.

    When the interferogram network is changed by pair QC, a reference point
    copied from the original stack may no longer be included in the common
    connected component. MintPy's reference_point step will skip reselection if
    REF_Y/X already exist, which later causes unwrap bridging to fail.
    """
    import h5py

    removed = []
    with h5py.File(str(ifgram_file), "r+") as f:
        has_qc = any(name in f for name in ("qc_pair_weight", "qc_action_code")) or \
            any(str(k).startswith("qc_") for k in f.attrs.keys())
        if not has_qc:
            return removed
        preserve_locked_ref = str(f.attrs.get("qc_preserve_reference_candidate", "false")).lower() == "true"
        has_locked_ref = ("REF_Y" in f.attrs) and ("REF_X" in f.attrs)
        if preserve_locked_ref and has_locked_ref:
            return removed
        for key in ("REF_Y", "REF_X", "REF_LAT", "REF_LON"):
            if key in f.attrs:
                removed.append(key)
                del f.attrs[key]
        if removed:
            f.attrs["qc_runtime_reference_reset"] = "true"
    return removed


def _set_qc_connected_reference_point(ifgram_file, geom_file=None, water_mask_file=None):
    """Pick a reference point guaranteed to survive MintPy bridging labeling."""
    import h5py
    import numpy as np
    from mintpy.objects.conncomp import label_boundary, label_conn_comp

    ifgram_file = Path(ifgram_file)
    geom_file = Path(geom_file) if geom_file else None
    water_mask_file = Path(water_mask_file) if water_mask_file else None

    with h5py.File(str(ifgram_file), "r") as f:
        has_qc = any(name in f for name in ("qc_pair_weight", "qc_action_code")) or \
            any(str(k).startswith("qc_") for k in f.attrs.keys())
        if not has_qc:
            return None
        preserve_locked_ref = str(f.attrs.get("qc_preserve_reference_candidate", "false")).lower() == "true"
        if preserve_locked_ref and ("REF_Y" in f.attrs) and ("REF_X" in f.attrs):
            return {
                "ref_y": int(f.attrs["REF_Y"]),
                "ref_x": int(f.attrs["REF_X"]),
                "locked_candidate_rank": int(f.attrs.get("qc_selected_reference_rank", "1")),
                "preserved_locked_candidate": True,
            }

        n_ifg = f["connectComponent"].shape[0]
        if "qc_action_code" in f:
            action_code = f["qc_action_code"][:]
            kept_idx = np.flatnonzero(action_code != 2)
        else:
            drop_flags = np.asarray(f["dropIfgram"][:]).astype(bool)
            kept_idx = np.flatnonzero(drop_flags)

        if kept_idx.size == 0:
            return None

        common_mask = np.ones(f["connectComponent"].shape[1:], dtype=bool)
        coh_sum = np.zeros(common_mask.shape, dtype=np.float64)
        for idx in kept_idx:
            cc = np.asarray(f["connectComponent"][idx, :, :])
            label_img, num_label = label_conn_comp(cc, min_area=2500.0, print_msg=False)
            label_img, num_label, _ = label_boundary(label_img, num_label, erosion_size=5, print_msg=False)
            common_mask &= np.asarray(label_img != 0)
            coh_sum += np.asarray(f["coherence"][idx, :, :], dtype=np.float64)

        mean_coh = coh_sum / float(len(kept_idx))

    if water_mask_file and water_mask_file.exists():
        with h5py.File(str(water_mask_file), "r") as fwm:
            mask_name = "waterMask" if "waterMask" in fwm else next(iter(fwm.keys()))
            wm = np.asarray(fwm[mask_name][:]).astype(bool)
            common_mask &= wm

    if geom_file and geom_file.exists():
        with h5py.File(str(geom_file), "r") as fg:
            if "latitude" in fg:
                lat = np.asarray(fg["latitude"][:], dtype=np.float64)
                common_mask &= np.isfinite(lat) & (lat > 0.1)

    # Avoid unstable borders.
    common_mask[:100, :] = False
    common_mask[-100:, :] = False
    common_mask[:, :100] = False
    common_mask[:, -100:] = False

    if not np.any(common_mask):
        return None

    score = np.where(common_mask, mean_coh, -np.inf)
    ry, rx = np.unravel_index(np.argmax(score), score.shape)
    selected_coh = float(mean_coh[ry, rx])

    with h5py.File(str(ifgram_file), "r+") as f:
        f.attrs["REF_Y"] = str(int(ry))
        f.attrs["REF_X"] = str(int(rx))
        f.attrs["qc_connected_reference"] = "true"

    print(f"  QC 公共连通分量参考点: y={int(ry)}, x={int(rx)}, mean_coh={selected_coh:.3f}")
    return {"ref_y": int(ry), "ref_x": int(rx), "mean_coh": selected_coh}


def _is_qc_ifgramstack(ifgram_file):
    import h5py

    with h5py.File(str(ifgram_file), "r") as f:
        return any(name in f for name in ("qc_pair_weight", "qc_action_code")) or \
            any(str(k).startswith("qc_") for k in f.attrs.keys())


# ---------------------------------------------------------------------------
# Run MintPy
# ---------------------------------------------------------------------------
def run_mintpy(template_path=None, work_dir=None, log_file=None, ifgram_stack_path=None):
    """
    Execute smallbaselineApp.py with the custom template.
    Monitors progress by tracking step output files.
    """
    template_path = Path(template_path or cfg.TEMPLATE_PATH)
    work_dir = Path(work_dir or cfg.MINTPY_DIR)
    log_file = Path(log_file or cfg.LOG_FILE)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Activate QC stack if requested
    if ifgram_stack_path is not None:
        requested_ifgram = Path(ifgram_stack_path)
        active_ifgram = work_dir / "inputs" / "ifgramStack.h5"
        if requested_ifgram.resolve() != active_ifgram.resolve():
            backup_ifgram = work_dir / "inputs" / "ifgramStack_pre_qc_backup.h5"
            if active_ifgram.exists() or active_ifgram.is_symlink():
                if active_ifgram.is_symlink():
                    active_ifgram.unlink()
                else:
                    if not backup_ifgram.exists():
                        active_ifgram.rename(backup_ifgram)
                    else:
                        active_ifgram.unlink()
            try:
                target = os.path.relpath(requested_ifgram, active_ifgram.parent)
                active_ifgram.symlink_to(target)
            except OSError:
                import shutil

                shutil.copy2(requested_ifgram, active_ifgram)

    # Checkpoint: check if velocity.h5 already exists
    velocity_file = work_dir / "velocity.h5"
    if velocity_file.exists():
        print(f"MintPy 结果已存在 ({velocity_file})。删除后可重新处理。")
        return

    active_ifgram = work_dir / "inputs" / "ifgramStack.h5"
    if active_ifgram.exists():
        removed_ref_attrs = _clear_qc_stale_reference_point(active_ifgram)
        if removed_ref_attrs:
            print(f"  检测到 QC 调整版 ifgramStack，已清除旧参考点属性: {removed_ref_attrs}")
        _set_qc_connected_reference_point(
            active_ifgram,
            geom_file=work_dir / "inputs" / "geometryRadar.h5",
            water_mask_file=work_dir / "waterMask.h5",
        )

    print("启动 MintPy 时序反演...")
    print(f"  模板: {template_path}")
    print(f"  工作目录: {work_dir}")
    mintpy_settings = recommend_mintpy_settings(requested_workers=cfg.N_WORKERS)
    print(f"  资源自适应: numWorker={mintpy_settings['num_worker']} maxMemory={mintpy_settings['max_memory_gb']}GB")

    cmd = [sys.executable, "-m", "mintpy.cli.smallbaselineApp",
           str(template_path), "--dir", str(work_dir)]

    # 如果 ifgramStack.h5 已存在 (由 build_mintpy_hdf5 创建), 跳过 load_data
    ifgram_exists = (work_dir / "inputs" / "ifgramStack.h5").exists()
    if ifgram_exists:
        start_step = "modify_network"
        if _is_qc_ifgramstack(work_dir / "inputs" / "ifgramStack.h5"):
            # QC has already decided keep/downweight/drop. Do not let MintPy
            # overwrite dropIfgram again in modify_network.
            start_step = "reference_point"
        cmd.extend(["--start", start_step])

    with open(log_file, "a") as lf:
        process = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(work_dir),
            env=build_thread_limited_env(os.environ.copy(), threads_per_process=1),
        )

        # Monitor progress
        _monitor_mintpy_progress(process, work_dir)

        _, stderr = process.communicate()

    if process.returncode != 0:
        logger.error(f"MintPy 失败: {stderr}")
        print(f"[ERROR] MintPy 失败，详情见 {log_file}")
        print(f"  stderr: {stderr[:500]}")
        raise RuntimeError(f"MintPy 失败，返回码 {process.returncode}")

    for key_file in ["timeseries.h5", "velocity.h5", "temporalCoherence.h5"]:
        if (work_dir / key_file).exists():
            print(f"  {key_file}")
        else:
            print(f"  [WARN] 缺失: {key_file}")

    # 板块运动校正（独立步骤，在 smallbaselineApp 完成后执行）
    _correct_plate_motion(work_dir, log_file)

    print("MintPy 处理完成。")


def _correct_plate_motion(work_dir, log_file):
    """
    对速率图施加板块运动校正（欧亚板块 ITRF2014）。

    MintPy 的板块运动校正是独立命令 plate_motion.py，不在 smallbaselineApp 流程内。
    此函数在 smallbaselineApp 完成后执行：
      1. 用 geocoded 几何文件计算欧亚板块 LOS 速率贡献
      2. 从 velocity.h5 中减去板块运动分量
      3. 输出 velocity_platMotionCorrected.h5 和更新 velocity.h5 的元数据

    输入: geo/geo_geometryRadar.h5（必须已由 geocode 步骤生成）
    输出: velocity_platMotionCorrected.h5
    """
    work_dir = Path(work_dir)
    velocity_file = work_dir / "velocity.h5"
    # 使用雷达坐标几何文件（与 velocity.h5 尺寸一致）
    # calc_plate_motion 内部会自动 geocode → 计算板块运动 → radar-code 回来
    geom_file = work_dir / "inputs" / "geometryRadar.h5"
    pmm_file = work_dir / "velocity_PMM_Eurasia.h5"
    corrected_file = work_dir / "velocity_platMotionCorrected.h5"

    if not velocity_file.exists():
        logger.warning("velocity.h5 不存在，跳过板块运动校正")
        return

    if corrected_file.exists():
        print(f"  板块运动校正结果已存在: {corrected_file.name}")
        return

    if not geom_file.exists():
        logger.warning(f"几何文件不存在 ({geom_file})，跳过板块运动校正")
        print(f"  [WARN] 板块运动校正需要 geometryRadar.h5")
        return

    print("\n板块运动校正 (欧亚板块 ITRF2014)...")

    try:
        from mintpy.plate_motion import calc_plate_motion
        from mintpy.objects.euler_pole import ITRF2014_PMM
        from mintpy.utils import readfile, writefile
        import numpy as np

        # 欧亚板块旋转极参数 (ITRF2014 Table 1)
        plate = ITRF2014_PMM["Eurasia"]
        omega_cart = [plate.omega_x, plate.omega_y, plate.omega_z]
        print(f"  Eurasia Euler pole: wx={plate.omega_x}, wy={plate.omega_y}, wz={plate.omega_z} mas/yr")

        # 计算欧亚板块在 LOS 方向的速率贡献
        # calc_plate_motion 返回 (ve, vn, vu, vlos) 四元组
        # 雷达坐标输入 → 内部自动 geocode → 计算 → 自动 radar-code 回来
        _, _, _, vlos = calc_plate_motion(
            geom_file=str(geom_file),
            omega_cart=omega_cart,
            pmm_file=str(pmm_file),
            pmm_comp="enu2los",
        )
        vlos = np.array(vlos)
        print(f"  板块 LOS 速率范围: {vlos.min()*1000:.2f} ~ {vlos.max()*1000:.2f} mm/yr")

        # 从速率场中减去板块运动
        vel_data, vel_atr = readfile.read(str(velocity_file), datasetName="velocity")
        vel_corrected = vel_data - vlos.astype(vel_data.dtype)

        # 写出校正后速率
        vel_atr["FILE_PATH"] = str(corrected_file)
        vel_atr["plate_motion_correction"] = "Eurasia ITRF2014"
        writefile.write(
            {"velocity": vel_corrected},
            out_file=str(corrected_file),
            metadata=vel_atr,
            ref_file=str(velocity_file),
        )

        valid = vel_corrected[np.isfinite(vel_corrected) & (vel_corrected != 0)]
        print(f"  校正后速率范围: {valid.min()*1000:.1f} ~ {valid.max()*1000:.1f} mm/yr")
        print(f"  已保存: {corrected_file.name}")

    except Exception as e:
        logger.error(f"板块运动校正失败: {e}")
        print(f"  [ERROR] 板块运动校正失败: {e}")


def _monitor_mintpy_progress(process, work_dir):
    """Monitor MintPy progress by checking step indicator files."""
    pbar = tqdm(MINTPY_STEPS, desc="MintPy steps", unit="step")
    completed = set()

    while process.poll() is None:
        for step in MINTPY_STEPS:
            if step in completed:
                continue
            # Check for various output indicators
            indicators = [
                work_dir / f"{step}.h5",
                work_dir / "inputs" / "ifgramStack.h5",
                work_dir / "velocity.h5",
                work_dir / "timeseries.h5",
            ]
            for ind in indicators:
                if ind.exists() and step not in completed:
                    completed.add(step)
                    pbar.update(1)
                    break
        time.sleep(3)

    # Final pass
    for step in MINTPY_STEPS:
        if step not in completed:
            completed.add(step)
            pbar.update(1)
    pbar.close()
