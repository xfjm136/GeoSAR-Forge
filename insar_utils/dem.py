"""
DEM download and preparation for ISCE2.
"""
import logging
import subprocess
from pathlib import Path

import requests
from tqdm.auto import tqdm

from .config import OPENTOPO_KEY, logger, require_config_vars

# ---------------------------------------------------------------------------
# SRTM 30m DEM via OpenTopography API
# ---------------------------------------------------------------------------
def download_srtm_dem(bounds, output_dir, api_key=None):
    """
    Download SRTM GL1 (30m) DEM from OpenTopography and prepare for ISCE2.

    Args:
        bounds: dict with N, S, W, E keys
        output_dir: Path to DEM directory
        api_key: OpenTopography API key (default from config)
    """
    api_key = api_key or OPENTOPO_KEY or require_config_vars("OPENTOPO_KEY")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dem_tif = output_dir / "dem.tif"
    dem_wgs84 = output_dir / "dem.dem.wgs84"

    if dem_tif.exists():
        print(f"DEM already exists: {dem_tif}")
        return dem_tif

    # Pad bounds slightly for safe coverage
    pad = 0.1
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "SRTMGL1",
        "south": bounds["S"] - pad,
        "north": bounds["N"] + pad,
        "west": bounds["W"] - pad,
        "east": bounds["E"] + pad,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }

    print(f"Downloading SRTM 30m DEM for "
          f"N={bounds['N']:.3f} S={bounds['S']:.3f} "
          f"W={bounds['W']:.3f} E={bounds['E']:.3f} ...")

    resp = requests.get(url, params=params, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dem_tif, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="DEM download"
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"DEM saved: {dem_tif}")

    # Convert to ISCE2 format using gdal_translate + fixImageXml.py
    _prepare_isce2_dem(dem_tif, dem_wgs84)

    return dem_tif


def _prepare_isce2_dem(dem_tif, dem_wgs84):
    """
    将 GeoTIFF DEM 转换为 ISCE2 兼容格式 (.dem.wgs84 + .xml + .vrt)。
    使用 ISCE2 Python API 直接生成 XML 元数据。
    """
    import numpy as np

    try:
        import rasterio
        import isce
        import isceobj
        from isceobj.Image.DemImage import DemImage

        with rasterio.open(str(dem_tif)) as src:
            data = src.read(1).astype(np.float32)
            width = src.width
            length = src.height
            x_first = src.transform.c
            y_first = src.transform.f
            x_step = src.transform.a
            y_step = src.transform.e

        # 写入 big-endian float32 二进制 (ISCE2 标准格式)
        data.astype('>f4').tofile(str(dem_wgs84))

        # 用 ISCE2 API 生成 XML + VRT
        img = DemImage()
        img.setFilename(str(dem_wgs84))
        img.setWidth(width)
        img.setLength(length)
        img.scheme = 'BIL'
        img.dataType = 'FLOAT'
        img.setAccessMode('READ')
        img.firstLatitude = y_first + y_step / 2   # 像素中心
        img.firstLongitude = x_first + x_step / 2
        img.deltaLatitude = y_step
        img.deltaLongitude = x_step
        img.renderHdr()

        logger.info(f"ISCE2 DEM 已准备: {dem_wgs84}")
        print(f"ISCE2 DEM 已准备: {dem_wgs84}")
    except Exception as e:
        logger.warning(f"ISCE2 DEM 准备失败: {e}")
        print(f"[WARN] ISCE2 DEM 准备失败: {e}")
