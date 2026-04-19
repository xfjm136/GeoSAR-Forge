"""
大气校正方案选择与外部大气数据准备引导。

当前策略是：
- 在 SLC 下载前记录项目级大气方案
- 在 MintPy 前的主链 QC 阶段再做外部数据准备与覆盖检查

支持方案：
- 数据驱动: GACOS / ERA5 / MERRA-2
- 算法驱动: 相位-高程拟合 / 时空滤波 / 不校正
"""
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Iterable

from tqdm.auto import tqdm

from . import config as cfg
from .config import logger


# ============================================================================
# 交互式大气校正方案选择
# ============================================================================
def choose_atmo_correction(scenes, bounds):
    """
    场景筛选完成后、SLC 下载前，引导用户选择大气校正方案。

    Returns:
        dict: {"method": str, "model": str|None, "dir": Path|None}
              用于后续主链 QC 与 generate_mintpy_template() 消费
    """
    print("\n" + "=" * 70)
    print(" 对流层大气延迟校正方案选择")
    print("=" * 70)

    print("""
 ┌─────────────────────────────────────────────────────────────────┐
 │             需要外部天气数据 (精度高, 推荐)                     │
 ├─────────────────────────────────────────────────────────────────┤
 │ [1] GACOS (推荐, 精确下载范围后置到 MintPy 前 QC)              │
 │     高分辨率迭代对流层分解, 空间分辨率 ~90m                     │
 │     当前阶段只记录方案; MintPy 前再给精确范围与日期清单        │
 │     适用: 局部区域, 精度要求高                                  │
 │                                                                 │
 │ [2] ERA5 (MintPy 前自动准备)                                    │
 │     ECMWF 全球再分析, 空间分辨率 ~31km, 时间分辨率 1h          │
 │     在主链 QC 阶段通过 CDS API 按月批量准备                     │
 │     适用: 大范围, 长时序, 全球可用                               │
 │                                                                 │
 │ [3] MERRA-2 (MintPy 前自动准备)                                 │
 │     NASA 全球再分析, 空间分辨率 ~50km, 时间分辨率 3h            │
 │     在主链 QC 阶段通过 PyAPS + wget 准备, 需 Earthdata 账号     │
 │     适用: ERA5 不可用时的替代方案, 北美地区尤佳                  │
 ├─────────────────────────────────────────────────────────────────┤
 │             无需外部数据 (纯算法)                                │
 ├─────────────────────────────────────────────────────────────────┤
 │ [4] 相位-高程经验拟合 (Phase-DEM Correlation)                   │
 │     假设: 大气延迟主要与地形高程线性相关                        │
 │     原理: φ_atm ≈ a·h + b, 逐景拟合高程-相位关系后扣除        │
 │     优点: 无需任何外部数据, 计算快                               │
 │     局限: 仅能去除地形相关的大气分量, 无法去除湍流分量          │
 │     适用: 高差大的山区, 水汽分层效应显著的场景                   │
 │                                                                 │
 │ [5] 时空滤波 (Spatial-Temporal Filter)                          │
 │     假设: 大气信号空间平滑+时间随机, 形变信号时间相关           │
 │     原理: 对时序残差施加高通时间滤波(去趋势) +                  │
 │            低通空间滤波(平滑大气), 分离形变与大气               │
 │     MintPy 通过 residual_RMS 和 deramp 步骤部分实现             │
 │     优点: 无需外部数据, 对湍流大气也有一定效果                   │
 │     局限: 可能平滑掉空间尺度小的真实形变信号                    │
 │     适用: 景数 ≥20, 城市/平原区域, 形变信号空间尺度远大于大气   │
 │                                                                 │
 │ [6] 不校正                                                      │
 │     跳过大气校正, 仅依赖时序平均抑制大气噪声                    │
 │     适用: 快速预览, 或形变速率远大于大气残差 (>50 mm/yr)        │
 └─────────────────────────────────────────────────────────────────┘
""")

    choice_map = {
        "1": "gacos", "gacos": "gacos",
        "2": "era5",  "era5": "era5",
        "3": "merra2", "merra-2": "merra2", "merra": "merra2",
        "4": "height_correlation", "dem": "height_correlation",
        "5": "spatial_filter", "filter": "spatial_filter",
        "6": "no", "skip": "no", "none": "no",
    }

    while True:
        choice = input("请选择大气校正方案 (1-6): ").strip().lower()
        method = choice_map.get(choice)
        if method:
            break
        print("无效输入，请输入 1-6")

    result = {"method": method, "model": None, "dir": None}

    if method == "gacos":
        print("\n已选择: GACOS")
        print("  当前阶段只记录方案，不要求立刻下载 GACOS 数据。")
        print("  在 MintPy 前的主链 QC 阶段，会根据实际 Dolphin 网格和 MintPy 子窗口")
        print("  自动计算更精确的 GACOS 建议下载范围，并生成日期清单/下载指引。")
        print("  建议等到该阶段拿到精确范围后再下载，这样通常只需要下载一次。")
        result["dir"] = cfg.GACOS_DIR

    elif method == "era5":
        print("\n已选择: ERA5")
        print("  当前阶段只记录方案，不立即下载天气数据。")
        print("  在 MintPy 前的主链 QC 阶段，会按当前场景日期和 AOI 自动准备 ERA5 数据。")
        result["model"] = "ERA5"
        result["dir"] = cfg.ERA5_DIR

    elif method == "merra2":
        print("\n已选择: MERRA-2")
        print("  当前阶段只记录方案，不立即下载天气数据。")
        print("  在 MintPy 前的主链 QC 阶段，会按当前场景日期和 AOI 自动准备 MERRA-2 数据。")
        result["model"] = "MERRA"
        result["dir"] = cfg.WORK_DIR / "MERRA2"

    elif method == "height_correlation":
        print("\n已选择: 相位-高程经验拟合")
        print("  MintPy 将逐景拟合 φ = a·h + b 并扣除高程相关的大气分量")

    elif method == "spatial_filter":
        print("\n已选择: 时空滤波")
        print("  MintPy 将通过 deramp + 时序残差分析 分离大气与形变")

    elif method == "no":
        print("\n已选择: 不校正大气")
        print("  时序平均可部分抑制随机大气噪声，但系统性大气误差将残留")

    print(f"\n大气校正方案: {method.upper()}")
    return result


# ============================================================================
# GACOS 引导
# ============================================================================
def compute_recommended_gacos_bounds(bounds, margin_ratio=0.20, min_margin_deg=0.06):
    """
    为 GACOS 下载给出一个比严格 AOI 更保守的建议范围。

    设计原则:
    - 不直接等于严格 AOI bbox，避免后续 AOI 像素裁剪、Dolphin 下采样和
      MintPy 子窗口检查时出现“日期齐全但空间覆盖不足”的情况。
    - 默认给出 20% 相对外扩，且每个方向至少 0.06 度。
    """
    north = float(bounds["N"])
    south = float(bounds["S"])
    west = float(bounds["W"])
    east = float(bounds["E"])

    lat_span = max(north - south, 1e-6)
    lon_span = max(east - west, 1e-6)
    lat_margin = max(float(min_margin_deg), lat_span * float(margin_ratio))
    lon_margin = max(float(min_margin_deg), lon_span * float(margin_ratio))

    return {
        "N": min(90.0, north + lat_margin),
        "S": max(-90.0, south - lat_margin),
        "W": max(-180.0, west - lon_margin),
        "E": min(180.0, east + lon_margin),
        "margin_ratio": float(margin_ratio),
        "min_margin_deg": float(min_margin_deg),
    }


def print_gacos_guide(scenes, bounds):
    """打印 GACOS 网站所需的提交参数。"""
    dates = sorted(set(_extract_date_str(s) for s in scenes))
    recommended = compute_recommended_gacos_bounds(bounds)

    utc_time = scenes[0].properties.get("startTime", "")
    utc_hms = utc_time.split("T")[1][:8] if "T" in utc_time else "Unknown"

    print("\n" + "=" * 60)
    print(" GACOS 对流层延迟数据下载引导")
    print(" 提交地址: http://www.gacos.net/")
    print("=" * 60)
    print()
    print(f"严格 AOI 边界 (Bounding Box):")
    print(f"  North: {bounds['N']:.4f}")
    print(f"  South: {bounds['S']:.4f}")
    print(f"  West:  {bounds['W']:.4f}")
    print(f"  East:  {bounds['E']:.4f}")
    print()
    print("建议提交范围 (推荐使用，适配主链 QC / AOI 像素裁剪 / MintPy 子窗口):")
    print(f"  North: {recommended['N']:.4f}")
    print(f"  South: {recommended['S']:.4f}")
    print(f"  West:  {recommended['W']:.4f}")
    print(f"  East:  {recommended['E']:.4f}")
    print(f"  说明: 在严格 AOI 基础上按 20% 外扩，且每个方向至少扩 0.06°。")
    print()
    print(f"SAR 获取 UTC 时间: {utc_hms}")
    print()
    print(f"日期列表 ({len(dates)} 个日期，请复制下方内容):")
    print("-" * 20)
    for d in dates:
        print(d)
    print("-" * 20)
    print()
    print(f"请将下载的 .ztd 和 .ztd.rsc 文件放入:")
    print(f"  {cfg.GACOS_DIR}")
    print("=" * 60)


def convert_gacos_tif_to_ztd(gacos_dir=None):
    """
    将 GACOS .ztd.tif (GeoTIFF) 自动转换为 MintPy 所需的 .ztd + .ztd.rsc。
    MintPy 需要: 二进制 float32 (.ztd) + ROI_PAC 元数据 (.ztd.rsc)。
    """
    import numpy as np
    gacos_dir = Path(gacos_dir or cfg.GACOS_DIR)
    if not gacos_dir.exists():
        return 0
    tifs = sorted(gacos_dir.glob("*.ztd.tif"))
    if not tifs:
        return 0
    try:
        import rasterio
    except ImportError:
        return 0

    converted = 0
    for tif in tifs:
        date_str = tif.name.split(".")[0]
        ztd_file = gacos_dir / f"{date_str}.ztd"
        rsc_file = gacos_dir / f"{date_str}.ztd.rsc"
        if ztd_file.exists() and rsc_file.exists():
            continue
        with rasterio.open(tif) as src:
            data = src.read(1).astype(np.float32)
            t = src.transform
            w, h = src.width, src.height
        data.tofile(str(ztd_file))
        rsc_file.write_text(
            f"WIDTH           {w}\nFILE_LENGTH     {h}\n"
            f"X_FIRST         {t.c}\nY_FIRST         {t.f}\n"
            f"X_STEP          {t.a}\nY_STEP          {t.e}\n"
            f"X_UNIT          degrees\nY_UNIT          degrees\n"
            f"Z_OFFSET        0\nZ_SCALE         1\n"
            f"PROJECTION      LATLON\nDATUM           WGS84\n"
        )
        converted += 1
    if converted:
        print(f"GACOS 格式转换: {converted} 个 .ztd.tif → .ztd + .ztd.rsc")
    return converted


def _extract_date_str(scene):
    """提取 YYYYMMDD 日期。"""
    if isinstance(scene, str):
        text = scene.strip()
        if len(text) == 8 and text.isdigit():
            return text
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            return text[:10].replace("-", "")
    t = scene.properties.get("startTime", "")
    return t[:10].replace("-", "")


def prepare_external_atmo_inputs(
    atmo_config: dict | None,
    scenes_or_dates: Iterable,
    bounds: dict,
) -> dict:
    """
    在 MintPy 前按已选择方案自动准备外部大气数据。

    当前策略:
    - GACOS: 不自动下载，只保留为后续覆盖检查对象
    - ERA5: 在此阶段自动下载
    - MERRA-2: 在此阶段自动下载
    - 其他方案: 无需准备
    """
    method = (atmo_config or {}).get("method", "no")
    summary = {
        "method": method,
        "prepared": False,
        "output_dir": None,
        "n_dates": 0,
        "existing_count": 0,
        "final_count": 0,
        "message": "no_external_data_needed",
    }
    date_list = sorted(set(_extract_date_str(s) for s in scenes_or_dates))
    summary["n_dates"] = len(date_list)

    if method == "gacos":
        summary["output_dir"] = str(cfg.GACOS_DIR)
        summary["message"] = "defer_to_qc_guidance"
        return summary

    if method == "era5":
        output_dir = Path((atmo_config or {}).get("dir") or cfg.ERA5_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary["output_dir"] = str(output_dir)
        summary["existing_count"] = sum(1 for d in date_list if (output_dir / f"ERA5_{d}.grb").exists())
        setup_era5_credentials()
        download_era5(date_list, bounds, output_dir=output_dir)
        summary["final_count"] = sum(1 for d in date_list if (output_dir / f"ERA5_{d}.grb").exists())
        summary["prepared"] = summary["final_count"] >= len(date_list)
        summary["message"] = "auto_download_era5"
        return summary

    if method == "merra2":
        output_dir = Path((atmo_config or {}).get("dir") or (cfg.WORK_DIR / "MERRA2"))
        output_dir.mkdir(parents=True, exist_ok=True)
        summary["output_dir"] = str(output_dir)
        summary["existing_count"] = sum(1 for d in date_list if (output_dir / f"merra-{d}-12.nc4").exists())
        setup_merra2_credentials()
        download_merra2(date_list, bounds, output_dir=output_dir)
        summary["final_count"] = sum(1 for d in date_list if (output_dir / f"merra-{d}-12.nc4").exists())
        summary["prepared"] = summary["final_count"] >= len(date_list)
        summary["message"] = "auto_download_merra2"
        return summary

    summary["message"] = "no_external_data_needed"
    return summary


# ============================================================================
# ERA5 凭证 + 批量下载
# ============================================================================
def setup_era5_credentials(url=None, key=None):
    """写入 ~/.cdsapirc。"""
    url = url or cfg.ERA5_URL
    key = key or cfg.ERA5_KEY
    if not key:
        key = cfg.require_config_vars("ERA5_KEY")
    rc_path = Path.home() / ".cdsapirc"

    content = f"url: {url}\nkey: {key}\n"
    if rc_path.exists() and rc_path.read_text().strip() == content.strip():
        print("ERA5 凭证已配置。")
        return
    rc_path.write_text(content)
    print(f"ERA5 凭证已写入 {rc_path}")


def download_era5(scenes, bounds, output_dir=None):
    """
    批量下载 ERA5 气压层数据。按月分组，一次 API 请求获取同月所有日期。
    """
    try:
        import cdsapi
    except ImportError:
        print("[WARN] cdsapi 未安装，ERA5 下载跳过。")
        return

    output_dir = Path(output_dir or cfg.ERA5_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_list = sorted(set(_extract_date_str(s) for s in scenes))
    needed = [d for d in date_list if not (output_dir / f"ERA5_{d}.grb").exists()]
    skipped = len(date_list) - len(needed)

    if not needed:
        print(f"ERA5: 全部 {len(date_list)} 个日期已存在，跳过")
        return

    area = [bounds["N"] + 1, bounds["W"] - 1, bounds["S"] - 1, bounds["E"] + 1]

    monthly = defaultdict(list)
    for d in needed:
        monthly[d[:6]].append(d)

    client = cdsapi.Client(quiet=True)

    print(f"ERA5 批量下载: {len(needed)} 个日期, {len(monthly)} 个月份批次 "
          f"({skipped} 已跳过)")

    for ym in tqdm(sorted(monthly.keys()), desc="ERA5 批量下载", unit="月"):
        dates_in_month = monthly[ym]
        year, month = ym[:4], ym[4:6]
        days = sorted(set(d[6:8] for d in dates_in_month))

        batch_file = output_dir / f"ERA5_batch_{ym}.grb"
        try:
            client.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis", "format": "grib",
                    "variable": ["geopotential", "specific_humidity", "temperature"],
                    "pressure_level": [str(p) for p in range(100, 1050, 25)],
                    "year": year, "month": month, "day": days,
                    "time": [f"{h:02d}:00" for h in range(0, 24, 6)],
                    "area": area,
                },
                str(batch_file),
            )

            if len(days) == 1:
                batch_file.rename(output_dir / f"ERA5_{dates_in_month[0]}.grb")
            else:
                _split_grib_by_date(batch_file, dates_in_month, output_dir, "ERA5")
                batch_file.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"ERA5 批次下载失败 {ym}: {e}")
            batch_file.unlink(missing_ok=True)
            for d in dates_in_month:
                _download_single_era5(client, d, area, output_dir)

    downloaded = sum(1 for d in needed if (output_dir / f"ERA5_{d}.grb").exists())
    print(f"ERA5 完成: {downloaded} 新下载, {skipped} 已跳过")


def _download_single_era5(client, date_str, area, output_dir):
    """单日 ERA5 下载 (批量失败的后备)。"""
    output_file = output_dir / f"ERA5_{date_str}.grb"
    if output_file.exists():
        return
    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
    try:
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis", "format": "grib",
                "variable": ["geopotential", "specific_humidity", "temperature"],
                "pressure_level": [str(p) for p in range(100, 1050, 25)],
                "year": year, "month": month, "day": day,
                "time": [f"{h:02d}:00" for h in range(0, 24, 6)],
                "area": area,
            },
            str(output_file),
        )
    except Exception as e:
        logger.error(f"ERA5 单日下载失败 {date_str}: {e}")
        output_file.unlink(missing_ok=True)


# ============================================================================
# MERRA-2 凭证 + 自动下载 (通过 PyAPS wget)
# ============================================================================
def setup_merra2_credentials(force=False):
    """
    配置 ~/.netrc + PyAPS model.cfg 中的 NASA Earthdata 凭证。
    force=True 时强制重新输入。
    """
    netrc_path = Path.home() / ".netrc"

    if not force and netrc_path.exists():
        content = netrc_path.read_text()
        if "urs.earthdata.nasa.gov" in content:
            import re
            m = re.search(r'machine\s+urs\.earthdata\.nasa\.gov.*?login\s+(\S+)',
                          content, re.DOTALL)
            if m:
                print(f"MERRA-2 凭证已配置: {m.group(1)}")
                _sync_netrc_to_pyaps()
                return

    print("\n" + "=" * 50)
    print(" MERRA-2 需要 NASA Earthdata 账号")
    print(" 注册: https://urs.earthdata.nasa.gov/")
    print(" 还需在账号中授权 'NASA GESDISC DATA ARCHIVE' 应用")
    print("=" * 50)
    user = input("NASA Earthdata 用户名: ").strip()
    pwd = input("NASA Earthdata 密码: ").strip()

    # 清除旧条目后重写 .netrc
    if netrc_path.exists():
        import re
        content = netrc_path.read_text()
        content = re.sub(
            r'machine\s+urs\.earthdata\.nasa\.gov\s*\n(\s+\w+\s+\S+\n)*',
            '', content
        )
    else:
        content = ""
    content += f"\nmachine urs.earthdata.nasa.gov\n  login {user}\n  password {pwd}\n"
    netrc_path.write_text(content)
    netrc_path.chmod(0o600)

    # 清除旧 cookies
    (Path.home() / ".urs_cookies").unlink(missing_ok=True)

    _sync_netrc_to_pyaps()
    print(f"凭证已更新: {netrc_path}")


def _sync_netrc_to_pyaps():
    """将 ~/.netrc 凭证同步到 PyAPS model.cfg。"""
    try:
        import netrc as _netrc
        creds = _netrc.netrc()
        auth = creds.authenticators("urs.earthdata.nasa.gov")
        if not auth:
            return

        import pyaps3 as pa
        import configparser
        cfg_file = Path(pa.__file__).parent / "model.cfg"
        if not cfg_file.exists():
            return

        cfg = configparser.RawConfigParser(delimiters='=')
        cfg.read(str(cfg_file))
        if not cfg.has_section("MERRA"):
            cfg.add_section("MERRA")
        cfg.set("MERRA", "user", auth[0])
        cfg.set("MERRA", "password", auth[2])
        with open(cfg_file, 'w') as f:
            cfg.write(f)
    except Exception:
        pass


def download_merra2(scenes, bounds, output_dir=None):
    """
    直接从 NASA GES DISC 下载 MERRA-2 气压层数据。
    自动尝试 MERRA2_400/401 两种文件前缀 (NASA 2020 年流变更)。
    断连自动重试 + 断点续传。
    """
    import requests
    import netrc as _netrc
    import time

    output_dir = Path(output_dir or cfg.WORK_DIR / "MERRA2")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_list = sorted(set(_extract_date_str(s) for s in scenes))
    needed = [d for d in date_list if not _valid_merra_file(output_dir, d)]

    if not needed:
        print(f"MERRA-2: 全部 {len(date_list)} 个日期已存在，跳过")
        return

    # 清理之前的空文件
    _cleanup_empty_merra(output_dir, needed)

    # 获取凭证
    try:
        creds = _netrc.netrc()
        auth = creds.authenticators("urs.earthdata.nasa.gov")
        if not auth:
            raise ValueError("无凭证")
    except Exception:
        print("[ERROR] ~/.netrc 中未找到 urs.earthdata.nasa.gov 凭证")
        setup_merra2_credentials(force=True)
        creds = _netrc.netrc()
        auth = creds.authenticators("urs.earthdata.nasa.gov")
        if not auth:
            return

    session = requests.Session()
    session.auth = (auth[0], auth[2])
    session.trust_env = True

    base = "https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I6NPANA.5.12.4"
    prefixes = ["MERRA2_400", "MERRA2_401"]

    print(f"MERRA-2 下载: {len(needed)} 个日期 (HTTP 直接下载)")

    success, fail = 0, 0
    for d in tqdm(needed, desc="MERRA-2 下载", unit="日"):
        year, month = d[:4], d[4:6]
        out_file = output_dir / f"merra-{d}-12.nc4"

        downloaded = False
        for prefix in prefixes:
            fname = f"{prefix}.inst6_3d_ana_Np.{d}.nc4"
            url = f"{base}/{year}/{month}/{fname}"

            for attempt in range(1, 4):  # 最多 3 次重试
                try:
                    # 断点续传
                    existing = out_file.stat().st_size if out_file.exists() else 0
                    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}

                    resp = session.get(url, stream=True, timeout=120,
                                       allow_redirects=True, headers=headers)

                    if resp.status_code == 404:
                        break  # 该前缀不存在，试下一个
                    if resp.status_code == 401:
                        print(f"\n[WARN] 认证失败")
                        retry = input("重新输入凭证? (y/n): ").strip().lower()
                        if retry in ("y", "yes", ""):
                            setup_merra2_credentials(force=True)
                            creds = _netrc.netrc()
                            auth = creds.authenticators("urs.earthdata.nasa.gov")
                            session.auth = (auth[0], auth[2])
                            continue
                        else:
                            return
                    if resp.status_code == 500:
                        if attempt < 3:
                            time.sleep(10 * attempt)
                            continue
                        break

                    # HTML 登录页检测
                    if "text/html" in resp.headers.get("content-type", ""):
                        break

                    resp.raise_for_status()

                    mode = "ab" if existing > 0 and resp.status_code == 206 else "wb"
                    with open(out_file, mode) as f:
                        for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                            f.write(chunk)

                    if _valid_merra_file(output_dir, d):
                        downloaded = True
                        break
                    else:
                        out_file.unlink(missing_ok=True)
                        if attempt < 3:
                            time.sleep(5)

                except requests.exceptions.ConnectionError:
                    if attempt < 3:
                        time.sleep(10 * attempt)
                    else:
                        out_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"MERRA-2 {prefix} {d} 尝试{attempt}: {e}")
                    if attempt >= 3:
                        out_file.unlink(missing_ok=True)

            if downloaded:
                success += 1
                break

        if not downloaded:
            fail += 1

    print(f"MERRA-2 完成: {success} 成功, {fail} 失败")


def _valid_merra_file(output_dir, date_str, min_size=1_000_000):
    """检查 MERRA-2 文件是否有效 (存在且 > 1MB)。"""
    f = output_dir / f"merra-{date_str}-12.nc4"
    return f.exists() and f.stat().st_size > min_size


def _cleanup_empty_merra(output_dir, dates):
    """删除空的/过小的 MERRA-2 文件。"""
    cleaned = 0
    for d in dates:
        f = output_dir / f"merra-{d}-12.nc4"
        if f.exists() and f.stat().st_size < 1_000_000:
            f.unlink()
            cleaned += 1
    if cleaned:
        print(f"  清理 {cleaned} 个无效文件 (0KB/过小)")


# ============================================================================
# GRIB 拆分工具
# ============================================================================
def _split_grib_by_date(batch_file, dates, output_dir, prefix):
    """按 dataDate 拆分 GRIB 文件为逐日文件。"""
    try:
        import eccodes
        writers = {}
        with open(batch_file, 'rb') as f:
            while True:
                msgid = eccodes.codes_grib_new_from_file(f)
                if msgid is None:
                    break
                data_date = str(eccodes.codes_get(msgid, 'dataDate'))
                if data_date not in writers:
                    writers[data_date] = open(output_dir / f"{prefix}_{data_date}.grb", 'wb')
                eccodes.codes_write(msgid, writers[data_date])
                eccodes.codes_release(msgid)
        for w in writers.values():
            w.close()
    except ImportError:
        import shutil
        for d in dates:
            target = output_dir / f"{prefix}_{d}.grb"
            if not target.exists():
                shutil.copy2(batch_file, target)
