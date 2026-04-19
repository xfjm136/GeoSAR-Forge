"""
智能选片: 行政区划 AOI 解析、均匀采样、健壮下载。
支持输入中文市/县级行政区名称，按省-市-县层级人工复审，检查空间连续性。
"""
import os
import json
import zipfile
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict

import asf_search as asf
from geopy.geocoders import Nominatim
from shapely.geometry import box, shape, MultiPolygon, Polygon
from shapely.ops import unary_union
from tqdm.auto import tqdm

from .config import logger, require_config_vars
from .hardware import recommend_cpu_workers


# ---------------------------------------------------------------------------
# 行政区划 AOI 解析
# ---------------------------------------------------------------------------
def resolve_aoi(name_or_bounds, buffer_deg=0.05):
    """
    解析研究区域 (AOI)。支持三种输入方式:

    1. 单个中文地名字符串: "兰州市" → 地理编码+缓冲区
    2. 行政区名称列表: ["城关区", "七里河区"] → 多区域合并+连续性检查
    3. NESW 边界字典: {"N":36.2, "S":35.8, "W":103.5, "E":104.1}

    对于列表输入 (方式2)，流程:
      1) 逐个地理编码，获取行政边界多边形
      2) 按 省-市-县 层级输出供人工复审
      3) 人工确认后检查空间连续性
      4) 不连续则终止

    Args:
        name_or_bounds: str, list[str], 或 dict
        buffer_deg: 缓冲区 (度), 仅用于单点地理编码

    Returns:
        (wkt_str, bounds_dict) 其中 bounds_dict = {"N","S","W","E"}
    """
    # 方式三: 直接边界
    if isinstance(name_or_bounds, dict):
        b = name_or_bounds
        poly = box(b["W"], b["S"], b["E"], b["N"])
        print(f"AOI 直接指定: N={b['N']}, S={b['S']}, W={b['W']}, E={b['E']}")
        return poly.wkt, b

    # 方式一: 单个地名
    if isinstance(name_or_bounds, str):
        return _resolve_single_name(name_or_bounds, buffer_deg)

    # 方式二: 行政区名称列表
    if isinstance(name_or_bounds, (list, tuple)):
        return _resolve_admin_regions(name_or_bounds, buffer_deg)

    raise ValueError(f"不支持的 AOI 类型: {type(name_or_bounds)}")


def _resolve_single_name(name, buffer_deg):
    """单个中文地名 → 地理编码 + 缓冲矩形。"""
    geolocator = Nominatim(user_agent="insar_pipeline_v1", timeout=15)
    location = geolocator.geocode(name, language="zh", addressdetails=True)
    if location is None:
        raise ValueError(f"无法地理编码: {name}")

    lat, lon = location.latitude, location.longitude

    # 尝试获取行政边界
    poly = _get_admin_polygon(location)
    if poly is not None:
        b = poly.bounds  # (minx, miny, maxx, maxy)
        bounds = {"W": b[0], "S": b[1], "E": b[2], "N": b[3]}
        # 打印行政层级
        addr = location.raw.get("address", {})
        province = addr.get("state", addr.get("province", ""))
        district = (addr.get("city_district") or addr.get("county")
                    or addr.get("suburb") or addr.get("district") or "")
        city = addr.get("city", "")
        # 去重
        if city and (city == district or city == name):
            city = ""
        if not city:
            sd = addr.get("state_district", "")
            if sd and sd != district and sd != name:
                city = sd
        parts = [p for p in [province, city, district] if p and p != name]
        if parts:
            print(f"  {' → '.join(parts)} → {name}")
        else:
            print(f"  {name}")
        print(f"\nAOI 边界: N={bounds['N']:.4f}, S={bounds['S']:.4f}, "
              f"W={bounds['W']:.4f}, E={bounds['E']:.4f}")
        return poly.convex_hull.wkt, bounds

    # 后备: 点+缓冲
    bounds = {
        "N": lat + buffer_deg,
        "S": lat - buffer_deg,
        "W": lon - buffer_deg,
        "E": lon + buffer_deg,
    }
    poly = box(bounds["W"], bounds["S"], bounds["E"], bounds["N"])
    print(f"'{name}' → ({lat:.4f}, {lon:.4f}), 缓冲 {buffer_deg}°")
    print(f"AOI: N={bounds['N']:.4f}, S={bounds['S']:.4f}, "
          f"W={bounds['W']:.4f}, E={bounds['E']:.4f}")
    return poly.wkt, bounds


def _resolve_admin_regions(names, buffer_deg):
    """
    多个行政区名称 → 合并区域 + 连续性检查。

    流程:
    1. 逐个地理编码，获取行政边界
    2. 按 省-市-县 层级格式化输出
    3. 等待人工确认
    4. 检查空间连续性 (unary_union 是否为单一多边形)
    5. 不连续 → 终止
    """
    geolocator = Nominatim(user_agent="insar_pipeline_v1", timeout=15)

    regions = {}     # name -> {"location", "polygon", "address"}
    failed = []

    print(f"正在解析 {len(names)} 个行政区...")
    for name in names:
        location = geolocator.geocode(
            name, language="zh", addressdetails=True,
            exactly_one=True,
        )
        if location is None:
            failed.append(name)
            print(f"  [WARN] 无法识别: {name}")
            continue

        poly = _get_admin_polygon(location)
        if poly is None:
            # 后备: 使用 boundingbox
            bb = location.raw.get("boundingbox", [])
            if len(bb) == 4:
                s, n, w, e = [float(x) for x in bb]
                poly = box(w, s, e, n)
            else:
                poly = box(
                    location.longitude - buffer_deg,
                    location.latitude - buffer_deg,
                    location.longitude + buffer_deg,
                    location.latitude + buffer_deg,
                )

        addr = location.raw.get("address", {})
        regions[name] = {
            "location": location,
            "polygon": poly,
            "address": addr,
        }

    if failed:
        print(f"\n[ERROR] 以下区域无法识别: {', '.join(failed)}")
        if not regions:
            raise ValueError("所有区域均无法识别")

    # ---- 按 省-市-县 层级输出 ----
    print("\n" + "=" * 50)
    print(" 行政区划复审")
    print("=" * 50)

    hierarchy = OrderedDict()
    for name, info in regions.items():
        addr = info["address"]

        # 中国行政区 Nominatim 键名优先级:
        #   省级: state
        #   市级: city (仅此键可靠表示地级市)
        #   区县级: city_district > county > suburb
        province = addr.get("state", addr.get("province", "未知省"))

        # 先提取区县级 (最可靠)
        district = (addr.get("city_district")
                    or addr.get("county")
                    or addr.get("suburb")
                    or addr.get("district")
                    or name)

        # 市级: 只取 city 键 (state_district/municipality 在中国常常是区县级别)
        city = addr.get("city", "")

        # 关键去重: 如果 city 和 district 一样, 或 city 就是用户输入的名字, 清空 city
        if city and (city == district or city == name):
            city = ""

        # 如果 city 仍为空, 尝试 state_district 但必须与 district 不同
        if not city:
            sd = addr.get("state_district", "")
            if sd and sd != district and sd != name:
                city = sd

        key = (province, city or "直辖/未知市")
        if key not in hierarchy:
            hierarchy[key] = []
        if district:
            hierarchy[key].append(district)
        else:
            hierarchy[key].append(name)  # 用原始输入名称作后备

    for (prov, city), districts in hierarchy.items():
        print(f"\n  {prov}")
        print(f"    └─ {city}")
        for i, d in enumerate(districts):
            connector = "└─" if i == len(districts) - 1 else "├─"
            print(f"        {connector} {d}")

    poly_bounds = [r["polygon"].bounds for r in regions.values()]
    all_w = min(b[0] for b in poly_bounds)
    all_s = min(b[1] for b in poly_bounds)
    all_e = max(b[2] for b in poly_bounds)
    all_n = max(b[3] for b in poly_bounds)
    print(f"\n  合并边界: N={all_n:.4f} S={all_s:.4f} W={all_w:.4f} E={all_e:.4f}")
    print("=" * 50)

    # ---- 人工确认 ----
    confirm = input("\n以上行政区是否正确？(y/n): ").strip().lower()
    if confirm not in ("y", "yes", ""):
        raise RuntimeError("用户取消，流水线终止。")

    # ---- 空间连续性检查 ----
    polygons = [r["polygon"] for r in regions.values()]
    # 对多边形做小幅缓冲以允许微小间隙 (~500m)
    buffered = [p.buffer(0.005) for p in polygons]
    merged = unary_union(buffered)

    if isinstance(merged, MultiPolygon):
        print("\n[ERROR] 所选行政区在空间上不连续！")
        print("  存在多个不相邻的区域，无法作为一个连续的 InSAR 研究区。")
        print("  请重新选择空间相邻的行政区。")
        # 显示各分离区域
        for i, part in enumerate(merged.geoms):
            b = part.bounds
            print(f"  区域 {i+1}: W={b[0]:.3f} S={b[1]:.3f} E={b[2]:.3f} N={b[3]:.3f}")
        raise RuntimeError("行政区空间不连续，流水线终止。")

    print("空间连续性检查通过。")

    # 使用精确合并边界 (非缓冲)
    actual_merged = unary_union(polygons)
    b = actual_merged.bounds
    bounds = {"W": b[0], "S": b[1], "E": b[2], "N": b[3]}

    return actual_merged.convex_hull.wkt, bounds


def _get_admin_polygon(location):
    """
    通过 Nominatim API 获取行政区边界多边形。
    geopy 的 geocode 不直接返回多边形，需用 raw API。
    """
    osm_id = location.raw.get("osm_id")
    osm_type = location.raw.get("osm_type", "")

    if not osm_id:
        return None

    # Nominatim lookup with polygon
    type_map = {"node": "N", "way": "W", "relation": "R"}
    osm_type_char = type_map.get(osm_type, "")
    if not osm_type_char:
        return None

    try:
        url = "https://nominatim.openstreetmap.org/lookup"
        params = {
            "osm_ids": f"{osm_type_char}{osm_id}",
            "format": "json",
            "polygon_geojson": 1,
        }
        headers = {"User-Agent": "insar_pipeline_v1"}
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data and "geojson" in data[0]:
            geojson = data[0]["geojson"]
            poly = shape(geojson)
            if poly.is_valid and not poly.is_empty:
                return poly
    except Exception as e:
        logger.debug(f"获取行政边界失败 (osm_id={osm_id}): {e}")

    return None


def _print_address_hierarchy(addr_dict):
    """打印地址层级。"""
    for name, addr in addr_dict.items():
        province = addr.get("state", addr.get("province", ""))
        city = addr.get("city", addr.get("town", ""))
        district = addr.get("suburb", addr.get("district",
                   addr.get("county", "")))
        parts = [p for p in [province, city, district] if p]
        print(f"  {' → '.join(parts)}")


# ---------------------------------------------------------------------------
# Scene Search
# ---------------------------------------------------------------------------
def search_scenes(aoi_wkt, date_range, flight_direction="DESCENDING"):
    """
    在 ASF 搜索 Sentinel-1 SLC 场景。
    按 Path/Frame 分组，自动选择景数最多的组。
    """
    start, end = date_range
    print(f"搜索 ASF: {start} ~ {end}, {flight_direction}...")

    results = asf.search(
        platform=[asf.PLATFORM.SENTINEL1],
        processingLevel=asf.PRODUCT_TYPE.SLC,
        intersectsWith=aoi_wkt,
        start=start,
        end=end,
        flightDirection=flight_direction.upper(),
    )

    if not results:
        raise RuntimeError("未找到场景。请检查 AOI 和时间范围。")

    groups = defaultdict(list)
    for r in results:
        key = (r.properties.get("pathNumber"), r.properties.get("frameNumber"))
        groups[key].append(r)

    best_key = max(groups, key=lambda k: len(groups[k]))
    best = groups[best_key]
    best.sort(key=lambda r: r.properties["startTime"])

    print(f"共找到 {len(results)} 景, {len(groups)} 个 Path/Frame 组合")
    print(f"选定 Path={best_key[0]}, Frame={best_key[1]} "
          f"({len(best)} 景, {best[0].properties['startTime'][:10]} "
          f"~ {best[-1].properties['startTime'][:10]})")

    print(f"\n{'Path':>6} {'Frame':>6} {'景数':>6}  时间范围")
    for k in sorted(groups, key=lambda k: -len(groups[k]))[:5]:
        g = sorted(groups[k], key=lambda r: r.properties["startTime"])
        print(f"{k[0]:>6} {k[1]:>6} {len(g):>6}  "
              f"{g[0].properties['startTime'][:10]} ~ "
              f"{g[-1].properties['startTime'][:10]}")

    return best


# ---------------------------------------------------------------------------
# Temporal Uniform Sampling
# ---------------------------------------------------------------------------
def uniform_temporal_sample(scenes, target_n):
    """
    时间均匀采样 + 基线优化。

    算法:
    1. 按日期排序
    2. 理想间隔 = (末-首) / (目标-1)
    3. 贪心: 每个时隙选时间最近+基线偏差最小的场景
    4. 强制包含首尾场景
    """
    if len(scenes) <= target_n:
        print(f"总景数 ({len(scenes)}) ≤ 目标 ({target_n})，全部使用。")
        return scenes

    scenes_sorted = sorted(scenes, key=lambda r: r.properties["startTime"])

    def parse_date(s):
        return datetime.fromisoformat(s.properties["startTime"][:19])

    def get_baseline(s):
        """从 asf_search 结果提取垂直基线。搜索结果通常不含基线信息，返回 None。"""
        try:
            bl = s.baseline
            if isinstance(bl, dict):
                val = bl.get("perpendicular", {}).get("value", None)
                if val is not None and val != 0:
                    return float(val)
            elif hasattr(bl, 'perpendicular'):
                val = getattr(bl.perpendicular, 'value', None)
                if val is not None and val != 0:
                    return float(val)
        except Exception:
            pass
        # asf_search 搜索结果不含基线信息 (需要 baseline API 单独计算)
        return None

    dates = [parse_date(s) for s in scenes_sorted]
    baselines = [get_baseline(s) for s in scenes_sorted]
    # 过滤 None 基线，无基线信息时仅按时间采样
    has_baseline = any(b is not None for b in baselines)
    baselines_num = [b if b is not None else 0.0 for b in baselines]
    median_bl = sorted(baselines_num)[len(baselines_num) // 2] if has_baseline else 0

    start_dt, end_dt = dates[0], dates[-1]
    interval = (end_dt - start_dt) / (target_n - 1)

    selected_indices = {0, len(scenes_sorted) - 1}

    for i in range(1, target_n - 1):
        ideal_time = start_dt + interval * i
        best_idx, best_score = None, float("inf")
        for j, (dt, bl) in enumerate(zip(dates, baselines_num)):
            if j in selected_indices:
                continue
            time_diff = abs((dt - ideal_time).total_seconds())
            bl_diff = abs(bl - median_bl) if has_baseline else 0
            score = time_diff + bl_diff * 3600
            if score < best_score:
                best_score = score
                best_idx = j
        if best_idx is not None:
            selected_indices.add(best_idx)

    selected = [scenes_sorted[i] for i in sorted(selected_indices)]
    print(f"\n均匀采样: {len(scenes_sorted)} → {len(selected)} 景 "
          f"(间隔 ~{interval.days} 天)")
    for s in selected:
        dt = s.properties["startTime"][:10]
        bl = get_baseline(s)
        bl_str = f"Bperp={bl:+7.1f}m" if bl is not None else "Bperp=N/A"
        print(f"  {dt}  {bl_str}")
    return selected


# ---------------------------------------------------------------------------
# ASF Session
# ---------------------------------------------------------------------------
def create_asf_session():
    asf_user, asf_pass = require_config_vars("ASF_USER", "ASF_PASS")
    session = asf.ASFSession().auth_with_creds(asf_user, asf_pass)
    return session


# ---------------------------------------------------------------------------
# Robust Download
# ---------------------------------------------------------------------------
def _download_one(scene, dest_dir, session, max_retries=3, show_bar=True):
    """单景下载: 分块+进度条+完整性校验+重试。"""
    filename = scene.properties["fileName"]
    dest = Path(dest_dir) / filename

    if dest.exists():
        try:
            with zipfile.ZipFile(dest, "r") as zf:
                if zf.testzip() is None:
                    logger.info(f"已存在且有效: {filename}")
                    return dest
        except (zipfile.BadZipFile, Exception):
            dest.unlink(missing_ok=True)

    url = scene.properties["url"]
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            if show_bar:
                pbar = tqdm(total=total, unit="B", unit_scale=True,
                            desc=f"{filename[:40]}", leave=False)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
                    if show_bar:
                        pbar.update(len(chunk))
            if show_bar:
                pbar.close()

            with zipfile.ZipFile(dest, "r") as zf:
                bad = zf.testzip()
                if bad:
                    raise zipfile.BadZipFile(f"损坏: {bad}")
            logger.info(f"下载完成: {filename}")
            return dest

        except Exception as e:
            logger.warning(f"下载尝试 {attempt}/{max_retries} 失败 "
                           f"{filename}: {e}")
            dest.unlink(missing_ok=True)

    logger.error(f"全部 {max_retries} 次尝试失败: {filename}")
    return None


def download_all(scenes, dest_dir, session, parallel=4):
    """
    并行下载所有选中场景。
    并发时禁用内层进度条避免 ANSI 转义码乱码，只显示外层总进度。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    parallel = recommend_cpu_workers("download", requested=parallel, n_items=len(scenes))
    downloaded, failed = [], []
    # 并发时禁用内层进度条
    show_inner_bar = (parallel <= 1)

    print(f"下载 {len(scenes)} 景 SLC ({parallel} 路并发)")

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(_download_one, scene, dest_dir, session,
                            show_bar=show_inner_bar): scene
            for scene in scenes
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="下载 SLC", unit="scene"):
            result = future.result()
            scene = futures[future]
            if result:
                downloaded.append(result)
            else:
                failed.append(scene.properties["fileName"])

    print(f"下载完成: {len(downloaded)} 成功, {len(failed)} 失败")
    if failed:
        print(f"  失败: {', '.join(failed)}")
    return downloaded
