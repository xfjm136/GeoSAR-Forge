"""
ISCE2 topsStack coregistration driver.
Runs coregistration-only workflow (-W slc) with geometry-based alignment.
Includes automatic precise orbit download and parallel execution.
"""
import os
import sys
import re
import subprocess
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm

from . import config as cfg
from .config import N_WORKERS, logger, require_config_vars
from .hardware import build_thread_limited_env, recommend_cpu_workers

# ISCE2 脚本路径 (conda 安装位置: $CONDA_PREFIX/share/isce2/topsStack)
import isce
_CONDA_PREFIX = Path(sys.executable).parent.parent  # e.g. /data/InSAR/env
_ISCE_STACK_DIR = _CONDA_PREFIX / "share" / "isce2" / "topsStack"
STACK_SENTINEL_PY = str(_ISCE_STACK_DIR / "stackSentinel.py")


# ---------------------------------------------------------------------------
# Directory preparation
# ---------------------------------------------------------------------------
def prepare_isce_dirs(work_dir=None):
    """Create ISCE2 directory structure."""
    base = Path(work_dir or cfg.ISCE_WORK_DIR)
    for sub in ["SLC", "DEM", "orbits", "configs", "run_files", "merged"]:
        (base / sub).mkdir(parents=True, exist_ok=True)
    print(f"ISCE2 目录结构已创建: {base}")


# ---------------------------------------------------------------------------
# Precise Orbit Download (EOF)
# ---------------------------------------------------------------------------
def download_orbits(scenes, orbit_dir=None):
    """
    从 ASF 批量并行下载 Sentinel-1 精密轨道文件 (EOF)。
    使用 sentineleof 包或 ASF API 获取精轨。

    Args:
        scenes: list of ASFProduct (选中的场景)
        orbit_dir: 精轨输出目录
    """
    orbit_dir = Path(orbit_dir or cfg.ORBIT_DIR)
    orbit_dir.mkdir(parents=True, exist_ok=True)
    asf_user, asf_pass = require_config_vars("ASF_USER", "ASF_PASS")

    # 检查已有精轨文件
    existing = set(f.name for f in orbit_dir.glob("*.EOF"))

    # 尝试使用 sentineleof (逐景调用，批量调用有 bug 只返回 1 个)
    try:
        from eof.download import download_eofs
        print(f"使用 sentineleof 逐景下载精密轨道...")

        for s in tqdm(scenes, desc="精轨下载", unit="file"):
            fname = s.properties.get("fileName", "")
            safe_name = fname.replace(".zip", ".SAFE") if fname.endswith(".zip") else fname
            try:
                download_eofs(
                    sentinel_file=[safe_name],
                    save_dir=str(orbit_dir),
                    orbit_type="precise",
                    asf_user=asf_user,
                    asf_password=asf_pass,
                )
            except Exception as e:
                logger.warning(f"精轨下载失败 {safe_name[:30]}: {e}")

        new_eofs = set(f.name for f in orbit_dir.glob("*.EOF")) - existing
        print(f"精轨下载完成: {len(new_eofs)} 个新文件")
        return

    except ImportError:
        logger.info("sentineleof 不可用，使用 ASF API 下载精轨")

    # 后备方案: 通过 ASF 搜索精轨产品
    import asf_search as asf

    print(f"通过 ASF 搜索并下载 {len(scenes)} 个场景的精密轨道...")
    session = asf.ASFSession().auth_with_creds(asf_user, asf_pass)

    def _download_orbit_for_scene(scene):
        """为单个场景下载精轨。"""
        try:
            granule = scene.properties.get("fileID", "")
            platform = "S1A" if "S1A" in granule else "S1B"
            start_time = scene.properties.get("startTime", "")

            # 搜索精轨
            orbits = asf.search(
                platform=[platform],
                processingLevel="POEORB",
                start=start_time[:10],
                end=start_time[:10],
                maxResults=1,
            )
            if orbits:
                orbit_file = orbit_dir / orbits[0].properties["fileName"]
                if not orbit_file.exists():
                    orbits[0].download(path=str(orbit_dir), session=session)
                    return True
            return False
        except Exception as e:
            logger.warning(f"精轨下载失败 {granule}: {e}")
            return False

    # 并行下载精轨
    downloaded = 0
    orbit_workers = recommend_cpu_workers("download", requested=8, n_items=len(scenes))
    with ThreadPoolExecutor(max_workers=orbit_workers) as executor:
        futures = {executor.submit(_download_orbit_for_scene, s): s for s in scenes}
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc="精轨下载", unit="file"):
            if f.result():
                downloaded += 1

    total_eofs = len(list(orbit_dir.glob("*.EOF")))
    print(f"精轨下载完成: 新增 {downloaded} 个, 目录共 {total_eofs} 个 EOF 文件")


# ---------------------------------------------------------------------------
# Parallel SLC Extraction
# ---------------------------------------------------------------------------
def _extract_single_zip(args):
    """解压单个 ZIP 文件 (用于并行调用)。"""
    zp, slc_dir = args
    safe_name = zp.stem + ".SAFE"
    safe_dir = slc_dir / safe_name
    if safe_dir.exists():
        return "skipped"
    try:
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(slc_dir)
        return "ok"
    except Exception as e:
        return f"error: {e}"


def extract_slc_zips(zip_dir, slc_dir):
    """
    并行解压 Sentinel-1 SLC .zip 文件 → .SAFE 目录。
    使用 N_WORKERS 个进程并行，大幅加速 I/O 密集型解压。
    """
    zip_dir = Path(zip_dir)
    slc_dir = Path(slc_dir)
    slc_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(zip_dir.glob("*.zip"))
    if not zips:
        print(f"未找到 .zip 文件: {zip_dir}")
        return

    # 并行解压 (I/O 密集, ThreadPool 避免 fork 开销)
    extracted, skipped, errors = 0, 0, 0
    args_list = [(zp, slc_dir) for zp in zips]

    extract_workers = recommend_cpu_workers("isce_extract", requested=N_WORKERS, n_items=len(zips))
    with ThreadPoolExecutor(max_workers=extract_workers) as executor:
        futures = {executor.submit(_extract_single_zip, a): a[0] for a in args_list}
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc="解压 SLC", unit="file"):
            result = f.result()
            if result == "ok":
                extracted += 1
            elif result == "skipped":
                skipped += 1
            else:
                errors += 1
                logger.error(f"解压失败 {futures[f].name}: {result}")

    print(f"解压完成: {extracted} 新解压, {skipped} 已跳过, {errors} 失败")


# ---------------------------------------------------------------------------
# Generate ISCE2 Stack
# ---------------------------------------------------------------------------
def generate_stack(slc_dir=None, dem_path=None, orbit_dir=None, work_dir=None,
                   bbox=None):
    """
    运行 stackSentinel.py 生成配准 run 文件。
    使用 -W slc (仅配准) + -C geometry (几何配准)。

    Args:
        bbox: 裁剪区域 [S, N, W, E]，可选
    """
    slc_dir = Path(slc_dir or cfg.ISCE_SLC_DIR)
    dem_path = Path(dem_path or cfg.DEM_PATH)
    orbit_dir = Path(orbit_dir or cfg.ORBIT_DIR)
    work_dir = Path(work_dir or cfg.ISCE_WORK_DIR)

    # aux 目录 (S1 辅助校准文件, 可为空目录)
    aux_dir = work_dir / "aux_cal"
    aux_dir.mkdir(parents=True, exist_ok=True)

    # 提前检测所需 swath（用于参数变化检测）
    needed_swaths = None
    if bbox:
        needed_swaths = _detect_swaths_for_bbox(slc_dir, bbox)

    run_dir = work_dir / "run_files"
    existing_runs = sorted(run_dir.glob("run_*")) if run_dir.exists() else []
    if existing_runs:
        # 检查 bbox/swath 参数是否与已有 run 文件一致
        # 如果不一致（用户修改了 AOI），必须重新生成并删除旧 .done 标记
        if not _run_params_changed(work_dir, bbox, needed_swaths):
            print(f"Run 文件已存在 ({len(existing_runs)} 个)，参数未变，跳过重新生成。")
            # 恢复保存的 bbox 到 config 供地理裁剪使用
            if bbox:
                cfg._AOI_BBOX = list(bbox)
            else:
                saved = _load_run_params(work_dir)
                if saved and saved.get("bbox"):
                    cfg._AOI_BBOX = saved["bbox"]
            return run_dir
        else:
            print("[INFO] 检测到 AOI/swath 参数变化，重新生成 run 文件...")
            import shutil
            shutil.rmtree(run_dir)
    # stackSentinel.py 要求 run_files 目录不存在
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
        logger.info("清理空的 run_files 目录")

    cmd = [
        sys.executable, STACK_SENTINEL_PY,
        "-s", str(slc_dir),
        "-d", str(dem_path),
        "-o", str(orbit_dir),
        "-a", str(aux_dir),
        "-w", str(work_dir),
        "-C", "geometry",
        "-W", "slc",
        "--num_proc", str(recommend_cpu_workers("isce_generate_stack", requested=N_WORKERS)),
    ]

    # 可选: 裁剪 bbox [S, N, W, E] — 同时自动选择覆盖 AOI 的 swath
    if bbox:
        bbox_str = " ".join(str(x) for x in bbox)
        cmd.extend(["-b", bbox_str])
        print(f"  AOI 裁剪: S={bbox[0]:.3f} N={bbox[1]:.3f} W={bbox[2]:.3f} E={bbox[3]:.3f}")

        if needed_swaths:
            swath_str = " ".join(str(s) for s in needed_swaths)
            cmd.extend(["--swath_num", swath_str])
            print(f"  自动选择 swath: {needed_swaths}")

    # 设置 ISCE2 topsStack 所需环境变量
    env = _get_isce_env()

    print(f"生成 ISCE2 配准脚本...")
    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(work_dir), env=env,
    )

    if result.returncode != 0:
        err_msg = result.stderr.strip() or result.stdout.strip() or "(无输出)"
        logger.error(f"stackSentinel.py 失败:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        raise RuntimeError(f"stackSentinel.py 失败:\n{err_msg[:800]}")

    logger.info(f"stackSentinel.py stdout:\n{result.stdout}")
    run_files = sorted(run_dir.glob("run_*"))
    print(f"已生成 {len(run_files)} 个 run 文件。")

    # 保存参数快照 (用于下次运行时的变化检测) 和 AOI bbox (用于地理裁剪)
    _save_run_params(work_dir, bbox, needed_swaths)
    if bbox:
        cfg._AOI_BBOX = list(bbox)

    return run_dir


def _load_run_params(work_dir):
    """读取上次保存的参数快照。"""
    import json
    params_file = work_dir / ".stack_params.json"
    if not params_file.exists():
        return None
    try:
        return json.loads(params_file.read_text())
    except Exception:
        return None


def _save_run_params(work_dir, bbox, swaths):
    """保存本次 generate_stack 使用的参数到 .stack_params.json。"""
    import json
    params = {
        "bbox": list(bbox) if bbox else None,
        "swaths": swaths,
    }
    (work_dir / ".stack_params.json").write_text(
        json.dumps(params, indent=2))


def _run_params_changed(work_dir, bbox, swaths):
    """
    对比当前参数与上次 generate_stack 保存的参数。
    若参数不同（AOI 改变），返回 True（需要重新生成）。
    若无历史记录，返回 False（保守地信任已有 run 文件）。
    """
    import json
    params_file = work_dir / ".stack_params.json"
    if not params_file.exists():
        return False  # 无历史记录 → 信任已有 run 文件
    try:
        saved = json.loads(params_file.read_text())
    except Exception:
        return False

    saved_bbox = saved.get("bbox")
    saved_swaths = saved.get("swaths")

    current_bbox = list(bbox) if bbox else None
    current_swaths = swaths

    # bbox 对比（允许微小浮点误差）
    if current_bbox != saved_bbox:
        if current_bbox is None or saved_bbox is None:
            return True
        if len(current_bbox) != len(saved_bbox):
            return True
        if any(abs(a - b) > 0.001 for a, b in zip(current_bbox, saved_bbox)):
            return True

    # swath 对比
    if current_swaths != saved_swaths:
        return True

    return False


# ---------------------------------------------------------------------------
# Auto-detect swaths covering AOI bbox
# ---------------------------------------------------------------------------
def _detect_swaths_for_bbox(slc_dir, bbox):
    """
    从 Sentinel-1 SAFE 目录的 annotation XML 中读取各 swath 的覆盖范围,
    返回覆盖 AOI bbox 的 swath 编号列表。

    Args:
        slc_dir: SAFE 目录所在路径
        bbox: [S, N, W, E]

    Returns:
        list[int]: 如 [2] 或 [1, 2], 失败时返回 None (使用默认全部 swath)
    """
    import xml.etree.ElementTree as ET
    from shapely.geometry import box as shapely_box, Polygon

    slc_dir = Path(slc_dir)
    safe_dirs = sorted(slc_dir.glob("*.SAFE"))
    if not safe_dirs:
        return None

    # 用第一个 SAFE 目录检测 swath 覆盖
    safe = safe_dirs[0]
    aoi_box = shapely_box(bbox[2], bbox[0], bbox[3], bbox[1])  # W, S, E, N

    needed = []
    for swath_num in [1, 2, 3]:
        ann_pattern = f"s1?-iw{swath_num}-slc-vv-*.xml"
        ann_files = sorted((safe / "annotation").glob(ann_pattern))
        if not ann_files:
            continue

        try:
            tree = ET.parse(str(ann_files[0]))
            root = tree.getroot()
            # 从 GCP (geolocationGrid) 提取 swath 覆盖范围
            points = root.findall('.//geolocationGridPoint')
            if not points:
                continue

            lats = [float(p.find('latitude').text) for p in points]
            lons = [float(p.find('longitude').text) for p in points]

            swath_box = shapely_box(min(lons), min(lats), max(lons), max(lats))
            if swath_box.intersects(aoi_box):
                needed.append(swath_num)
        except Exception as e:
            logger.warning(f"Swath {swath_num} 检测失败: {e}")
            return None  # 检测失败，不限制 swath

    if not needed:
        logger.warning("AOI 未与任何 swath 相交，使用全部 swath")
        return None

    return needed


# ---------------------------------------------------------------------------
# Execute ISCE2 Run Files (fully parallel)
# ---------------------------------------------------------------------------
def _parse_run_file(run_file):
    """解析 run 文件提取独立命令列表。

    ISCE2 的 run 文件每行命令末尾带 ' &'（后台符号），用于 shell 中批量并行提交。
    我们用 ProcessPoolExecutor 管理并行，不需要 shell '&'，必须去除否则
    subprocess(shell=True) 会立即返回 0 而真实进程在后台游离，导致 .done
    标记提前写入、实际输出文件缺失。
    """
    commands = []
    with open(run_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # 去除 ISCE2 run 文件中的后台提交符 ' &'
                line = line.rstrip("& ").rstrip()
                if line:
                    commands.append(line)
    return commands


def _get_isce_env():
    """构建包含 ISCE2 topsStack 完整路径的环境变量。"""
    env = build_thread_limited_env(os.environ.copy(), threads_per_process=1)
    repo_root = str(Path(__file__).resolve().parents[1])
    isce2_share = str(_CONDA_PREFIX / "share" / "isce2")
    tops_stack = str(_CONDA_PREFIX / "share" / "isce2" / "topsStack")
    isce_apps = str(_CONDA_PREFIX / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "isce" / "applications")
    conda_bin = str(_CONDA_PREFIX / "bin")

    # PYTHONPATH:
    # 1. repo_root 让 sitecustomize.py 等兼容补丁对子进程生效
    # 2. isce2_share 让 topsStack 相关模块可导入
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(filter(None, [repo_root, isce2_share, pythonpath]))

    # PATH: 包含 topsStack 脚本 + isce applications + conda bin
    path = env.get("PATH", "")
    env["PATH"] = ":".join(filter(None, [tops_stack, isce_apps, conda_bin, path]))

    return env


def _run_single_command(cmd_str, log_file, cwd):
    """执行单条 ISCE2 命令，日志写独立临时文件避免并发冲突。"""
    import tempfile
    env = _get_isce_env()
    tmp_log = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False,
                                          dir=cwd, prefix='isce_')
    tmp_path = tmp_log.name
    try:
        result = subprocess.run(
            cmd_str, shell=True,
            stdout=tmp_log, stderr=tmp_log,
            cwd=cwd, env=env, timeout=7200,
        )
        tmp_log.close()
        # 合并到主日志
        with open(log_file, "a") as main, open(tmp_path) as tmp:
            main.write(tmp.read())
        return result.returncode
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def run_isce_steps(run_dir=None, log_file=None):
    """
    按序执行 ISCE2 run 文件。每个 run 文件内的多条独立命令
    用 ProcessPoolExecutor(N_WORKERS) 全并行执行，充分利用 40 核。

    断点续传: 已完成的 run 文件以 .done 标记跳过。
    跳过前会验证关键输出文件是否存在；若缺失则删除 .done 并重新执行。
    失败时抛出 RuntimeError 阻止后续步骤继续。
    """
    run_dir = Path(run_dir or cfg.RUN_DIR)
    log_file = Path(log_file or cfg.LOG_FILE)
    work_dir = run_dir.parent

    # 过滤掉 .done 标记文件，只保留 run_XX 脚本
    run_files = sorted([f for f in run_dir.glob("run_*")
                        if not f.suffix == ".done"])
    if not run_files:
        print("未找到 run 文件。请先运行 generate_stack()。")
        return

    total_cmds = sum(len(_parse_run_file(rf)) for rf in run_files)
    default_workers = recommend_cpu_workers("general", requested=N_WORKERS)
    print(f"执行 {len(run_files)} 个 run 步骤 (共 {total_cmds} 条命令, "
          f"默认并行上限 {default_workers})...")

    for rf in tqdm(run_files, desc="ISCE2 配准步骤", unit="step"):
        marker = rf.with_suffix(".done")
        if marker.exists():
            # 验证关键输出是否真实存在，防止残留旧 .done 造成跳过
            if _validate_step_outputs(rf.name, work_dir):
                logger.info(f"跳过已完成: {rf.name}")
                continue
            else:
                logger.warning(f"{rf.name} 标记为完成但输出缺失，重新执行")
                marker.unlink()

        # 在每个步骤执行前确保 coreg_secondarys 符号链接存在
        _ensure_coreg_symlink(work_dir)

        commands = _parse_run_file(rf)
        if not commands:
            marker.touch()
            continue

        logger.info(f"运行 {rf.name}: {len(commands)} 条命令")

        if len(commands) == 1:
            rc = _run_single_command(commands[0], str(log_file), str(work_dir))
            if rc != 0:
                logger.error(f"ISCE2 命令失败: {rf.name}")
                raise RuntimeError(
                    f"ISCE2 步骤 {rf.name} 失败 (返回码 {rc})。"
                    f"详情见 {log_file}"
                )
        else:
            # 多条命令 → 按步骤复杂度与当前可用内存限流
            failed_cmds = []
            step_workers = recommend_cpu_workers(
                f"isce:{rf.name}",
                requested=N_WORKERS,
                n_items=len(commands),
            )
            logger.info(f"{rf.name} 使用 {step_workers} 并行 worker")
            with ProcessPoolExecutor(max_workers=step_workers) as executor:
                futures = {
                    executor.submit(
                        _run_single_command, cmd, str(log_file), str(work_dir)
                    ): cmd
                    for cmd in commands
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"  {rf.name} ({len(commands)} cmd)",
                    leave=False,
                    unit="cmd",
                ):
                    rc = future.result()
                    if rc != 0:
                        cmd = futures[future]
                        logger.error(f"失败: {cmd[:100]}")
                        failed_cmds.append(cmd[:80])

            if failed_cmds:
                raise RuntimeError(
                    f"ISCE2 步骤 {rf.name} 中 {len(failed_cmds)} 条命令失败。"
                    f"详情见 {log_file}"
                )

        marker.touch()
        logger.info(f"完成: {rf.name}")

    print("ISCE2 配准全部完成。")


def _validate_step_outputs(step_name, work_dir):
    """
    验证 ISCE2 步骤的关键输出是否存在。
    仅对关键步骤做检查，其余步骤默认信任 .done 标记。

    Returns:
        True  — 输出完整，可安全跳过
        False — 输出缺失，需要重新执行
    """
    coreg_dir = work_dir / "coreg_secondarys"

    if "resample" in step_name or "run_05" in step_name:
        # run_05: 每个 secondary 必须有 IW*.xml
        if not coreg_dir.exists():
            return False
        date_dirs = [d for d in coreg_dir.iterdir() if d.is_dir()
                     and d.name.isdigit() and len(d.name) == 8]
        if not date_dirs:
            return False
        for d in date_dirs:
            # 至少需要一个 IW*.xml 文件
            iw_xmls = list(d.glob("IW*.xml"))
            if not iw_xmls:
                logger.debug(f"  缺少 IW*.xml: {d.name}")
                return False
        return True

    if "geo2rdr" in step_name or "run_04" in step_name:
        # run_04: 每个 secondary 必须有 geo2rdr offset 文件
        if not coreg_dir.exists():
            return False
        date_dirs = [d for d in coreg_dir.iterdir() if d.is_dir()
                     and d.name.isdigit() and len(d.name) == 8]
        if not date_dirs:
            return False
        for d in date_dirs:
            offsets = list(d.glob("IW*/azimuth_01.off"))
            if not offsets:
                return False
        return True

    if "merge" in step_name or "run_07" in step_name:
        # run_07: merged/SLC 必须有日期子目录和 .slc 文件
        merged_slc = work_dir / "merged" / "SLC"
        if not merged_slc.exists():
            return False
        slc_dates = [d for d in merged_slc.iterdir() if d.is_dir()]
        if not slc_dates:
            return False
        for d in slc_dates:
            if not list(d.glob("*.slc.full.vrt")) and not list(d.glob("*.slc")):
                return False
        return True

    # 默认: 信任 .done 标记
    return True


def _ensure_coreg_symlink(work_dir):
    """确保 coreg_secondarys -> secondarys 符号链接存在。"""
    secondarys = work_dir / "secondarys"
    coreg_link = work_dir / "coreg_secondarys"
    if not secondarys.exists():
        return
    # 处理损坏的符号链接: .exists() 对 broken symlink 返回 False,
    # 但 os.symlink 会因路径已存在而抛出 FileExistsError
    if coreg_link.is_symlink() or coreg_link.exists():
        return
    os.symlink(str(secondarys), str(coreg_link))
    logger.info("创建符号链接: coreg_secondarys -> secondarys")


# ---------------------------------------------------------------------------
# 后处理: 修复 merge + 转 GeoTIFF + 裁剪
# 测试中发现的三个关键问题的自动修复
# ---------------------------------------------------------------------------
def postprocess_slc(work_dir=None):
    """
    ISCE2 配准后处理 (自动修复测试中发现的问题):
    1. 创建 coreg_secondarys 符号链接 (ISCE2 版本兼容)
    2. 修复 merge VRT 自引用 bug — 从 burst TIFF 重新拼接 SLC
    3. 转换为 Dolphin 可读的 tiled GeoTIFF
    4. 裁剪所有 SLC 到公共最小尺寸
    """
    import numpy as np
    import xml.etree.ElementTree as ET

    work_dir = Path(work_dir or cfg.ISCE_WORK_DIR)
    merged_slc = work_dir / "merged" / "SLC"

    # --- 1. coreg_secondarys 符号链接 ---
    _ensure_coreg_symlink(work_dir)

    # --- 2. 修复 merge: 从 burst 重新拼接 SLC ---
    if not merged_slc.exists():
        merged_slc.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] merged/SLC 目录不存在，已创建。请确认 ISCE2 配准已完成。")
        return

    dates = sorted([d.name for d in merged_slc.iterdir() if d.is_dir()])
    if not dates:
        print("[WARN] merged/SLC 下无日期子目录，跳过后处理")
        return

    # 找一个有效的 secondary VRT 作为模板 (非参考日期)
    ref_date = dates[0]
    template_vrt = None
    for d in dates[1:]:
        vrt = merged_slc / d / f"{d}.slc.full.vrt"
        if vrt.exists():
            tree = ET.parse(str(vrt))
            sources = tree.getroot().find('.//VRTRasterBand').findall('.//SimpleSource')
            if len(sources) > 1:  # 多 source = 正常的 burst 拼接 VRT
                template_vrt = vrt
                break

    if template_vrt is None:
        print("[WARN] 无法找到有效的模板 VRT")
        return

    print(f"  模板 VRT: {template_vrt.name} ({len(sources)} sources)")

    for date in tqdm(dates, desc="修复 SLC merge", unit="date"):
        tif_out = merged_slc / date / f"{date}.slc.tif"
        if tif_out.exists():
            continue

        slc_vrt = merged_slc / date / f"{date}.slc.full.vrt"
        if not slc_vrt.exists():
            continue

        # 检查是否是自引用 VRT (参考场景的 bug)
        tree = ET.parse(str(slc_vrt))
        root = tree.getroot()
        band = root.find('.//VRTRasterBand')
        sources = band.findall('.//SimpleSource')

        needs_fix = False
        if len(sources) <= 1:
            # 参考场景或损坏的 VRT → 用模板修复
            needs_fix = True
        else:
            # 检查源文件是否可读
            fn = sources[0].find('SourceFilename')
            src_fn = fn.text
            if fn.get('relativeToVRT', '0') == '1':
                src_fn = str((slc_vrt.parent / src_fn).resolve())
            if src_fn.endswith('.vrt') and src_fn == str(slc_vrt):
                needs_fix = True  # 自引用

        if needs_fix and date == ref_date:
            # 参考场景: 用模板 VRT 结构, 替换路径为 reference 目录
            from copy import deepcopy
            new_tree = ET.parse(str(template_vrt))
            new_root = new_tree.getroot()
            for src in new_root.find('.//VRTRasterBand').findall('.//SimpleSource'):
                fn_elem = src.find('SourceFilename')
                old = fn_elem.text
                # 替换 secondary 路径为 reference
                for d2 in dates[1:]:
                    if d2 in old:
                        fn_elem.text = old.replace(
                            f'coreg_secondarys/{d2}',
                            'reference'
                        ).replace(
                            f'secondarys/{d2}',
                            'reference'
                        )
                        break
            fixed_vrt = slc_vrt.with_suffix('.fixed.vrt')
            new_tree.write(str(fixed_vrt))
            slc_vrt = fixed_vrt

        # 用 gdal.Translate 直接转换 (避免全量内存加载 OOM)
        try:
            from osgeo import gdal
            gdal.UseExceptions()
            gdal.Translate(
                str(tif_out), str(slc_vrt),
                format="GTiff",
                creationOptions=["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"],
            )
            logger.info(f"SLC GeoTIFF: {date}")
        except Exception as e:
            logger.warning(f"SLC 转换失败 {date}: {e}")

    # --- 3. 按 AOI 经纬度裁剪 SLC 到研究区像素范围 ---
    # 先找所有 TIF（不管尺寸是否一致）
    tifs = sorted(merged_slc.glob("*/*.slc.tif"))
    if not tifs:
        print("[WARN] 无 SLC TIF 文件可裁剪")
        return

    from osgeo import gdal
    gdal.UseExceptions()

    # 从 geom_reference 的 lat/lon 网格确定 AOI 对应的像素行列范围
    # 这是真正的地理裁剪：把 45381×7336 的全景缩小到呈贡区的几千像素
    geom_ref = work_dir / "merged" / "geom_reference"
    lat_vrt = geom_ref / "lat.rdr.full.vrt"
    lon_vrt = geom_ref / "lon.rdr.full.vrt"

    aoi_col_off, aoi_row_off, aoi_cols, aoi_rows = None, None, None, None

    if lat_vrt.exists() and lon_vrt.exists() and cfg._AOI_BBOX is not None:
        try:
            import numpy as np
            bbox = cfg._AOI_BBOX  # [S, N, W, E]
            S, N, W, E = bbox[0], bbox[1], bbox[2], bbox[3]

            ds_lat = gdal.Open(str(lat_vrt))
            lat_arr = ds_lat.ReadAsArray()
            ds_lat = None
            ds_lon = gdal.Open(str(lon_vrt))
            lon_arr = ds_lon.ReadAsArray()
            ds_lon = None

            # AOI 像素级 padding（不使用百分比缓冲）
            # 百分比缓冲会导致处理区域远大于 GACOS/ERA5 等大气数据覆盖范围，
            # MintPy 在覆盖范围外填 0 后经参考减法会产生 ~2000mm 虚假校正
            pad = 20   # ~240m in range, ~280m in azimuth
            mask = ((lat_arr >= S) & (lat_arr <= N) &
                    (lon_arr >= W) & (lon_arr <= E) &
                    (lat_arr > 0.1))

            if mask.any():
                rows, cols = np.where(mask)
                aoi_row_off = max(0, int(rows.min()) - pad)
                aoi_col_off = max(0, int(cols.min()) - pad)
                aoi_rows = min(lat_arr.shape[0], int(rows.max()) + pad + 1) - aoi_row_off
                aoi_cols = min(lat_arr.shape[1], int(cols.max()) + pad + 1) - aoi_col_off
                print(f"  AOI 地理裁剪: 行 {aoi_row_off}-{aoi_row_off+aoi_rows}, "
                      f"列 {aoi_col_off}-{aoi_col_off+aoi_cols} "
                      f"({aoi_cols}×{aoi_rows} px)")
        except Exception as e:
            logger.warning(f"AOI 像素范围计算失败: {e}")

    import shutil
    sizes = []
    for t in tifs:
        ds = gdal.Open(str(t))
        sizes.append((ds.RasterXSize, ds.RasterYSize))
        ds = None

    # 如果有 AOI 地理范围，裁剪到 AOI；否则退回到公共最小尺寸
    if aoi_col_off is not None:
        for t, (sx, sy) in tqdm(zip(tifs, sizes), desc="AOI 地理裁剪 SLC",
                                total=len(tifs), unit="date"):
            if sx == aoi_cols and sy == aoi_rows:
                continue
            tmp = str(t) + ".tmp.tif"
            ds = gdal.Open(str(t))
            gdal.Translate(tmp, ds,
                           srcWin=[aoi_col_off, aoi_row_off, aoi_cols, aoi_rows],
                           format="GTiff",
                           creationOptions=["TILED=YES", "BLOCKXSIZE=512",
                                            "BLOCKYSIZE=512"])
            ds = None
            shutil.move(tmp, str(t))
        final_x, final_y = aoi_cols, aoi_rows
        # 将裁剪偏移写入 config，供 build_mintpy_hdf5 几何对齐使用
        cfg._AOI_CROP_OFFSET = (aoi_row_off, aoi_col_off)
        # 同时持久化到 project.json
        from .config import save_project_progress, _PROJECT_NAME
        if _PROJECT_NAME:
            save_project_progress("isce2_done")
        logger.info(f"AOI crop offset saved: row_off={aoi_row_off}, col_off={aoi_col_off}")
    else:
        # 退回到公共最小尺寸（无 AOI bbox 信息时的后备）
        min_x = min(s[0] for s in sizes)
        min_y = min(s[1] for s in sizes)
        if len(set(sizes)) > 1:
            for t, (sx, sy) in zip(tifs, sizes):
                if sx != min_x or sy != min_y:
                    tmp = str(t) + ".tmp.tif"
                    ds = gdal.Open(str(t))
                    gdal.Translate(tmp, ds, srcWin=[0, 0, min_x, min_y],
                                   format="GTiff",
                                   creationOptions=["TILED=YES", "BLOCKXSIZE=512",
                                                    "BLOCKYSIZE=512"])
                    ds = None
                    shutil.move(tmp, str(t))
        final_x, final_y = min_x, min_y

    print(f"  SLC 后处理完成: {len(tifs)} 景, 最终尺寸 {final_x}×{final_y}")
