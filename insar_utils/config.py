"""
Global configuration, logging, proxy, credentials, and utility functions.
支持项目隔离: 每个项目独立目录, 防止数据"串味"。
"""
import os
import sys
import json
import shutil
import logging
import time
from pathlib import Path
from glob import glob

from .hardware import detect_hardware_profile, recommend_cpu_workers, summarize_hardware

# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------
def setup_proxy():
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    os.environ["all_proxy"] = "socks5://127.0.0.1:7890"

# ---------------------------------------------------------------------------
# Root directory (all projects live under this)
# ---------------------------------------------------------------------------
ROOT_DIR = Path("/data/InSAR")
ENV_FILE = ROOT_DIR / ".env"
PROJECTS_DIR = ROOT_DIR / "projects"


def _strip_env_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, _strip_env_quotes(value))


def _get_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def require_config_vars(*names: str):
    missing = []
    values = []
    for name in names:
        value = globals().get(name, os.getenv(name))
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(name)
        values.append(value)
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"缺少配置: {missing_str}。请在 {ENV_FILE} 中设置，或导出为环境变量。"
        )
    if len(values) == 1:
        return values[0]
    return tuple(values)


_load_dotenv(ENV_FILE)

# ---------------------------------------------------------------------------
# Project-level paths (initialized by init_project)
# ---------------------------------------------------------------------------
WORK_DIR      = ROOT_DIR   # will be overwritten by init_project
SLC_ZIP_DIR   = ROOT_DIR / "SLC_zip"
ISCE_WORK_DIR = ROOT_DIR / "isce2"
ISCE_SLC_DIR  = ROOT_DIR / "isce2" / "SLC"
ORBIT_DIR     = ROOT_DIR / "isce2" / "orbits"
RUN_DIR       = ROOT_DIR / "isce2" / "run_files"
DEM_DIR       = ROOT_DIR / "DEM"
DEM_PATH      = ROOT_DIR / "DEM" / "dem.dem.wgs84"
DEM_FILE      = ROOT_DIR / "DEM" / "dem.tif"
GACOS_DIR     = ROOT_DIR / "GACOS"
ERA5_DIR      = ROOT_DIR / "ERA5"
DOLPHIN_DIR   = ROOT_DIR / "dolphin_work"
MINTPY_DIR    = ROOT_DIR / "mintpy"
EXPORT_DIR    = ROOT_DIR / "export"
FORECAST_DIR  = ROOT_DIR / "forecast"
LOG_FILE      = ROOT_DIR / "pipeline.log"
TEMPLATE_PATH = ROOT_DIR / "mintpy" / "custom_template.txt"
SLC_VRT_PATTERN = ""

# ---------------------------------------------------------------------------
# Project state
# ---------------------------------------------------------------------------
_PROJECT_NAME = None
_AUTO_CLEANUP = False
_AOI_BBOX = None        # [S, N, W, E] — AOI 经纬度边界，供 postprocess_slc 地理裁剪
_AOI_CROP_OFFSET = None # (row_off, col_off) — SLC 地理裁剪的像素偏移，供 build_mintpy_hdf5 几何对齐

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
HARDWARE_PROFILE = detect_hardware_profile()
N_WORKERS = recommend_cpu_workers("general")


def refresh_hardware_profile():
    global HARDWARE_PROFILE, N_WORKERS
    HARDWARE_PROFILE = detect_hardware_profile(refresh=True)
    N_WORKERS = recommend_cpu_workers("general")
    return HARDWARE_PROFILE


def set_max_cpu_workers(limit: int):
    if int(limit) <= 0:
        raise ValueError("limit 必须大于 0")
    os.environ["INSAR_FORGE_MAX_CPU_WORKERS"] = str(int(limit))
    refresh_hardware_profile()
    return N_WORKERS


def print_hardware_summary():
    summary = summarize_hardware()
    print("硬件概况:")
    print(f"  CPU: {summary['physical_cpus']} 物理核 / {summary['logical_cpus']} 逻辑核")
    print(f"  内存: {summary['available_memory_gb']:.1f} / {summary['total_memory_gb']:.1f} GB 可用")
    if summary["gpu_names"]:
        for idx, name in enumerate(summary["gpu_names"]):
            total = summary["gpu_total_memory_gb"][idx]
            free = summary["gpu_free_memory_gb"][idx]
            print(f"  GPU{idx}: {name} ({free:.1f} / {total:.1f} GB 可用)")
    else:
        print("  GPU: 未检测到 CUDA 设备")
    print(f"  默认并行上限: {N_WORKERS}")


def _cleanup_root_isce_log():
    """Migrate any root-level isce.log into the active project, then remove it."""
    root_log = ROOT_DIR / "isce.log"
    target_log = ISCE_WORK_DIR / "isce.log"
    target_log.parent.mkdir(parents=True, exist_ok=True)

    if root_log.is_symlink():
        try:
            if root_log.resolve() == target_log.resolve():
                root_log.unlink()
                target_log.touch(exist_ok=True)
                return
        except FileNotFoundError:
            pass
        root_log.unlink()

    if root_log.exists() and not root_log.is_symlink():
        try:
            existing_text = root_log.read_bytes()
            if existing_text:
                with target_log.open("ab") as f:
                    f.write(existing_text)
            root_log.unlink()
        except Exception:
            fallback = ROOT_DIR / f"isce.log.migrated_to_{_PROJECT_NAME or 'project'}"
            shutil.move(str(root_log), str(fallback))

    target_log.touch(exist_ok=True)

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
ASF_USER = _get_env("ASF_USER")
ASF_PASS = _get_env("ASF_PASS")

ESA_USER = _get_env("ESA_USER")
ESA_PASS = _get_env("ESA_PASS")

OPENTOPO_KEY = _get_env("OPENTOPO_KEY")

ERA5_URL = _get_env("ERA5_URL", "https://cds.climate.copernicus.eu/api")
ERA5_KEY = _get_env("ERA5_KEY")


# ============================================================================
# 项目初始化 (核心函数)
# ============================================================================
def init_project(name=None):
    """
    初始化或恢复一个项目。所有数据文件隔离在 /data/InSAR/projects/<name>/ 下。

    Args:
        name: 项目名称。为 None 时交互式输入。

    Returns:
        project_name: str

    流程:
    1. 显示已有项目列表, 支持选择恢复或新建
    2. 创建项目目录结构
    3. 询问是否启用中间文件自动清理
    4. 保存项目配置到 project.json (断点恢复)
    """
    global _PROJECT_NAME, _AUTO_CLEANUP, _AOI_BBOX, _AOI_CROP_OFFSET
    global WORK_DIR, SLC_ZIP_DIR, ISCE_WORK_DIR, ISCE_SLC_DIR, ORBIT_DIR
    global RUN_DIR, DEM_DIR, DEM_PATH, DEM_FILE, GACOS_DIR, ERA5_DIR
    global DOLPHIN_DIR, MINTPY_DIR, EXPORT_DIR, FORECAST_DIR, LOG_FILE, TEMPLATE_PATH
    global SLC_VRT_PATTERN

    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    # 列出已有项目
    existing = sorted([d.name for d in PROJECTS_DIR.iterdir()
                       if d.is_dir() and (d / "project.json").exists()])

    if existing:
        print("\n" + "=" * 50)
        print(" 已有项目")
        print("=" * 50)
        for i, p in enumerate(existing, 1):
            cfg = _load_project_json(PROJECTS_DIR / p / "project.json")
            status = cfg.get("last_step", "未开始")
            aoi = cfg.get("aoi", "")
            print(f"  [{i}] {p}  (进度: {status}, AOI: {aoi})")
        print(f"  [N] 新建项目")
        print()

    if name is None:
        if existing:
            choice = input("选择项目编号或输入新项目名称: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(existing):
                name = existing[int(choice) - 1]
                print(f"恢复项目: {name}")
            elif choice.upper() == "N":
                name = input("新项目名称: ").strip()
            else:
                name = choice
        else:
            name = input("项目名称: ").strip()

    if not name:
        raise ValueError("项目名称不能为空")

    # 清理项目名 (移除非法字符)
    import re
    name = re.sub(r'[^\w\-.]', '_', name)

    # 设置项目路径
    WORK_DIR = PROJECTS_DIR / name
    _PROJECT_NAME = name

    SLC_ZIP_DIR   = WORK_DIR / "SLC_zip"
    ISCE_WORK_DIR = WORK_DIR / "isce2"
    ISCE_SLC_DIR  = ISCE_WORK_DIR / "SLC"
    ORBIT_DIR     = ISCE_WORK_DIR / "orbits"
    RUN_DIR       = ISCE_WORK_DIR / "run_files"
    DEM_DIR       = WORK_DIR / "DEM"
    DEM_PATH      = DEM_DIR / "dem.dem.wgs84"
    DEM_FILE      = DEM_DIR / "dem.tif"
    GACOS_DIR     = WORK_DIR / "GACOS"
    ERA5_DIR      = WORK_DIR / "ERA5"
    DOLPHIN_DIR   = WORK_DIR / "dolphin_work"
    MINTPY_DIR    = WORK_DIR / "mintpy"
    EXPORT_DIR    = WORK_DIR / "export"
    FORECAST_DIR  = WORK_DIR / "forecast"
    LOG_FILE      = WORK_DIR / "pipeline.log"
    TEMPLATE_PATH = MINTPY_DIR / "custom_template.txt"
    SLC_VRT_PATTERN = str(ISCE_WORK_DIR / "merged" / "SLC" / "*" / "*.slc.full.vrt")

    # 创建目录
    for d in [SLC_ZIP_DIR, ISCE_SLC_DIR, ORBIT_DIR, DEM_DIR, GACOS_DIR,
              ERA5_DIR, DOLPHIN_DIR, MINTPY_DIR, EXPORT_DIR, FORECAST_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 加载或创建项目配置
    cfg_file = WORK_DIR / "project.json"
    if cfg_file.exists():
        cfg = _load_project_json(cfg_file)
        _AUTO_CLEANUP = cfg.get("auto_cleanup", False)
        # 恢复 AOI bbox（供 postprocess_slc 地理裁剪使用）
        if cfg.get("aoi_bbox"):
            _AOI_BBOX = cfg["aoi_bbox"]
        # 恢复 AOI crop offset（供 build_mintpy_hdf5 几何对齐使用）
        if cfg.get("aoi_crop_offset"):
            _AOI_CROP_OFFSET = tuple(cfg["aoi_crop_offset"])
        print(f"\n项目 [{name}] 已恢复")
        print(f"  目录: {WORK_DIR}")
        print(f"  自动清理: {'开启' if _AUTO_CLEANUP else '关闭'}")
        if cfg.get("last_step"):
            print(f"  上次进度: {cfg['last_step']}")
    else:
        # 新项目 — 询问是否自动清理
        print(f"\n新建项目: {name}")
        print(f"  目录: {WORK_DIR}")
        print()
        print("是否启用中间文件自动清理?")
        print("  [Y] 开启 — 每步完成后自动删除不再需要的中间数据，节省磁盘")
        print("  [N] 关闭 — 保留所有中间文件，方便调试回溯 (需更多磁盘空间)")
        cleanup_choice = input("自动清理 (Y/N, 默认Y): ").strip().lower()
        _AUTO_CLEANUP = cleanup_choice not in ("n", "no")

        _save_project_json(cfg_file, {
            "name": name,
            "auto_cleanup": _AUTO_CLEANUP,
            "created": str(Path(".")),  # will be filled with timestamp
            "last_step": "",
            "aoi": "",
        })
        print(f"  自动清理: {'开启' if _AUTO_CLEANUP else '关闭'}")

    _cleanup_root_isce_log()

    return name


def save_project_progress(step_name, **extra):
    """保存项目进度 (用于断点恢复)。同时持久化 _AOI_BBOX 和 _AOI_CROP_OFFSET。"""
    cfg_file = WORK_DIR / "project.json"
    cfg = _load_project_json(cfg_file) if cfg_file.exists() else {}
    cfg["last_step"] = step_name
    if _AOI_BBOX is not None:
        cfg["aoi_bbox"] = _AOI_BBOX
    if _AOI_CROP_OFFSET is not None:
        cfg["aoi_crop_offset"] = list(_AOI_CROP_OFFSET)
    cfg.update(extra)
    _save_project_json(cfg_file, cfg)


def _load_project_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def _save_project_json(path, cfg):
    from datetime import datetime
    cfg["updated"] = datetime.now().isoformat()
    Path(path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_logger_configured = False

def setup_logging():
    global _logger_configured
    if _logger_configured:
        return
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(str(LOG_FILE), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(ch)

    # 压制第三方库的冗余 DEBUG 日志 (否则淹没 Jupyter)
    for noisy in ['rasterio', 'rasterio.env', 'rasterio._io', 'rasterio._base',
                   'rasterio._env', 'rasterio.session', 'rasterio._filepath',
                   'urllib3', 'matplotlib', 'h5py', 'jax', 'jax._src',
                   'botocore', 'boto3', 's3fs', 'fsspec',
                   'numba', 'numba.core', 'numba.core.byteflow',
                   'numba.core.ssa_rewrite', 'numba.core.interpreter',
                   'shapely', 'fiona', 'pyproj']:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _logger_configured = True

logger = logging.getLogger("insar_pipeline")

# ---------------------------------------------------------------------------
# Disk space check
# ---------------------------------------------------------------------------
def disk_check(min_gb=100):
    usage = shutil.disk_usage("/data")
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    print(f"磁盘 /data: {free_gb:.1f} GB 可用 / {total_gb:.1f} GB 总计")
    if free_gb < min_gb:
        print(f"[WARN] 剩余空间不足 {min_gb} GB!")
    return free_gb

# ---------------------------------------------------------------------------
# Cleanup (conditional on _AUTO_CLEANUP)
# ---------------------------------------------------------------------------
def cleanup_temp():
    """手动清理所有中间文件。"""
    cleanup_after_step("all", force=True)


def cleanup_after_step(step_name, force=False):
    """
    分步清理中间数据。仅在 auto_cleanup 开启或 force=True 时执行。

    注意:
      - ``after_dolphin`` 只清理体积较大的 wrapped interferograms / scratch 文件，
        不再删除 ``phase_linking``。后续的 pre-MintPy QC 仍需要其中的
        temporal coherence / PS 辅助栅格。
      - ``after_isce2`` 不再删除 ``isce2/baselines``。Dolphin→MintPy 桥接阶段仍要
        读取星型基线文本来恢复 ``bperp``；这些文件延后到 ``after_hdf5_bridge`` 再删。
      - ``after_hdf5_bridge`` 发生在 Dolphin→MintPy HDF5 桥接完成之后，
        此时才允许整体移除 ``dolphin_work``，并回收 ``isce2/baselines``。
    """
    if not force and not _AUTO_CLEANUP:
        return

    freed = 0

    def _dir_size_bytes(d: Path) -> int:
        total = 0
        for f in d.rglob("*"):
            if not f.is_file():
                continue
            try:
                total += f.stat().st_size
            except OSError:
                continue
        return total

    def _list_dir_entries(d: Path):
        try:
            return list(d.iterdir())
        except OSError:
            return []

    def _has_only_fuse_hidden_files(d: Path) -> bool:
        entries = _list_dir_entries(d)
        return bool(entries) and all(p.name.startswith(".fuse_hidden") for p in entries)

    def _rm_dir(d, label=""):
        nonlocal freed
        d = Path(d)
        if d.exists() and d.is_dir():
            size = _dir_size_bytes(d)
            last_err = None
            for attempt in range(3):
                try:
                    shutil.rmtree(d)
                    freed += size
                    logger.info(f"清理 {label or d.name}: {size/1e9:.1f} GB")
                    return
                except OSError as exc:
                    last_err = exc
                    if not d.exists():
                        freed += size
                        logger.info(f"清理 {label or d.name}: {size/1e9:.1f} GB")
                        return
                    if _has_only_fuse_hidden_files(d):
                        leftovers = ", ".join(p.name for p in _list_dir_entries(d))
                        logger.warning(
                            "清理 %s 时仅剩 FUSE 占位文件，暂缓删除并继续流程: %s",
                            label or d.name,
                            leftovers,
                        )
                        logger.warning(
                            "这通常表示文件句柄尚未释放，不是中间产物被误删；稍后可再次执行 cleanup_after_step()。"
                        )
                        return
                    if attempt < 2:
                        time.sleep(1.0 + attempt)
            raise last_err

    def _rm_glob(parent, pattern, label=""):
        nonlocal freed
        parent = Path(parent)
        for f in parent.glob(pattern):
            if f.is_file():
                freed += f.stat().st_size
                f.unlink()
            elif f.is_dir():
                size = sum(x.stat().st_size for x in f.rglob("*") if x.is_file())
                freed += size
                shutil.rmtree(f)

    steps = [step_name] if step_name != "all" else [
        "after_extract", "after_isce2", "after_postprocess",
        "after_dolphin", "after_hdf5_bridge",
    ]

    for step in steps:
        if step == "after_extract":
            _rm_glob(SLC_ZIP_DIR, "*.zip", "SLC ZIPs")
        elif step == "after_isce2":
            _rm_glob(ISCE_SLC_DIR, "*.SAFE", "SAFE dirs")
            for sub in ["stack", "configs", "run_files", "aux_cal"]:
                _rm_dir(ISCE_WORK_DIR / sub)
        elif step == "after_postprocess":
            merged_slc = ISCE_WORK_DIR / "merged" / "SLC"
            if merged_slc.exists():
                for pat in ["*/*.slc.full", "*/*.slc.full.vrt", "*/*.slc.full.xml",
                            "*/*.slc.full.aux.xml", "*/*.slc.fixed.vrt", "*/*.slc.hdr"]:
                    _rm_glob(merged_slc, pat)
            _rm_dir(ISCE_WORK_DIR / "secondarys")
            _rm_dir(ISCE_WORK_DIR / "reference")
            coreg = ISCE_WORK_DIR / "coreg_secondarys"
            if coreg.is_symlink():
                coreg.unlink()
        elif step == "after_dolphin":
            _rm_glob(DOLPHIN_DIR / "interferograms", "*.int.tif", "wrapped ifgrams")
            _rm_dir(DOLPHIN_DIR / "unwrapped" / "scratch", "unwrapped scratch")
        elif step == "after_hdf5_bridge":
            _rm_dir(DOLPHIN_DIR, "dolphin_work")
            _rm_dir(ISCE_WORK_DIR / "baselines", "ISCE baselines")
            merged_slc = ISCE_WORK_DIR / "merged" / "SLC"
            if merged_slc.exists():
                _rm_glob(merged_slc, "*/*.slc.tif", "SLC TIFs")
        elif step == "after_mintpy":
            ifg = MINTPY_DIR / "inputs" / "ifgramStack.h5"
            if ifg.exists():
                freed += ifg.stat().st_size
                ifg.unlink()

    if freed > 0:
        gb = freed / (1024**3)
        free_now = disk_check()
        print(f"清理 [{step_name}]: 释放 {gb:.1f} GB, 当前可用 {free_now:.1f} GB")
