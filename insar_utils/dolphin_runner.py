"""
Dolphin Phase Linking and parallel unwrapping driver.
"""
import subprocess
import time
from pathlib import Path
from glob import glob

import yaml
from tqdm.auto import tqdm

from . import config as cfg
from .config import N_WORKERS, logger
from .hardware import detect_hardware_profile, recommend_cpu_workers


# ---------------------------------------------------------------------------
# Build Dolphin Configuration
# ---------------------------------------------------------------------------
def build_dolphin_config(
    slc_pattern=None,
    work_dir=None,
    n_workers=None,
    half_window=None,
    strides=None,
    max_bandwidth=3,
    unwrap_method="snaphu",
):
    """
    Build Dolphin workflow configuration YAML.

    Configures:
    - Phase Linking with half_window and strides for multi-looking
    - SNAPHU unwrapping with n_workers parallel tiles
    - Out-of-core block processing (512x512)

    Args:
        slc_pattern: glob pattern for coregistered SLC VRT files
        work_dir: Dolphin output directory
        n_workers: number of parallel workers
        half_window: optional tuple[int, int] for (x, y)
        strides: optional tuple[int, int] for (x, y)
        max_bandwidth: interferogram network bandwidth
        unwrap_method: unwrapping method, default snaphu

    Returns:
        Path to generated dolphin_config.yaml
    """
    slc_pattern = slc_pattern or cfg.SLC_VRT_PATTERN
    work_dir = Path(work_dir or cfg.DOLPHIN_DIR)
    n_workers = n_workers or N_WORKERS
    work_dir.mkdir(parents=True, exist_ok=True)

    config_path = work_dir / "dolphin_config.yaml"
    half_window = half_window or (11, 5)
    strides = strides or (6, 3)

    # Find SLC files
    slc_files = sorted(glob(slc_pattern))
    if not slc_files:
        # Try alternative pattern
        alt_pattern = str(cfg.ISCE_WORK_DIR / "merged" / "SLC" / "*" / "*.slc.full")
        slc_files = sorted(glob(alt_pattern))
    if not slc_files:
        raise FileNotFoundError(
            f"No SLC files found matching {slc_pattern}. "
            f"Run ISCE2 coregistration first."
        )

    print(f"Found {len(slc_files)} coregistered SLC files.")

    # Write SLC file list
    slc_list_file = work_dir / "slc_files.txt"
    slc_list_file.write_text("\n".join(slc_files) + "\n")

    # Build YAML config
    # Try using dolphin CLI to generate template first, then modify
    try:
        _build_via_cli(
            slc_list_file,
            work_dir,
            n_workers,
            config_path,
            half_window=half_window,
            strides=strides,
            max_bandwidth=max_bandwidth,
            unwrap_method=unwrap_method,
        )
    except Exception as e:
        logger.warning(f"dolphin CLI config failed ({e}), building YAML manually")
        _build_manual_yaml(
            slc_files,
            work_dir,
            n_workers,
            config_path,
            half_window=half_window,
            strides=strides,
            max_bandwidth=max_bandwidth,
            unwrap_method=unwrap_method,
        )

    print(f"Dolphin config saved: {config_path}")
    return config_path


def _build_via_cli(
    slc_list_file,
    work_dir,
    n_workers,
    config_path,
    half_window,
    strides,
    max_bandwidth,
    unwrap_method,
):
    """Generate config using dolphin CLI, then patch parameters."""
    # Generate base config
    cmd = [
        "dolphin", "config",
        "--slc-files", f"@{slc_list_file}",
        "--work-dir", str(work_dir),
        "--strides", str(strides[0]), str(strides[1]),
        "--unwrap-method", str(unwrap_method),
        "--n-workers", str(n_workers),
        "--threads-per-worker", "1",
        "--block-shape", "512", "512",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    # Read generated config and patch phase linking window
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 根据硬件自动调优, 覆盖 CLI 生成的默认并行参数
    # 读取 SLC 数量来计算合理参数
    slc_count = sum(1 for line in open(slc_list_file) if line.strip())
    snap_jobs, pl_threads, blk = _auto_tune_hardware(n_workers, slc_count)

    # Ensure phase linking settings
    if "phase_linking" not in cfg:
        cfg["phase_linking"] = {}
    cfg["phase_linking"]["half_window"] = {"x": int(half_window[0]), "y": int(half_window[1])}

    # 网络型干涉图（max_bandwidth=3，代替单参考星型）
    if "interferogram_network" not in cfg:
        cfg["interferogram_network"] = {}
    cfg["interferogram_network"]["reference_idx"] = None
    cfg["interferogram_network"]["max_bandwidth"] = int(max_bandwidth)

    # Ensure unwrapping parallelism uses tuned value
    if "unwrap_options" not in cfg:
        cfg["unwrap_options"] = {}
    cfg["unwrap_options"]["n_parallel_jobs"] = snap_jobs

    # Fix worker settings
    if "worker_settings" not in cfg:
        cfg["worker_settings"] = {}
    cfg["worker_settings"]["threads_per_worker"] = pl_threads

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def _build_manual_yaml(
    slc_files,
    work_dir,
    n_workers,
    config_path,
    half_window,
    strides,
    max_bandwidth,
    unwrap_method,
):
    """Build Dolphin config YAML matching v0.42+ schema.

    interferogram_network 使用网络型连接 (max_bandwidth=3)：
    每景与时间上最近的3景组成干涉对，提供相位闭合冗余约束，
    支持 MintPy 的解缠误差自动检测与校正（网络型优于无冗余星型）。
    """
    snap_jobs, pl_threads, blk = _auto_tune_hardware(n_workers, len(slc_files))

    config = {
        "cslc_file_list": slc_files,
        "work_directory": str(work_dir),
        "phase_linking": {
            "half_window": {"x": int(half_window[0]), "y": int(half_window[1])},
            "ministack_size": 15,
        },
        "interferogram_network": {
            "reference_idx": None,       # None = 网络型，不使用单参考星型
            "max_bandwidth": int(max_bandwidth),  # 每景与最近几景组成干涉对
            "max_temporal_baseline": None,
        },
        "unwrap_options": {
            "unwrap_method": str(unwrap_method),
            "n_parallel_jobs": snap_jobs,
        },
        "output_options": {
            "strides": {"x": int(strides[0]), "y": int(strides[1])},
        },
        "worker_settings": {
            "block_shape": [blk, blk],
            "n_parallel_bursts": 1,
            "threads_per_worker": pl_threads,
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# 硬件自动调优
# ---------------------------------------------------------------------------
def _auto_tune_hardware(n_workers, n_slc):
    """
    根据 CPU 核数和内存自动计算最优并行参数。

    关键内存约束:
      Phase Linking 每个 worker 加载 block_size^2 * n_slc * 8 bytes (complex64)
      threads_per_worker 个 worker 同时运行
      总内存 ≈ threads * block^2 * n_slc * 8

    SNAPHU: 单进程单线程, 每实例 ~1-3GB
    """
    profile = detect_hardware_profile()
    n_cpu = recommend_cpu_workers("dolphin", requested=n_workers)
    mem_gb = max(4.0, float(profile.available_memory_gb))

    n_ifgrams = max(n_slc - 1, 1)

    # --- Phase Linking 内存预算 ---
    # threads_per_worker = 同时处理的 block 数
    # 每个 block 内存 = block^2 * n_slc * 8 bytes (complex64) * ~3x 安全系数
    # PL 和 SNAPHU 顺序执行, PL 可以用更多内存
    mem_for_pl = mem_gb * 0.7
    block = 512  # 固定 512, 1024 内存开销翻 4 倍
    mem_per_block = (block**2 * n_slc * 8) / 1e9 * 3

    # worker 数 = min(核数, 内存能支撑的 block 数)
    pl_threads = min(n_cpu, max(1, int(mem_for_pl / mem_per_block)))
    pl_threads = max(pl_threads, 2)

    # --- SNAPHU 并行数 ---
    # 实测: SNAPHU 每实例峰值 5-8 GB (AOI 裁剪后 ~3-5 GB)
    # 保守按 6GB/实例, 用可用内存的 70% (留 30% 给 PL 残留+系统)
    mem_for_snaphu = mem_gb * 0.7
    snaphu_by_mem = max(1, int(mem_for_snaphu / 6))
    snaphu_jobs = min(n_cpu, snaphu_by_mem, n_ifgrams)
    snaphu_jobs = max(snaphu_jobs, 2)  # 最少 2

    return snaphu_jobs, pl_threads, block


# ---------------------------------------------------------------------------
# Run Dolphin
# ---------------------------------------------------------------------------
def run_dolphin(config_path_or_slc_pattern=None, log_file=None, n_workers=None):
    """
    执行 Dolphin 工作流。优先使用 Python API (稳定)，CLI 作后备。
    实现断点续传: 检查已有 unwrapped/ 输出则跳过。
    """
    log_file = Path(log_file or cfg.LOG_FILE)
    n_workers = n_workers or N_WORKERS
    work_dir = Path(config_path_or_slc_pattern).parent if config_path_or_slc_pattern else cfg.DOLPHIN_DIR

    # 若传入的是 config yaml
    config_path = Path(config_path_or_slc_pattern) if config_path_or_slc_pattern else None
    if config_path and config_path.suffix in ('.yaml', '.yml'):
        work_dir = config_path.parent

    # Checkpoint
    unwrap_dir = work_dir / "unwrapped"
    if unwrap_dir.exists():
        unw_files = list(unwrap_dir.glob("*.unw.tif"))
        if unw_files:
            print(f"Dolphin 输出已存在 ({len(unw_files)} 个解缠文件)。"
                  f"删除 {unwrap_dir} 以重新处理。")
            return

    # ---- 优先使用 Python API (稳定, 无 PATH 问题) ----
    print("启动 Dolphin Phase Linking + 解缠...")
    print(f"  工作目录: {work_dir}")
    print(f"  并行解缠: {n_workers} jobs")

    # 读取或构建配置
    slc_list_file = work_dir / "slc_files.txt"
    if config_path and config_path.exists() and config_path.suffix in ('.yaml', '.yml'):
        if slc_list_file.exists():
            slc_files = [l.strip() for l in slc_list_file.read_text().splitlines() if l.strip()]
        else:
            # 从 YAML 中读取 SLC 列表
            import yaml as _yaml
            with open(config_path) as f:
                cfg_data = _yaml.safe_load(f)
            slc_files = cfg_data.get('cslc_file_list', [])
    else:
        slc_files = sorted(glob(str(config_path) if config_path else cfg.SLC_VRT_PATTERN))

    try:
        from dolphin.workflows.config import DisplacementWorkflow
        from dolphin.workflows import displacement

        # 压制 rasterio/GDAL/jax 的冗余 DEBUG 日志 (否则淹没 Jupyter)
        import logging as _logging
        for noisy in ['rasterio', 'rasterio.env', 'rasterio._io', 'rasterio._base',
                       'rasterio._env', 'rasterio.session', 'rasterio._filepath',
                       'urllib3', 'matplotlib', 'h5py', 'jax', 'jax._src',
                       'numba', 'numba.core', 'numba.core.byteflow',
                       'numba.core.ssa_rewrite', 'numba.core.interpreter']:
            _logging.getLogger(noisy).setLevel(_logging.WARNING)

        # 根据硬件自动调优
        snap_jobs, pl_threads, blk = _auto_tune_hardware(n_workers, len(slc_files))
        print(f"  硬件自动调优: SNAPHU {snap_jobs} 并行 | Phase Linking {pl_threads} 线程 | 块 {blk}x{blk}")

        dolphin_cfg = DisplacementWorkflow(
            cslc_file_list=slc_files,
            work_directory=str(work_dir),
            phase_linking={"half_window": {"x": 11, "y": 5}, "ministack_size": 15},
            interferogram_network={
                "reference_idx": None,   # 网络型连接
                "max_bandwidth": 3,      # 每景与最近3景
            },
            unwrap_options={
                "unwrap_method": "snaphu",
                "n_parallel_jobs": snap_jobs,
            },
            output_options={"strides": {"x": 6, "y": 3}},
            worker_settings={
                "block_shape": [blk, blk],
                "n_parallel_bursts": 1,
                "threads_per_worker": pl_threads,
            },
        )
        displacement.run(dolphin_cfg)

    except Exception as e:
        logger.error(f"Dolphin Python API 失败: {e}")
        print(f"[ERROR] Dolphin 失败: {e}")
        raise

    unw_files = list((work_dir / "unwrapped").glob("*.unw.tif"))
    print(f"Dolphin 完成: {len(unw_files)} 对解缠干涉图")
