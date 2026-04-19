"""Hardware-aware runtime tuning helpers for the InSAR pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import shutil
import subprocess
from typing import Any


@dataclass(frozen=True)
class HardwareProfile:
    logical_cpus: int
    physical_cpus: int
    total_memory_gb: float
    available_memory_gb: float
    gpu_names: tuple[str, ...]
    gpu_total_memory_gb: tuple[float, ...]
    gpu_free_memory_gb: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PROFILE_CACHE: HardwareProfile | None = None


def _read_proc_meminfo() -> tuple[float, float]:
    total_gb = 32.0
    avail_gb = 24.0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            values = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    values[parts[0].rstrip(":")] = float(parts[1])
        if "MemTotal" in values:
            total_gb = values["MemTotal"] / (1024.0**2)
        if "MemAvailable" in values:
            avail_gb = values["MemAvailable"] / (1024.0**2)
        else:
            avail_gb = total_gb * 0.75
    except Exception:
        pass
    return float(total_gb), float(avail_gb)


def _detect_gpu_profile() -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    if shutil.which("nvidia-smi") is None:
        return (), (), ()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return (), (), ()

    names: list[str] = []
    total_mem: list[float] = []
    free_mem: list[float] = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            names.append(parts[0])
            total_mem.append(float(parts[1]) / 1024.0)
            free_mem.append(float(parts[2]) / 1024.0)
        except Exception:
            continue
    return tuple(names), tuple(total_mem), tuple(free_mem)


def detect_hardware_profile(*, refresh: bool = False) -> HardwareProfile:
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None and not refresh:
        return _PROFILE_CACHE

    logical_cpus = os.cpu_count() or 4
    physical_cpus = logical_cpus
    try:
        import psutil

        logical_cpus = psutil.cpu_count(logical=True) or logical_cpus
        physical_cpus = psutil.cpu_count(logical=False) or physical_cpus
        vm = psutil.virtual_memory()
        total_gb = float(vm.total) / (1024.0**3)
        avail_gb = float(vm.available) / (1024.0**3)
    except Exception:
        total_gb, avail_gb = _read_proc_meminfo()

    gpu_names, gpu_total, gpu_free = _detect_gpu_profile()
    _PROFILE_CACHE = HardwareProfile(
        logical_cpus=int(logical_cpus),
        physical_cpus=int(physical_cpus or logical_cpus),
        total_memory_gb=float(total_gb),
        available_memory_gb=float(avail_gb),
        gpu_names=gpu_names,
        gpu_total_memory_gb=gpu_total,
        gpu_free_memory_gb=gpu_free,
    )
    return _PROFILE_CACHE


def _env_override_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def build_thread_limited_env(base_env: dict[str, str] | None = None, *, threads_per_process: int = 1) -> dict[str, str]:
    env = dict(base_env or os.environ.copy())
    thread_value = str(max(1, int(threads_per_process)))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = thread_value
    return env


def _stage_memory_per_worker_gb(stage: str) -> float:
    stage = stage.lower()
    stage_map = {
        "general": 6.0,
        "download": 1.0,
        "isce_extract": 1.0,
        "isce_generate_stack": 2.0,
        "isce:run_01_unpack_topo_reference": 1.5,
        "isce:run_02_unpack_secondary_slc": 2.0,
        "isce:run_03_average_baseline": 1.5,
        "isce:run_04_fullburst_geo2rdr": 5.0,
        "isce:run_05_fullburst_resample": 8.0,
        "isce:run_06_extract_stack_valid_region": 6.0,
        "isce:run_07_merge_reference_secondary_slc": 10.0,
        "isce:run_08_grid_baseline": 3.0,
        "mintpy": 24.0,
        "tree_models": 4.0,
        "dolphin": 6.0,
    }
    return float(stage_map.get(stage, stage_map["general"]))


def recommend_cpu_workers(
    stage: str = "general",
    *,
    requested: int | None = None,
    n_items: int | None = None,
) -> int:
    profile = detect_hardware_profile()
    override = _env_override_int("INSAR_FORGE_MAX_CPU_WORKERS")
    cpu_cap = max(1, int(profile.physical_cpus or profile.logical_cpus or 1))
    if override is not None:
        cpu_cap = min(cpu_cap, override)

    requested_val = int(requested) if requested and requested > 0 else cpu_cap
    requested_val = max(1, min(requested_val, cpu_cap))

    usable_mem_gb = max(2.0, profile.available_memory_gb * 0.75)
    mem_per_worker = _stage_memory_per_worker_gb(stage)
    mem_cap = max(1, int(usable_mem_gb // max(mem_per_worker, 0.5)))

    worker_count = min(requested_val, cpu_cap, mem_cap)
    if n_items is not None and n_items > 0:
        worker_count = min(worker_count, int(n_items))
    return max(1, int(worker_count))


def recommend_mintpy_settings(*, requested_workers: int | None = None) -> dict[str, int]:
    profile = detect_hardware_profile()
    num_worker = min(4, recommend_cpu_workers("mintpy", requested=requested_workers))
    max_memory = int(max(4, min(16, math.floor(profile.available_memory_gb * 0.12))))
    return {
        "num_worker": max(1, num_worker),
        "max_memory_gb": max(4, max_memory),
    }


def _target_gpu_memory_gb(device: str) -> float | None:
    profile = detect_hardware_profile()
    if device != "cuda" or not profile.gpu_total_memory_gb:
        return None
    return float(profile.gpu_total_memory_gb[0])


def recommend_torch_batch_size(stage: str, *, requested: int | None, device: str = "auto") -> int:
    profile = detect_hardware_profile()
    if device == "auto":
        device = "cuda" if profile.gpu_total_memory_gb else "cpu"

    stage = stage.lower()
    gpu_mem = _target_gpu_memory_gb(device)

    if "train" in stage:
        if gpu_mem is not None:
            if gpu_mem <= 8:
                recommended = 64
            elif gpu_mem <= 12:
                recommended = 96
            elif gpu_mem <= 16:
                recommended = 128
            elif gpu_mem <= 24:
                recommended = 192
            else:
                recommended = 256
        else:
            recommended = 32 if profile.available_memory_gb < 32 else 64
    else:
        if gpu_mem is not None:
            if gpu_mem <= 8:
                recommended = 256
            elif gpu_mem <= 12:
                recommended = 512
            elif gpu_mem <= 16:
                recommended = 768
            elif gpu_mem <= 24:
                recommended = 1024
            else:
                recommended = 1536
        else:
            recommended = 128 if profile.available_memory_gb < 32 else 256

    if requested is None or requested <= 0:
        return int(recommended)
    return int(max(1, min(int(requested), int(recommended))))


def summarize_hardware() -> dict[str, Any]:
    profile = detect_hardware_profile()
    return {
        "logical_cpus": profile.logical_cpus,
        "physical_cpus": profile.physical_cpus,
        "total_memory_gb": round(profile.total_memory_gb, 2),
        "available_memory_gb": round(profile.available_memory_gb, 2),
        "gpu_names": list(profile.gpu_names),
        "gpu_total_memory_gb": [round(x, 2) for x in profile.gpu_total_memory_gb],
        "gpu_free_memory_gb": [round(x, 2) for x in profile.gpu_free_memory_gb],
    }
