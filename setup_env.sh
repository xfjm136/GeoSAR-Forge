#!/bin/bash
# GeoSAR-Forge environment setup / rebuild
# Usage:
#   bash setup_env.sh
#   bash setup_env.sh --recreate
#   USE_LOCAL_PROXY=1 bash setup_env.sh --recreate
set -euo pipefail

ENV_DIR="${ENV_DIR:-/data/InSAR/env}"
KERNEL_NAME="${KERNEL_NAME:-insar}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-InSAR Pipeline}"
USE_LOCAL_PROXY="${USE_LOCAL_PROXY:-0}"
LOCAL_HTTP_PROXY="${LOCAL_HTTP_PROXY:-http://127.0.0.1:7890}"
LOCAL_ALL_PROXY="${LOCAL_ALL_PROXY:-socks5://127.0.0.1:7890}"
TORCH_COMPUTE_PLATFORM="${TORCH_COMPUTE_PLATFORM:-auto}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
TORCH_PACKAGE_SPEC="${TORCH_PACKAGE_SPEC:-torch torchvision torchaudio}"

RECREATE=0
for arg in "$@"; do
    case "$arg" in
        --recreate|--force)
            RECREATE=1
            ;;
        *)
            echo "[ERROR] Unknown argument: $arg"
            echo "Usage: bash setup_env.sh [--recreate]"
            exit 1
            ;;
    esac
done

if [ "$USE_LOCAL_PROXY" = "1" ]; then
    export https_proxy="$LOCAL_HTTP_PROXY"
    export http_proxy="$LOCAL_HTTP_PROXY"
    export all_proxy="$LOCAL_ALL_PROXY"
    echo "[INFO] Using local proxy:"
    echo "       http(s): $LOCAL_HTTP_PROXY"
    echo "       all_proxy: $LOCAL_ALL_PROXY"
else
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY
    echo "[INFO] Proxy disabled for environment setup"
fi

have_env_python=0
if [ -x "$ENV_DIR/bin/python" ]; then
    have_env_python=1
fi

if [ -d "$ENV_DIR" ] && [ "$RECREATE" -ne 1 ] && [ "$have_env_python" -eq 1 ]; then
    echo "[WARN] Environment already exists at $ENV_DIR"
    echo "       Use: bash setup_env.sh --recreate"
    exit 1
fi

if [ -d "$ENV_DIR" ] && [ "$RECREATE" -eq 1 -o "$have_env_python" -eq 0 ]; then
    echo "[INFO] Removing existing environment at $ENV_DIR"
    rm -rf "$ENV_DIR" 2>/dev/null || true
    if [ -d "$ENV_DIR" ]; then
        backup_dir="${ENV_DIR}.broken.$(date +%Y%m%d_%H%M%S)"
        echo "[WARN] Direct removal did not finish cleanly, moving old environment to $backup_dir"
        mv "$ENV_DIR" "$backup_dir"
    fi
fi

if ! command -v mamba >/dev/null 2>&1; then
    echo "[ERROR] mamba not found. Please install mamba or expose it in PATH."
    exit 1
fi

detect_torch_index_url() {
    if [ -n "$TORCH_INDEX_URL" ]; then
        echo "$TORCH_INDEX_URL"
        return
    fi

    case "$TORCH_COMPUTE_PLATFORM" in
        auto)
            if command -v nvidia-smi >/dev/null 2>&1; then
                echo "https://download.pytorch.org/whl/cu128"
            else
                echo "https://download.pytorch.org/whl/cpu"
            fi
            ;;
        gpu|cuda|cu128|cuda12.8|cuda-12.8)
            echo "https://download.pytorch.org/whl/cu128"
            ;;
        cpu)
            echo "https://download.pytorch.org/whl/cpu"
            ;;
        *)
            echo "[ERROR] Unsupported TORCH_COMPUTE_PLATFORM=$TORCH_COMPUTE_PLATFORM" >&2
            exit 1
            ;;
    esac
}

TORCH_INDEX_URL="$(detect_torch_index_url)"
echo "[INFO] PyTorch wheel index: $TORCH_INDEX_URL"

run_mamba_install() {
    local label="$1"
    shift
    echo "[INFO] $label"
    mamba install -p "$ENV_DIR" -c conda-forge -y "$@"
}

echo "[1/7] Creating base conda environment..."
mamba create -p "$ENV_DIR" -c conda-forge -y python=3.11 pip

echo "[2/7] Installing notebook and scientific basics..."
run_mamba_install "Installing base scientific stack" \
    numpy scipy pandas matplotlib h5py pyyaml requests \
    tqdm geopy scikit-learn joblib psutil \
    jupyterlab notebook ipykernel nbformat

echo "[3/7] Installing geospatial stack..."
run_mamba_install "Installing geospatial packages" \
    gdal rasterio shapely geopandas contextily simplekml

echo "[4/7] Installing InSAR support stack..."
run_mamba_install "Installing core InSAR support packages" \
    isce2 mintpy snaphu pysolid asf_search cdsapi pyaps3 eccodes

echo "[5/7] Installing Dolphin..."
if ! mamba install -p "$ENV_DIR" -c conda-forge -y dolphin; then
    echo "[WARN] conda dolphin install failed, trying pip fallback: dolphin-opera"
    "$ENV_DIR/bin/pip" install dolphin-opera
fi

echo "[6/7] Installing XGBoost and extra Python packages..."
"$ENV_DIR/bin/pip" install -q --upgrade pip setuptools wheel
"$ENV_DIR/bin/pip" install -q sentineleof xgboost==2.1.4
"$ENV_DIR/bin/pip" install --upgrade $TORCH_PACKAGE_SPEC --index-url "$TORCH_INDEX_URL"

echo "[7/7] Registering Jupyter kernel..."
"$ENV_DIR/bin/python" -m ipykernel install \
    --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

echo "[VERIFY] Verifying imports..."
"$ENV_DIR/bin/python" - <<'PY'
checks = [
    ("isce", "import isce"),
    ("dolphin", "import dolphin"),
    ("mintpy", "import mintpy"),
    ("asf_search", "import asf_search"),
    ("pysolid", "import pysolid"),
    ("torch", "import torch"),
    ("xgboost", "import xgboost"),
    ("h5py", "import h5py"),
    ("rasterio", "import rasterio"),
    ("geopandas", "import geopandas"),
    ("contextily", "import contextily"),
    ("simplekml", "import simplekml"),
    ("yaml", "import yaml"),
    ("cdsapi", "import cdsapi"),
    ("pyaps3", "import pyaps3"),
    ("eof.download", "from eof.download import download_eofs"),
    ("joblib", "import joblib"),
    ("sklearn", "import sklearn"),
    ("psutil", "import psutil"),
    ("nbformat", "import nbformat"),
    ("osgeo", "from osgeo import gdal"),
]

failed = []
for name, stmt in checks:
    try:
        exec(stmt, {})
        print(f"OK   {name}")
    except Exception as exc:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        failed.append(name)

import torch
print("torch cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
print("torch device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch first device:", torch.cuda.get_device_name(0))

if failed:
    raise SystemExit(f"Import verification failed: {failed}")
PY

echo "[DONE] Environment setup finished."
echo
echo "Environment deployed successfully."
echo "Jupyter kernel: $KERNEL_DISPLAY_NAME"
echo "Python path: $ENV_DIR/bin/python"
echo "Activate with: conda activate $ENV_DIR"
