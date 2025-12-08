#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -x  # Print commands and their arguments as they are executed (for debugging)

# ===================== 1. Ensure required tools are installed =====================
CONDA_INSTALL_PATH="/opt/anaconda3"
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed! Exiting..."
    exit 1
fi

# Check if make is installed
if ! command -v make &> /dev/null; then
    echo "ERROR: make is not installed! Exiting..."
    exit 1
fi

# Check if docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "ERROR: docker is not installed! Exiting..."
    exit 1
fi
if ! sudo systemctl is-active --quiet docker; then
    echo "ERROR: docker service is not running! Exiting..."
    exit 1
fi

if [ ! -d "${CONDA_INSTALL_PATH}" ]; then
  echo "ERROR: Conda installation not found at ${CONDA_INSTALL_PATH}! Exiting..."
  exit 1
fi

# ===================== 2. Create and activate Conda virtual environment =====================
source "${CONDA_INSTALL_PATH}/bin/activate"
if conda info --envs | grep -q "^build_pytorch\s"; then
    echo "Conda environment 'build_pytorch' already exists, activating directly..."
else
    echo "Conda environment 'build_pytorch' not found, creating and activating..."
    conda create -y -n build_pytorch python=3.11
fi
conda activate build_pytorch

# ===================== 3. Pull PyTorch source code and sync submodules =====================
## Get the pytorch root directory, is the parent directory of this script's directory
PYTORCH_DIR="$(cd "$(dirname "$0")"/.. ; pwd -P)"
cd "${PYTORCH_DIR}"
git submodule sync
git submodule update --init --recursive

# ===================== 4. Compile PyTorch Docker image (adapt to notes) =====================
# Execute compilation
commit_hash=$(git rev-parse --short HEAD)
PYTORCH_VERSION="v2.8.0-${commit_hash}" CUDA_VERSION_SHORT="12.8" CUDA_VERSION="12.8.1"  PYTHON_VERSION="3.11" \
    make -f docker.Makefile devel-image

echo "PyTorch image compilation completed!"