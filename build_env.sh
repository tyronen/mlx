#!/usr/bin/env bash
# run like `source setup.sh` on any remote to ensure active shell is set up

# --- 1. System Setup ---
cp ~/.env .env
set -a; source .env; set +a

mkdir -p data

source .env
if [[ -n "$SSH_CONNECTION" && -d /workspace ]]; then
  echo "🐧 Running on remote runpod with storage attached - setting custom hf cache dir"
  set -a
  export HF_HOME="/dev/shm/.cache"
  export HF_DATASETS_CACHE="$HF_HOME/datasets_cache"
  export PIP_CACHE_DIR="$HF_HOME/pip_cache"
  export KAGGLEHUB_CACHE="$HF_HOME/kagglehub"
  mkdir -p $HF_HOME
  mkdir -p $HF_DATASETS_CACHE
  mkdir -p $PIP_CACHE_DIR
  set +a
fi

echo "🔍 Detecting system package versions..."

echo "# This file is auto-generated to constrain torch and numpy to the system versions." > training/constraints.txt

# Execute Python commands to get the exact version strings
function getversion() {
  VERSION=$(python -c "import $1; print($1.__version__)")
  # Check that the commands succeeded
  if [[ -z "$VERSION" ]]; then
    echo "❌ Error: Failed to detect $1 . Ensure it is installed in the system's Python."
    exit 1
  fi
  echo "  ✅ Detected $1: $VERSION"
  echo "$1==$VERSION" >> training/constraints.txt
}

getversion torch
getversion numpy
getversion torchvision

echo "🎉 Success! 'training/constraints.txt' has been created."

# cd /workspace/flash-attention

# 2  Tell the build which GPU architecture to target
#    (compute 12.0 ➜ sm_120; change if your `torch.cuda.get_device_capability(0)`
#     reports something else)
export FORCE_CUDA_ARCH=120
# optional but nice: keep Torch’s JIT in sync
export TORCH_CUDA_ARCH_LIST="12.0"
export FLASH_ATTENTION_FORCE_BUILD=TRUE   # skip the 404 wheel fetch
export FLASH_ATTN_CUDA_ARCHS=120          # just Blackwell
export MAX_JOBS=8                         # ninja job slots  (keeps RAM down)
export NVCC_THREADS=8                     # threads per nvcc compile
export NINJAFLAGS="-j8"
export TMPDIR=/dev/shm
#fallocate -l 32G /swap && mkswap /swap && swapon /swap
# 3  Build & install against the Torch that’s already in the venv
# python -m pip install -v --no-build-isolation .

# 4  Quick import test
#python - <<'PY'
#import torch, flash_attn_cuda
#print("Flash-Attention OK on", torch.version.cuda,
#      "with", torch.cuda.get_device_name(0))
#PY
cd /workspace/mlx

pip install --upgrade pip

pip install -r training/requirements.txt -c training/constraints.txt
pip install -r api/requirements.txt -c training/constraints.txt

# Only needed for synthetic-generation (image-captioning), api server (audio) and SFT (fine-tuning)
# pip install flash-attn --no-build-isolation
wandb login $WANDB_KEY

# May be needed on RTX 5090
# pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129