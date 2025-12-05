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

# 2  Tell the build which GPU architecture to target
#    Auto-detect via nvidia-smi if available
if command -v nvidia-smi &> /dev/null; then
    RAW_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
    CLEAN_COMPUTE_CAP=$(echo "$RAW_COMPUTE_CAP" | tr -d '.')
    echo "Auto-detected GPU architecture: $RAW_COMPUTE_CAP (sm_$CLEAN_COMPUTE_CAP)"
else
    echo "nvidia-smi not found. Defaulting to 12.8 (Blackwell)"
    RAW_COMPUTE_CAP="12.8"
    CLEAN_COMPUTE_CAP="128"
fi

# 3. Detect local CUDA Compiler version (Mechanic)
#    We need to match PyTorch version to this, NOT the driver version.
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    # Format: 12.8
    CLEAN_NVCC_VERSION=$(echo "$NVCC_VERSION" | tr -d '.')
    # Format: 128
    echo "Auto-detected CUDA Compiler (nvcc): $NVCC_VERSION"
else
    echo "nvcc not found! Cannot determine compiler version."
    NVCC_VERSION="12.8"
    CLEAN_NVCC_VERSION="128"
fi

export CUDA_HOME=/usr/local/cuda
export CUDA_COMPILER_VERSION=$NVCC_VERSION
export CUDA_COMPILER_TAG="cu$CLEAN_NVCC_VERSION"

export FORCE_CUDA_ARCH=$CLEAN_COMPUTE_CAP
# optional but nice: keep Torch’s JIT in sync
export TORCH_CUDA_ARCH_LIST="$RAW_COMPUTE_CAP"
export FLASH_ATTENTION_FORCE_BUILD=TRUE   # skip the 404 wheel fetch
export FLASH_ATTN_CUDA_ARCHS=$CLEAN_COMPUTE_CAP          # matches detected arch
export MAX_JOBS=16                         # ninja job slots  (keeps RAM down)
export NVCC_THREADS=8                     # threads per nvcc compile
export NINJAFLAGS="-j8"
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

export TORCHINDUCTOR_CACHE_DIR=/tmp/.torchinductor
export TRITON_CACHE_DIR=/tmp/.triton