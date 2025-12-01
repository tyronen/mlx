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
#    (compute 12.0 ➜ sm_120; change if your `torch.cuda.get_device_capability(0)`
#     reports something else)
export FORCE_CUDA_ARCH=128
# optional but nice: keep Torch’s JIT in sync
export TORCH_CUDA_ARCH_LIST="12.8"
export FLASH_ATTENTION_FORCE_BUILD=TRUE   # skip the 404 wheel fetch
export FLASH_ATTN_CUDA_ARCHS=128          # just Blackwell
export MAX_JOBS=8                         # ninja job slots  (keeps RAM down)
export NVCC_THREADS=8                     # threads per nvcc compile
export NINJAFLAGS="-j8"
export TMPDIR=/dev/shm


export TORCHINDUCTOR_CACHE_DIR=/tmp/.torchinductor
export TRITON_CACHE_DIR=/tmp/.triton