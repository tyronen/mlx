#!/bin/bash
# Ensure we have the correct environment variables for the build
source ./env_vars.sh

# Optional: Go to workspace if it exists, but not strictly required for pip install unless building local clone
if [ -d "/workspace/flash-attention" ]; then
    cd /workspace/flash-attention
fi
# Force uninstall ONLY if specifically requested (e.g. ./setup_5090.sh clean)
if [ "$1" == "clean" ]; then
    echo "Cleaning up old installations..."
    pip uninstall -y flash-attn || true
fi

# ---------------------------------------------------------------------------
# CRITICAL: Version Alignment Strategy
# ---------------------------------------------------------------------------
# 1. Driver: 12.9 (Provided by Host/RunPod)
# 2. Compiler: Auto-detected in env_vars.sh (e.g., 12.8)
# 3. PyTorch: MUST MATCH COMPILER, NOT DRIVER
#
# If we install PyTorch cu129, it will look for a 12.9 compiler which doesn't exist
# in this container, causing the Flash Attention build to fail.
# We must downgrade/pin PyTorch to the compiler version (e.g., cu128).
# ---------------------------------------------------------------------------

# Check if PyTorch is already installed correctly (basic check)
if python -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "$CUDA_COMPILER_TAG"; then
    echo "PyTorch already matches $CUDA_COMPILER_TAG. Skipping reinstall."
else
    echo "Installing PyTorch for CUDA Compiler: $CUDA_COMPILER_TAG"
    pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/$CUDA_COMPILER_TAG
fi

# Now that PyTorch matches the local compiler, we can build Flash Attention
# It will use the env vars from env_vars.sh to target the 5090 hardware (sm_120)

# Check if flash-attn is already installed
if pip show flash-attn > /dev/null 2>&1; then
    echo "✅ Flash Attention is already installed."
    echo "   Run './setup_5090.sh clean' if you want to force a rebuild."
else
    echo "⏳ Building Flash Attention for arch $FLASH_ATTN_CUDA_ARCHS... (This takes 5-10 mins)"
    # Removed --no-cache-dir to allow pip to reuse previous builds if available
    pip install flash-attn --no-build-isolation --no-binary flash-attn -v
fi

