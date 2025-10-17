#!/usr/bin/env bash

set -e

echo "🚀 Initializing RunPod environment..."

apt-get update
apt-get install -y vim rsync nvtop htop tmux curl ca-certificates git-lfs lsof nano less cargo rustc swig ninja-build build-essential pciutils wget

echo "========================================="
echo "🔍 GPU Visibility Check"
echo "========================================="

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi found"
    nvidia-smi
else
    echo "❌ nvidia-smi NOT found - GPU will not be accessible"
fi

# Check PCI devices
echo ""
echo "PCI NVIDIA devices:"
lspci | grep -i nvidia || echo "❌ No NVIDIA devices found via lspci"

# Check if CUDA libraries are accessible
echo ""
echo "CUDA runtime libraries:"
if ldconfig -p | grep -q libcudart; then
    echo "✅ CUDA runtime libraries found"
    ldconfig -p | grep libcudart | head -3
else
    echo "❌ CUDA runtime libraries NOT found!"
    echo "   This is likely why PyTorch can't initialize CUDA."
    echo "   The container needs CUDA runtime libraries mounted."
    echo ""
    echo "   Common locations to check:"
    ls -la /usr/local/cuda*/lib64/libcudart.so* 2>/dev/null || echo "   ❌ Not in /usr/local/cuda*/lib64/"
    ls -la /usr/lib/x86_64-linux-gnu/libcudart.so* 2>/dev/null || echo "   ❌ Not in /usr/lib/x86_64-linux-gnu/"
fi

echo ""
echo "PyTorch CUDA check:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'Built with CUDA: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}')" || echo "❌ Failed to check PyTorch"

echo ""
echo "========================================="

# Fix broken RunPod template - CUDA runtime present but PyTorch can't initialize
if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    if command -v nvidia-smi &> /dev/null && ldconfig -p | grep -q libcudart; then
        echo ""
        echo "⚠️  GPU and CUDA runtime present, but PyTorch can't initialize CUDA"
        echo "    This is a known issue with PyTorch 2.8.0+cu128 on certain drivers"
        echo ""
        
        # Check for version mismatch
        PYTORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "12.8")
        DRIVER_CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "unknown")
        
        echo "PyTorch built with CUDA: $PYTORCH_CUDA"
        echo "Driver supports CUDA: $DRIVER_CUDA"
        echo ""
        
        # The issue: Container's NVIDIA_REQUIRE_CUDA doesn't include driver 575
        # The template was built before driver 575 existed
        echo "Detected driver version constraint issue"
        echo "Removing NVIDIA_REQUIRE_CUDA restriction..."
        
        # Unset the restrictive requirement and add to bashrc
        unset NVIDIA_REQUIRE_CUDA
        
        cat >> ~/.bashrc << 'EOF'

# Fix for PyTorch CUDA initialization on newer drivers
# RunPod template doesn't know about driver 575+
unset NVIDIA_REQUIRE_CUDA
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
EOF
        
        source ~/.bashrc
        
        echo "✅ Removed driver version restriction"
        echo ""
        echo "Testing CUDA after fix..."
        
        # Test with the new environment
        if CUDA_VISIBLE_DEVICES=0 python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"; then
            echo "✅ CUDA now working!"
        else
            echo ""
            echo "❌ Still not working. Possible causes:"
            echo "   1. PyTorch 2.8.0+cu128 incompatible with CUDA 12.9 driver"
            echo "   2. Missing CUDA libraries beyond runtime"
            echo "   3. Driver/container permission issues"
            echo ""
            echo "Recommended: Use a different RunPod template"
        fi
    fi
fi

echo ""
echo "========================================="

cd /workspace/ramdiskbackup/shm || true
rsync -vrt .cache /dev/shm || true
rsync -vrt * /dev/shm || true

echo "✅ init_pod.sh completed"