#!/usr/bin/env bash
# Deep dive into CUDA initialization failure

echo "========================================="
echo "🔬 Deep CUDA Diagnostic"
echo "========================================="

echo ""
echo "1. CUDA Libraries Check:"
echo "------------------------"
ldconfig -p | grep -E "libcuda|libcudart|libnvidia" | head -20

echo ""
echo "2. CUDA Installation Paths:"
echo "------------------------"
ls -la /usr/local/cuda*/lib64/ 2>/dev/null | head -20 || echo "No /usr/local/cuda"
ls -la /usr/lib/x86_64-linux-gnu/ | grep -E "cuda|nvidia" | head -20

echo ""
echo "3. Environment Variables:"
echo "------------------------"
env | grep -E "CUDA|LD_LIBRARY" || echo "No CUDA env vars"

echo ""
echo "4. Detailed PyTorch CUDA Test:"
echo "------------------------"
python3 << 'PYEOF'
import sys
import torch
import os

print(f"PyTorch: {torch.__version__}")
print(f"Python: {sys.version}")
print(f"CUDA compiled version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

print("\nCUDA availability check:")
try:
    is_avail = torch.cuda.is_available()
    print(f"  torch.cuda.is_available(): {is_avail}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nDevice count check:")
try:
    count = torch.cuda.device_count()
    print(f"  torch.cuda.device_count(): {count}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nTrying to get device properties:")
try:
    if torch.cuda.device_count() > 0:
        props = torch.cuda.get_device_properties(0)
        print(f"  Device 0: {props.name}")
        print(f"  Compute: {props.major}.{props.minor}")
        print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nChecking CUDA library loading:")
try:
    # Try to manually load CUDA
    import ctypes
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        print("  ✅ libcuda.so.1 loaded successfully")
    except Exception as e:
        print(f"  ❌ Failed to load libcuda.so.1: {e}")
    
    try:
        libcudart = ctypes.CDLL("libcudart.so.12")
        print("  ✅ libcudart.so.12 loaded successfully")
    except Exception as e:
        print(f"  ❌ Failed to load libcudart.so.12: {e}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nLD_LIBRARY_PATH:")
print(f"  {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

PYEOF

echo ""
echo "5. Check for conflicting CUDA installations:"
echo "------------------------"
find /usr -name "libcudart.so*" 2>/dev/null

echo ""
echo "========================================="

