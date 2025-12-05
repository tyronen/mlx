#!/usr/bin/env python3
"""
Comprehensive GPU diagnostic script.
Run this on the machine where you are experiencing issues.
"""

import sys
import os
import subprocess
import ctypes


def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True).decode().strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Failed: {e}"


print("=" * 60)
print("GPU Visibility Diagnostic v2")
print("=" * 60)

print(f"\n1. Environment Variables:")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
print(f"   LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")
print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not Set')}")
print(f"   PATH: {os.environ.get('PATH', 'Not Set')[:200]}...")  # Truncated

print(f"\n2. System Checks:")
print(f"   Python executable: {sys.executable}")
print(
    f"   NVIDIA-SMI check: {run_cmd('nvidia-smi --query-gpu=gpu_name,driver_version --format=csv,noheader')}"
)
print(f"   NVCC check: {run_cmd('nvcc --version | grep release')}")
print(f"   /dev/nvidia* check: {run_cmd('ls -l /dev/nvidia* 2>/dev/null')}")

print(f"\n3. PyTorch Checks:")
try:
    import torch

    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   PyTorch Built with CUDA: {torch.version.cuda}")
    print(f"   PyTorch Install Path: {os.path.dirname(torch.__file__)}")

    is_available = torch.cuda.is_available()
    print(f"   torch.cuda.is_available(): {is_available}")

    if is_available:
        print(f"   Device Count: {torch.cuda.device_count()}")
        print(f"   Current Device: {torch.cuda.current_device()}")
        print(f"   Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"   ❌ CUDA not available in PyTorch.")

        # Deep dive diagnosis
        print(f"\n4. Deep Dive Diagnosis:")

        # Check if libcuda.so can be loaded manually
        libnames = ["libcuda.so", "libcuda.so.1", "libcudart.so"]
        for name in libnames:
            try:
                ctypes.CDLL(name)
                print(f"   ✅ ctypes.CDLL('{name}') loaded successfully")
            except OSError as e:
                print(f"   ❌ Failed to load '{name}': {e}")

        # Check torch specific libraries
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
        print(f"   Checking PyTorch libs in: {torch_lib_path}")
        if os.path.exists(torch_lib_path):
            print(
                f"   Found libs: {', '.join(f for f in os.listdir(torch_lib_path) if 'cuda' in f)[:100]}..."
            )

except ImportError as e:
    print(f"   ❌ Failed to import torch: {e}")

print("=" * 60)
