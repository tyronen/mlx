#!/usr/bin/env python3
"""Quick GPU visibility diagnostic script."""

import sys

print("=" * 60)
print("GPU Visibility Diagnostic")
print("=" * 60)

# Check PyTorch
try:
    import torch

    print(f"\n✅ PyTorch imported successfully")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"\nGPU Devices:")
        for i in range(torch.cuda.device_count()):
            print(f"   [{i}] {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"       Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"       Compute capability: {props.major}.{props.minor}")

        # Try a simple CUDA operation
        print(f"\n✅ Testing CUDA operation...")
        x = torch.randn(100, 100).cuda()
        y = x @ x
        print(f"   Success! Result shape: {y.shape}")
    else:
        print(f"\n❌ CUDA is NOT available!")
        print(f"\nPossible causes:")
        print(f"   1. PyTorch not built with CUDA support")
        print(f"      Run: python -c 'import torch; print(torch.version.cuda)'")
        print(f"   2. NVIDIA driver not accessible in container")
        print(f"      Run: nvidia-smi")
        print(f"   3. Container not started with GPU access")
        print(f"      Docker: needs --gpus all")
        print(f"      RunPod: check template GPU settings")
        sys.exit(1)

except ImportError as e:
    print(f"❌ Failed to import PyTorch: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All checks passed!")
print("=" * 60)
