#!/bin/bash
# Script to compile bitsandbytes for CUDA 12.8

set -e

echo "Installing build dependencies..."
apt-get install -y build-essential cmake

echo "Cloning bitsandbytes repository..."
rm -rf bitsandbytes
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes

echo "Setting up CUDA environment..."
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VERSION=128

echo "Building bitsandbytes for CUDA 12.8..."
python setup.py install
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install .

echo "Testing installation..."
python -m bitsandbytes

echo "Done! bitsandbytes should now work with CUDA 12.8"

