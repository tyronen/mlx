#!/usr/bin/env bash
# run like `source setup.sh` on any remote to ensure active shell is set up

# --- 1. System Setup ---
cp ~/.env .env
set -a; source .env; set +a

mkdir -p data

source .env


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

cd /workspace/mlx

pip install --upgrade pip

pip install -r training/requirements.txt -c training/constraints.txt
pip install -r api/requirements.txt -c training/constraints.txt

wandb login $WANDB_KEY

export TORCHINDUCTOR_CACHE_DIR=/tmp/.torchinductor
export TRITON_CACHE_DIR=/tmp/.triton

