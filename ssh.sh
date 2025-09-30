#!/usr/bin/env bash
# run like `source ssh.sh` on tmp runpod

apt-get update
apt-get install -y vim rsync nvtop htop tmux curl ca-certificates git-lfs lsof nano less cargo rustc swig ninja-build build-essential

cd /workspace/mlx

pip install --upgrade pip
source .env
wandb login $WANDB_API_KEY
source setup.sh