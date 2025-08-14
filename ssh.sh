#!/usr/bin/env bash
# run like `source ssh.sh` on tmp runpod

apt-get update
apt-get install -y vim rsync nvtop htop tmux curl ca-certificates git-lfs lsof nano less cargo rustc swig ninja-build build-essential

cd /workspace

mkdir -p mlx
pip install --upgrade pip
