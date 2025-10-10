#!/usr/bin/env bash
# run like `source ssh.sh` on tmp runpod, after sending your private key and this script

# ensure we have git, clone repo, cd in etc.
apt-get update
apt-get install -y vim rsync git nvtop htop tmux curl ca-certificates git-lfs lsof nano

# start ssh agent, add key, go to /workspace
eval "$(ssh-agent -s)"
ssh-add .ssh/id_ed25519
cd /workspace

git clone git@github.com:tyronen/gg-clip-vit.git || true
cd gg-clip-vit
git pull
git status

# chain into setup.sh
echo "Chaining into setup.sh..."
source setup.sh
