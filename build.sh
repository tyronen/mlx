#!/bin/zsh

set -e

echo "🚀 Building application for production..."
docker buildx build --platform linux/amd64,linux/arm64 -t docker.io/$DOCKER_USERNAME/ui:latest --target prod --push .

echo "✅ Build completed successfully!"
