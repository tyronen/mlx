#!/bin/zsh

set -e

echo "🚀 Building application for production..."
echo "📦 Building for platform: linux/arm64"
echo "🏷️  Tag: docker.io/$DOCKER_USERNAME/ui:latest"

# Build first without pushing
echo "🔨 Building image..."
docker buildx build --platform linux/arm64 -t docker.io/$DOCKER_USERNAME/ui:latest -f ui/Dockerfile .

echo "✅ Build completed successfully!"
