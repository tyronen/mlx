#!/bin/bash
set -e
SERVER="lightsail"

# Build with compression and smaller layers
echo "🔨 Building optimized Docker image..."
docker buildx build --platform linux/amd64 \
    --compress \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -t docker.io/$DOCKER_USERNAME/ui:latest \
    -f ui/Dockerfile .

# Push with retry logic for slow connections
echo "📤 Pushing to Docker Hub (with retry logic)..."
max_retries=3
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    if docker push --platform linux/amd64 docker.io/$DOCKER_USERNAME/ui:latest; then
        echo "✅ Push successful!"
        break
    else
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            echo "❌ Push failed, retrying in 30 seconds... (attempt $retry_count/$max_retries)"
            sleep 30
        else
            echo "❌ Push failed after $max_retries attempts"
            exit 1
        fi
    fi
done

# Deploy to server
echo "🚀 Deploying to server..."
scp .env.prod $SERVER:.env.prod
scp ui_prod.sh $SERVER:ui_prod.sh
scp docker-compose.yml $SERVER:docker-compose.yml
ssh $SERVER "./ui_prod.sh"