#!/bin/bash
set -e
SERVER="lightsail"

# Build with compression and smaller layers
echo "🔨 Building optimized Docker image..."
docker buildx build --platform linux/amd64 \
    -t docker.io/$DOCKER_USERNAME/ui:latest \
    -f ui/Dockerfile .

docker push --platform linux/amd64 docker.io/$DOCKER_USERNAME/ui:latest

# Deploy to server
echo "🚀 Deploying to server..."
scp .env.prod $SERVER:.env.prod
scp ui_prod.sh $SERVER:ui_prod.sh
scp docker-compose.yml $SERVER:docker-compose.yml
ssh $SERVER "./ui_prod.sh"