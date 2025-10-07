#!/bin/zsh

source .env.dev
export PYTHONPATH=$PWD:$PYTHONPATH

# Function to run docker compose with env file
docker_compose() {
    docker compose --env-file .env.dev "$@"
}

echo "🚀 Starting Redis with Docker Compose..."
docker_compose up -d redis || exit 1

# Wait for Redis to be ready
echo "⏳ Waiting for Redis to be ready..."
until redis-cli ping > /dev/null 2>&1; do
    sleep 1
done

echo "✅ Redis is ready, starting Streamlit..."

# Start Streamlit locally (not in Docker)
streamlit run ui/webserver.py --server.port=$PORT --server.address=0.0.0.0 --server.fileWatcherType=poll

# Cleanup on exit
echo "🛑 Shutting down Redis..."
docker_compose down
