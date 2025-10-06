#!/bin/bash

# Wait for Redis to be ready
echo "⏳ Waiting for Redis to be ready..."
until redis-cli -h $REDIS_HOST -p $REDIS_PORT ping > /dev/null 2>&1; do
    sleep 1
done

echo "✅ Redis is ready"

# Start the Streamlit application
echo "🌐 Starting Streamlit application..."
exec streamlit run ui/webserver.py --server.port=80 --server.address=0.0.0.0
