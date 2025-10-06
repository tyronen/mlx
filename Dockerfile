FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install production dependencies with optimizations
COPY ui/requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt && \
    pip cache purge

# Create necessary directories
RUN mkdir -p /app/data /app/data/hn_predict /app/common /app/models /app/ui /data

# Copy data files
COPY data/mnist_cnn.pth /app/data/
COPY data/train_vocab.json /app/data/
COPY data/hn_predict/word_to_ix.json /app/data/hn_predict/
COPY data/scaler.npz /app/data/
COPY data/inference_cache.pkl /app/data/
COPY data/best_quantile.pth /app/data/
COPY data/word2vec_skipgram.pth /app/data/
COPY data/twin_towers.pth /app/data/
COPY data/redis_dump.rdb /app/data/

# Copy source code
COPY common/ /app/common/
COPY models/ /app/models/
COPY ui/ /app/ui/

# Production optimizations
RUN find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true


EXPOSE 80
