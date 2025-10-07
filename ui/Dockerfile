FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
# Install your requirements
COPY ui/requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

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
COPY data/complex_mnist.pth /app/data/

# Copy source code
COPY common/ /app/common/
COPY models/ /app/models/
COPY ui/ /app/ui/

EXPOSE 80
