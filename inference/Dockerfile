FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

COPY inference/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data
COPY data/mnist_cnn.pth /app/data/

COPY common/ /app/common/
COPY models/ /app/models/
COPY inference/ /app/inference/

CMD ["python", "inference/serverless.py", "--model_path", "/app/data/mnist_cnn.pth"]
