FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

COPY ui/requirements.txt .

RUN pip install --prefer-binary -r requirements.txt

RUN mkdir -p /app/data
COPY data/mnist_cnn.pth /app/data/
COPY common/ /app/common/
COPY models/ /app/models/
COPY ui/ /app/ui/

EXPOSE $PORT
CMD sh -c "streamlit run ui/webserver.py --server.port=${PORT:-80} --server.address=0.0.0.0"
