FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

COPY ui/requirements.txt .

RUN pip install --prefer-binary -r requirements.txt

COPY ui/ /app/
EXPOSE 80
CMD sh -c "streamlit run webserver.py --server.port=${PORT:-80} --server.address=0.0.0.0"