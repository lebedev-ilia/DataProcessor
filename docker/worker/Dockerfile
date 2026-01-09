FROM python:3.11-slim

WORKDIR /app

# Minimal deps for PR-0 bootstrap checks (not full ML pipeline deps).
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
      boto3==1.34.162 \
      celery==5.4.0 \
      fastapi==0.115.8 \
      redis==5.0.8 \
      requests==2.32.3 \
      uvicorn==0.30.6

CMD ["python", "/app/docker/worker/bootstrap.py"]


