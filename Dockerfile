# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    XDG_CACHE_HOME=/cache

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY serverless/ /app/serverless/

RUN pip install --no-cache-dir \
    runpod==1.6.0 \
    faster-whisper==1.2.1 \
    requests==2.32.3

CMD ["python", "-u", "serverless/handler.py"]
