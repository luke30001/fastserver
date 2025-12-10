# syntax=docker/dockerfile:1

# GPU base image with CUDA + cuDNN runtime
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

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

# Pre-download the Turbo weights into the image cache to avoid cold start fetches.
RUN python - <<'PY'
import os
from faster_whisper import WhisperModel
os.environ.setdefault("HF_HOME", "/cache/hf")
os.environ.setdefault("XDG_CACHE_HOME", "/cache")
# Use CPU for download; runtime will use CUDA.
WhisperModel("turbo", device="cpu", compute_type="int8")
print("Turbo model cached.")
PY

CMD ["python", "-u", "serverless/handler.py"]
