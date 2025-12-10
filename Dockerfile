# syntax=docker/dockerfile:1

# GPU base image with CUDA + cuDNN runtime
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Use /app/models for baked-in model (won't be shadowed by RunPod volume mounts)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models \
    XDG_CACHE_HOME=/app/models \
    HUGGINGFACE_HUB_CACHE=/app/models

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the current directory contents (handler + dependencies) into the image.
COPY . /app/serverless/

RUN pip install --no-cache-dir \
    runpod==1.6.0 \
    faster-whisper==1.2.1 \
    requests==2.32.3

# Pre-download the Turbo weights into the image cache to avoid cold start fetches.
# Create the models directory first to ensure it exists
# Cache bust: v2
RUN mkdir -p /app/models && python - <<'PY'
import os
os.environ["HF_HOME"] = "/app/models"
os.environ["XDG_CACHE_HOME"] = "/app/models"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/app/models"
from faster_whisper import WhisperModel
# Use CPU for download; runtime will use CUDA.
# Explicitly set download_root to match runtime handler
model = WhisperModel("turbo", device="cpu", compute_type="int8", download_root="/app/models")
print("Turbo model cached.")
# Verify the cache location
import subprocess
subprocess.run(["ls", "-la", "/app/models"], check=True)
subprocess.run(["find", "/app/models", "-type", "d"], check=True)
PY

CMD ["python", "-u", "serverless/handler.py"]
