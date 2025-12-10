# syntax=docker/dockerfile:1

# GPU base image with CUDA 12.4 + cuDNN 9 runtime (compatible with RunPod CUDA 12.x)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Use /app/models for baked-in model (won't be shadowed by RunPod volume mounts)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models \
    XDG_CACHE_HOME=/app/models \
    HUGGINGFACE_HUB_CACHE=/app/models

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy serverless code
COPY handler.py /app/serverless/handler.py
COPY README.md /app/README.md

# Python dependencies:
# - faster-whisper for ASR
# - nvidia-cudnn-cu12 and nvidia-cuda-nvrtc-cu12 provide the libcudnn_ops.so.9.* and NVRTC libs
#   that CTranslate2 / faster-whisper expect when running on CUDA 12.
RUN pip install --no-cache-dir \
      "runpod==1.7.1" \
      "requests" \
      "faster-whisper==1.0.3" \
      "nvidia-cudnn-cu12==9.5.0.50" \
      "nvidia-cuda-nvrtc-cu12==12.4.127"

# Pre-download the 'turbo' model at build time to reduce cold start.
# We use device=cpu here so that the build does not require a physical GPU;
# at runtime the handler will load the same model on CUDA with float16.
RUN python - << 'PY'
import os
from faster_whisper import WhisperModel

os.makedirs("/app/models", exist_ok=True)
os.environ["HF_HOME"] = "/app/models"
os.environ["XDG_CACHE_HOME"] = "/app/models"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/app/models"

print("Pre-caching faster-whisper 'turbo' model to /app/models ...")
model = WhisperModel("turbo", device="cpu", compute_type="int8", download_root="/app/models")
print("Model cached.")
PY

CMD ["python", "-u", "serverless/handler.py"]
