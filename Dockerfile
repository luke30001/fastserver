FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY handler.py .

# Tweak at deploy time:
#   MODEL_SIZE=small|medium|large-v3
#   COMPUTE_TYPE=float16|int8_float16|int8
ENV MODEL_SIZE=medium \
    COMPUTE_TYPE=float16 \
    VAD_FILTER=true

CMD ["python3", "-u", "handler.py"]
