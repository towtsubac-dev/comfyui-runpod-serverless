FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone https://github.com/Comfy-Org/ComfyUI.git /workspace/ComfyUI

WORKDIR /workspace/ComfyUI

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod requests

WORKDIR /workspace

COPY handler.py /workspace/handler.py

CMD ["python", "/workspace/handler.py"]
