FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git

WORKDIR /workspace/ComfyUI

RUN python3 -m pip install --break-system-packages \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install --break-system-packages \
    -r requirements.txt

RUN pip install --break-system-packages runpod requests

WORKDIR /workspace

COPY handler.py /workspace/handler.py

CMD ["python3", "/workspace/handler.py"]
