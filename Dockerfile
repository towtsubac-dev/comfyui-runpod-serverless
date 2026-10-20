# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/workspace/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ffmpeg \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---------------------------------------------------------
# ComfyUI
# ---------------------------------------------------------

RUN git clone --depth 1 \
    https://github.com/comfyanonymous/ComfyUI.git \
    /workspace/ComfyUI

# ---------------------------------------------------------
# MiniMax H3 video extension
# ---------------------------------------------------------

RUN git clone --depth 1 \
    https://github.com/kat3ri/ComfyUI-MiniMax-H3-Extend.git \
    /workspace/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Extend

# Video input decoding for H3 continuation
RUN git clone --depth 1 \
    https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite

WORKDIR /workspace/ComfyUI

RUN python --version && \
    pip --version && \
    python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"

RUN pip install --break-system-packages \
    --no-cache-dir \
    -r requirements.txt

RUN pip install --break-system-packages \
    --no-cache-dir \
    runpod \
    requests \
    huggingface_hub

RUN pip install --break-system-packages \
    --no-cache-dir \
    -r /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt

# ---------------------------------------------------------
# SageAttention 2.2.0
#
# PyTorch 2.12.0
# CUDA 13.0
# Python 3.12
# RTX 4090 / SM89
# ---------------------------------------------------------

RUN pip install --break-system-packages \
    --no-cache-dir \
    https://github.com/tcpassos/sage-wheels-linux/releases/download/sage-2.2.0-torch-2.12.0-cu130-py312/sageattention-2.2.0-89-cp312-cp312-linux_x86_64.whl

# ---------------------------------------------------------
# MiniMax H3 directories
# ---------------------------------------------------------

RUN mkdir -p \
    /workspace/ComfyUI/models/diffusion_models \
    /workspace/ComfyUI/models/text_encoders \
    /workspace/ComfyUI/models/vae \
    /workspace/ComfyUI/models/loras

# ---------------------------------------------------------
# MiniMax H3 diffusion model
# ---------------------------------------------------------

RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download( \
    repo_id='Comfy-Org/MiniMax-H3', \
    filename='minimax_h3_fl2va_pruned_int8_convrot.safetensors', \
    subfolder='diffusion_models', \
    local_dir='/workspace/ComfyUI/models' \
)"

# ---------------------------------------------------------
# MiniMax H3 text encoder
# ---------------------------------------------------------

RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download( \
    repo_id='Comfy-Org/MiniMax-H3', \
    filename='qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', \
    subfolder='text_encoders', \
    local_dir='/workspace/ComfyUI/models' \
)"

# ---------------------------------------------------------
# MiniMax H3 Video VAE
# ---------------------------------------------------------

RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download( \
    repo_id='Comfy-Org/MiniMax-H3', \
    filename='minimax_h3_video_vae_int8_convrot.safetensors', \
    subfolder='vae', \
    local_dir='/workspace/ComfyUI/models' \
)"

# ---------------------------------------------------------
# MiniMax H3 Audio VAE
# ---------------------------------------------------------

RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download( \
    repo_id='Comfy-Org/MiniMax-H3', \
    filename='minimax_h3_audio_vae_fp32.safetensors', \
    subfolder='vae', \
    local_dir='/workspace/ComfyUI/models' \
)"

# ---------------------------------------------------------
# MiniMax H3 Turbo LoRA
# 8-step generation
# ---------------------------------------------------------

RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download( \
    repo_id='lightx2v/Minimax-h3-Turbo', \
    filename='minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors', \
    local_dir='/workspace/ComfyUI/models/loras' \
)"

# ---------------------------------------------------------
# Handler
# ---------------------------------------------------------

WORKDIR /workspace

COPY handler.py /workspace/handler.py

# ---------------------------------------------------------
# Start
# ---------------------------------------------------------

CMD ["python", "/workspace/handler.py"]
