# syntax=docker/dockerfile:1
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS cuda_toolkit
FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime AS sage_builder
COPY --from=cuda_toolkit /usr/local/cuda-13.0 /usr/local/cuda-13.0
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV PATH=/usr/local/cuda-13.0/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST="8.9;12.0"
ENV MAX_JOBS=2
RUN apt-get update && apt-get install -y --no-install-recommends git gcc g++ ninja-build libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --break-system-packages --no-cache-dir packaging ninja setuptools wheel
RUN git clone --depth 1 --branch v2.2.0 https://github.com/thu-ml/SageAttention.git /tmp/SageAttention
RUN cd /tmp/SageAttention && pip wheel --no-build-isolation --no-deps -w /tmp/sage-wheels .

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
# SageAttention 2.2.0, built for SM89 and SM120
# ---------------------------------------------------------
COPY --from=sage_builder /tmp/sage-wheels/ /tmp/sage-wheels/
RUN pip install --break-system-packages --no-deps /tmp/sage-wheels/sageattention*.whl && rm -rf /tmp/sage-wheels
RUN python -c "import sageattention, torch; print('SageAttention import OK; torch=', torch.__version__)"

# ---------------------------------------------------------
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
# Veda Sparse Attention (isolated test branch)
# Predictor is baked into the image; no worker-time download.
# ---------------------------------------------------------
RUN git clone --depth 1 https://github.com/veda-sparse/Veda-on-ComfyUI.git \
    /workspace/ComfyUI/custom_nodes/Veda-on-ComfyUI
# The upstream node supports clone-and-restart; editable pip install fails
# because setuptools discovers multiple top-level asset directories.
# PyTorch already supplies Triton on Linux; verify its import without
# upgrading a Torch-coupled dependency. Fail early if it is missing.
RUN python -c "import torch, triton; print('Veda runtime:', 'torch', torch.__version__, 'triton', triton.__version__)"
RUN python -m compileall -q /workspace/ComfyUI/custom_nodes/Veda-on-ComfyUI/veda_comfy
# Diagnostic build: keep Veda files but prevent ComfyUI from importing the node.
# Move outside custom_nodes; restore only after startup is verified.
RUN mv /workspace/ComfyUI/custom_nodes/Veda-on-ComfyUI /workspace/Veda-on-ComfyUI-disabled
RUN mkdir -p /workspace/ComfyUI/models/veda
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download( \
repo_id='Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview', \
filename='minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors', \
revision='9a1fd3a41b4a754a7886e64e82edbddf599fd1bd', \
local_dir='/workspace/ComfyUI/models/veda' \
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
