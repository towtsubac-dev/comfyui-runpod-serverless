FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone --depth 1 \
    https://github.com/comfyanonymous/ComfyUI.git \
    /workspace/ComfyUI

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
    requests
    
RUN pip install --break-system-packages \
    --no-cache-dir \
    huggingface_hub

ENV HF_HOME=/workspace/huggingface

WORKDIR /workspace

COPY handler.py /workspace/handler.py

CMD ["python", "/workspace/handler.py"]
