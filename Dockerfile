FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_PREFER_BINARY=1

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-dev \
    git \
    git-lfs \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# setup python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python3 -m pip install --upgrade pip setuptools wheel

# ============================================================================
# dependencies stage
# ============================================================================
FROM base AS dependencies

WORKDIR /app

# install pytorch with cuda support first (large, rarely changes)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# install xformers matching the pytorch/cuda version
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu121

# copy and install requirements
COPY requirements_versions.txt .
RUN pip install -r requirements_versions.txt

# install k-diffusion dependencies not in requirements
# - dctorch: discrete cosine transforms for layers.py
# - clip-anytorch: CLIP model for evaluation.py (provides 'clip' module)
# - wandb: experiment tracking (may be imported)
# - scipy: scientific computing utilities
# - lpips: perceptual loss for evaluation
RUN pip install dctorch clip-anytorch wandb scipy lpips

# ============================================================================
# final stage
# ============================================================================
FROM dependencies AS final

WORKDIR /app

# copy application code
COPY . .

# clone required repositories (using forks where originals were deleted)
RUN mkdir -p repositories && \
    git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git \
        repositories/stable-diffusion-webui-assets && \
    git clone --depth 1 https://github.com/w-e-w/stablediffusion.git \
        repositories/stable-diffusion-stability-ai && \
    git clone --depth 1 https://github.com/Stability-AI/generative-models.git \
        repositories/generative-models && \
    git clone --depth 1 https://github.com/crowsonkb/k-diffusion.git \
        repositories/k-diffusion && \
    git clone --depth 1 https://github.com/salesforce/BLIP.git \
        repositories/BLIP

# create directories for runtime data
RUN mkdir -p /app/models/Stable-diffusion \
             /app/models/VAE \
             /app/models/Lora \
             /app/models/ControlNet \
             /app/models/DAT \
             /app/outputs \
             /app/extensions \
             /app/embeddings \
             /app/log

# download DAT upscaler models (avoiding Git LFS issues)
# Source: HuggingFace mirrors of official DAT models
RUN cd /app/models/DAT && \
    # DAT (original) - x2, x3, x4
    wget -q https://huggingface.co/licyk/sd-upscaler-models/resolve/main/DAT/DAT_x2.pth && \
    wget -q https://huggingface.co/licyk/sd-upscaler-models/resolve/main/DAT/DAT_x3.pth && \
    wget -q https://huggingface.co/licyk/sd-upscaler-models/resolve/main/DAT/DAT_x4.pth && \
    # DAT-2 - x2, x3, x4
    wget -q https://huggingface.co/licyk/sd-upscaler-models/resolve/main/DAT/DAT_2_x2.pth && \
    wget -q https://huggingface.co/licyk/sd-upscaler-models/resolve/main/DAT/DAT_2_x3.pth && \
    wget -q https://huggingface.co/licyk/sd-upscaler-models/resolve/main/DAT/DAT_2_x4.pth && \
    # DAT-S (small) - x2, x3, x4
    wget -q https://huggingface.co/licyk/sd-upscaler-models/resolve/main/DAT/DAT_S_x2.pth && \
    wget -q https://huggingface.co/licyk/sd-upscaler-models/resolve/main/DAT/DAT_S_x3.pth && \
    wget -q https://huggingface.co/licyk/sd-upscaler-models/resolve/main/DAT/DAT_S_x4.pth && \
    # DAT-light - x2, x3, x4
    wget -q https://huggingface.co/licyk/sd-upscaler-models/resolve/main/DAT/DAT_light_x2.pth && \
    wget -q https://huggingface.co/licyk/sd-upscaler-models/resolve/main/DAT/DAT_light_x3.pth && \
    wget -q https://huggingface.co/licyk/sd-upscaler-models/resolve/main/DAT/DAT_light_x4.pth

# environment variables for gpu memory management
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10737418240
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# default port
EXPOSE 7860

# health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# run webui directly (repos already cloned)
CMD [ \
    "python", "webui.py", \
    "--api", "--xformers", "--listen", "--port", "7860", \
    "--skip-version-check", "--disable-safe-unpickle" \
]
