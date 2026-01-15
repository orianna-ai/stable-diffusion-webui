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

# install xformers for memory-efficient attention
RUN pip install xformers

# copy and install requirements
COPY requirements_versions.txt .
RUN pip install -r requirements_versions.txt

# ============================================================================
# clone required repositories
# ============================================================================
FROM dependencies AS repos

WORKDIR /app

# clone repositories at pinned commits
RUN mkdir -p repositories && \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git \
        repositories/stable-diffusion-webui-assets && \
    cd repositories/stable-diffusion-webui-assets && \
    git checkout 6f7db241d2f8ba7457bac5ca9753331f0c266917

RUN git clone https://github.com/Stability-AI/stablediffusion.git \
        repositories/stable-diffusion-stability-ai && \
    cd repositories/stable-diffusion-stability-ai && \
    git checkout cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf

RUN git clone https://github.com/Stability-AI/generative-models.git \
        repositories/generative-models && \
    cd repositories/generative-models && \
    git checkout 45c443b316737a4ab6e40413d7794a7f5657c19f

RUN git clone https://github.com/crowsonkb/k-diffusion.git \
        repositories/k-diffusion && \
    cd repositories/k-diffusion && \
    git checkout ab527a9a6d347f364e3d185ba6d714e22d80cb3c

RUN git clone https://github.com/salesforce/BLIP.git \
        repositories/BLIP && \
    cd repositories/BLIP && \
    git checkout 48211a1594f1321b00f14c9f7a5b4813144b2fb9

# ============================================================================
# final stage
# ============================================================================
FROM repos AS final

WORKDIR /app

# copy application code
COPY . .

# create directories for runtime data
RUN mkdir -p /app/models/Stable-diffusion \
             /app/models/VAE \
             /app/models/Lora \
             /app/models/ControlNet \
             /app/outputs \
             /app/extensions \
             /app/embeddings \
             /app/log

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

# run webui directly (repos already cloned at build time)
CMD [ \
    "python", "webui.py", \
    "--api", "--xformers", "--listen", "--port", "7860", \
    "--skip-version-check", "--disable-safe-unpickle" \
]
