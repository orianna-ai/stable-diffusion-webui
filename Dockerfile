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
# final stage
# ============================================================================
FROM dependencies AS final

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

# default command
CMD [ \
    "python", "launch.py", \
    "--api", "--xformers", "--listen", "--port", "7860", \
    "--skip-version-check", "--disable-safe-unpickle" \
]
