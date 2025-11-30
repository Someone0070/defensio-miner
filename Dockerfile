# Defensio Miner - Docker Image
# =============================
# Supports both CPU and NVIDIA GPU mining

# Base image with CUDA support
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy CUDA kernel source
WORKDIR /build
COPY cuda/ashmaize_kernel.cu ./cuda/

# Build CUDA library for multiple architectures
RUN nvcc -O3 -arch=sm_70 -Xcompiler -fPIC --shared \
    -o cuda/libashmaize_cuda_sm70.so cuda/ashmaize_kernel.cu && \
    nvcc -O3 -arch=sm_75 -Xcompiler -fPIC --shared \
    -o cuda/libashmaize_cuda_sm75.so cuda/ashmaize_kernel.cu && \
    nvcc -O3 -arch=sm_80 -Xcompiler -fPIC --shared \
    -o cuda/libashmaize_cuda_sm80.so cuda/ashmaize_kernel.cu && \
    nvcc -O3 -arch=sm_86 -Xcompiler -fPIC --shared \
    -o cuda/libashmaize_cuda_sm86.so cuda/ashmaize_kernel.cu && \
    nvcc -O3 -arch=sm_89 -Xcompiler -fPIC --shared \
    -o cuda/libashmaize_cuda_sm89.so cuda/ashmaize_kernel.cu

# Runtime image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash miner

# Set working directory
WORKDIR /app

# Copy Python requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir cupy-cuda12x

# Copy CUDA libraries from builder
COPY --from=builder /build/cuda/*.so /app/cuda/

# Copy application files
COPY miner.py .
COPY src/ ./src/
COPY cuda/ashmaize_kernel.cu ./cuda/

# Create a script to select the right library based on GPU
RUN echo '#!/bin/bash\n\
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ".")\n\
case $ARCH in\n\
  70|72) LIB="sm70";;\n\
  75) LIB="sm75";;\n\
  80) LIB="sm80";;\n\
  86|87) LIB="sm86";;\n\
  89|90) LIB="sm89";;\n\
  *) LIB="sm80";;\n\
esac\n\
ln -sf /app/cuda/libashmaize_cuda_${LIB}.so /app/cuda/libashmaize_cuda.so\n\
exec python3 /app/miner.py "$@"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Create data directories
RUN mkdir -p /data/wallets /data/logs && \
    chown -R miner:miner /data /app

# Set default environment variables
ENV DEFENSIO_WALLET_DIR=/data/wallets
ENV DEFENSIO_LOG_DIR=/data/logs
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER miner

# Volume for persistent data
VOLUME ["/data"]

# Entry point
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden)
CMD ["--cpu-workers", "4"]
