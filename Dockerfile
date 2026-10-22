# ==============================================================================
# Multi-stage Dockerfile for NVIDIA Nemotron 3.5 Streaming STT Service (TorchScript)
# Self-Contained: Exports TorchScript model, builds Rust tch-rs binary, runs service
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Model Exporter (Exports HF model to TorchScript if not already present)
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS model-exporter

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /export

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir transformers soundfile huggingface_hub librosa scipy

ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

COPY scripts/export_clean_torchscript.py /export/export_clean_torchscript.py

RUN echo "Exporting TorchScript model from Hugging Face..." && \
    OUTPUT_DIR=/export/torchscript_model python3 /export/export_clean_torchscript.py

# ------------------------------------------------------------------------------
# Stage 2: Rust Service Builder (tch-rs with LibTorch C++)
# ------------------------------------------------------------------------------
FROM rust:1.85-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    pkg-config \
    libssl-dev \
    libasound2-dev \
    g++ \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG LIBTORCH_URL="https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip"

RUN mkdir -p /usr/local/libtorch && \
    echo "Downloading LibTorch from ${LIBTORCH_URL}..." && \
    for i in 1 2 3 4 5 6 7 8 9 10; do \
        echo "Download attempt $i/10..." && \
        wget -c --progress=dot:giga -O /tmp/libtorch.zip "${LIBTORCH_URL}" && break || sleep 3; \
    done && \
    test -s /tmp/libtorch.zip && \
    unzip -q /tmp/libtorch.zip -d /tmp && \
    mv /tmp/libtorch/* /usr/local/libtorch/ && \
    rm -rf /tmp/libtorch.zip /tmp/libtorch

ENV LIBTORCH=/usr/local/libtorch
ENV LD_LIBRARY_PATH=/usr/local/libtorch/lib:${LD_LIBRARY_PATH}

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY static ./static

# Build the release binary
RUN cargo build --release --bin api_server

# ------------------------------------------------------------------------------
# Stage 3: Minimal Production Runtime with NVIDIA CUDA
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/libtorch/lib:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    libasound2 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Copy LibTorch runtime libraries
COPY --from=builder /usr/local/libtorch/lib/ /usr/local/libtorch/lib/
RUN ldconfig /usr/local/libtorch/lib

# 2. Copy compiled Rust API server
COPY --from=builder /app/target/release/api_server /app/api_server

# 3. Copy TorchScript model
COPY --from=model-exporter /export/torchscript_model /app/torchscript_model

EXPOSE 8099

ENV RUST_LOG=info
ENV MODEL_PATH=/app/torchscript_model
ENV PORT=8099
ENV TORCH_CPP_LOG_LEVEL=ERROR

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8099/health || exit 1

CMD ["/app/api_server"]
