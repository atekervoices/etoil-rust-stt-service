# ==============================================================================
# Multi-stage Dockerfile for NVIDIA Nemotron 3.5 Streaming STT Service (GPU/CUDA)
# Completely Self-Contained: Downloads original .nemo, exports to ONNX & compiles Rust
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Model Downloader & ONNX Exporter
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS model-exporter

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /export

# Upgrade pip and install wheel/setuptools build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install lightweight CPU PyTorch using --extra-index-url
RUN pip install --no-cache-dir torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Install export requirements and NeMo toolkit
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    "pyarrow<18.0.0" \
    onnx \
    onnxruntime \
    soundfile \
    sentencepiece \
    huggingface_hub \
    "nemo_toolkit[asr]"

# Copy export helper scripts
COPY scripts/download_model.py /export/download_model.py
COPY scripts/export_nemotron.py /export/export_nemotron.py

# Download original NVIDIA Nemotron model (.nemo) from Hugging Face
RUN python3 /export/download_model.py --output-dir /export/models

# Export original NVIDIA .nemo to parakeet-rs ONNX format
RUN python3 /export/export_nemotron.py /export/models/nemotron-3.5-asr-streaming-0.6b.nemo /export/nemotron-3.5-asr-streaming-0.6b-onnx

# Clean up raw .nemo checkpoint to minimize disk usage
RUN rm -rf /export/models

# ------------------------------------------------------------------------------
# Stage 2: Rust Service Builder
# ------------------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    pkg-config \
    libssl-dev \
    libasound2-dev \
    g++ \
    git \
    ca-certificates \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy source code and static web assets
COPY . .

# Build the release binary with CUDA acceleration
RUN cargo build --release --features cuda --bin api_server

# ------------------------------------------------------------------------------
# Stage 3: Minimal Production Runtime (GPU / CUDA)
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies (no Python or build toolchains in final image!)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3t64 \
    libasound2t64 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Copy compiled Rust binary
COPY --from=builder /app/target/release/api_server /app/api_server

# 2. Copy the exported ONNX model files from stage 1
COPY --from=model-exporter /export/nemotron-3.5-asr-streaming-0.6b-onnx /app/nemotron-3.5-asr-streaming-0.6b-onnx

# Expose HTTP / WebSocket API port
EXPOSE 8099

# Environment variables
ENV RUST_LOG=info
ENV MODEL_PATH=/app/nemotron-3.5-asr-streaming-0.6b-onnx
ENV PORT=8099

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8099/health || exit 1

# Launch the STT API server
CMD ["/app/api_server"]
