# Multi-stage Dockerfile for Canary STT Service
# Stage 1: Build
FROM ubuntu:24.04 AS builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies and Rust
RUN apt-get update && apt-get install -y \
    curl \
    pkg-config \
    libssl-dev \
    libasound2-dev \
    g++ \
    ca-certificates \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy source code
COPY . .

# Build the application
RUN cargo build --release

# Stage 2: Runtime (with GPU support)
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3t64 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /app/target/release/api_server /app/api_server
COPY --from=builder /app/target/release/main /app/main

# Copy model directory (if it exists in the build context)
COPY canary-180m-flash-int8 ./canary-180m-flash-int8

# Create directory for test audio
RUN mkdir -p /app/audio

# Expose the API port
EXPOSE 8080

# Set environment variables
ENV RUST_LOG=info
ENV MODEL_PATH=/app/canary-180m-flash-int8

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the API server
CMD ["/app/api_server"]
