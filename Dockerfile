# Multi-stage Dockerfile for Canary STT Service
# Stage 1: Build
FROM rust:nightly as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Cargo files (Cargo.lock is optional)
COPY Cargo.toml ./
COPY Cargo.lock* ./

# Create a dummy main.rs to cache dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy actual source code
COPY src ./src

# Build the application
RUN cargo build --release

# Stage 2: Runtime (with GPU support)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
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
