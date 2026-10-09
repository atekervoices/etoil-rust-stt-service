# Docker Deployment Guide

This guide explains how to deploy the Crane Nemotron Streaming ASR service (Luganda & English) with Earshot Neural VAD using Docker.

## Prerequisites

### For GPU Support (Recommended)
- NVIDIA GPU (CUDA compatible)
- NVIDIA drivers installed
- Docker installed
- NVIDIA Container Toolkit (`nvidia-docker2`) installed

### For CPU-Only
- Docker installed

---

## Quick Start

### With GPU Support (CUDA Acceleration)

1. **Build and start service:**
```bash
docker compose up -d
```

2. **Check logs:**
```bash
docker compose logs -f
```

3. **Verify health check:**
```bash
curl http://localhost:8080/health
```

Expected output:
```json
{
  "status": "healthy",
  "model": "CraneAILabs/crane-nemo-asr-1.1 (Nemotron 3.5 Streaming)",
  "supported_languages": ["lg (Luganda)", "en (English)", "auto (Auto-detect)"],
  "excluded_languages": ["sn (Shona - disabled)"],
  "chunk_size_samples": 8960,
  "chunk_duration_ms": 560
}
```

---

## Manual Docker Commands

### 1. Build Docker Images

**GPU version (CUDA):**
```bash
docker build -t crane-nemotron-stt:latest -f Dockerfile .
```

**CPU version:**
```bash
docker build -t crane-nemotron-stt:cpu -f Dockerfile.cpu .
```

### 2. Run Container (No Host Volumes Required)

Because the model weights and Rust binary are both baked into the container during the multi-stage build, no host volume flags are required:

**GPU Version (CUDA):**
```bash
docker run -d \
  --name crane-nemotron-stt \
  --gpus all \
  -p 8080:8080 \
  crane-nemotron-stt:latest
```

**CPU Version:**
```bash
docker run -d \
  --name crane-nemotron-stt \
  -p 8080:8080 \
  crane-nemotron-stt:cpu
```

*(Optional: If you ever want to override the baked-in model with external weights, you can pass `-v $(pwd)/custom-model:/app/crane-nemo-asr-1.1-onnx:ro`)*

---

## Testing Containerized Endpoints

### 1. Health Check
```bash
curl http://localhost:8080/health
```

### 2. Luganda Transcription (Single File)
```bash
curl -X POST \
  "http://localhost:8080/v1/transcribe?language=lg" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio/luganda_sample.wav
```

### 3. Batch Transcription (Multiple Files)
```bash
curl -X POST \
  "http://localhost:8080/v1/transcribe/batch?language=en" \
  -F "file1=@audio/sample1.wav" \
  -F "file2=@audio/sample2.wav"
```

### 4. Verify Shona is Excluded
```bash
curl -X POST \
  "http://localhost:8080/v1/transcribe?language=sn" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio/sample.wav
```
Response:
```json
{"error": "Language 'shona' (sn) is explicitly disabled in this service configuration."}
```

### 5. WebSocket Streaming & Documentation
- **Live WebSocket**: `ws://localhost:8080/v1/realtime?language=lg`
- **Interactive Documentation**: `http://localhost:8080/docs`
- **OpenAPI 3.0 Spec**: `http://localhost:8080/docs/openapi.json`
- **Web Demo Interface**: `http://localhost:8080/`
