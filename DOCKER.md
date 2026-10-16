# Docker Deployment Guide

This guide explains how to deploy the PHOSAI Nemotron 3.5 Streaming ASR service (Luganda & Multilingual) using Docker.

## Prerequisites

### For GPU Support (Recommended)
- NVIDIA GPU (CUDA compatible)
- NVIDIA drivers installed
- Docker & Docker Compose installed
- NVIDIA Container Toolkit (`nvidia-docker2`) installed
- **Hugging Face Token** (`HF_TOKEN`): required because `phosai/nemotron-3.5-asr-streaming-luganda` is a private repository.

### For CPU-Only
- Docker installed
- **Hugging Face Token** (`HF_TOKEN`)

---

## Quick Start (Docker Compose)

1. **Configure your Hugging Face Token:**
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Edit `.env` and set your token:
```env
HF_TOKEN=hf_your_actual_token_here
```

2. **Build and start service:**
```bash
docker compose build --build-arg HF_TOKEN=${HF_TOKEN}
docker compose up -d
```
Or simply:
```bash
HF_TOKEN=hf_your_actual_token_here docker compose up -d --build
```

3. **Check logs:**
```bash
docker compose logs -f
```

4. **Verify health check:**
```bash
curl http://localhost:8099/health
```

Expected output:
```json
{
  "status": "healthy",
  "model": "phosai-streaming-voice-v1",
  "supported_languages": [
    "en (English US)",
    "en-gb (English Commonwealth)",
    "es (Spanish)",
    "fr (French)",
    "de (German)",
    "it (Italian)",
    "pt (Portuguese)",
    "ru (Russian)",
    "zh (Chinese)",
    "ja (Japanese)",
    "ko (Korean)",
    "hi (Hindi)",
    "ar (Arabic)",
    "tr (Turkish)",
    "sw (Swahili)",
    "lug (Luganda)",
    "auto (Auto-detect)"
  ]
}
```

---

## Manual Docker Commands

### 1. Build Docker Images with HF_TOKEN

**GPU version (CUDA):**
```bash
docker build \
  --build-arg HF_TOKEN=hf_your_token_here \
  -t nemotron-stt:latest \
  -f Dockerfile .
```

**CPU version:**
```bash
docker build \
  --build-arg HF_TOKEN=hf_your_token_here \
  -t nemotron-stt:cpu \
  -f Dockerfile.cpu .
```

### 2. Run Container (No Host Volumes Required)

Because the model weights and Rust binary are both baked into the container during the multi-stage build, no host volume flags are required:

**GPU Version (CUDA):**
```bash
docker run -d \
  --name nemotron-stt \
  --gpus all \
  -p 8099:8099 \
  nemotron-stt:latest
```

**CPU Version:**
```bash
docker run -d \
  --name nemotron-stt \
  -p 8099:8099 \
  nemotron-stt:cpu
```

*(Optional: If you ever want to override the baked-in model with external weights, you can pass `-v $(pwd)/custom-model:/app/nemotron-3.5-asr-streaming-luganda-onnx:ro`)*

---

## Testing Endpoints

### 1. Health Check
```bash
curl http://localhost:8099/health
```

### 2. Luganda Transcription (Single File)
```bash
curl -X POST \
  "http://localhost:8099/v1/transcribe?language=lug" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio/luganda_sample.wav
```
*(Also accepts `language=lg` or `language=luganda`)*

### 3. Batch Transcription (Multiple Files)
```bash
curl -X POST \
  "http://localhost:8099/v1/transcribe/batch?language=lug" \
  -F "file1=@audio/sample1.wav" \
  -F "file2=@audio/sample2.wav"
```

### 4. WebSocket Streaming & Documentation
- **Live WebSocket (Luganda)**: `ws://localhost:8099/v1/realtime?language=lug`
- **Interactive Documentation**: `http://localhost:8099/docs`
- **OpenAPI 3.0 Spec**: `http://localhost:8099/docs/openapi.json`
- **Web Demo Interface**: `http://localhost:8099/`
