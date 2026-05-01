# Docker Deployment Guide

This guide explains how to deploy the Canary STT service using Docker with GPU support.

## Prerequisites

### For GPU Support (Recommended)
- NVIDIA GPU (tested with RTX 2060)
- NVIDIA drivers installed
- Docker installed
- nvidia-docker2 installed

### For CPU-Only
- Docker installed

## Quick Start

### With GPU Support (Recommended)

1. **Build and run with docker-compose:**
```bash
docker-compose up -d
```

2. **Check logs:**
```bash
docker-compose logs -f
```

3. **Test the service:**
```bash
curl http://localhost:8080/health
```

### CPU-Only Version

If you don't have NVIDIA GPU or nvidia-docker2:

1. **Edit docker-compose.yml** to use the CPU version:
```yaml
services:
  canary-stt:
    build:
      context: .
      dockerfile: Dockerfile.cpu  # Change to CPU version
```

2. **Build and run:**
```bash
docker-compose up -d
```

## Manual Docker Commands

### Build the Image

**GPU version:**
```bash
docker build -t canary-stt:latest .
```

**CPU version:**
```bash
docker build -f Dockerfile.cpu -t canary-stt:cpu .
```

### Run the Container

**With GPU support:**
```bash
docker run -d \
  --name canary-stt \
  --gpus all \
  -p 8080:8080 \
  -v $(pwd)/canary-180m-flash-int8:/app/canary-180m-flash-int8:ro \
  -v $(pwd)/audio:/app/audio:ro \
  canary-stt:latest
```

**CPU only:**
```bash
docker run -d \
  --name canary-stt \
  -p 8080:8080 \
  -v $(pwd)/canary-180m-flash-int8:/app/canary-180m-flash-int8:ro \
  -v $(pwd)/audio:/app/audio:ro \
  canary-stt:cpu
```

## Configuration

### Environment Variables

- `RUST_LOG` - Log level (default: `info`)
- `MODEL_PATH` - Path to the model directory (default: `/app/canary-180m-flash-int8`)

### Volumes

- `/app/canary-180m-flash-int8` - Model directory (read-only)
- `/app/audio` - Audio files directory (read-only)

## Testing the Deployment

### Health Check
```bash
curl http://localhost:8080/health
```

### Single File Transcription
```bash
curl -X POST \
  http://localhost:8080/v1/transcribe/canary?sample_rate=16000 \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio/test.wav
```

### Batch Upload
```bash
curl -X POST \
  http://localhost:8080/v1/transcribe/batch?sample_rate=16000 \
  -F "files=@audio/test1.wav" \
  -F "files=@audio/test2.wav"
```

### WebSocket
```bash
wscat -c ws://localhost:8080/ws
```

## GPU Detection

The service automatically detects and uses GPU if available. You'll see in the logs:
- `✅ Using CUDA GPU acceleration` - GPU is being used
- `⚠️ CUDA not available, falling back to CPU` - CPU fallback

## Troubleshooting

### GPU Not Detected

If you see "CUDA not available" message:
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify nvidia-docker2 installation: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
3. Check Docker GPU support: `docker run --rm --gpus all ubuntu nvidia-smi`

### Build Errors

If you encounter build errors:
1. Ensure Docker has enough memory (at least 4GB)
2. Check Rust version compatibility
3. Try cleaning Docker cache: `docker system prune -a`

### Model Not Found

If the model directory is missing:
1. Ensure `canary-180m-flash-int8` directory exists
2. Check volume mounting in docker-compose.yml
3. Verify the model path is correct

## Performance

### Expected Performance with GPU (RTX 2060)
- **Single file:** ~100-500ms latency
- **Batch processing:** 10-50x faster than CPU
- **Concurrent requests:** High throughput with GPU acceleration

### CPU Performance
- **Single file:** ~3-4 seconds latency
- **Batch processing:** Limited by CPU cores
- **Concurrent requests:** Moderate throughput

## Production Deployment

For production deployment, consider:

1. **Resource Limits:**
```yaml
deploy:
  resources:
    limits:
      memory: 4G
      reservations:
        memory: 2G
```

2. **Logging:**
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

3. **Security:**
- Use HTTPS/TLS
- Add authentication
- Network isolation
- Regular security updates

## Monitoring

### Container Health
```bash
docker ps
docker inspect canary-stt
```

### Logs
```bash
docker logs -f canary-stt
```

### Resource Usage
```bash
docker stats canary-stt
```

## Support

For issues related to:
- **Docker:** Check Docker documentation
- **NVIDIA GPU:** Check NVIDIA Docker documentation
- **Canary STT:** Check the main project documentation
