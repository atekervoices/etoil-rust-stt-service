# Canary STT Service

A high-performance Speech-to-Text service built with Rust, featuring industry-standard endpoints, GPU acceleration, and real-time streaming capabilities.

## Features

- 🚀 **High Performance** - GPU acceleration with automatic detection (CUDA/RTX support)
- 🎯 **Multiple Endpoints** - Single file, multipart batch, and WebSocket streaming
- 🔄 **Concurrent Processing** - Parallel batch transcription for high throughput
- 🌐 **Industry Standard** - Multipart batch upload like AssemblyAI/Deepgram
- 📡 **Real-time Streaming** - WebSocket endpoint for live transcription
- 🐳 **Docker Ready** - Containerized deployment with GPU support
- 🧪 **Comprehensive Testing** - Benchmark scripts and test utilities

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [GPU Support](#gpu-support)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- **Rust** 1.80 or later
- **NVIDIA GPU** (optional, for GPU acceleration)
- **CUDA Toolkit** (optional, for GPU support)
- **Docker** (optional, for containerized deployment)

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd etoil-rust-stt-service

# Build the project
cargo build --release

# Run the API server
cargo run --bin api_server
```

### Model Download

The service uses the `canary-180m-flash-int8` model. The model directory is excluded from git due to its large size.

**Download the model:**
```bash
# Clone the model repository or download from Hugging Face
# Example (adjust based on actual model source):
git clone https://huggingface.co/nvidia/canary-180m-flash-int8 canary-180m-flash-int8

# Or download manually and extract
# Ensure the directory structure is:
# ./canary-180m-flash-int8/
#   ├── model.onnx
#   ├── config.json
#   └── other model files...
```

**Verify model installation:**
```bash
# The model should be in the canary-180m-flash-int8 directory
ls canary-180m-flash-int8
```

**Note:** The model directory is excluded from git via `.gitignore` to keep the repository size manageable.

## Quick Start

### Start the Server

```bash
cargo run --bin api_server
```

The server will start on `http://localhost:8080`

### Test the Service

```bash
# Health check
curl http://localhost:8080/health

# Single file transcription
curl -X POST \
  http://localhost:8080/v1/transcribe/canary?sample_rate=16000 \
  -H "Content-Type: application/octet-stream" \
  --data-binary @test-audio.wav
```

## API Endpoints

### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_enabled": true
}
```

### Single File Transcription

```http
POST /v1/transcribe/canary?sample_rate=16000&source_lang=en&target_lang=en
Content-Type: application/octet-stream
```

**Parameters:**
- `sample_rate` - Audio sample rate (default: 16000)
- `source_lang` - Source language code (default: "en")
- `target_lang` - Target language code (default: "en")

**Request Body:** Raw PCM audio data (16-bit, little-endian)

**Response:**
```json
{
  "text": "Transcription result here",
  "processing_time": 1.234
}
```

### Batch Transcription (Industry Standard)

```http
POST /v1/transcribe/batch?sample_rate=16000&source_lang=en&target_lang=en
Content-Type: multipart/form-data
```

**Parameters:**
- `sample_rate` - Audio sample rate (default: 16000)
- `source_lang` - Source language code (default: "en")
- `target_lang` - Target language code (default: "en")

**Request Body:** Multipart form data with `files` field containing multiple audio files

**Response:**
```json
{
  "results": [
    {
      "name": "audio1.wav",
      "text": "Transcription result",
      "processing_time": 1.234,
      "audio_duration": 5.5,
      "success": true,
      "error": null
    }
  ],
  "total_processing_time": 1.234,
  "total_files": 1,
  "successful_files": 1,
  "failed_files": 0
}
```

### WebSocket Streaming

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

// Send configuration
ws.send(JSON.stringify({
  type: 'config',
  sample_rate: 16000,
  source_lang: 'en',
  target_lang: 'en'
}));

// Send audio data (base64 encoded)
ws.send(JSON.stringify({
  type: 'audio',
  data: 'base64_encoded_audio',
  sample_rate: 16000
}));

// Receive partial results
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === 'partial') {
    console.log('Partial:', message.text);
  } else if (message.type === 'final') {
    console.log('Final:', message.text);
  }
};
```

## GPU Support

The service automatically detects and uses GPU acceleration when available.

### Supported GPUs

- **NVIDIA GPUs** (tested with RTX 2060)
- **CUDA 12.1** or later
- **TensorRT** (optional, for optimized inference)

### GPU Detection

The service will automatically:
1. Try to initialize CUDA GPU
2. Fall back to CPU if GPU is unavailable
3. Log the detected execution provider

**Console Output:**
- `✅ Using CUDA GPU acceleration` - GPU detected and in use
- `⚠️ CUDA not available, falling back to CPU` - CPU fallback

### GPU Requirements

- **NVIDIA Drivers** - Latest drivers for your GPU
- **CUDA Toolkit** - Version 12.1 or later
- **Docker GPU Support** - nvidia-docker2 for containerized deployment

### Performance Comparison

| Configuration | Latency | Throughput | Speedup |
|--------------|---------|------------|---------|
| CPU (i7)     | ~3-4s   | 0.3 req/s  | 1x      |
| GPU (RTX 2060) | ~100-500ms | 10-50 req/s | 10-50x  |

## Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### Manual Docker Build

```bash
# Build GPU version
docker build -t canary-stt:latest .

# Build CPU-only version
docker build -f Dockerfile.cpu -t canary-stt:cpu .

# Run with GPU support
docker run -d \
  --name canary-stt \
  --gpus all \
  -p 8080:8080 \
  -v $(pwd)/canary-180m-flash-int8:/app/canary-180m-flash-int8:ro \
  canary-stt:latest
```

For detailed Docker deployment instructions, see [DOCKER.md](DOCKER.md).

## Testing

### Run Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name
```

### Test HTML Interface

Open `test.html` in your browser to test all endpoints through a web interface.

### WebSocket Test

```bash
python test_websocket.py
```

## Benchmarking

### Run Benchmark Script

```bash
# Install dependencies
pip install requests websocket-client

# Benchmark single file endpoint
python benchmark.py canary 8 3 test-audio.wav

# Benchmark batch endpoint
python benchmark.py batch 8 3 test-audio.wav

# Benchmark WebSocket endpoint
python benchmark.py websocket 2 1 test-audio.wav
```

**Parameters:**
- `endpoint` - canary, batch, or websocket
- `num_requests` - Number of concurrent requests
- `num_rounds` - Number of test rounds
- `audio_file` - Path to test audio file

### Benchmark Output

The benchmark provides:
- **Sequential vs Concurrent** performance comparison
- **Latency metrics** - min, avg, max, p50, p95
- **Throughput** - requests per second
- **Speedup** - Concurrent vs sequential improvement
- **Accuracy verification** - Ensures consistent results

## Configuration

### Environment Variables

- `RUST_LOG` - Log level (default: `info`)
- `MODEL_PATH` - Path to model directory (default: `./canary-180m-flash-int8`)
- `SERVER_PORT` - Server port (default: `8080`)

### Server Configuration

Edit `src/api_server.rs` to customize:
- Port number
- CORS settings
- Request size limits
- Timeout settings

## API Documentation

### OpenAPI Specification

Open `swagger.html` in your browser to view the interactive API documentation.

### OpenAPI JSON

The OpenAPI specification is available at:
```
http://localhost:8080/openapi.json
```

## Troubleshooting

### GPU Not Detected

**Problem:** Service falls back to CPU despite having NVIDIA GPU

**Solutions:**
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA installation
3. Ensure canary-rs has GPU support enabled
4. Check Docker GPU support (if using Docker)

### Model Loading Errors

**Problem:** "Failed to load model" errors

**Solutions:**
1. Verify model directory exists: `ls canary-180m-flash-int8`
2. Check file permissions
3. Ensure sufficient disk space
4. Verify model integrity

### Build Errors

**Problem:** Rust compilation errors

**Solutions:**
1. Update Rust: `rustup update`
2. Clean build: `cargo clean`
3. Check dependencies: `cargo check`
4. Verify Rust version compatibility

### Docker Build Errors

**Problem:** Docker build fails

**Solutions:**
1. Check Docker version: `docker --version`
2. Clean Docker cache: `docker system prune -a`
3. Verify network connectivity
4. Check Docker disk space

### WebSocket Connection Issues

**Problem:** WebSocket connection fails

**Solutions:**
1. Check server is running
2. Verify WebSocket endpoint URL
3. Check firewall settings
4. Test with WebSocket client tools

## Performance Optimization

### GPU Optimization

1. **Use TensorRT** - Further optimize inference speed
2. **Batch Processing** - Process multiple files concurrently
3. **Memory Management** - Adjust GPU memory allocation

### CPU Optimization

1. **Increase Threads** - Adjust thread pool size
2. **Use Smaller Models** - Trade accuracy for speed
3. **Optimize Audio** - Pre-process audio files

### Network Optimization

1. **Enable Compression** - Use gzip for responses
2. **Keep-Alive Connections** - Reduce connection overhead
3. **Load Balancing** - Distribute requests across instances

## Production Deployment

### Security Considerations

1. **Authentication** - Add API key authentication
2. **HTTPS/TLS** - Encrypt all communications
3. **Rate Limiting** - Prevent abuse
4. **Input Validation** - Sanitize all inputs
5. **Network Isolation** - Use private networks

### Monitoring

1. **Health Checks** - Implement comprehensive health monitoring
2. **Logging** - Centralized log aggregation
3. **Metrics** - Track performance metrics
4. **Alerting** - Set up alerting for failures

### Scaling

1. **Horizontal Scaling** - Deploy multiple instances
2. **Load Balancing** - Use a load balancer
3. **Auto-scaling** - Scale based on demand
4. **GPU Clustering** - Distribute GPU workloads

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Specify your license here]

## Support

For issues and questions:
- **Documentation** - Check this README and DOCKER.md
- **Issues** - Open an issue on GitHub
- **Discussions** - Use GitHub Discussions for questions

## Acknowledgments

- **canary-rs** - Speech-to-Text library
- **Warp** - Web framework for Rust
- **ONNX Runtime** - Machine learning inference engine
- **NVIDIA** - GPU acceleration technology

## Roadmap

- [ ] Add more language support
- [ ] Implement speaker diarization
- [ ] Add word-level timestamps
- [ ] Improve WebSocket streaming
- [ ] Add batch job scheduling
- [ ] Implement caching layer
- [ ] Add authentication/authorization
- [ ] Improve error handling
- [ ] Add monitoring dashboard
- [ ] Support more audio formats
