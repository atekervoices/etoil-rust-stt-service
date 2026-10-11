# NVIDIA Nemotron 3.5 Cache-Aware Streaming ASR Service

A high-performance, ultra-low latency Automatic Speech Recognition (ASR) service built in Rust using **[parakeet-rs](https://github.com/altunenes/parakeet-rs)** and **[nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)** (FastConformer RNN-Transducer, 600M parameters).

Optimized for real-time conversational AI, voice agents, live captioning, and telephony. Powered by **CUDA GPU acceleration**, **160 ms streaming chunks** (`right-context = 1`), and a **zero-GC Rust runtime**.

---

## Key Highlights

- **Ultra-Low Latency Streaming**:
  - **160 ms Chunk Pacing**: 2,560 audio samples per chunk @ 16kHz with `right-context = 1` for minimal lookahead delay.
  - **Time-to-First-Token (TTFB / TTFT)**: **~38 ms** pure GPU inference step (**~132 ms** client-to-client over WAN).
  - **End-of-Turn (EOT) Latency**: **~370 ms** from speech end to final turn completion.
  - **Real-Time Factor (RTF)**: **`0.2545`** (~**3.9× faster than real-time** on an entry-level RTX 2060 GPU).
- **Concurrency & Resource Isolation**:
  - Built with `NemotronHandle` + `from_shared`: The heavy 2.4GB neural network is loaded **once** in GPU VRAM.
  - Each WebSocket connection gets an independent, lightweight stream with isolated acoustic buffers, decoder states, and language selection. **Zero cross-talk and zero VRAM duplication.**
- **Dual WebSocket Streaming Modes**:
  - **Raw Binary Mode**: Direct raw 16-bit PCM LE byte streaming for maximum throughput and zero base64 CPU overhead.
  - **JSON Base64 Mode**: Standard JSON audio messages compatible with browser AudioWorklet and speech SDKs.
- **Multilingual Support**:
  - 40 languages supported natively via one-hot prompt embedding (English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Hindi, Arabic, Turkish, Swahili, Luganda, and Auto-Detect).
- **Production-Ready REST & WebSocket Interfaces**:
  - `GET /health`: Health and model readiness check.
  - `POST /v1/transcribe`: Synchronous single-file and multi-file batch upload (`audio/wav`, `application/octet-stream`, `multipart/form-data`).
  - `GET /v1/realtime`: Low-latency streaming WebSocket.
  - `GET /docs` & `GET /docs/openapi.json`: Embedded Swagger UI & OpenAPI 3.0 specification.
  - `GET /`: Interactive web demo with real-time waveform visualizer and microphone streaming.

---

## Benchmarks & Latency Milestones

Tested on staging host (`stt-staging.atekervoices.com`, NVIDIA GeForce RTX 2060 8GB VRAM) across independent trials using standardized streaming latency specifications:

### 1. Real-Time Latency Milestones

| Milestone / Metric | Industry Target (Voice Agents & Captioning) | Our Nemotron Service (RTX 2060) | Evaluation |
| :--- | :--- | :--- | :--- |
| **Network Transit Time (TCP RTT)** | 20 – 200 ms | **85 – 108 ms** | *Normal internet WAN* |
| **Pure GPU Compute Step (per 160ms)** | 150 – 300 ms | **24.2 ms** | **~6× faster than cloud baselines** |
| **Total Transcript Latency (p50 Median)** | 200 – 500 ms | **132.6 ms** | **Ultra-low latency (<150ms)** |
| **Total Transcript Latency (p95 Tail)** | < 500 ms | **153.0 ms** | **Ultra-tight tail (±11ms jitter)** |
| **End-of-Turn (EOT) Latency** | 100 – 500 ms | **372.7 ± 13.4 ms** | **Passes voice-agent budget** |
| **Batch Real-Time Factor (RTF)** | < 1.0 | **0.2545 (3.93×)** | **High throughput** |
| **Accuracy Match** | Ground truth | **100% Identical** | *Zero drift across runs* |

---

## Architecture Overview

```
                      Client Microphone / Audio Stream (16kHz 16-bit Mono)
                                            │
                                            ▼
                    WebSocket Endpoint (/v1/realtime) or HTTP (/v1/transcribe)
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               │                                                         │
         Client 1 Stream                                           Client 2 Stream
   (Nemotron::from_shared)                                   (Nemotron::from_shared)
   • 160ms chunk buffer (2560 smp)                           • 160ms chunk buffer (2560 smp)
   • Isolated encoder cache                                  • Isolated encoder cache
   • Isolated decoder LSTM states                            • Isolated decoder LSTM states
   • Target language: "en"                                   • Target language: "fr"
               │                                                         │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                           NemotronHandle (Arc<Mutex<NemotronModel>>)
                                2.4 GB Single GPU Allocation
                         ONNX Runtime CUDA Provider (libcublasLt 13)
```

---

## Supported Languages

| Language | Query Param | Prompt Token | Status |
| :--- | :--- | :--- | :--- |
| **English (US)** | `en`, `en-US` | `en-US` | ✅ Active |
| **English (Commonwealth)** | `en-gb` | `en-GB` | ✅ Active |
| **Spanish** | `es`, `es-ES` | `es-ES` | ✅ Active |
| **French** | `fr`, `fr-FR` | `fr-FR` | ✅ Active |
| **German** | `de`, `de-DE` | `de-DE` | ✅ Active |
| **Italian** | `it`, `it-IT` | `it-IT` | ✅ Active |
| **Portuguese** | `pt`, `pt-PT` | `pt-PT` | ✅ Active |
| **Russian** | `ru`, `ru-RU` | `ru-RU` | ✅ Active |
| **Chinese** | `zh`, `zh-CN` | `zh-CN` | ✅ Active |
| **Japanese** | `ja`, `ja-JP` | `ja-JP` | ✅ Active |
| **Korean** | `ko`, `ko-KR` | `ko-KR` | ✅ Active |
| **Hindi** | `hi`, `hi-IN` | `hi-IN` | ✅ Active |
| **Arabic** | `ar`, `ar-AR` | `ar-AR` | ✅ Active |
| **Turkish** | `tr`, `tr-TR` | `tr-TR` | ✅ Active |
| **Swahili** | `sw`, `sw-KE` | `sw-KE` | ✅ Active |
| **Luganda** | `lg`, `lug` | `auto` | ✅ Active (Auto Prompting) |
| **Auto-Detect** | `auto` | `auto` | ✅ Active |

---

## API Reference

### 1. Synchronous HTTP Transcription (`/v1/transcribe`)

#### A. Single Audio File (Raw Binary or Multipart)
```bash
curl -X POST "http://localhost:8099/v1/transcribe?language=en" \
  -H "Content-Type: audio/wav" \
  --data-binary @test_audio/eng_1.wav
```

**Response**:
```json
{
  "text": "How do you access market information for sweet potato",
  "processing_time": 1.065,
  "audio_duration": 4.14,
  "real_time_factor": 0.2574,
  "language": "en"
}
```

#### B. Multi-File Batch Transcription
Upload multiple files simultaneously via `multipart/form-data`:
```bash
curl -X POST "http://localhost:8099/v1/transcribe?language=en" \
  -F "file1=@audio_1.wav" \
  -F "file2=@audio_2.wav"
```

**Response**:
```json
{
  "results": [
    {
      "name": "audio_1.wav",
      "text": "How do you access market information",
      "processing_time": 0.82,
      "audio_duration": 3.20,
      "real_time_factor": 0.256,
      "success": true,
      "error": null
    },
    {
      "name": "audio_2.wav",
      "text": "Sweet potato market pricing analysis",
      "processing_time": 0.65,
      "audio_duration": 2.50,
      "real_time_factor": 0.260,
      "success": true,
      "error": null
    }
  ],
  "total_processing_time": 1.47,
  "total_files": 2,
  "successful_files": 2,
  "failed_files": 0
}
```

---

### 2. Real-Time Streaming WebSocket (`/v1/realtime`)

Connect via WebSocket to `ws://localhost:8099/v1/realtime?language=en`.

#### Connection Handshake
On successful connection, the server sends a `ready` message:
```json
{
  "type": "ready",
  "model": "phosai-streaming-voice-v1",
  "supported_languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "hi", "ar", "tr", "sw", "auto"]
}
```

#### Streaming Audio
* **Binary Framing (Recommended)**: Send raw 16-bit PCM little-endian byte chunks (5,120 bytes per 160ms chunk) as WebSocket binary frames.
* **JSON Base64 Framing**:
  ```json
  {
    "type": "audio",
    "data": "<BASE64_PCM16_BYTES>",
    "sample_rate": 16000
  }
  ```

#### Interim and Final Transcript Events
```json
// Interim token partial:
{
  "type": "partial",
  "text": "market",
  "accumulated_text": "How do you access market",
  "confidence": 0.95,
  "timestamp": 3.15
}

// End-of-turn final event:
{
  "type": "final",
  "text": "How do you access market information for sweet potato",
  "confidence": 0.95,
  "timestamp": 4.14,
  "processing_time": 4.14
}
```

#### Control Messages
- **Flush / End-of-Turn**: `{"type": "flush"}` — Forces right-context lookahead flush and returns the finalized sentence.
- **Soft Reset**: `{"type": "reset", "finalize": false}` — Emits current text without wiping model cache.
- **Hard Reset**: `{"type": "reset", "finalize": true}` — Clears all audio buffers and resets encoder/decoder states.

---

## Docker Deployment (GPU / CUDA)

The service is fully containerized using a self-contained multi-stage build.

### Run with Docker Compose
```bash
# Start GPU service
docker compose up -d

# Check startup logs
docker compose logs -f nemotron-stt
```

### Manual Build & Run
```bash
# Build the image
docker compose build nemotron-stt

# Run container with NVIDIA GPU acceleration
docker compose up -d --force-recreate nemotron-stt
```

### Hardware Requirements
- **Host GPU**: NVIDIA Turing, Ampere, Ada Lovelace, or Hopper architecture (RTX 2060, RTX 3060+, RTX 4070+, L4, A10, A100).
- **VRAM**: Minimum 6 GB VRAM (8 GB+ recommended).
- **Driver**: NVIDIA Driver 535+ with `nvidia-container-toolkit`.

---

## Latency Measurement Suite

Run the included standardized streaming latency benchmark script:

```bash
# Run 3-iteration statistical benchmark:
python measure_latency.py
```

Outputs:
- DNS, TCP connect (RTT), and TLS handshake breakdowns.
- Synchronous HTTP GPU time and RTF.
- Time-to-First-Token (TTFB / TTFT).
- Interim Transcript Latency (p50, p90, p95, Mean, Max).
- End-of-Turn (EOT) Latency.

---

## License

This project is licensed under the Apache 2.0 License. Model weights are governed by the [NVIDIA Open Model License](https://developer.nvidia.com/downloads/license/nvidia-open-model-license-agreement).
