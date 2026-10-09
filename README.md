# Crane Nemotron 3.5 Cache-Aware Streaming ASR Service

A high-performance, real-time Speech-to-Text (ASR) service built in Rust using **parakeet-rs** and **NVIDIA Nemotron-3.5 0.6B FastConformer RNN-Transducer**, fine-tuned for **Luganda** and **English** by **Crane AI Labs** ([CraneAILabs/crane-nemo-asr-1.1](https://huggingface.co/CraneAILabs/crane-nemo-asr-1.1)).

---

## Key Features

- **Cache-Aware Stateful Streaming** — Real-time speech-to-text with rolling encoder cache and decoder states across 560ms chunks (8,960 samples @ 16kHz).
- **Language Specialization**:
  - **Luganda (`lg`)**: High-accuracy Luganda ASR (WER 20.9%, CER 5.1%, Gemini 4.95/5 intelligibility), using automatic prompt conditioning.
  - **English (`en`)**: Clean English transcription with punctuation and capitalization preserved.
  - **Auto-Detect (`auto`)**: Automatic language identification.
-    **Multi-Interface Support**:
  - **WebSocket (`/v1/realtime`)**: Ultra-low latency bidirectional real-time audio streaming
  - **REST API (`/v1/transcribe`)**: Single audio file transcription.
  - **Batch Multipart (`/v1/transcribe/batch`)**: High-throughput multi-file batch processing.
  - **Interactive Documentation (`/docs` & `/docs/openapi.json`)**: Built-in Swagger UI and OpenAPI 3.0 specification.
  - **Interactive Web Demo (`/`)**: Dark-mode browser tester with live microphone streaming visualizer.
  - **CLI Tools**: `live` (direct microphone transcription) and `file_transcribe` (audio file batch tool).
- **Hardware Acceleration** — Runs on CPU (fast real-time factor) or NVIDIA GPU via CUDA with ONNX Runtime.
- **Docker & Compose Ready** — Multi-stage builds for both GPU (CUDA 12.6) and lightweight CPU execution.

---

## Architecture & Streaming Principles

Nemotron-3.5 0.6B Streaming uses a **FastConformer RNN-Transducer** with cache-aware streaming:

```
Audio Input (16kHz Mono)
       │
       ▼
Chunking Buffer (560ms / 8960 samples)
       │
       ├──► VAD (Energy/Silence Detection & Utterance Turn Tracking)
       │
       ▼
Nemotron Cache-Aware Step
  ├── Encoder Cache (att_context_size: [56, 6] or [56, 13])
  ├── Language Conditioning (Prompt Index: 'auto' for Luganda, 'en-US' for English)
  └── Decoder Joint Network (Predictor & Joint Emission)
       │
       ▼
Token Stream -> Normalized Transcript Deltas
```

- **Audio Rate**: 16,000 Hz (16-bit PCM mono or 32-bit float).
- **Chunk Size**: `8960` samples (560 ms).
- **State Reset**: Calling `model.reset()` flushes encoder and decoder caches cleanly between distinct utterances.

---

## Language Support Matrix

| Language | Code | Prompt Mapping | Status |
| :--- | :--- | :--- | :--- |
| **Luganda** | `lg`, `lug`, `luganda` | `auto` (Index 101) | ✅ Fully Supported (Crane AI fine-tuned) |
| **English** | `en`, `eng`, `en-US` | `en-US` (Index 0) | ✅ Fully Supported |
| **Auto-Detect** | `auto` | `auto` (Index 101) | ✅ Fully Supported |
| **Shona** | `sn`, `shona` | N/A | ❌ **Excluded / Disabled** |

> **Note on Luganda Prompting**: As detailed in Crane AI Labs' model card:
> *"Best results use the language-ID prompt set to auto-detect; Luganda and Shona do not have a dedicated prompt slot, so auto-detect is recommended for them."*
> This service automatically routes Luganda requests to the optimal `auto` prompt slot. Requests requesting Shona (`sn`) will be rejected with an informative error message.

---

## Model Setup & ONNX Export

The service requires the ONNX-exported weights of `CraneAILabs/crane-nemo-asr-1.1`.

### 1. Download the Pre-trained `.nemo` Model

```bash
# Using Python helper script:
python scripts/download_model.py --output-dir ./models
```

Or download directly via Hugging Face Hub:
```bash
huggingface-cli download CraneAILabs/crane-nemo-asr-1.1 crane-nemo-asr-1.1.nemo --local-dir ./models
```

### 2. Export to ONNX for `parakeet-rs`

Export the `.nemo` weights to the ONNX directory structure:

```bash
# Install export dependencies
pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
pip install "numpy<2.0" onnx onnxruntime soundfile sentencepiece torch

# Run the export script
python scripts/export_crane_nemotron.py ./models/crane-nemo-asr-1.1.nemo ./crane-nemo-asr-1.1-onnx
```

Expected files in `./crane-nemo-asr-1.1-onnx`:
- `encoder.onnx`
- `encoder.onnx.data` (external tensor weights)
- `decoder_joint.onnx`
- `tokenizer.model` (SentencePiece vocab)
- `config.json`

---

## Running the Service

### 1. Start the API Server

```bash
# Using default model directory (crane-nemo-asr-1.1-onnx) on port 8080:
cargo run --release --bin api_server

# Or with custom model path / port:
MODEL_PATH=./crane-nemo-asr-1.1-onnx PORT=8080 cargo run --release --bin api_server
```

Once started:
- **Interactive Web Demo**: [http://localhost:8080/](http://localhost:8080/)
- **Interactive Documentation**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **OpenAPI 3.0 JSON Spec**: [http://localhost:8080/docs/openapi.json](http://localhost:8080/docs/openapi.json)
- **Health Check**: [http://localhost:8080/health](http://localhost:8080/health)
- **Live Streaming WebSocket**: `ws://localhost:8080/v1/realtime`

---

### 2. Live Microphone Streaming (CLI)

Test live microphone streaming in real-time right from your terminal:

```bash
# Live microphone transcription in Luganda:
cargo run --release --bin live lg ./crane-nemo-asr-1.1-onnx

# Live microphone transcription in English:
cargo run --release --bin live en ./crane-nemo-asr-1.1-onnx

# Auto-detect language:
cargo run --release --bin live auto ./crane-nemo-asr-1.1-onnx
```

---

### 3. Audio File Transcription (CLI)

Transcribe a WAV file chunk-by-chunk using cache-aware streaming:

```bash
# Transcribe WAV file in Luganda:
cargo run --release --bin file_transcribe -- test_luganda.wav lg ./crane-nemo-asr-1.1-onnx

# Transcribe WAV file in English:
cargo run --release --bin file_transcribe -- sample.wav en ./crane-nemo-asr-1.1-onnx
```

---

## API Reference (Standardized v1 Endpoints)

### 1. Synchronous Audio Transcription
```http
POST /v1/transcribe?language=lg
Content-Type: application/octet-stream | audio/wav | multipart/form-data
```

#### A. Single File Upload (Raw Audio or 1-file Multipart)
```bash
# Via raw binary:
curl -X POST "http://localhost:8080/v1/transcribe?language=lg" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @luganda_audio.wav

# Or via AssemblyAI-style multipart upload:
curl -X POST "http://localhost:8080/v1/transcribe?language=lg" \
  -F "audio=@luganda_audio.wav"
```
**Response**:
```json
{
  "text": "Twasanyuse nnyo okukulaba.",
  "processing_time": 0.145,
  "audio_duration": 2.80,
  "real_time_factor": 0.051,
  "language": "lg"
}
```

#### B. Multi-File Batch Transcription
Uploading 2 or more files to `POST /v1/transcribe` (or to explicit `POST /v1/transcribe/batch`):
```bash
curl -X POST "http://localhost:8080/v1/transcribe?language=en" \
  -F "file1=@clip1.wav" \
  -F "file2=@clip2.wav"
```
**Response**:
```json
{
  "results": [
    {
      "name": "clip1.wav",
      "text": "Welcome to the real-time speech service.",
      "processing_time": 0.12,
      "audio_duration": 2.50,
      "real_time_factor": 0.048,
      "success": true,
      "error": null
    },
    {
      "name": "clip2.wav",
      "text": "Processing speech with high accuracy.",
      "processing_time": 0.14,
      "audio_duration": 3.00,
      "real_time_factor": 0.046,
      "success": true,
      "error": null
    }
  ],
  "total_processing_time": 0.26,
  "total_files": 2,
  "successful_files": 2,
  "failed_files": 0
}
```

---

### 2. Real-Time Streaming WebSocket (`/v1/realtime`)

Connect to `ws://localhost:8080/v1/realtime`.

#### Connection Handshake with Query Parameters
Connect directly with target language:
```text
ws://localhost:8080/v1/realtime?language=lg&sample_rate=16000
```

#### In-Band Configuration (Optional)
```json
{
  "type": "config",
  "language": "lg",
  "sample_rate": 16000
}
```

#### Stream Audio (Client -> Server)
Send base64-encoded 16-bit PCM chunks (16kHz mono):
```json
{
  "type": "audio",
  "data": "<BASE64_PCM16_BYTES>",
  "sample_rate": 16000
}
```

#### Streaming Responses (Server -> Client)
```json
// Partial token emission (each 560ms chunk):
{
  "type": "partial",
  "text": "Twasanyuse",
  "accumulated_text": "Twasanyuse",
  "confidence": 0.92,
  "timestamp": 0.56
}

// Final completed utterance (on Earshot VAD silence hangover ~288ms):
{
  "type": "final",
  "text": "Twasanyuse nnyo okukulaba.",
  "confidence": 0.95,
  "timestamp": 2.80,
  "processing_time": 2.80
}
```

---

### 3. Interactive Documentation & OpenAPI

- **Interactive Swagger UI**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **OpenAPI 3.0 JSON Specification**: [http://localhost:8080/docs/openapi.json](http://localhost:8080/docs/openapi.json)
- **Interactive Web Demo UI**: [http://localhost:8080/](http://localhost:8080/)
- **Health Check**: [http://localhost:8080/health](http://localhost:8080/health)

#### WebSocket Control Messages
- **Flush**: `{ "type": "flush" }` — Pads remaining audio buffer and outputs final transcript.
- **Reset**: `{ "type": "reset" }` — Flushes Nemotron streaming cache between turns.

---

## Docker Deployment

### Run with Docker Compose

```bash
# Start GPU service
docker compose up -d

# Check service logs
docker compose logs -f
```

### Build Docker Image Manually

```bash
# Build CUDA GPU container:
docker build -t crane-nemotron-stt -f Dockerfile .

# Build CPU-only container:
docker build -t crane-nemotron-stt-cpu -f Dockerfile.cpu .

# Run with GPU (Self-contained, no host volume required):
docker run --gpus all -p 8080:8080 crane-nemotron-stt

# Or run with CPU:
docker run -p 8080:8080 crane-nemotron-stt-cpu
```

---

## Verification & Testing

Verify that language validation rejects Shona (`sn`):
```bash
curl -X POST "http://localhost:8080/v1/transcribe?language=sn" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @test.wav
```
Response:
```json
{
  "error": "Language 'shona' (sn) is explicitly disabled in this service configuration."
}
```

---

## License

Inherits the [NVIDIA Open Model License](https://developer.nvidia.com/downloads/license/nvidia-open-model-license-agreement) from the base Nemotron-3.5 model.
