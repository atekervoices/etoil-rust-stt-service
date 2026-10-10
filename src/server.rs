use bytes::Bytes;
use multer::Multipart;
use parakeet_rs::{Nemotron, NemotronMode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use warp::{ws::Ws, Filter};

use crate::language::{CHUNK_DURATION_MS, CHUNK_SIZE, SAMPLE_RATE, SupportedLanguage};
use crate::websocket::WebSocketHandler;

#[derive(Debug, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub struct TranscriptionResponse {
    pub text: String,
    pub processing_time: f64,
    pub audio_duration: f64,
    pub real_time_factor: f64,
    pub language: String,
}

#[derive(Debug, Deserialize, utoipa::IntoParams, utoipa::ToSchema)]
pub struct TranscriptionQuery {
    pub sample_rate: Option<u32>,
    pub language: Option<String>,
    pub source_lang: Option<String>,
    pub target_lang: Option<String>,
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct BatchAudioFile {
    pub name: String,
    pub data: Vec<u8>,
    pub sample_rate: Option<u32>,
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct BatchTranscriptionRequest {
    pub files: Vec<BatchAudioFile>,
    pub language: Option<String>,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct BatchTranscriptionResult {
    pub name: String,
    pub text: String,
    pub processing_time: f64,
    pub audio_duration: f64,
    pub real_time_factor: f64,
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct BatchTranscriptionResponse {
    pub results: Vec<BatchTranscriptionResult>,
    pub total_processing_time: f64,
    pub total_files: usize,
    pub successful_files: usize,
    pub failed_files: usize,
}

/// Unified response: returns single object if 1 file was uploaded, or batch object if multiple files.
#[derive(Debug, Serialize, utoipa::ToSchema)]
#[serde(untagged)]
pub enum TranscribeResponse {
    Single(TranscriptionResponse),
    Batch(BatchTranscriptionResponse),
}

#[derive(utoipa::ToSchema, Serialize)]
struct HealthResponse {
    status: String,
    model: String,
    supported_languages: Vec<String>,
    excluded_languages: Vec<String>,
    chunk_size_samples: usize,
    chunk_duration_ms: usize,
}

async fn health_check() -> impl warp::Reply {
    warp::reply::json(&HealthResponse {
        status: "healthy".to_string(),
        model: "CraneAILabs/crane-nemo-asr-1.1 (Nemotron 3.5 Streaming)".to_string(),
        supported_languages: vec![
            "lg (Luganda)".to_string(),
            "en (English US)".to_string(),
            "en-gb (English Commonwealth)".to_string(),
            "auto (Auto-detect)".to_string(),
        ],
        excluded_languages: vec!["sn (Shona - disabled)".to_string()],
        chunk_size_samples: CHUNK_SIZE,
        chunk_duration_ms: CHUNK_DURATION_MS,
    })
}

fn resample_to_16k(samples: &[f32], src_rate: u32) -> Vec<f32> {
    if src_rate == 16000 || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = src_rate as f64 / 16000.0;
    let target_len = (samples.len() as f64 / ratio).round() as usize;
    let mut resampled = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let src_idx = i as f64 * ratio;
        let idx0 = src_idx.floor() as usize;
        let frac = (src_idx - idx0 as f64) as f32;
        let s0 = samples.get(idx0).copied().unwrap_or(0.0);
        let s1 = samples.get(idx0 + 1).copied().unwrap_or(s0);
        resampled.push(s0 + frac * (s1 - s0));
    }
    resampled
}

fn parse_audio_samples(data: &[u8]) -> Result<Vec<f32>, String> {
    // Check if data is a WAV file container
    if data.starts_with(b"RIFF") && data.len() > 44 {
        let cursor = std::io::Cursor::new(data);
        if let Ok(mut reader) = hound::WavReader::new(cursor) {
            let spec = reader.spec();
            let mut samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => {
                    reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
                }
                hound::SampleFormat::Int => reader
                    .samples::<i16>()
                    .map(|s| s.unwrap_or(0) as f32 / 32768.0)
                    .collect(),
            };
            if spec.channels > 1 {
                samples = samples.chunks(spec.channels as usize).map(|c| c[0]).collect();
            }
            if spec.sample_rate != 16000 {
                samples = resample_to_16k(&samples, spec.sample_rate);
            }
            return Ok(samples);
        }
    }

    // Default to raw 16-bit PCM little-endian (assumed 16kHz)
    let mut samples = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
        samples.push(sample);
    }
    Ok(samples)
}

/// Unified transcription handler (AssemblyAI / Industry Standard):
/// - Accepts raw binary audio (application/octet-stream, audio/wav)
/// - Accepts multipart/form-data:
///     - 1 file uploaded -> returns single TranscriptionResponse directly
///     - 2+ files uploaded -> returns BatchTranscriptionResponse
async fn transcribe_unified_handler(
    content_type: String,
    query: TranscriptionQuery,
    body: Bytes,
    service: Arc<NemotronService>,
) -> Result<impl warp::Reply, warp::Rejection> {
    let lang_str = query
        .language
        .or(query.source_lang)
        .or(query.target_lang)
        .unwrap_or_else(|| "auto".to_string());

    let language = SupportedLanguage::parse(&lang_str)
        .map_err(|e| warp::reject::custom(ApiError::LanguageError(e)))?;

    if content_type.starts_with("multipart/form-data") {
        let boundary = content_type
            .split("boundary=")
            .nth(1)
            .ok_or_else(|| warp::reject::custom(ApiError::InvalidContentType))?;

        let stream = futures::stream::once(async move { Ok::<_, multer::Error>(body) });
        let mut multipart = Multipart::new(stream, boundary);

        let mut files = Vec::new();

        while let Some(field) = multipart.next_field().await.map_err(|e| {
            warp::reject::custom(ApiError::TranscriptionFailed(format!("Multipart error: {e}")))
        })? {
            let name = field.file_name().unwrap_or("audio.wav").to_string();
            let data = field.bytes().await.map_err(|e| {
                warp::reject::custom(ApiError::TranscriptionFailed(format!(
                    "Failed to read file: {e}"
                )))
            })?;

            files.push(BatchAudioFile {
                name,
                data: data.to_vec(),
                sample_rate: query.sample_rate,
            });
        }

        if files.is_empty() {
            return Err(warp::reject::custom(ApiError::TranscriptionFailed(
                "No audio files found in request".to_string(),
            )));
        }

        if files.len() == 1 {
            // Single file uploaded via multipart (AssemblyAI Sync STT style)
            let single_file = files.into_iter().next().unwrap();
            match service
                .transcribe_audio_bytes(&single_file.data, language)
                .await
            {
                Ok(resp) => {
                    let reply = warp::reply::json(&TranscribeResponse::Single(resp));
                    Ok(warp::reply::with_status(reply, warp::http::StatusCode::OK))
                }
                Err(e) => Err(warp::reject::custom(ApiError::TranscriptionFailed(
                    e.to_string(),
                ))),
            }
        } else {
            // Multiple files uploaded (batch)
            let batch_request = BatchTranscriptionRequest {
                files,
                language: Some(language.code().to_string()),
            };
            match service.transcribe_batch(batch_request, language).await {
                Ok(resp) => {
                    let reply = warp::reply::json(&TranscribeResponse::Batch(resp));
                    Ok(warp::reply::with_status(reply, warp::http::StatusCode::OK))
                }
                Err(e) => Err(warp::reject::custom(ApiError::TranscriptionFailed(
                    e.to_string(),
                ))),
            }
        }
    } else if content_type.starts_with("application/octet-stream")
        || content_type.starts_with("audio/")
    {
        // Raw audio bytes (PCM16 or WAV)
        match service.transcribe_audio_bytes(body.as_ref(), language).await {
            Ok(resp) => {
                let reply = warp::reply::json(&TranscribeResponse::Single(resp));
                Ok(warp::reply::with_status(reply, warp::http::StatusCode::OK))
            }
            Err(e) => Err(warp::reject::custom(ApiError::TranscriptionFailed(
                e.to_string(),
            ))),
        }
    } else {
        Err(warp::reject::custom(ApiError::InvalidContentType))
    }
}

/// Explicit batch endpoint: always returns BatchTranscriptionResponse
async fn transcribe_explicit_batch_handler(
    content_type: String,
    query: TranscriptionQuery,
    body: Bytes,
    service: Arc<NemotronService>,
) -> Result<impl warp::Reply, warp::Rejection> {
    if !content_type.starts_with("multipart/form-data") {
        return Err(warp::reject::custom(ApiError::InvalidContentType));
    }

    let boundary = content_type
        .split("boundary=")
        .nth(1)
        .ok_or_else(|| warp::reject::custom(ApiError::InvalidContentType))?;

    let stream = futures::stream::once(async move { Ok::<_, multer::Error>(body) });
    let mut multipart = Multipart::new(stream, boundary);

    let mut files = Vec::new();
    let lang_str = query
        .language
        .or(query.source_lang)
        .or(query.target_lang)
        .unwrap_or_else(|| "auto".to_string());

    let language = SupportedLanguage::parse(&lang_str)
        .map_err(|e| warp::reject::custom(ApiError::LanguageError(e)))?;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        warp::reject::custom(ApiError::TranscriptionFailed(format!("Multipart error: {e}")))
    })? {
        let name = field.file_name().unwrap_or("audio.wav").to_string();
        let data = field.bytes().await.map_err(|e| {
            warp::reject::custom(ApiError::TranscriptionFailed(format!(
                "Failed to read file: {e}"
            )))
        })?;

        files.push(BatchAudioFile {
            name,
            data: data.to_vec(),
            sample_rate: query.sample_rate,
        });
    }

    if files.is_empty() {
        return Err(warp::reject::custom(ApiError::TranscriptionFailed(
            "No audio files found in request".to_string(),
        )));
    }

    let batch_request = BatchTranscriptionRequest {
        files,
        language: Some(language.code().to_string()),
    };

    match service.transcribe_batch(batch_request, language).await {
        Ok(response) => {
            let reply = warp::reply::json(&response);
            Ok(warp::reply::with_status(reply, warp::http::StatusCode::OK))
        }
        Err(e) => Err(warp::reject::custom(ApiError::TranscriptionFailed(
            e.to_string(),
        ))),
    }
}

pub struct NemotronService {
    pub model_path: Arc<String>,
    shared_model: Arc<Mutex<Nemotron>>,
}

impl NemotronService {
    pub async fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        println!("Loading Crane Nemotron model from: {model_path}");

        let model = Nemotron::from_pretrained(model_path, None)?;
        println!(
            "Crane Nemotron model loaded successfully! Mode: {:?}",
            model.mode()
        );

        Ok(Self {
            model_path: Arc::new(model_path.to_string()),
            shared_model: Arc::new(Mutex::new(model)),
        })
    }

    pub async fn transcribe_audio_bytes(
        &self,
        audio_data: &[u8],
        language: SupportedLanguage,
    ) -> Result<TranscriptionResponse, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = Instant::now();
        let samples = parse_audio_samples(audio_data)?;
        let audio_duration = samples.len() as f64 / SAMPLE_RATE as f64;

        let mut model = self.shared_model.lock().await;
        model.reset();

        if model.mode() == NemotronMode::Multilingual {
            model.set_target_lang(language.to_nemotron_lang())?;
        }

        let mut full_text = String::new();

        // Process in 560ms chunks (8960 samples)
        for chunk in samples.chunks(CHUNK_SIZE) {
            let chunk_data: Vec<f32> = if chunk.len() < CHUNK_SIZE {
                let mut padded = chunk.to_vec();
                padded.resize(CHUNK_SIZE, 0.0);
                padded
            } else {
                chunk.to_vec()
            };

            let delta = model.transcribe_chunk(&chunk_data)?;
            if !delta.is_empty() {
                full_text.push_str(&delta);
            }
        }

        // Flush 3 chunks to release lookahead tokens from RNNT joint network
        let flush_chunk = vec![0.0f32; CHUNK_SIZE];
        for _ in 0..3 {
            if let Ok(flush_delta) = model.transcribe_chunk(&flush_chunk) {
                if !flush_delta.is_empty() {
                    full_text.push_str(&flush_delta);
                }
            }
        }

        model.reset();

        let processing_time = start_time.elapsed().as_secs_f64();
        let real_time_factor = if audio_duration > 0.0 {
            processing_time / audio_duration
        } else {
            0.0
        };

        Ok(TranscriptionResponse {
            text: full_text.trim().to_string(),
            processing_time,
            audio_duration,
            real_time_factor,
            language: language.code().to_string(),
        })
    }

    pub async fn transcribe_batch(
        &self,
        request: BatchTranscriptionRequest,
        language: SupportedLanguage,
    ) -> Result<BatchTranscriptionResponse, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = Instant::now();
        let mut results = Vec::new();
        let mut successful_files = 0;
        let mut failed_files = 0;

        for file in request.files {
            let file_start = Instant::now();
            match self.transcribe_audio_bytes(&file.data, language).await {
                Ok(resp) => {
                    successful_files += 1;
                    results.push(BatchTranscriptionResult {
                        name: file.name,
                        text: resp.text,
                        processing_time: resp.processing_time,
                        audio_duration: resp.audio_duration,
                        real_time_factor: resp.real_time_factor,
                        success: true,
                        error: None,
                    });
                }
                Err(e) => {
                    failed_files += 1;
                    results.push(BatchTranscriptionResult {
                        name: file.name,
                        text: String::new(),
                        processing_time: file_start.elapsed().as_secs_f64(),
                        audio_duration: 0.0,
                        real_time_factor: 0.0,
                        success: false,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        let total_processing_time = start_time.elapsed().as_secs_f64();
        let total_files = results.len();

        Ok(BatchTranscriptionResponse {
            results,
            total_processing_time,
            total_files,
            successful_files,
            failed_files,
        })
    }
}

pub fn with_service(
    service: Arc<NemotronService>,
) -> impl Filter<Extract = (Arc<NemotronService>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || service.clone())
}

pub async fn run_server(service: Arc<NemotronService>, port: u16) {
    let cors = warp::cors()
        .allow_any_origin()
        .allow_headers(vec!["content-type", "authorization"])
        .allow_methods(vec!["GET", "POST", "OPTIONS"]);

    // GET /health
    let health = warp::path("health")
        .and(warp::get())
        .and_then(|| async { Ok::<_, warp::Rejection>(health_check().await) });

    // POST /v1/transcribe (Single audio transcription)
    let transcribe_v1 = warp::path("v1")
        .and(warp::path("transcribe"))
        .and(warp::path::end())
        .and(warp::post())
        .and(warp::header::<String>("content-type"))
        .and(warp::query::<TranscriptionQuery>())
        .and(warp::body::bytes())
        .and(with_service(service.clone()))
        .and_then(transcribe_unified_handler);

    // POST /v1/transcribe/batch (Batch audio transcription)
    let batch_transcribe = warp::path("v1")
        .and(warp::path("transcribe"))
        .and(warp::path("batch"))
        .and(warp::path::end())
        .and(warp::post())
        .and(warp::header::<String>("content-type"))
        .and(warp::query::<TranscriptionQuery>())
        .and(warp::body::bytes())
        .and(with_service(service.clone()))
        .and_then(transcribe_explicit_batch_handler);

    // WebSocket: GET /v1/realtime (Real-time live streaming audio)
    let ws_model_path = service.model_path.clone();
    let realtime_ws = warp::path("v1")
        .and(warp::path("realtime"))
        .and(warp::path::end())
        .and(warp::ws())
        .and(warp::query::<TranscriptionQuery>())
        .and(warp::any().map(move || ws_model_path.clone()))
        .map(|ws: Ws, query: TranscriptionQuery, model_path: Arc<String>| {
            let initial_lang = query.language.or(query.source_lang);
            ws.on_upgrade(move |socket| async move {
                let ws_handler = WebSocketHandler::new(model_path);
                ws_handler.handle_websocket(socket, initial_lang).await;
            })
        });

    // OpenAPI Spec JSON: GET /docs/openapi.json
    let openapi = warp::path("docs")
        .and(warp::path("openapi.json"))
        .and(warp::path::end())
        .and(warp::get())
        .map(|| {
            let openapi_spec = serde_json::json!({
                "openapi": "3.0.0",
                "info": {
                    "title": "Crane Nemotron Cache-Aware Streaming ASR API",
                    "description": "Production-grade Speech-to-Text API specializing in Luganda ('lg') and English ('en'), using Crane Nemotron-3.5 0.6B FastConformer RNN-Transducer.",
                    "version": "1.2.0",
                    "contact": { "name": "API Support" }
                },
                "servers": [
                    { "url": "/", "description": "Local server" }
                ],
                "paths": {
                    "/v1/transcribe": {
                        "post": {
                            "summary": "Transcribe Audio (Single or Batch)",
                            "description": "Synchronous STT (AssemblyAI style). Accepts raw 16-bit PCM/WAV binary or multipart/form-data. If 1 file is sent, returns a single transcript object. If multiple files are uploaded, returns a batch results object.",
                            "parameters": [
                                {
                                    "name": "language",
                                    "in": "query",
                                    "required": false,
                                    "description": "Target language: 'lg' (Luganda), 'en' (English), or 'auto' (Auto-detect). 'sn' (Shona) is explicitly rejected.",
                                    "schema": { "type": "string", "default": "auto" }
                                },
                                {
                                    "name": "sample_rate",
                                    "in": "query",
                                    "required": false,
                                    "description": "Audio sample rate in Hz (default: 16000)",
                                    "schema": { "type": "integer", "default": 16000 }
                                }
                            ],
                            "requestBody": {
                                "description": "Raw audio binary (audio/wav, application/octet-stream) or multipart/form-data files",
                                "required": true,
                                "content": {
                                    "application/octet-stream": { "schema": { "type": "string", "format": "binary" } },
                                    "audio/wav": { "schema": { "type": "string", "format": "binary" } },
                                    "multipart/form-data": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "audio": { "type": "string", "format": "binary", "description": "Audio file" }
                                            }
                                        }
                                    }
                                }
                            },
                            "responses": {
                                "200": {
                                    "description": "Transcription successful",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "oneOf": [
                                                    { "$ref": "#/components/schemas/TranscriptionResponse" },
                                                    { "$ref": "#/components/schemas/BatchTranscriptionResponse" }
                                                ]
                                            }
                                        }
                                    }
                                },
                                "400": { "description": "Invalid audio or unsupported language (e.g. Shona)" },
                                "500": { "description": "Internal server error" }
                            }
                        }
                    },
                    "/v1/transcribe/batch": {
                        "post": {
                            "summary": "Explicit Batch Transcription",
                            "description": "Upload multiple audio files via multipart/form-data. Always returns BatchTranscriptionResponse array.",
                            "parameters": [
                                {
                                    "name": "language",
                                    "in": "query",
                                    "required": false,
                                    "description": "Target language: 'lg', 'en', or 'auto'",
                                    "schema": { "type": "string", "default": "auto" }
                                }
                            ],
                            "responses": {
                                "200": {
                                    "description": "Batch transcription results",
                                    "content": {
                                        "application/json": {
                                            "schema": { "$ref": "#/components/schemas/BatchTranscriptionResponse" }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "/v1/realtime": {
                        "get": {
                            "summary": "Real-time Streaming WebSocket (AssemblyAI style)",
                            "description": "Bidirectional WebSocket connection for live 560ms cache-aware streaming. Connect with query param e.g. ws://localhost:8099/v1/realtime?language=lg. Supported messages: 'audio' (base64 PCM), 'partial' (delta text), 'final' (completed utterance), 'speech_started', 'speech_ended', 'reset', 'flush'.",
                            "parameters": [
                                {
                                    "name": "language",
                                    "in": "query",
                                    "required": false,
                                    "description": "Target language: 'lg', 'en', or 'auto'",
                                    "schema": { "type": "string", "default": "auto" }
                                },
                                {
                                    "name": "sample_rate",
                                    "in": "query",
                                    "required": false,
                                    "schema": { "type": "integer", "default": 16000 }
                                }
                            ]
                        }
                    },
                    "/health": {
                        "get": {
                            "summary": "Service Health Check",
                            "description": "Returns model status, supported languages, chunk sizes.",
                            "responses": { "200": { "description": "Healthy" } }
                        }
                    }
                },
                "components": {
                    "schemas": {
                        "TranscriptionResponse": {
                            "type": "object",
                            "properties": {
                                "text": { "type": "string", "example": "Twasanyuse nnyo okukulaba." },
                                "processing_time": { "type": "number", "example": 0.142 },
                                "audio_duration": { "type": "number", "example": 2.80 },
                                "real_time_factor": { "type": "number", "example": 0.051 },
                                "language": { "type": "string", "example": "lg" }
                            }
                        },
                        "BatchTranscriptionResponse": {
                            "type": "object",
                            "properties": {
                                "results": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": { "type": "string" },
                                            "text": { "type": "string" },
                                            "processing_time": { "type": "number" },
                                            "audio_duration": { "type": "number" },
                                            "real_time_factor": { "type": "number" },
                                            "success": { "type": "boolean" },
                                            "error": { "type": "string", "nullable": true }
                                        }
                                    }
                                },
                                "total_processing_time": { "type": "number" },
                                "total_files": { "type": "integer" },
                                "successful_files": { "type": "integer" },
                                "failed_files": { "type": "integer" }
                            }
                        }
                    }
                }
            });
            warp::reply::json(&openapi_spec)
        });

    let swagger_html = r#"<!DOCTYPE html>
<html>
<head>
    <title>Crane Nemotron ASR API - Interactive Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.5.0/swagger-ui.css" />
    <style>
        body { margin: 0; background: #fafafa; }
        .topbar { display: none; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@4.5.0/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {
            SwaggerUIBundle({
                url: '/docs/openapi.json',
                dom_id: '#swagger-ui',
                presets: [SwaggerUIBundle.presets.apis],
                layout: 'BaseLayout'
            });
        };
    </script>
</body>
</html>"#;

    // Interactive Swagger UI: GET /docs
    let docs = warp::path("docs")
        .and(warp::path::end())
        .and(warp::get())
        .map(move || warp::reply::html(swagger_html));

    // Interactive Demo UI (at root /)
    let demo_ui = warp::path::end().and(warp::get()).map(|| {
        let html = include_str!("../static/index.html");
        warp::reply::html(html)
    });

    let routes = health
        .or(demo_ui)
        .or(realtime_ws)
        .or(transcribe_v1)
        .or(batch_transcribe)
        .or(openapi)
        .or(docs)
        .with(cors)
        .with(warp::log("api"));

    println!("============================================================");
    println!(" Crane Nemotron 3.5 Streaming ASR Service running on port {port}");
    println!(" Engine: parakeet-rs | FastConformer Native Transducer");
    println!(" Languages: Luganda ('lg'), English ('en', 'en-gb'), Auto ('auto')");
    println!(" Shona ('sn'): EXCLUDED / DISABLED");
    println!("------------------------------------------------------------");
    println!(" [WebSocket Streaming]  ws://localhost:{port}/v1/realtime");
    println!(" [Single Transcribe]    POST http://localhost:{port}/v1/transcribe");
    println!(" [Batch Transcribe]     POST http://localhost:{port}/v1/transcribe/batch");
    println!(" [Health Check]         GET http://localhost:{port}/health");
    println!(" [Interactive Docs]     http://localhost:{port}/docs");
    println!(" [OpenAPI Spec JSON]    http://localhost:{port}/docs/openapi.json");
    println!(" [Demo Web Interface]   http://localhost:{port}/");
    println!("============================================================");

    warp::serve(routes).run(([0, 0, 0, 0], port)).await;
}

#[derive(Debug)]
pub enum ApiError {
    InvalidContentType,
    LanguageError(String),
    TranscriptionFailed(String),
}

impl warp::reject::Reject for ApiError {}

impl warp::Reply for ApiError {
    fn into_response(self) -> warp::reply::Response {
        let (code, message) = match self {
            ApiError::InvalidContentType => (
                warp::http::StatusCode::BAD_REQUEST,
                "Invalid content type. Expected application/octet-stream, audio/wav, or multipart/form-data".to_string(),
            ),
            ApiError::LanguageError(msg) => (warp::http::StatusCode::BAD_REQUEST, msg),
            ApiError::TranscriptionFailed(msg) => (
                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Transcription failed: {msg}"),
            ),
        };

        let json = warp::reply::json(&serde_json::json!({
            "error": message
        }));

        warp::reply::with_status(json, code).into_response()
    }
}
