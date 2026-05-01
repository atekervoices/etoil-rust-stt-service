use canary_rs::{Canary, ExecutionConfig, ExecutionProvider};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use warp::{Filter, ws::Ws};
use utoipa::OpenApi;
use futures::future;
use multer::Multipart;
use bytes::Bytes;

#[derive(Debug, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub struct TranscriptionResponse {
    pub text: String,
    pub processing_time: f64,
    pub audio_duration: f64,
}

#[derive(Debug, Deserialize, utoipa::IntoParams, utoipa::ToSchema)]
pub struct TranscriptionQuery {
    pub sample_rate: Option<u32>,
    pub source_lang: Option<String>,
    pub target_lang: Option<String>,
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct BatchAudioFile {
    pub name: String,
    pub data: Vec<u8>, // raw binary audio data
    pub sample_rate: Option<u32>,
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct BatchTranscriptionRequest {
    pub files: Vec<BatchAudioFile>,
    pub source_lang: Option<String>,
    pub target_lang: Option<String>,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct BatchTranscriptionResult {
    pub name: String,
    pub text: String,
    pub processing_time: f64,
    pub audio_duration: f64,
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

#[derive(utoipa::ToSchema, Serialize)]
struct HealthResponse {
    status: String,
    model: String,
}

#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Health check successful", body = HealthResponse)
    ),
    tag = "Health"
)]
async fn health_check() -> impl warp::Reply {
    warp::reply::json(&HealthResponse {
        status: "healthy".to_string(),
        model: "canary-rs".to_string(),
    })
}

#[utoipa::path(
    post,
    path = "/v1/transcribe/canary",
    request_body(content = Vec<u8>, description = "Raw audio data in 16-bit PCM format", content_type = "application/octet-stream"),
    params(
        TranscriptionQuery
    ),
    responses(
        (status = 200, description = "Transcription successful", body = TranscriptionResponse),
        (status = 400, description = "Invalid content type"),
        (status = 500, description = "Transcription failed")
    ),
    tag = "Transcription"
)]
async fn transcribe_audio(
    content_type: String,
    query: TranscriptionQuery,
    body: bytes::Bytes,
    service: Arc<CanaryService>,
) -> Result<impl warp::Reply, warp::Rejection> {
    if !content_type.starts_with("application/octet-stream") {
        return Err(warp::reject::custom(ApiError::InvalidContentType));
    }

    let sample_rate = query.sample_rate.unwrap_or(16000);
    let source_lang = query.source_lang.unwrap_or_else(|| "en".to_string());
    let target_lang = query.target_lang.unwrap_or_else(|| "en".to_string());
    
    match service
        .transcribe_raw_audio(body.to_vec(), sample_rate, &source_lang, &target_lang)
        .await
    {
        Ok(response) => {
            let reply = warp::reply::json(&response);
            Ok(warp::reply::with_status(reply, warp::http::StatusCode::OK))
        }
        Err(e) => {
            eprintln!("Transcription error: {}", e);
            Err(warp::reject::custom(ApiError::TranscriptionFailed(
                e.to_string(),
            )))
        }
    }
}


async fn transcribe_multipart_batch(
    content_type: String,
    query: TranscriptionQuery,
    body: Bytes,
    service: Arc<CanaryService>,
) -> Result<impl warp::Reply, warp::Rejection> {
    // Validate content type for multipart
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
    let source_lang = query.source_lang.unwrap_or_else(|| "en".to_string());
    let target_lang = query.target_lang.unwrap_or_else(|| "en".to_string());

    // Process each file in the multipart request
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        warp::reject::custom(ApiError::TranscriptionFailed(format!("Multipart error: {}", e)))
    })? {
        let name = field.file_name()
            .unwrap_or("unknown")
            .to_string();
        
        let data = field.bytes().await.map_err(|e| {
            warp::reject::custom(ApiError::TranscriptionFailed(format!("Failed to read file: {}", e)))
        })?;

        files.push(BatchAudioFile {
            name,
            data: data.to_vec(),
            sample_rate: query.sample_rate,
        });
    }

    if files.is_empty() {
        return Err(warp::reject::custom(ApiError::TranscriptionFailed(
            "No files found in request".to_string(),
        )));
    }

    let batch_request = BatchTranscriptionRequest {
        files,
        source_lang: Some(source_lang),
        target_lang: Some(target_lang),
    };

    match service.transcribe_batch(batch_request).await {
        Ok(response) => {
            let reply = warp::reply::json(&response);
            Ok(warp::reply::with_status(reply, warp::http::StatusCode::OK))
        }
        Err(e) => {
            eprintln!("Multipart batch transcription error: {}", e);
            Err(warp::reject::custom(ApiError::TranscriptionFailed(
                e.to_string(),
            )))
        }
    }
}

// ApiDoc struct is used for manual OpenAPI specification

#[derive(Clone)]
pub struct CanaryService {
    model: Arc<RwLock<Canary>>,
}

impl CanaryService {
    pub async fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        println!("Loading Canary model from: {}", model_path);
        
        // Auto-detect best execution provider: CUDA GPU if available, otherwise CPU
        let config = ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cuda);
        let model = match Canary::from_pretrained(model_path, Some(config)) {
            Ok(model) => {
                println!("✅ Using CUDA GPU acceleration");
                model
            }
            Err(_) => {
                println!("⚠️  CUDA not available, falling back to CPU");
                let cpu_config = ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cpu);
                Canary::from_pretrained(model_path, Some(cpu_config))?
            }
        };
        
        println!("✅ Canary model loaded successfully!");
        
        Ok(Self {
            model: Arc::new(RwLock::new(model)),
        })
    }

    pub async fn transcribe_raw_audio(
        &self,
        audio_data: Vec<u8>,
        sample_rate: u32,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<TranscriptionResponse, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = Instant::now();
        
        // Convert bytes to f32 samples (assuming 16-bit PCM)
        let mut samples = Vec::new();
        for chunk in audio_data.chunks_exact(2) {
            if chunk.len() == 2 {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                samples.push(sample);
            }
        }
        
        let audio_duration = samples.len() as f64 / sample_rate as f64;
        
        // Get model session
        let model = self.model.read().await;
        let mut session = model.session();
        
        // Transcribe
        let result = session.transcribe_samples(&samples, sample_rate as usize, 1, source_lang, target_lang)?;
        
        let processing_time = start_time.elapsed().as_secs_f64();
        
        Ok(TranscriptionResponse {
            text: result.text,
            processing_time,
            audio_duration,
        })
    }

    pub async fn transcribe_batch(
        &self,
        request: BatchTranscriptionRequest,
    ) -> Result<BatchTranscriptionResponse, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = Instant::now();
        let source_lang = request.source_lang.unwrap_or_else(|| "en".to_string());
        let target_lang = request.target_lang.unwrap_or_else(|| "en".to_string());
        
        let mut results = Vec::new();
        let mut successful_files = 0;
        let mut failed_files = 0;
        
        // Process files in parallel using tokio
        let source_lang_clone = source_lang.clone();
        let target_lang_clone = target_lang.clone();
        
        let futures: Vec<_> = request.files.into_iter().map(|file| {
            let service = self.clone();
            let source_lang = source_lang_clone.clone();
            let target_lang = target_lang_clone.clone();
            async move {
                let file_start_time = std::time::Instant::now();
                
                // Use binary audio data directly
                let audio_data = file.data;
                let sample_rate = file.sample_rate.unwrap_or(16000);
                
                match service.transcribe_raw_audio(audio_data, sample_rate, &source_lang, &target_lang).await {
                    Ok(response) => {
                        BatchTranscriptionResult {
                            name: file.name,
                            text: response.text,
                            processing_time: response.processing_time,
                            audio_duration: response.audio_duration,
                            success: true,
                            error: None,
                        }
                    }
                    Err(e) => {
                        BatchTranscriptionResult {
                            name: file.name,
                            text: String::new(),
                            processing_time: file_start_time.elapsed().as_secs_f64(),
                            audio_duration: 0.0,
                            success: false,
                            error: Some(e.to_string()),
                        }
                    }
                }
            }
        }).collect();
        
        // Wait for all transcriptions to complete
        let batch_results = future::join_all(futures).await;
        
        for result in batch_results {
            if result.success {
                successful_files += 1;
            } else {
                failed_files += 1;
            }
            results.push(result);
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
    service: Arc<CanaryService>,
) -> impl Filter<Extract = (Arc<CanaryService>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || service.clone())
}

pub async fn run_server(service: Arc<CanaryService>, port: u16) {
    let cors = warp::cors()
        .allow_any_origin()
        .allow_headers(vec!["content-type"])
        .allow_methods(vec!["GET", "POST", "OPTIONS"]);

    // Health check endpoint
    let health = warp::path("health")
        .and(warp::get())
        .and_then(|| async {
            Ok::<_, warp::Rejection>(health_check().await)
        });

    // Transcription endpoint
    let transcribe = warp::path("v1")
        .and(warp::path("transcribe"))
        .and(warp::path("canary"))
        .and(warp::post())
        .and(warp::header::<String>("content-type"))
        .and(warp::query::<TranscriptionQuery>())
        .and(warp::body::bytes())
        .and(with_service(service.clone()))
        .and_then(transcribe_audio);

    // Batch transcription endpoint (multipart/form-data - industry standard like AssemblyAI/Deepgram)
    let batch_transcribe = warp::path("v1")
        .and(warp::path("transcribe"))
        .and(warp::path("batch"))
        .and(warp::post())
        .and(warp::header::<String>("content-type"))
        .and(warp::query::<TranscriptionQuery>())
        .and(warp::body::bytes())
        .and(with_service(service.clone()))
        .and_then(transcribe_multipart_batch);

    // OpenAPI JSON endpoint
    let openapi = warp::path("api-docs")
        .and(warp::path("openapi.json"))
        .and(warp::get())
        .map(|| {
            let openapi_spec = serde_json::json!({
                "openapi": "3.0.0",
                "info": {
                    "title": "Canary STT API",
                    "description": "Speech-to-Text API using Canary-RS model with real-time streaming support",
                    "version": "1.0.0",
                    "contact": {
                        "name": "Canary STT Service"
                    }
                },
                "paths": {
                    "/health": {
                        "get": {
                            "tags": ["Health"],
                            "summary": "Health check",
                            "description": "Check if the server is running and the model is loaded",
                            "responses": {
                                "200": {
                                    "description": "Health check successful",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "$ref": "#/components/schemas/HealthResponse"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "/v1/transcribe/canary": {
                        "post": {
                            "tags": ["Transcription"],
                            "summary": "Transcribe audio",
                            "description": "Transcribe audio data using the REST API",
                            "parameters": [
                                {
                                    "name": "sample_rate",
                                    "in": "query",
                                    "description": "Audio sample rate",
                                    "required": false,
                                    "schema": {
                                        "type": "integer",
                                        "default": 16000
                                    }
                                },
                                {
                                    "name": "source_lang",
                                    "in": "query",
                                    "description": "Source language",
                                    "required": false,
                                    "schema": {
                                        "type": "string",
                                        "default": "en"
                                    }
                                },
                                {
                                    "name": "target_lang",
                                    "in": "query",
                                    "description": "Target language",
                                    "required": false,
                                    "schema": {
                                        "type": "string",
                                        "default": "en"
                                    }
                                }
                            ],
                            "requestBody": {
                                "description": "Raw audio data in 16-bit PCM format",
                                "required": true,
                                "content": {
                                    "application/octet-stream": {
                                        "schema": {
                                            "type": "string",
                                            "format": "binary"
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
                                                "$ref": "#/components/schemas/TranscriptionResponse"
                                            }
                                        }
                                    }
                                },
                                "400": {
                                    "description": "Invalid content type"
                                },
                                "500": {
                                    "description": "Transcription failed"
                                }
                            }
                        }
                    },
                    "/v1/transcribe/batch": {
                        "post": {
                            "tags": ["Transcription"],
                            "summary": "Upload and transcribe multiple audio files",
                            "description": "Upload multiple audio files using multipart/form-data for batch transcription. Industry standard format used by AssemblyAI, Deepgram, and other voice APIs.",
                            "parameters": [
                                {
                                    "name": "sample_rate",
                                    "in": "query",
                                    "description": "Audio sample rate for all files",
                                    "required": false,
                                    "schema": {
                                        "type": "integer",
                                        "default": 16000
                                    }
                                },
                                {
                                    "name": "source_lang",
                                    "in": "query",
                                    "description": "Source language for all files",
                                    "required": false,
                                    "schema": {
                                        "type": "string",
                                        "default": "en"
                                    }
                                },
                                {
                                    "name": "target_lang",
                                    "in": "query",
                                    "description": "Target language for all files",
                                    "required": false,
                                    "schema": {
                                        "type": "string",
                                        "default": "en"
                                    }
                                }
                            ],
                            "requestBody": {
                                "description": "Multiple audio files to transcribe using multipart/form-data",
                                "required": true,
                                "content": {
                                    "multipart/form-data": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "files": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string",
                                                        "format": "binary"
                                                    },
                                                    "description": "Audio files to transcribe"
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "responses": {
                                "200": {
                                    "description": "Batch transcription completed",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "$ref": "#/components/schemas/BatchTranscriptionResponse"
                                            }
                                        }
                                    }
                                },
                                "400": {
                                    "description": "Invalid request format"
                                },
                                "500": {
                                    "description": "Batch transcription failed"
                                }
                            }
                        }
                    }
                },
                "components": {
                    "schemas": {
                        "HealthResponse": {
                            "type": "object",
                            "properties": {
                                "status": {
                                    "type": "string",
                                    "example": "healthy"
                                },
                                "model": {
                                    "type": "string",
                                    "example": "canary-rs"
                                }
                            }
                        },
                        "TranscriptionResponse": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "Transcribed text",
                                    "example": "Hello, world!"
                                },
                                "processing_time": {
                                    "type": "number",
                                    "format": "float",
                                    "description": "Processing time in seconds",
                                    "example": 1.23
                                },
                                "audio_duration": {
                                    "type": "number",
                                    "format": "float",
                                    "description": "Audio duration in seconds",
                                    "example": 2.5
                                }
                            }
                        },
                        "TranscriptionQuery": {
                            "type": "object",
                            "properties": {
                                "sample_rate": {
                                    "type": "integer",
                                    "description": "Audio sample rate",
                                    "example": 16000
                                },
                                "source_lang": {
                                    "type": "string",
                                    "description": "Source language",
                                    "example": "en"
                                },
                                "target_lang": {
                                    "type": "string",
                                    "description": "Target language",
                                    "example": "en"
                                }
                            }
                        },
                        "BatchAudioFile": {
                            "type": "object",
                            "required": ["name", "data"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "File name for identification",
                                    "example": "audio1.wav"
                                },
                                "data": {
                                    "type": "string",
                                    "format": "binary",
                                    "description": "Raw binary audio data (16-bit PCM, mono)",
                                    "example": "Binary audio data..."
                                },
                                "sample_rate": {
                                    "type": "integer",
                                    "description": "Audio sample rate (optional, defaults to 16000)",
                                    "example": 16000
                                }
                            }
                        },
                        "BatchTranscriptionRequest": {
                            "type": "object",
                            "required": ["files"],
                            "properties": {
                                "files": {
                                    "type": "array",
                                    "description": "Array of audio files to transcribe",
                                    "items": {
                                        "$ref": "#/components/schemas/BatchAudioFile"
                                    }
                                },
                                "source_lang": {
                                    "type": "string",
                                    "description": "Source language for all files",
                                    "example": "en"
                                },
                                "target_lang": {
                                    "type": "string",
                                    "description": "Target language for all files",
                                    "example": "en"
                                }
                            }
                        },
                        "BatchTranscriptionResult": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "File name",
                                    "example": "audio1.wav"
                                },
                                "text": {
                                    "type": "string",
                                    "description": "Transcribed text",
                                    "example": "Hello, world!"
                                },
                                "processing_time": {
                                    "type": "number",
                                    "format": "float",
                                    "description": "Processing time in seconds",
                                    "example": 1.23
                                },
                                "audio_duration": {
                                    "type": "number",
                                    "format": "float",
                                    "description": "Audio duration in seconds",
                                    "example": 2.5
                                },
                                "success": {
                                    "type": "boolean",
                                    "description": "Whether transcription was successful",
                                    "example": true
                                },
                                "error": {
                                    "type": "string",
                                    "description": "Error message if transcription failed",
                                    "example": "Failed to decode audio"
                                }
                            }
                        },
                        "BatchTranscriptionResponse": {
                            "type": "object",
                            "properties": {
                                "results": {
                                    "type": "array",
                                    "description": "Array of transcription results",
                                    "items": {
                                        "$ref": "#/components/schemas/BatchTranscriptionResult"
                                    }
                                },
                                "total_processing_time": {
                                    "type": "number",
                                    "format": "float",
                                    "description": "Total processing time for all files",
                                    "example": 5.67
                                },
                                "total_files": {
                                    "type": "integer",
                                    "description": "Total number of files processed",
                                    "example": 3
                                },
                                "successful_files": {
                                    "type": "integer",
                                    "description": "Number of successfully transcribed files",
                                    "example": 2
                                },
                                "failed_files": {
                                    "type": "integer",
                                    "description": "Number of failed transcriptions",
                                    "example": 1
                                }
                            }
                        }
                    }
                },
                "tags": [
                    {
                        "name": "Health",
                        "description": "Health check endpoints"
                    },
                    {
                        "name": "Transcription",
                        "description": "Audio transcription endpoints"
                    }
                ]
            });
            warp::reply::json(&openapi_spec)
        });

    // Swagger UI endpoint
    let swagger_ui = warp::path("swagger-ui")
        .and(warp::path::end())
        .and(warp::get())
        .map(|| {
            let html = include_str!("../swagger.html");
            warp::reply::html(html)
        });

    // WebSocket endpoint
    let ws_service = service.clone();
    let websocket = warp::path("ws")
        .and(warp::ws())
        .and(warp::any().map(move || ws_service.clone()))
        .map(|ws: Ws, service: Arc<CanaryService>| {
            ws.on_upgrade(move |websocket| async move {
                // Create WebSocket handler
                let model = service.model.clone();
                let ws_handler = crate::websocket::WebSocketHandler::new(model);
                ws_handler.handle_websocket(websocket).await;
            })
        });

    let routes = health
        .or(websocket)
        .or(transcribe)
        .or(batch_transcribe)
        .or(openapi)
        .or(swagger_ui)
        .with(cors)
        .with(warp::log("api"));

    println!("Canary STT Server starting on http://0.0.0.0:{}", port);
    println!("Available endpoints:");
    println!("  GET  /health - Health check");
    println!("  GET  /swagger-ui/ - Interactive API documentation (Swagger UI)");
    println!("  GET  /api-docs/openapi.json - OpenAPI specification");
    println!("  WS   /ws - WebSocket streaming transcription");
    println!("  POST /v1/transcribe/canary?sample_rate=16000 - Transcribe single audio");
    println!("  POST /v1/transcribe/batch?sample_rate=16000 - Upload multiple audio files");
    println!();
    println!("REST API Usage:");
    println!("  Content-Type: application/octet-stream");
    println!("  Body: Raw audio data (16-bit PCM, mono)");
    println!("  Query: ?sample_rate=16000&source_lang=en&target_lang=en");
    println!();
    println!("WebSocket Usage:");
    println!("  Connect: ws://localhost:{}/ws", port);
    println!("  Send config: {{\"type\":\"config\",\"sample_rate\":16000,\"source_lang\":\"en\",\"target_lang\":\"en\"}}");
    println!("  Send audio: {{\"type\":\"audio\",\"data\":\"<base64_audio>\",\"sample_rate\":16000}}");

    warp::serve(routes).run(([0, 0, 0, 0], port)).await;
}

#[derive(Debug)]
pub enum ApiError {
    InvalidContentType,
    TranscriptionFailed(String),
}

impl warp::reject::Reject for ApiError {}

impl warp::Reply for ApiError {
    fn into_response(self) -> warp::reply::Response {
        let (code, message) = match self {
            ApiError::InvalidContentType => (
                warp::http::StatusCode::BAD_REQUEST,
                "Invalid content type. Expected application/octet-stream".to_string(),
            ),
            ApiError::TranscriptionFailed(msg) => (
                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Transcription failed: {}", msg),
            ),
        };

        let json = warp::reply::json(&serde_json::json!({
            "error": message
        }));

        warp::reply::with_status(json, code).into_response()
    }
}
