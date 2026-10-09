use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use parakeet_rs::{Nemotron, NemotronMode};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::mpsc;
use warp::ws::{Message as WarpMessage, WebSocket};

use crate::language::{CHUNK_SIZE, SupportedLanguage};
use crate::vad_implementation::{normalize_punctuation_spacing, EarshotVAD, VADResult};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WSMessage {
    #[serde(rename = "config")]
    Config {
        sample_rate: Option<u32>,
        language: Option<String>,
        source_lang: Option<String>,
        target_lang: Option<String>,
    },
    #[serde(rename = "audio")]
    Audio {
        data: String, // base64 encoded 16-bit PCM audio data
        sample_rate: Option<u32>,
    },
    #[serde(rename = "reset")]
    Reset,
    #[serde(rename = "flush")]
    Flush,
    #[serde(rename = "partial")]
    Partial {
        text: String,
        accumulated_text: String,
        confidence: f32,
        timestamp: f64,
    },
    #[serde(rename = "final")]
    Final {
        text: String,
        confidence: f32,
        timestamp: f64,
        processing_time: f64,
    },
    #[serde(rename = "speech_started")]
    SpeechStarted {
        timestamp: f64,
    },
    #[serde(rename = "speech_ended")]
    SpeechEnded {
        timestamp: f64,
    },
    #[serde(rename = "error")]
    Error {
        message: String,
    },
    #[serde(rename = "ready")]
    Ready {
        model: String,
        vad_engine: String,
        supported_languages: Vec<String>,
        chunk_size_samples: usize,
    },
}

#[derive(Clone)]
pub struct WebSocketHandler {
    model_path: Arc<String>,
}

impl WebSocketHandler {
    pub fn new(model_path: Arc<String>) -> Self {
        Self { model_path }
    }

    pub async fn handle_websocket(&self, ws: WebSocket, initial_language: Option<String>) {
        let (mut ws_tx, mut ws_rx) = ws.split();
        let (tx, mut rx) = mpsc::channel::<WSMessage>(100);

        let mut active_language = SupportedLanguage::Auto;
        if let Some(ref l) = initial_language {
            if let Ok(parsed) = SupportedLanguage::parse(l) {
                active_language = parsed;
            }
        }

        // Send ready message on connection
        let ready_msg = WSMessage::Ready {
            model: "CraneAILabs/crane-nemo-asr-1.1 (Nemotron 3.5 Streaming)".to_string(),
            vad_engine: "Earshot (pyke.io, 3,600x RTF neural VAD)".to_string(),
            supported_languages: vec![
                "lg (Luganda)".to_string(),
                "en (English)".to_string(),
                "auto (Auto-detect)".to_string(),
            ],
            chunk_size_samples: CHUNK_SIZE,
        };
        if ws_tx
            .send(WarpMessage::text(
                serde_json::to_string(&ready_msg).unwrap_or_default(),
            ))
            .await
            .is_err()
        {
            return;
        }

        // Handle outgoing messages
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if let Ok(text) = serde_json::to_string(&msg) {
                    if ws_tx.send(WarpMessage::text(text)).await.is_err() {
                        break;
                    }
                }
            }
        });

        // Connection state
        let mut audio_buffer: VecDeque<f32> = VecDeque::with_capacity(CHUNK_SIZE * 4);
        let mut accumulated_transcript = String::new();
        let mut vad = EarshotVAD::new();
        let start_time = std::time::Instant::now();

        // Load default model session
        let mut nemotron_model = match Nemotron::from_pretrained(self.model_path.as_str(), None) {
            Ok(mut model) => {
                if model.mode() == NemotronMode::Multilingual {
                    let _ = model.set_target_lang(active_language.to_nemotron_lang());
                }
                Some(model)
            }
            Err(e) => {
                let _ = tx
                    .send(WSMessage::Error {
                        message: format!("Failed to load Nemotron model: {e}"),
                    })
                    .await;
                return;
            }
        }

        while let Some(msg) = ws_rx.next().await {
            match msg {
                Ok(message) => {
                    if message.is_text() {
                        let text = match message.to_str() {
                            Ok(t) => t,
                            Err(_) => continue,
                        };

                        let ws_msg = match serde_json::from_str::<WSMessage>(text) {
                            Ok(m) => m,
                            Err(e) => {
                                let _ = tx
                                    .send(WSMessage::Error {
                                        message: format!("Invalid message format: {e}"),
                                    })
                                    .await;
                                continue;
                            }
                        };

                        match ws_msg {
                            WSMessage::Config {
                                language,
                                source_lang,
                                target_lang,
                                ..
                            } => {
                                let lang_str = language
                                    .or(source_lang)
                                    .or(target_lang)
                                    .unwrap_or_else(|| "auto".to_string());

                                match SupportedLanguage::parse(&lang_str) {
                                    Ok(lang) => {
                                        active_language = lang;
                                        println!(
                                            "WebSocket: Configured language '{}' (Nemotron target: '{}')",
                                            active_language.name(),
                                            active_language.to_nemotron_lang()
                                        );

                                        if let Some(ref mut model) = nemotron_model {
                                            if model.mode() == NemotronMode::Multilingual {
                                                if let Err(e) = model.set_target_lang(
                                                    active_language.to_nemotron_lang(),
                                                ) {
                                                    let _ = tx
                                                        .send(WSMessage::Error {
                                                            message: format!(
                                                                "Failed to set model language: {e}"
                                                            ),
                                                        })
                                                        .await;
                                                }
                                            }
                                        }
                                    }
                                    Err(err) => {
                                        let _ = tx.send(WSMessage::Error { message: err }).await;
                                    }
                                }
                            }

                            WSMessage::Audio { data, .. } => {
                                // Decode base64 PCM16 audio
                                let audio_bytes = match base64::engine::general_purpose::STANDARD
                                    .decode(&data)
                                {
                                    Ok(b) => b,
                                    Err(e) => {
                                        let _ = tx
                                            .send(WSMessage::Error {
                                                message: format!("Base64 audio decode error: {e}"),
                                            })
                                            .await;
                                        continue;
                                    }
                                };

                                // Convert 16-bit PCM LE to f32 samples
                                for chunk in audio_bytes.chunks_exact(2) {
                                    let sample =
                                        i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                                    audio_buffer.push_back(sample);
                                }

                                // Process full 560ms chunks (8960 samples)
                                while audio_buffer.len() >= CHUNK_SIZE {
                                    let chunk: Vec<f32> =
                                        audio_buffer.drain(..CHUNK_SIZE).collect();

                                    // Earshot Neural VAD processing (256-sample / 16ms frames)
                                    let vad_res = vad.process_chunk(&chunk);
                                    match vad_res {
                                        VADResult::SpeechStarted => {
                                            let _ = tx
                                                .send(WSMessage::SpeechStarted {
                                                    timestamp: start_time.elapsed().as_secs_f64(),
                                                })
                                                .await;
                                        }
                                        VADResult::EndOfUtterance {
                                            should_finalize, ..
                                        } => {
                                            if should_finalize {
                                                let _ = tx
                                                    .send(WSMessage::SpeechEnded {
                                                        timestamp: start_time
                                                            .elapsed()
                                                            .as_secs_f64(),
                                                    })
                                                    .await;

                                                let final_text = normalize_punctuation_spacing(
                                                    accumulated_transcript.trim(),
                                                );
                                                if !final_text.is_empty() {
                                                    let _ = tx
                                                        .send(WSMessage::Final {
                                                            text: final_text,
                                                            confidence: 0.95,
                                                            timestamp: start_time
                                                                .elapsed()
                                                                .as_secs_f64(),
                                                            processing_time: start_time
                                                                .elapsed()
                                                                .as_secs_f64(),
                                                        })
                                                        .await;
                                                }

                                                // Reset Nemotron cache and Earshot internal state for next utterance
                                                if let Some(ref mut model) = nemotron_model {
                                                    model.reset();
                                                }
                                                vad.reset();
                                                accumulated_transcript.clear();
                                            }
                                        }
                                        _ => {}
                                    }

                                    // Stream through Nemotron cache-aware inference
                                    if let Some(ref mut model) = nemotron_model {
                                        match model.transcribe_chunk(&chunk) {
                                            Ok(delta) => {
                                                if !delta.trim().is_empty() {
                                                    let delta_norm =
                                                        normalize_punctuation_spacing(delta.trim());
                                                    if !delta_norm.is_empty() {
                                                        if !accumulated_transcript.is_empty()
                                                            && !accumulated_transcript
                                                                .ends_with(' ')
                                                        {
                                                            accumulated_transcript.push(' ');
                                                        }
                                                        accumulated_transcript
                                                            .push_str(&delta_norm);

                                                        let _ = tx
                                                            .send(WSMessage::Partial {
                                                                text: delta_norm,
                                                                accumulated_text:
                                                                    accumulated_transcript.clone(),
                                                                confidence: 0.92,
                                                                timestamp: start_time
                                                                    .elapsed()
                                                                    .as_secs_f64(),
                                                            })
                                                            .await;
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                let _ = tx
                                                    .send(WSMessage::Error {
                                                        message: format!(
                                                            "Nemotron inference error: {e}"
                                                        ),
                                                    })
                                                    .await;
                                            }
                                        }
                                    }
                                }
                            }

                            WSMessage::Reset => {
                                audio_buffer.clear();
                                accumulated_transcript.clear();
                                if let Some(ref mut model) = nemotron_model {
                                    model.reset();
                                }
                                vad.reset();
                                println!("WebSocket: Cache and Earshot VAD reset by client request");
                            }

                            WSMessage::Flush => {
                                // Pad remaining buffer with zeros to CHUNK_SIZE if any audio exists
                                if !audio_buffer.is_empty() {
                                    let mut final_chunk: Vec<f32> = audio_buffer.drain(..).collect();
                                    final_chunk.resize(CHUNK_SIZE, 0.0);

                                    if let Some(ref mut model) = nemotron_model {
                                        if let Ok(delta) = model.transcribe_chunk(&final_chunk) {
                                            let delta_norm =
                                                normalize_punctuation_spacing(delta.trim());
                                            if !delta_norm.is_empty() {
                                                if !accumulated_transcript.is_empty()
                                                    && !accumulated_transcript.ends_with(' ')
                                                {
                                                    accumulated_transcript.push(' ');
                                                }
                                                accumulated_transcript.push_str(&delta_norm);
                                            }
                                        }
                                    }
                                }

                                let final_text =
                                    normalize_punctuation_spacing(accumulated_transcript.trim());
                                if !final_text.is_empty() {
                                    let _ = tx
                                        .send(WSMessage::Final {
                                            text: final_text,
                                            confidence: 0.95,
                                            timestamp: start_time.elapsed().as_secs_f64(),
                                            processing_time: start_time.elapsed().as_secs_f64(),
                                        })
                                        .await;
                                }

                                if let Some(ref mut model) = nemotron_model {
                                    model.reset();
                                }
                                vad.reset();
                                accumulated_transcript.clear();
                            }

                            _ => {}
                        }
                    } else if message.is_binary() {
                        let _ = tx
                            .send(WSMessage::Error {
                                message:
                                    "Binary messages not supported. Use base64 encoded audio JSON."
                                        .to_string(),
                            })
                            .await;
                    } else if message.is_close() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    }
}
