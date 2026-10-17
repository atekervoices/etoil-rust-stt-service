use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use parakeet_rs::{Nemotron, NemotronMode};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::mpsc;
use warp::ws::{Message as WarpMessage, WebSocket};

use crate::language::{CHUNK_SIZE, SupportedLanguage};
use crate::server::NemotronService;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WSMessage {
    #[serde(rename = "config")]
    Config {
        sample_rate: Option<u32>,
        language: Option<String>,
        source_lang: Option<String>,
        target_lang: Option<String>,
        blank_penalty: Option<f32>,
    },
    #[serde(rename = "audio")]
    Audio {
        data: String, // base64 encoded 16-bit PCM audio data
        sample_rate: Option<u32>,
    },
    #[serde(rename = "reset")]
    Reset {
        #[serde(default)]
        finalize: Option<bool>,
    },
    #[serde(rename = "end")]
    End,
    #[serde(rename = "flush")]
    Flush,
    #[serde(rename = "transcript")]
    Transcript {
        text: String,
        is_final: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        finalize: Option<bool>,
    },
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
        supported_languages: Vec<String>,
    },
}

#[derive(Clone)]
pub struct WebSocketHandler {
    service: Arc<NemotronService>,
}

impl WebSocketHandler {
    pub fn new(service: Arc<NemotronService>) -> Self {
        Self { service }
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
            model: "phosai-streaming-voice-v1".to_string(),
            supported_languages: vec![
                "en (English US)".to_string(),
                "en-gb (English Commonwealth)".to_string(),
                "es (Spanish)".to_string(),
                "fr (French)".to_string(),
                "de (German)".to_string(),
                "it (Italian)".to_string(),
                "pt (Portuguese)".to_string(),
                "ru (Russian)".to_string(),
                "zh (Chinese)".to_string(),
                "ja (Japanese)".to_string(),
                "ko (Korean)".to_string(),
                "hi (Hindi)".to_string(),
                "ar (Arabic)".to_string(),
                "tr (Turkish)".to_string(),
                "sw (Swahili)".to_string(),
                "auto (Auto-detect)".to_string(),
            ],
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
        let mut last_emitted_text = String::new();
        let mut last_emitted_final = String::new();
        let mut silent_chunks_count: usize = 0;
        let start_time = std::time::Instant::now();

        // Initialize independent client-isolated Nemotron stream from shared GPU handle
        let mut model = Nemotron::from_shared(&self.service.handle);
        if model.mode() == NemotronMode::Multilingual {
            let _ = model.set_target_lang(active_language.to_nemotron_lang());
        }

        while let Some(msg) = ws_rx.next().await {
            let mut new_audio_received = false;
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
                                blank_penalty,
                                ..
                            } => {
                                if let Some(penalty) = blank_penalty {
                                    model.set_blank_penalty(penalty);
                                }
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

                                        // Reset model state and audio/transcript buffers for the new language
                                        audio_buffer.clear();
                                        accumulated_transcript.clear();
                                        last_emitted_text.clear();
                                        last_emitted_final.clear();
                                        silent_chunks_count = 0;
                                        model.reset();
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
                                new_audio_received = true;
                            }

                            WSMessage::Reset { finalize } => {
                                let should_finalize = finalize.unwrap_or(true);
                                if !should_finalize {
                                    // Soft reset (Modal pattern): return current text without clearing state
                                    let current_clean = accumulated_transcript.trim().to_string();
                                    let _ = tx
                                        .send(WSMessage::Transcript {
                                            text: current_clean,
                                            is_final: true,
                                            finalize: Some(false),
                                        })
                                        .await;
                                } else {
                                    // Hard reset: clear all buffers and reset model cache
                                    audio_buffer.clear();
                                    accumulated_transcript.clear();
                                    last_emitted_text.clear();
                                    last_emitted_final.clear();
                                    silent_chunks_count = 0;
                                    model.reset();
                                    println!("WebSocket: Model cache reset by client request");
                                }
                            }

                            WSMessage::End | WSMessage::Flush => {
                                // Process any remaining audio in the buffer
                                if !audio_buffer.is_empty() {
                                    let mut final_chunk: Vec<f32> = audio_buffer.drain(..).collect();
                                    final_chunk.resize(CHUNK_SIZE, 0.0);
                                    let _ = model.transcribe_chunk(&final_chunk);
                                }
                                // Flush trailing lookahead tokens (right-context flush)
                                let flush_chunk = vec![0.0f32; CHUNK_SIZE];
                                for _ in 0..2 {
                                    let _ = model.transcribe_chunk(&flush_chunk);
                                }

                                let model_transcript = model.get_transcript();
                                let final_text = if !model_transcript.is_empty() {
                                    model_transcript
                                } else {
                                    accumulated_transcript.trim().to_string()
                                };
                                model.reset();

                                if !final_text.is_empty() {
                                    // Server-side deduplication (Modal pattern):
                                    // Only emit the delta if final_text starts with last_emitted_final
                                    let delta_text = if !last_emitted_final.is_empty()
                                        && final_text.starts_with(&last_emitted_final)
                                    {
                                        final_text[last_emitted_final.len()..].trim().to_string()
                                    } else {
                                        final_text.clone()
                                    };

                                    let text_to_emit = if delta_text.is_empty() {
                                        final_text.clone()
                                    } else {
                                        delta_text
                                    };

                                    last_emitted_final = final_text.clone();
                                    println!("[Final  | {}] {}", active_language.code(), text_to_emit);

                                    let _ = tx
                                        .send(WSMessage::Transcript {
                                            text: text_to_emit.clone(),
                                            is_final: true,
                                            finalize: Some(true),
                                        })
                                        .await;

                                    let _ = tx
                                        .send(WSMessage::Final {
                                            text: text_to_emit,
                                            confidence: 0.95,
                                            timestamp: start_time.elapsed().as_secs_f64(),
                                            processing_time: start_time.elapsed().as_secs_f64(),
                                        })
                                        .await;
                                }
                                audio_buffer.clear();
                                accumulated_transcript.clear();
                                last_emitted_text.clear();
                                last_emitted_final.clear();
                                silent_chunks_count = 0;
                                model.reset();
                            }

                            _ => {}
                        }
                    } else if message.is_binary() {
                        // Support direct raw PCM 16-bit little-endian binary streaming
                        let audio_bytes = message.as_bytes();
                        for chunk in audio_bytes.chunks_exact(2) {
                            let sample =
                                i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                            audio_buffer.push_back(sample);
                        }
                        new_audio_received = true;
                    } else if message.is_close() {
                        break;
                    }

                    // Unified audio chunk processing
                    if new_audio_received {
                        while audio_buffer.len() >= CHUNK_SIZE {
                            let chunk: Vec<f32> = audio_buffer.drain(..CHUNK_SIZE).collect();

                            // Continuous stateful Nemotron streaming (native blank transducer)
                            let delta_res: Result<String, _> = model.transcribe_chunk(&chunk);

                            match delta_res {
                                Ok(delta) => {
                                    if !delta.is_empty() {
                                        silent_chunks_count = 0;
                                        accumulated_transcript.push_str(&delta);

                                        let current_clean = accumulated_transcript.trim().to_string();
                                        if !current_clean.is_empty() && current_clean != last_emitted_text {
                                            last_emitted_text = current_clean.clone();
                                            println!("[Stream | {}] {}", active_language.code(), current_clean);

                                            let _ = tx
                                                .send(WSMessage::Transcript {
                                                    text: current_clean.clone(),
                                                    is_final: false,
                                                    finalize: None,
                                                })
                                                .await;

                                            let _ = tx
                                                .send(WSMessage::Partial {
                                                    text: delta,
                                                    accumulated_text: current_clean,
                                                    confidence: 0.95,
                                                    timestamp: start_time.elapsed().as_secs_f64(),
                                                })
                                                .await;
                                        }
                                    } else {
                                        // Silence chunk: check for sentence boundary pause
                                        if !accumulated_transcript.is_empty() {
                                            silent_chunks_count += 1;
                                            // After ~800ms (5 chunks of 160ms) of pause after speech, finalize sentence
                                            if silent_chunks_count >= 5 {
                                                let final_sentence = accumulated_transcript.trim().to_string();
                                                if !final_sentence.is_empty() {
                                                    println!("[Sentence | {}] {}", active_language.code(), final_sentence);
                                                    let _ = tx
                                                        .send(WSMessage::Transcript {
                                                            text: final_sentence.clone(),
                                                            is_final: true,
                                                            finalize: Some(true),
                                                        })
                                                        .await;

                                                    let _ = tx
                                                        .send(WSMessage::Final {
                                                            text: final_sentence,
                                                            confidence: 0.95,
                                                            timestamp: start_time.elapsed().as_secs_f64(),
                                                            processing_time: start_time.elapsed().as_secs_f64(),
                                                        })
                                                        .await;
                                                }

                                                accumulated_transcript.clear();
                                                last_emitted_text.clear();
                                                last_emitted_final.clear();
                                                silent_chunks_count = 0;
                                                model.reset();
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = tx
                                        .send(WSMessage::Error {
                                            message: format!("Nemotron inference error: {e}"),
                                        })
                                        .await;
                                }
                            }
                        }
                    }
                }
                Err(_) => break,
            }
        }

        // Cleanup model cache when client disconnects
        model.reset();
    }
}
