use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use warp::ws::{Message as WarpMessage, WebSocket};

use crate::language::SupportedLanguage;
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

        let mut active_language = SupportedLanguage::Luganda;
        if let Some(ref l) = initial_language {
            if let Ok(parsed) = SupportedLanguage::parse(l) {
                active_language = parsed;
            }
        }

        // Send ready handshake on connection
        let model_id = std::env::var("MODEL_ID")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "phosai/elgon-streaming-v1".to_string());
        let ready_msg = WSMessage::Ready {
            model: model_id,
            supported_languages: vec![
                "en (English)".to_string(),
                "lug (Luganda)".to_string(),
                "auto (Auto-detect)".to_string(),
            ],
        };
        if let Ok(ready_text) = serde_json::to_string(&ready_msg) {
            let _ = ws_tx.send(WarpMessage::text(ready_text)).await;
        }

        // Dedicated outgoing WebSocket transmitter task
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if let Ok(text) = serde_json::to_string(&msg) {
                    if let Err(e) = ws_tx.send(WarpMessage::text(text)).await {
                        eprintln!("[WebSocket Out] Error sending frame: {e}");
                        break;
                    }
                }
            }
        });

        // Connection state
        let mut audio_buffer: Vec<f32> = Vec::with_capacity(16000 * 30);
        let mut pre_speech_buffer: Vec<f32> = Vec::with_capacity(16000);
        let mut last_emitted_text = String::new();
        let mut last_emitted_final = String::new();
        let mut last_inference_time = std::time::Instant::now();
        let start_time = std::time::Instant::now();
        let mut sample_rate: u32 = 16000;
        let mut silent_samples: usize = 0;
        let mut speech_detected: bool = false;
        let mut noise_floor_rms: f32 = 0.004;
        let mut speech_peak_rms: f32 = 0.03;

        while let Some(msg) = ws_rx.next().await {
            let mut new_audio_samples: Vec<f32> = Vec::new();
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
                                sample_rate: sr,
                            } => {
                                if let Some(s) = sr {
                                    sample_rate = s;
                                }
                                let lang_str = language
                                    .or(source_lang)
                                    .or(target_lang)
                                    .unwrap_or_else(|| "lug".to_string());

                                match SupportedLanguage::parse(&lang_str) {
                                    Ok(lang) => {
                                        if active_language != lang {
                                            last_emitted_text.clear();
                                            last_emitted_final.clear();
                                        }
                                        active_language = lang;
                                        println!(
                                            "WebSocket: Configured language '{}' (target: '{}')",
                                            active_language.name(),
                                            active_language.to_nemotron_lang()
                                        );
                                    }
                                    Err(err) => {
                                        let _ = tx.send(WSMessage::Error { message: err }).await;
                                    }
                                }
                            }

                            WSMessage::Audio { data, sample_rate: sr } => {
                                if let Some(s) = sr {
                                    sample_rate = s;
                                }
                                if let Ok(audio_bytes) = base64::engine::general_purpose::STANDARD.decode(&data) {
                                    for chunk in audio_bytes.chunks_exact(2) {
                                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                                        new_audio_samples.push(sample);
                                    }
                                }
                            }

                            WSMessage::Reset { finalize } => {
                                let should_finalize = finalize.unwrap_or(true);
                                if should_finalize && !last_emitted_text.is_empty() {
                                    let _ = tx
                                        .send(WSMessage::Transcript {
                                            text: last_emitted_text.clone(),
                                            is_final: true,
                                            finalize: Some(true),
                                        })
                                        .await;
                                }
                                audio_buffer.clear();
                                pre_speech_buffer.clear();
                                last_emitted_text.clear();
                                last_emitted_final.clear();
                                speech_detected = false;
                                silent_samples = 0;
                                noise_floor_rms = 0.004;
                                speech_peak_rms = 0.03;
                                println!("WebSocket: Session reset by client request");
                            }

                            WSMessage::End | WSMessage::Flush => {
                                if !audio_buffer.is_empty() {
                                    let model = self.service.model.clone();
                                    let buf_clone = audio_buffer.clone();
                                    let lang_target = active_language.to_nemotron_lang().to_string();
                                    let sr = sample_rate;

                                    let transcript_res = tokio::task::spawn_blocking(move || {
                                        model.transcribe(&buf_clone, sr, &lang_target)
                                    }).await;

                                    if let Ok(Ok(transcript)) = transcript_res {
                                        let final_text = transcript.trim().to_string();
                                        if !final_text.is_empty() {
                                            let delta_text = if !last_emitted_final.is_empty() && final_text.starts_with(&last_emitted_final) {
                                                final_text[last_emitted_final.len()..].trim().to_string()
                                            } else {
                                                final_text.clone()
                                            };
                                            let text_to_emit = if delta_text.is_empty() { final_text.clone() } else { delta_text };
                                            last_emitted_final = final_text.clone();
                                            println!("[Final  | {}] {}", active_language.code(), text_to_emit);

                                            let _ = tx.send(WSMessage::Final {
                                                text: text_to_emit,
                                                confidence: 0.95,
                                                timestamp: start_time.elapsed().as_secs_f64(),
                                                processing_time: start_time.elapsed().as_secs_f64(),
                                            }).await;
                                        }
                                    }
                                }
                                audio_buffer.clear();
                                pre_speech_buffer.clear();
                                last_emitted_text.clear();
                                speech_detected = false;
                                silent_samples = 0;
                            }

                            _ => {}
                        }
                    } else if message.is_binary() {
                        let audio_bytes = message.as_bytes();
                        for chunk in audio_bytes.chunks_exact(2) {
                            let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                            new_audio_samples.push(sample);
                        }
                    } else if message.is_close() {
                        break;
                    }

                    // Process new incoming audio samples
                    if !new_audio_samples.is_empty() {
                        let rms = (new_audio_samples.iter().map(|&s| s * s).sum::<f32>() / new_audio_samples.len().max(1) as f32).sqrt();
                        let sr_usize = sample_rate as usize;
                        let pre_roll_max = (sr_usize * 3) / 10; // 300ms pre-speech rollback buffer

                        if !speech_detected {
                            let onset_thresh = (noise_floor_rms * 2.5).clamp(0.008, 0.035);
                            if rms >= onset_thresh {
                                speech_detected = true;
                                silent_samples = 0;
                                speech_peak_rms = rms.max(0.02);
                                audio_buffer.extend_from_slice(&pre_speech_buffer);
                                pre_speech_buffer.clear();
                                audio_buffer.extend_from_slice(&new_audio_samples);
                            } else {
                                noise_floor_rms = (0.92 * noise_floor_rms + 0.08 * rms).clamp(0.002, 0.020);
                                pre_speech_buffer.extend_from_slice(&new_audio_samples);
                                if pre_speech_buffer.len() > pre_roll_max {
                                    let excess = pre_speech_buffer.len() - pre_roll_max;
                                    pre_speech_buffer.drain(0..excess);
                                }
                            }
                        } else {
                            audio_buffer.extend_from_slice(&new_audio_samples);
                            let silence_thresh = (noise_floor_rms * 1.8)
                                .max(speech_peak_rms * 0.22)
                                .clamp(0.006, 0.030);

                            if rms >= silence_thresh {
                                silent_samples = 0;
                                speech_peak_rms = (0.85 * speech_peak_rms + 0.15 * rms).clamp(0.015, 0.25);
                            } else {
                                silent_samples += new_audio_samples.len();
                                noise_floor_rms = (0.97 * noise_floor_rms + 0.03 * rms).clamp(0.002, 0.020);
                            }
                        }

                        // Sentence pause detection:
                        // 1. Standard pause: ~0.64s of silence after speech (>= 0.45s total audio)
                        // 2. Micro-pause on long continuous speech: ~0.24s of silence when buffer >= 7s
                        // 3. Hard ceiling: 14s of continuous uninterrupted speech
                        let standard_pause = speech_detected
                            && silent_samples >= (sr_usize * 64 / 100)
                            && audio_buffer.len() >= (sr_usize * 45 / 100);
                        let long_speech_micro_pause = speech_detected
                            && audio_buffer.len() >= (sr_usize * 7)
                            && silent_samples >= (sr_usize * 24 / 100);
                        let max_buffer_exceeded = speech_detected && audio_buffer.len() >= (sr_usize * 14);

                        if standard_pause || long_speech_micro_pause || max_buffer_exceeded {
                            let model = self.service.model.clone();
                            let buf_clone = audio_buffer.clone();
                            let lang_target = active_language.to_nemotron_lang().to_string();
                            let sr = sample_rate;

                            let transcript_res = tokio::task::spawn_blocking(move || {
                                model.transcribe(&buf_clone, sr, &lang_target)
                            }).await;

                            if let Ok(Ok(transcript)) = transcript_res {
                                let final_text = transcript.trim().to_string();
                                if !final_text.is_empty() {
                                    println!("[Sentence | {}] {}", active_language.code(), final_text);
                                    let _ = tx.send(WSMessage::Final {
                                        text: final_text,
                                        confidence: 0.95,
                                        timestamp: start_time.elapsed().as_secs_f64(),
                                        processing_time: start_time.elapsed().as_secs_f64(),
                                    }).await;
                                }
                            }

                            // Clear buffer cleanly without overlapping already-emitted audio
                            audio_buffer.clear();
                            pre_speech_buffer.clear();
                            speech_detected = false;
                            silent_samples = 0;
                            last_emitted_text.clear();
                            last_inference_time = std::time::Instant::now();
                        }
                        // Streaming partial transcription update every ~280ms after previous inference finishes
                        else if speech_detected && audio_buffer.len() >= 3840 && last_inference_time.elapsed().as_millis() >= 280 {
                            let model = self.service.model.clone();
                            let buf_clone = audio_buffer.clone();
                            let lang_target = active_language.to_nemotron_lang().to_string();
                            let sr = sample_rate;

                            let transcript_res = tokio::task::spawn_blocking(move || {
                                model.transcribe(&buf_clone, sr, &lang_target)
                            }).await;
                            last_inference_time = std::time::Instant::now();

                            if let Ok(Ok(transcript)) = transcript_res {
                                let current_clean = transcript.trim().to_string();
                                if !current_clean.is_empty() && current_clean != last_emitted_text {
                                    let delta = if current_clean.starts_with(&last_emitted_text) {
                                        current_clean[last_emitted_text.len()..].trim().to_string()
                                    } else {
                                        current_clean.clone()
                                    };
                                    last_emitted_text = current_clean.clone();
                                    println!("[Stream | {}] {}", active_language.code(), current_clean);

                                    let _ = tx.send(WSMessage::Partial {
                                        text: delta,
                                        accumulated_text: current_clean,
                                        confidence: 0.95,
                                        timestamp: start_time.elapsed().as_secs_f64(),
                                    }).await;
                                }
                            }
                        }
                    }
                }
                Err(_) => break,
            }
        }
    }
}
