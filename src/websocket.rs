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
        let mut last_emitted_text = String::new();
        let mut last_emitted_final = String::new();
        let mut last_inference_time = std::time::Instant::now();
        let start_time = std::time::Instant::now();
        let mut sample_rate: u32 = 16000;
        let mut silent_samples: usize = 0;
        let mut dip_samples: usize = 0;
        let mut speech_detected: bool = false;
        let mut utterance_rms_ema: f32 = 0.02;

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
                                last_emitted_text.clear();
                                last_emitted_final.clear();
                                speech_detected = false;
                                silent_samples = 0;
                                dip_samples = 0;
                                utterance_rms_ema = 0.02;
                                println!("WebSocket: Session reset by client request");
                            }

                            WSMessage::End | WSMessage::Flush => {
                                if !audio_buffer.is_empty() {
                                    let model = self.service.model.clone();
                                    let buf_clone = audio_buffer.clone();
                                    let lang_target = active_language.to_nemotron_lang().to_string();
                                    let sr = sample_rate;

                                    let transcript_res = tokio::task::spawn_blocking(move || {
                                        model.transcribe_final(&buf_clone, sr, &lang_target)
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
                                last_emitted_text.clear();
                                speech_detected = false;
                                silent_samples = 0;
                                dip_samples = 0;
                            }

                            _ => {}
                        }
                    } else if message.is_binary() {
                        let audio_bytes = message.as_bytes();
                        // If a WAV container header is sent over WebSocket, parse header sample_rate and skip 44-byte RIFF header
                        let pcm_slice = if audio_bytes.starts_with(b"RIFF") && audio_bytes.len() > 44 {
                            let wav_sr = u32::from_le_bytes([audio_bytes[24], audio_bytes[25], audio_bytes[26], audio_bytes[27]]);
                            if (8000..=96000).contains(&wav_sr) {
                                sample_rate = wav_sr;
                            }
                            &audio_bytes[44..]
                        } else {
                            audio_bytes
                        };
                        for chunk in pcm_slice.chunks_exact(2) {
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
                        audio_buffer.extend(&new_audio_samples);

                        // Sensitive threshold (0.002) so quiet speech syllables (e.g. 0.0035 RMS) are always captured
                        if rms >= 0.002 {
                            if !speech_detected {
                                utterance_rms_ema = rms.max(0.01);
                            } else {
                                utterance_rms_ema = 0.88 * utterance_rms_ema + 0.12 * rms;
                            }
                            speech_detected = true;
                            silent_samples = 0;

                            // Detect natural breath/pause dips during continuous speech when ambient noise > 0.002
                            let dip_thresh = (utterance_rms_ema * 0.38).clamp(0.0025, 0.015);
                            if rms < dip_thresh {
                                dip_samples += new_audio_samples.len();
                            } else {
                                dip_samples = 0;
                            }
                        } else if speech_detected {
                            silent_samples += new_audio_samples.len();
                            dip_samples += new_audio_samples.len();
                        } else {
                            // While waiting for speech onset, keep a rolling 0.64s pre-roll buffer
                            // so idle room silence does not accumulate, without ever clipping word onsets
                            let pre_roll_samples = (sr_usize * 64) / 100;
                            if audio_buffer.len() > pre_roll_samples {
                                let excess = audio_buffer.len() - pre_roll_samples;
                                audio_buffer.drain(0..excess);
                            }
                        }

                        // Sentence & clause boundary detection (keeps segments in the 4-9s RNN-T LSTM sweet spot):
                        // 1. Standard quiet pause: ~0.85s of silence (rms < 0.002) after speech (>= 0.6s total audio)
                        // 2. Breath pause on medium/long speech: ~0.32s energy dip when buffer >= 6.0s
                        // 3. Smart Valley Split at 9.5s: if continuous speech reaches 9.5s without a pause,
                        //    split at the quietest 80ms inter-word valley between 6.2s and 8.8s and carry over the remainder!
                        let standard_pause = speech_detected
                            && silent_samples >= (sr_usize * 85 / 100)
                            && audio_buffer.len() >= (sr_usize * 60 / 100);
                        let breath_pause = speech_detected
                            && audio_buffer.len() >= (sr_usize * 6)
                            && dip_samples >= (sr_usize * 32 / 100);
                        let valley_split_needed = speech_detected && audio_buffer.len() >= (sr_usize * 95 / 10);

                        if standard_pause || breath_pause || valley_split_needed {
                            // Determine exact split point so we NEVER cut through the middle of an active word
                            let split_pos = if valley_split_needed && !standard_pause && !breath_pause {
                                let search_start = (sr_usize * 62 / 10).min(audio_buffer.len());
                                let search_end = (sr_usize * 88 / 10).min(audio_buffer.len());
                                let win = (sr_usize * 8 / 100).max(640); // 80ms window
                                let step = win / 2;
                                let mut best_idx = search_end;
                                let mut min_energy = f32::MAX;
                                if search_end > search_start + win {
                                    let mut idx = search_start;
                                    while idx + win <= search_end {
                                        let e: f32 = audio_buffer[idx..idx + win].iter().map(|&x| x * x).sum();
                                        if e < min_energy {
                                            min_energy = e;
                                            best_idx = idx + win / 2;
                                        }
                                        idx += step;
                                    }
                                }
                                best_idx
                            } else {
                                audio_buffer.len()
                            };

                            let model = self.service.model.clone();
                            // Include up to 160ms past split_pos for encoder right-context lookahead when valley-splitting
                            let transcribe_end = (split_pos + (sr_usize * 16 / 100)).min(audio_buffer.len());
                            let buf_clone = audio_buffer[..transcribe_end].to_vec();
                            let lang_target = active_language.to_nemotron_lang().to_string();
                            let sr = sample_rate;

                            let transcript_res = tokio::task::spawn_blocking(move || {
                                model.transcribe_final(&buf_clone, sr, &lang_target)
                            }).await;

                            if let Ok(Ok(transcript)) = transcript_res {
                                let final_text = transcript.trim().to_string();
                                if !final_text.is_empty() {
                                    println!("[Pass 2 Final | {}] {}", active_language.code(), final_text);
                                    let _ = tx.send(WSMessage::Final {
                                        text: final_text,
                                        confidence: 0.98,
                                        timestamp: start_time.elapsed().as_secs_f64(),
                                        processing_time: start_time.elapsed().as_secs_f64(),
                                    }).await;
                                }
                            }

                            if split_pos < audio_buffer.len() {
                                // Valley split: retain all audio spoken after split_pos (plus 80ms lead-in) for the next segment!
                                let drain_to = split_pos.saturating_sub(sr_usize * 8 / 100);
                                audio_buffer.drain(0..drain_to);
                                speech_detected = true;
                                silent_samples = 0;
                                dip_samples = 0;
                            } else {
                                // Pause split: keep the trailing 0.32s of the pause as pre-roll for the next sentence
                                let tail_keep = (sr_usize * 32 / 100).min(dip_samples.max(silent_samples)).min(audio_buffer.len());
                                if tail_keep > 0 {
                                    let drain_len = audio_buffer.len() - tail_keep;
                                    audio_buffer.drain(0..drain_len);
                                } else {
                                    audio_buffer.clear();
                                }
                                speech_detected = false;
                                silent_samples = 0;
                                dip_samples = 0;
                            }
                            last_emitted_text.clear();
                            last_inference_time = std::time::Instant::now();
                        }
                        // Pass 1: Fast streaming partial transcription update every ~240ms while speech is ongoing
                        else if speech_detected && audio_buffer.len() >= 2560 && last_inference_time.elapsed().as_millis() >= 240 {
                            let model = self.service.model.clone();
                            let buf_clone = audio_buffer.clone();
                            let lang_target = active_language.to_nemotron_lang().to_string();
                            let sr = sample_rate;

                            let transcript_res = tokio::task::spawn_blocking(move || {
                                model.transcribe_streaming(&buf_clone, sr, &lang_target)
                            }).await;
                            last_inference_time = std::time::Instant::now();

                            if let Ok(Ok(transcript)) = transcript_res {
                                let current_clean = transcript.trim().to_string();
                                if !current_clean.is_empty() && current_clean != last_emitted_text {
                                    let prev_stem = last_emitted_text.trim_end_matches(&['.', '?', '!', ','][..]).trim();
                                    let delta = if !prev_stem.is_empty() && current_clean.starts_with(prev_stem) {
                                        current_clean[prev_stem.len()..].trim().to_string()
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
