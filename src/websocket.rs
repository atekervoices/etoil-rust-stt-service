use canary_rs::{Canary, StreamConfig};
use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc, Mutex};
use warp::ws::{WebSocket, Message as WarpMessage};
use crate::vad_implementation::{VADState, VADResult, rms, normalize_punctuation_spacing};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WSMessage {
    #[serde(rename = "config")]
    Config {
        sample_rate: Option<u32>,
        source_lang: Option<String>,
        target_lang: Option<String>,
    },
    #[serde(rename = "audio")]
    Audio {
        data: String, // base64 encoded audio data
        sample_rate: u32,
    },
    #[serde(rename = "partial")]
    Partial {
        text: String,
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
    Ready,
}

pub struct WebSocketHandler {
    model: Arc<RwLock<Canary>>,
}

impl WebSocketHandler {
    pub fn new(model: Arc<RwLock<Canary>>) -> Self {
        Self { model }
    }

    pub async fn handle_websocket(&self, ws: WebSocket) {
        let (mut ws_tx, mut ws_rx) = ws.split();
        let (tx, mut rx) = mpsc::channel::<WSMessage>(100);

        // Send ready message
        if let Err(_) = ws_tx.send(WarpMessage::text(
            serde_json::to_string(&WSMessage::Ready).unwrap()
        )).await {
            return;
        }

        // Handle outgoing messages
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if let Ok(text) = serde_json::to_string(&msg) {
                    ws_tx.send(WarpMessage::text(text)).await.ok();
                }
            }
        });

        // Handle incoming messages
        let mut audio_buffer = VecDeque::new();
        let mut source_lang = "en".to_string();
        let mut target_lang = "en".to_string();
        let mut stream_state = None;
        let mut full_session = None;
        let mut sample_rate = 16000_usize;
        let mut channels = 1_usize;
        
        // Create VAD state for proper turn detection
        let vad_state = Arc::new(Mutex::new(VADState::new()));

        while let Some(msg) = ws_rx.next().await {
            match msg {
                Ok(message) => {
                    if message.is_text() {
                        if let Ok(text) = message.to_str() {
                            if let Ok(ws_msg) = serde_json::from_str::<WSMessage>(&text) {
                                match ws_msg {
                                    WSMessage::Config { sample_rate: _sr, source_lang: sl, target_lang: tl } => {
                                        println!("Server: Received config message - sample_rate: {:?}, source_lang: {:?}, target_lang: {:?}", _sr, sl, tl);
                                        if let Some(sl) = sl { source_lang = sl; }
                                        if let Some(tl) = tl { target_lang = tl; }
                                        
                                        // Initialize stream state
                                        let model = self.model.read().await;
                                        let stream_cfg = StreamConfig::new()
                                            .with_window_duration(0.5)  // Minimum viable window for fast TTFB (was 8.0)
                                            .with_step_duration(0.1)    // Very frequent updates (was 0.5)
                                            .with_emit_partial(true)
                                            .with_pad_partial(true)     // Enable padding for faster first result
                                            .with_stability_window(1);   // Minimum stability for speed
                                        
                                        // Initialize streaming state for partial results
                                        match model.stream(source_lang.clone(), target_lang.clone(), stream_cfg) {
                                            Ok(state) => {
                                                stream_state = Some(state);
                                                println!("Server: Stream initialized successfully");
                                            },
                                            Err(e) => {
                                                println!("Server: Failed to initialize stream: {}", e);
                                                let _ = tx.send(WSMessage::Error {
                                                    message: format!("Failed to initialize stream: {}", e),
                                                }).await;
                                                continue;
                                            }
                                        }
                                        
                                        // Initialize full session for final re-decode (like Deepgram/ElevenLabs)
                                        full_session = Some(model.session());
                                    }
                                    WSMessage::Audio { data, sample_rate: audio_sr } => {
                                        // Decode base64 audio data
                                        if let Ok(audio_bytes) = base64::engine::general_purpose::STANDARD.decode(&data) {
                                            // Audio received successfully, processing...
                                            println!("Server: Received audio chunk: {} bytes, sample_rate: {}", audio_bytes.len(), audio_sr);
                                            
                                            // Convert bytes to f32 samples (assuming 16-bit PCM)
                                            let mut samples = Vec::new();
                                            for chunk in audio_bytes.chunks_exact(2) {
                                                if chunk.len() == 2 {
                                                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                                                    samples.push(sample);
                                                }
                                            }
                                            
                                            audio_buffer.extend(samples);
                                            
                                            // Process if we have enough audio and stream is initialized
                                            let min_samples = audio_sr as usize / 10; // 0.1 second minimum for fast TTFB
                                            
                                            if stream_state.is_some() && audio_buffer.len() >= min_samples {
                                                let chunk: Vec<f32> = audio_buffer.drain(..min_samples).collect();
                                                
                                                // Update sample_rate and channels for final decode
                                                sample_rate = audio_sr as usize;
                                                
                                                // Process audio through VAD
                                                let mut vad = vad_state.lock().await;
                                                let vad_result = vad.process_chunk(&chunk, sample_rate);
                                                
                                                match vad_result {
                                                    VADResult::SpeechStarted => {
                                                        println!("Server: Speech started detected");
                                                        let _ = tx.send(WSMessage::SpeechStarted {
                                                            timestamp: std::time::Instant::now().elapsed().as_secs_f64(),
                                                        }).await;
                                                    }
                                                    VADResult::EndOfUtterance { audio, should_finalize } => {
                                                        println!("Server: End of utterance detected, should_finalize: {}", should_finalize);
                                                        
                                                        if should_finalize {
                                                            // Send speech_ended
                                                            let _ = tx.send(WSMessage::SpeechEnded {
                                                                timestamp: std::time::Instant::now().elapsed().as_secs_f64(),
                                                            }).await;
                                                            
                                                            // Re-decode full utterance for accurate final transcript (like Deepgram/ElevenLabs)
                                                            if let Some(ref mut session) = full_session {
                                                                let start_time = std::time::Instant::now();
                                                                match session.transcribe_samples(&audio, sample_rate, 1, &source_lang, &target_lang) {
                                                                    Ok(result) => {
                                                                        let processing_time = start_time.elapsed().as_secs_f64();
                                                                        let final_text = normalize_punctuation_spacing(result.text.trim());
                                                                        
                                                                        if !final_text.is_empty() {
                                                                            println!("Server: Final transcript: {}", final_text);
                                                                            let _ = tx.send(WSMessage::Final {
                                                                                text: final_text,
                                                                                confidence: 0.95, // Higher confidence for full re-decode
                                                                                timestamp: processing_time,
                                                                                processing_time,
                                                                            }).await;
                                                                        }
                                                                    }
                                                                    Err(e) => {
                                                                        println!("Server: Error in final decode: {}", e);
                                                                    }
                                                                }
                                                            }
                                                            
                                                            // Reset streaming state for next utterance
                                                            if let Some(ref mut state) = stream_state {
                                                                state.reset();
                                                            }
                                                            vad.accumulated_text.clear();
                                                        }
                                                    }
                                                    _ => {} // Silence or SpeechContinues - no special action needed
                                                }
                                                
                                                // Process through streaming for partial results
                                                if let Some(ref mut state) = stream_state {
                                                    let start_time = std::time::Instant::now();
                                                    
                                                    match state.push_samples(&chunk, sample_rate, 1) {
                                                        Ok(results) => {
                                                            let processing_time = start_time.elapsed().as_secs_f64();
                                                            for result in results {
                                                                let delta_text = result.delta_text.clone();
                                                                
                                                                if !delta_text.trim().is_empty() {
                                                                    // Update accumulated text with normalization
                                                                    let normalized = normalize_punctuation_spacing(&delta_text);
                                                                    if !normalized.is_empty() {
                                                                        // Send partial
                                                                        let _ = tx.send(WSMessage::Partial {
                                                                            text: normalized,
                                                                            confidence: 0.8,
                                                                            timestamp: processing_time,
                                                                        }).await;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        Err(e) => {
                                                            let _ = tx.send(WSMessage::Error {
                                                                message: format!("Processing error: {}", e),
                                                            }).await;
                                                        }
                                                    }
                                                }
                                            } else if stream_state.is_none() {
                                                let _ = tx.send(WSMessage::Error {
                                                    message: "Stream not initialized. Send config message first.".to_string(),
                                                }).await;
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    } else if message.is_binary() {
                        let _ = tx.send(WSMessage::Error {
                            message: "Binary messages not supported. Use base64 encoded audio.".to_string(),
                        }).await;
                    } else if message.is_close() {
                        break;
                    }
                }
                Err(_) => {
                    break;
                }
            }
        }
    }
}
