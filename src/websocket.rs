use canary_rs::{Canary, StreamConfig};
use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use warp::ws::{WebSocket, Message as WarpMessage};

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

        while let Some(msg) = ws_rx.next().await {
            match msg {
                Ok(message) => {
                    if message.is_text() {
                        if let Ok(text) = message.to_str() {
                            if let Ok(ws_msg) = serde_json::from_str::<WSMessage>(&text) {
                                match ws_msg {
                                    WSMessage::Config { sample_rate: _sr, source_lang: sl, target_lang: tl } => {
                                        if let Some(sl) = sl { source_lang = sl; }
                                        if let Some(tl) = tl { target_lang = tl; }
                                        
                                        // Initialize stream state
                                        let model = self.model.read().await;
                                        let stream_cfg = StreamConfig::new()
                                            .with_window_duration(8.0)
                                            .with_step_duration(0.5)
                                            .with_emit_partial(true)
                                            .with_pad_partial(false)
                                            .with_stability_window(3);
                                        
                                        match model.stream(source_lang.clone(), target_lang.clone(), stream_cfg) {
                                            Ok(state) => stream_state = Some(state),
                                            Err(e) => {
                                                let _ = tx.send(WSMessage::Error {
                                                    message: format!("Failed to initialize stream: {}", e),
                                                }).await;
                                                continue;
                                            }
                                        }
                                    }
                                    WSMessage::Audio { data, sample_rate: audio_sr } => {
                                        // Decode base64 audio data
                                        if let Ok(audio_bytes) = base64::engine::general_purpose::STANDARD.decode(&data) {
                                            // Audio received successfully, processing...
                                            
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
                                            let min_samples = audio_sr as usize / 4; // 0.25 second minimum
                                            
                                            if stream_state.is_some() && audio_buffer.len() >= min_samples {
                                                let chunk: Vec<f32> = audio_buffer.drain(..min_samples).collect();
                                                
                                                if let Some(ref mut state) = stream_state {
                                                    let start_time = std::time::Instant::now();
                                                    match state.push_samples(&chunk, audio_sr as usize, 1) {
                                                        Ok(results) => {
                                                            let processing_time = start_time.elapsed().as_secs_f64();
                                                            for result in results {
                                                                // Send partial results with both full text and delta
                                                                let msg = WSMessage::Partial {
                                                                    text: result.delta_text.clone(),
                                                                    confidence: 0.8, // Placeholder confidence
                                                                    timestamp: processing_time,
                                                                };
                                                                let _ = tx.send(msg).await;
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
