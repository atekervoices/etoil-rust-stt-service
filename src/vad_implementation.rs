// Proper VAD implementation for WebSocket server
// Based on canary-rs streaming example

use std::sync::Arc;
use tokio::sync::Mutex;

pub struct VADState {
    pub in_utterance: bool,
    pub silence_samples: usize,
    pub noise_floor: f32,
    pub utterance_audio: Vec<f32>,
    pub accumulated_text: String,
    pub speech_active: bool,
    pub last_audio_time: std::time::Instant,
}

impl VADState {
    pub fn new() -> Self {
        Self {
            in_utterance: false,
            silence_samples: 0,
            noise_floor: 0.0,
            utterance_audio: Vec::new(),
            accumulated_text: String::new(),
            speech_active: false,
            last_audio_time: std::time::Instant::now(),
        }
    }

    pub fn process_chunk(&mut self, chunk: &[f32], sample_rate: usize) -> VADResult {
        let chunk_rms = rms(chunk);
        
        // Update noise floor
        if self.noise_floor == 0.0 {
            self.noise_floor = chunk_rms;
        } else if chunk_rms < self.noise_floor * 1.5 {
            self.noise_floor = self.noise_floor * 0.95 + chunk_rms * 0.05;
        }
        
        let min_silence_threshold = 0.0008_f32;
        let max_silence_threshold = 0.02_f32;
        let silence_threshold = (self.noise_floor * 3.0)
            .clamp(min_silence_threshold, max_silence_threshold);
        
        let silence_hold_seconds = 0.5_f32;  // Reduced from 0.8 for faster final detection
        let silence_hold_samples = ((sample_rate as f32 * silence_hold_seconds).round() as usize).max(1);
        
        let min_utterance_seconds = 0.3_f32;
        let min_utterance_samples = ((sample_rate as f32 * min_utterance_seconds).round() as usize).max(1);
        
        self.last_audio_time = std::time::Instant::now();
        
        if chunk_rms < silence_threshold {
            // Silence detected
            if self.in_utterance {
                self.silence_samples = self.silence_samples.saturating_add(chunk.len());
                self.utterance_audio.extend_from_slice(chunk);
                
                if self.silence_samples >= silence_hold_samples {
                    // End of utterance
                    let should_finalize = self.utterance_audio.len() >= min_utterance_samples;
                    
                    self.in_utterance = false;
                    self.speech_active = false;
                    self.silence_samples = 0;
                    
                    return VADResult::EndOfUtterance {
                        audio: std::mem::take(&mut self.utterance_audio),
                        should_finalize,
                    };
                }
            }
            return VADResult::Silence;
        }
        
        // Speech detected
        self.silence_samples = 0;
        
        let was_inactive = !self.in_utterance;
        self.in_utterance = true;
        self.speech_active = true;
        self.utterance_audio.extend_from_slice(chunk);
        
        if was_inactive {
            return VADResult::SpeechStarted;
        }
        
        VADResult::SpeechContinues
    }
}

#[derive(Debug)]
pub enum VADResult {
    Silence,
    SpeechStarted,
    SpeechContinues,
    EndOfUtterance { audio: Vec<f32>, should_finalize: bool },
}

pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|v| v * v).sum();
    (sum / samples.len() as f32).sqrt()
}

pub fn normalize_punctuation_spacing(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    
    for ch in text.chars() {
        if ch.is_whitespace() {
            pending_space = true;
            continue;
        }
        
        if is_tight_punct(ch) {
            if out.ends_with(' ') {
                out.pop();
            }
            out.push(ch);
            pending_space = false;
            continue;
        }
        
        if pending_space && !out.is_empty() {
            out.push(' ');
        }
        out.push(ch);
        pending_space = false;
    }
    
    out
}

pub fn is_tight_punct(ch: char) -> bool {
    matches!(ch, '.' | ',' | '!' | '?' | ';' | ':')
}
