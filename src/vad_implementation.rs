//! High-performance Neural Voice Activity Detection (VAD) powered by Earshot (pyke.io).
//! Achieves 3,600x real-time factor (0.0003 RTF) with pure-Rust neural VAD.

use earshot::Detector;

/// 256 samples per frame (16ms @ 16kHz) for Earshot neural VAD
pub const EARSHOT_FRAME_SIZE: usize = 256;

#[derive(Debug, PartialEq, Eq)]
pub enum VADResult {
    Silence,
    SpeechStarted,
    SpeechContinues,
    EndOfUtterance { should_finalize: bool },
}

pub struct EarshotVAD {
    detector: Box<Detector>,
    pub in_utterance: bool,
    pub silence_frames: usize,
    pub speech_frames: usize,
    /// Number of consecutive silent 16ms frames required to conclude an utterance.
    /// 18 frames * 16ms ~= 288ms silence hangover for snappy conversational turn-taking.
    silence_hangover_frames: usize,
    /// Number of voiced 16ms frames required to confirm speech has started (e.g. 2 frames = 32ms).
    min_speech_frames: usize,
    pub total_speech_frames_in_utterance: usize,
}

impl EarshotVAD {
    pub fn new() -> Self {
        Self {
            detector: Detector::default_boxed(),
            in_utterance: false,
            silence_frames: 0,
            speech_frames: 0,
            silence_hangover_frames: 75, // ~1200ms of silence for natural conversational pauses
            min_speech_frames: 2,        // ~32ms of speech
            total_speech_frames_in_utterance: 0,
        }
    }

    /// Customize silence hangover duration (in milliseconds, e.g. 250ms - 500ms).
    pub fn with_silence_hangover_ms(mut self, ms: usize) -> Self {
        self.silence_hangover_frames = (ms / 16).max(1);
        self
    }

    /// Reset VAD state and neural internal buffers
    pub fn reset(&mut self) {
        self.detector.reset();
        self.in_utterance = false;
        self.silence_frames = 0;
        self.speech_frames = 0;
        self.total_speech_frames_in_utterance = 0;
    }

    /// Process an audio chunk (e.g. 560ms / 8960 samples @ 16kHz).
    /// Iterates across 16ms (256-sample) frames using Earshot neural detector.
    pub fn process_chunk(&mut self, chunk: &[f32]) -> VADResult {
        let mut chunk_had_speech_start = false;
        let mut chunk_had_utterance_end = false;

        // Process in 256-sample frames (16ms @ 16kHz)
        for frame in chunk.chunks_exact(EARSHOT_FRAME_SIZE) {
            let score = self.detector.predict_f32(frame);
            let is_voice = score >= 0.5;

            if is_voice {
                self.speech_frames += 1;
                self.silence_frames = 0;

                if !self.in_utterance && self.speech_frames >= self.min_speech_frames {
                    self.in_utterance = true;
                    chunk_had_speech_start = true;
                }

                if self.in_utterance {
                    self.total_speech_frames_in_utterance += 1;
                }
            } else {
                self.speech_frames = 0;

                if self.in_utterance {
                    self.silence_frames += 1;

                    if self.silence_frames >= self.silence_hangover_frames {
                        self.in_utterance = false;
                        self.silence_frames = 0;
                        chunk_had_utterance_end = true;
                    }
                }
            }
        }

        if chunk_had_speech_start {
            VADResult::SpeechStarted
        } else if chunk_had_utterance_end {
            let should_finalize = self.total_speech_frames_in_utterance >= self.min_speech_frames;
            self.total_speech_frames_in_utterance = 0;
            VADResult::EndOfUtterance { should_finalize }
        } else if self.in_utterance {
            VADResult::SpeechContinues
        } else {
            VADResult::Silence
        }
    }
}

impl Default for EarshotVAD {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalizes spacing around punctuation (e.g. "word , next" -> "word, next").
pub fn normalize_punctuation_spacing(text: &str) -> String {
    let mut result = text.to_string();

    // Remove space before punctuation
    result = result.replace(" ,", ",");
    result = result.replace(" .", ".");
    result = result.replace(" ?", "?");
    result = result.replace(" !", "!");
    result = result.replace(" ;", ";");
    result = result.replace(" :", ":");

    // Clean multiple spaces
    while result.contains("  ") {
        result = result.replace("  ", " ");
    }

    result.trim().to_string()
}
