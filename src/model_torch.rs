use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use tch::{CModule, Device, IValue, Kind, Tensor};
use tokenizers::Tokenizer;

pub struct NemotronTorchModel {
    encoder: Arc<CModule>,
    decoder: Arc<CModule>,
    joint: Arc<CModule>,
    tokenizer: Tokenizer,
    prompt_dictionary: HashMap<String, i64>,
    blank_id: i64,
    default_prompt_id: i64,
    device: Device,
}

impl NemotronTorchModel {
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let device = if tch::Cuda::is_available() {
            println!("CUDA detected! Initializing Nemotron models on GPU (cuda:0)...");
            Device::Cuda(0)
        } else {
            println!("CUDA not detected. Initializing Nemotron models on CPU...");
            Device::Cpu
        };
        Self::load_with_device(dir, device)
    }

    pub fn load_with_device<P: AsRef<Path>>(
        dir: P,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let dir = dir.as_ref();
        let enc_path = dir.join("encoder.pt");
        let dec_path = dir.join("decoder.pt");
        let joint_path = dir.join("joint.pt");
        let tok_path = dir.join("tokenizer.json");
        let prompt_dict_path = dir.join("prompt_dictionary.json");
        let config_path = dir.join("config.json");

        println!("Loading TorchScript models from {} onto {:?}...", dir.display(), device);
        let mut encoder = CModule::load_on_device(&enc_path, device)
            .map_err(|e| format!("Failed to load {}: {e}", enc_path.display()))?;
        let mut decoder = CModule::load_on_device(&dec_path, device)
            .map_err(|e| format!("Failed to load {}: {e}", dec_path.display()))?;
        let mut joint = CModule::load_on_device(&joint_path, device)
            .map_err(|e| format!("Failed to load {}: {e}", joint_path.display()))?;

        encoder.set_eval();
        decoder.set_eval();
        joint.set_eval();

        println!("Loading tokenizer from {}...", tok_path.display());
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| format!("Failed to load {}: {e}", tok_path.display()))?;

        let prompt_dictionary: HashMap<String, i64> = if prompt_dict_path.exists() {
            let file = File::open(&prompt_dict_path)?;
            let reader = BufReader::new(file);
            serde_json::from_reader(reader).unwrap_or_default()
        } else {
            HashMap::new()
        };

        // Read blank_token_id from config.json if available (e.g., 2048 for phosai/elgon-streaming-v1, or 13087 for base Nemotron)
        let config_blank_id: Option<i64> = if config_path.exists() {
            File::open(&config_path)
                .ok()
                .and_then(|f| serde_json::from_reader::<_, serde_json::Value>(BufReader::new(f)).ok())
                .and_then(|v| {
                    v.get("blank_token_id")
                        .and_then(|b| b.as_i64())
                        .or_else(|| v.get("blank_id").and_then(|b| b.as_i64()))
                        .or_else(|| v.get("vocab_size").and_then(|b| b.as_i64()))
                })
        } else {
            None
        };

        let blank_id = config_blank_id.unwrap_or_else(|| tokenizer.get_vocab_size(true) as i64);
        // Default prompt id is 63 for Luganda
        let default_prompt_id = *prompt_dictionary.get("lug").unwrap_or(&63);

        println!("NemotronTorchModel loaded successfully on {:?} (blank_id: {blank_id}, default_prompt: {default_prompt_id}).", device);

        Ok(Self {
            encoder: Arc::new(encoder),
            decoder: Arc::new(decoder),
            joint: Arc::new(joint),
            tokenizer,
            prompt_dictionary,
            blank_id,
            default_prompt_id,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Resolve language code to model prompt ID.
    /// Note: phosai/elgon-streaming-v1 is trained specifically on English (prompt 0) and Luganda (prompt 63).
    /// We match explicit English/Luganda/auto tags first so "auto" never maps to untrained prompt 101.
    pub fn resolve_prompt_id(&self, lang: &str) -> i64 {
        let trimmed = lang.trim();
        let normalized = trimmed.to_lowercase().replace('_', "-");

        match normalized.as_str() {
            "lug" | "luganda" | "ganda" | "oluganda" | "lg" | "lug-ug" | "lg-ug" => *self
                .prompt_dictionary
                .get("lug")
                .or_else(|| self.prompt_dictionary.get("lug-UG"))
                .unwrap_or(&63),
            "en" | "eng" | "english" | "en-us" | "eng-us" | "en-ug" | "eng-ug" | "en-gb"
            | "eng-gb" | "ugandan-english" | "us" | "american" | "british" | "commonwealth" => *self
                .prompt_dictionary
                .get("en-US")
                .or_else(|| self.prompt_dictionary.get("en"))
                .unwrap_or(&0),
            // Map "auto" to English (0) rather than untrained prompt ID 101
            "auto" | "" | "default" => *self
                .prompt_dictionary
                .get("en-US")
                .or_else(|| self.prompt_dictionary.get("en"))
                .unwrap_or(&0),
            other if other.starts_with("en-") || other.starts_with("eng-") => *self
                .prompt_dictionary
                .get("en-US")
                .or_else(|| self.prompt_dictionary.get("en"))
                .unwrap_or(&0),
            other if other.starts_with("lug-") || other.starts_with("lg-") => *self
                .prompt_dictionary
                .get("lug")
                .or_else(|| self.prompt_dictionary.get("lug-UG"))
                .unwrap_or(&63),
            _ => {
                if let Some(&id) = self.prompt_dictionary.get(trimmed) {
                    return id;
                }
                if let Some(&id) = self.prompt_dictionary.get(&normalized) {
                    return id;
                }
                self.default_prompt_id
            }
        }
    }

    /// Resample audio to 16,000 Hz if necessary using high-quality bandlimited sinc interpolation
    pub fn resample_to_16k(audio: &[f32], orig_sr: u32) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        if orig_sr == 16000 || audio.is_empty() {
            return Ok(audio.to_vec());
        }

        use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let mut resampler = SincFixedIn::<f32>::new(
            16000.0 / orig_sr as f64,
            2.0,
            params,
            audio.len(),
            1,
        )?;

        let waves_in = vec![audio.to_vec()];
        let waves_out = resampler.process(&waves_in, None)?;
        Ok(waves_out.into_iter().next().unwrap_or_default())
    }

    /// Transcribe an audio waveform with RNN-T greedy decoding
    pub fn transcribe(
        &self,
        raw_audio: &[f32],
        orig_sr: u32,
        language: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        if raw_audio.is_empty() {
            return Ok(String::new());
        }

        // 1. Ensure 16kHz sample rate
        let mut audio_16k = Self::resample_to_16k(raw_audio, orig_sr)?;

        // 2. Peak-normalize quiet microphone streams to match training waveform distribution (max_amp ~ 0.5-0.95)
        // NemotronAsrStreamingFeatureExtractor computes un-normalized log-mel spectrograms, so quiet mic audio
        // (e.g. max_amp 0.05-0.20) shifts log-mel energies down by 15-20 dB compared to recorded WAV files.
        let max_amp = audio_16k.iter().copied().fold(0.0f32, |m, s| m.max(s.abs()));
        if max_amp >= 0.006 && max_amp < 0.55 {
            let gain = (0.75 / max_amp).min(8.0);
            for s in &mut audio_16k {
                *s = (*s * gain).clamp(-1.0, 1.0);
            }
        }

        // 3. Append 320ms (5,120 samples = 4 encoder frames = 1 full lookahead chunk) of trailing silence
        // so the final speech frames in streaming buffers receive full right-context lookahead and conv tail.
        audio_16k.resize(audio_16k.len() + 5120, 0.0);

        let _guard = tch::no_grad_guard();

        // 4. Prepare raw audio waveform tensor: shape [1, N] on target device
        let wav_tensor = Tensor::from_slice(&audio_16k)
            .unsqueeze(0)
            .to_kind(Kind::Float)
            .to_device(self.device);
        let prompt_id = self.resolve_prompt_id(language);
        let prompt_tensor = Tensor::from_slice(&[prompt_id])
            .to_kind(Kind::Int64)
            .to_device(self.device);

        // 5. Run End-to-End Encoder: waveform [1, N] -> audio features [1, T, 640]
        let enc_out = self.encoder.forward_ts(&[wav_tensor, prompt_tensor])?;
        let num_frames = enc_out.size()[1];

        // 6. Initialize Decoder LSTM states (2 layers, hidden size 640) on target device
        let mut h = Tensor::zeros([2, 1, 640], (Kind::Float, self.device));
        let mut c = Tensor::zeros([2, 1, 640], (Kind::Float, self.device));
        let mut last_token = Tensor::from_slice(&[self.blank_id])
            .unsqueeze(0)
            .to_kind(Kind::Int64)
            .to_device(self.device);

        let (mut dec_out, new_h, new_c) = self.step_decoder(&last_token, &h, &c)?;
        h = new_h;
        c = new_c;

        // 7. RNN-T Greedy Decoding Loop
        let mut emitted_tokens: Vec<u32> = Vec::new();

        for t in 0..num_frames {
            let frame = enc_out.slice(1, t, t + 1, 1); // [1, 1, 640]
            for _ in 0..6 {
                let logits = self.joint.forward_ts(&[&dec_out, &frame])?; // [1, Vocab]
                let token = logits.argmax(-1, false).int64_value(&[]);

                if token == self.blank_id {
                    // Blank emitted -> advance acoustic frame, keep decoder state intact
                    break;
                }

                // Prevent multi-token RNN-T repetition loops (never emit the exact same subword token >2 times consecutively)
                if emitted_tokens.len() >= 2
                    && emitted_tokens[emitted_tokens.len() - 1] == token as u32
                    && emitted_tokens[emitted_tokens.len() - 2] == token as u32
                {
                    break;
                }

                // Non-blank emitted -> record token and advance decoder LSTM
                emitted_tokens.push(token as u32);
                last_token = Tensor::from_slice(&[token])
                    .unsqueeze(0)
                    .to_kind(Kind::Int64)
                    .to_device(self.device);
                let (next_dec_out, next_h, next_c) = self.step_decoder(&last_token, &h, &c)?;
                dec_out = next_dec_out;
                h = next_h;
                c = next_c;
            }
        }

        // 6. Decode token IDs into text
        let transcript = self.tokenizer.decode(&emitted_tokens, true)
            .map_err(|e| format!("Tokenizer decode error: {e}"))?;

        Ok(transcript)
    }

    fn step_decoder(
        &self,
        token: &Tensor,
        h: &Tensor,
        c: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), Box<dyn std::error::Error + Send + Sync>> {
        let ivalue = self.decoder.forward_is(&[
            IValue::from(token.shallow_clone()),
            IValue::from(h.shallow_clone()),
            IValue::from(c.shallow_clone()),
        ])?;

        match ivalue {
            IValue::Tuple(elements) if elements.len() == 3 => {
                let dec_out = match &elements[0] {
                    IValue::Tensor(t) => t.shallow_clone(),
                    _ => return Err("Expected Tensor at tuple[0]".into()),
                };
                let new_h = match &elements[1] {
                    IValue::Tensor(t) => t.shallow_clone(),
                    _ => return Err("Expected Tensor at tuple[1]".into()),
                };
                let new_c = match &elements[2] {
                    IValue::Tensor(t) => t.shallow_clone(),
                    _ => return Err("Expected Tensor at tuple[2]".into()),
                };
                Ok((dec_out, new_h, new_c))
            }
            _ => Err("Decoder forward did not return a 3-element tuple".into()),
        }
    }
}
