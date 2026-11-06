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

    /// Run End-to-End Encoder with configurable chunk_size (e.g. 4 = 320ms Pass 1 streaming, 14 = 1120ms Pass 2 rescoring).
    /// Automatically falls back to 2-input forward_ts if an older encoder.pt without dynamic chunk_size is loaded.
    fn encode_waveform(
        &self,
        audio_16k: &[f32],
        language: &str,
        chunk_size: i64,
    ) -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
        let wav_tensor = Tensor::from_slice(audio_16k)
            .unsqueeze(0)
            .to_kind(Kind::Float)
            .to_device(self.device);
        let prompt_id = self.resolve_prompt_id(language);
        let prompt_tensor = Tensor::from_slice(&[prompt_id])
            .to_kind(Kind::Int64)
            .to_device(self.device);
        let chunk_tensor = Tensor::from_slice(&[chunk_size])
            .to_kind(Kind::Int64)
            .to_device(self.device);

        match self.encoder.forward_ts(&[&wav_tensor, &prompt_tensor, &chunk_tensor]) {
            Ok(enc_out) => Ok(enc_out),
            Err(_) => {
                let enc_out = self.encoder.forward_ts(&[&wav_tensor, &prompt_tensor])?;
                Ok(enc_out)
            }
        }
    }

    /// Pass 1 (Streaming Partials): Fast 320ms lookahead (chunk_size = 4) + RNN-T Greedy Decoding (beam = 1)
    pub fn transcribe_streaming(
        &self,
        raw_audio: &[f32],
        orig_sr: u32,
        language: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        if raw_audio.is_empty() {
            return Ok(String::new());
        }

        let audio_16k = Self::resample_to_16k(raw_audio, orig_sr)?;
        let _guard = tch::no_grad_guard();

        // Pass 1 uses chunk_size = 4 (320ms causal lookahead)
        let enc_out = self.encode_waveform(&audio_16k, language, 4)?;
        let num_frames = enc_out.size()[1];

        let mut h = Tensor::zeros([2, 1, 640], (Kind::Float, self.device));
        let mut c = Tensor::zeros([2, 1, 640], (Kind::Float, self.device));
        let mut last_token = Tensor::from_slice(&[self.blank_id])
            .unsqueeze(0)
            .to_kind(Kind::Int64)
            .to_device(self.device);

        let (mut dec_out, new_h, new_c) = self.step_decoder(&last_token, &h, &c)?;
        h = new_h;
        c = new_c;

        let mut emitted_tokens: Vec<u32> = Vec::new();

        for t in 0..num_frames {
            let frame = enc_out.slice(1, t, t + 1, 1);
            let mut prev_token_in_frame: Option<i64> = None;
            for _ in 0..6 {
                let logits = self.joint.forward_ts(&[&dec_out, &frame])?;
                let token = logits.argmax(-1, false).int64_value(&[]);

                if token == self.blank_id {
                    break;
                }
                if prev_token_in_frame == Some(token) {
                    break;
                }
                if emitted_tokens.len() >= 2
                    && emitted_tokens[emitted_tokens.len() - 1] == token as u32
                    && emitted_tokens[emitted_tokens.len() - 2] == token as u32
                {
                    break;
                }

                prev_token_in_frame = Some(token);
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

        let transcript = self.tokenizer.decode(&emitted_tokens, true)
            .map_err(|e| format!("Tokenizer decode error: {e}"))?;
        Ok(transcript)
    }

    /// Pass 2 (Endpoint Rescoring & Sync HTTP): Extended 1,120ms lookahead (chunk_size = 14) + RNN-T Beam Search (beam = 4)
    /// Resolves co-articulated subword ambiguities ("totroduce" -> "to introduce", "transction" -> "transcription", "uble" -> "double")
    pub fn transcribe_final(
        &self,
        raw_audio: &[f32],
        orig_sr: u32,
        language: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        if raw_audio.is_empty() {
            return Ok(String::new());
        }

        let audio_16k = Self::resample_to_16k(raw_audio, orig_sr)?;
        let _guard = tch::no_grad_guard();

        // Pass 2 uses chunk_size = 14 (13 lookahead frames = 1,120ms right-context across word boundaries)
        let enc_out = self.encode_waveform(&audio_16k, language, 14)?;
        let num_frames = enc_out.size()[1];

        const BEAM_WIDTH: usize = 4;
        const LENGTH_BONUS: f64 = 0.16;
        const PRUNE_MARGIN: f64 = 3.2;

        struct BeamState {
            tokens: Vec<u32>,
            dec_out: Tensor,
            h: Tensor,
            c: Tensor,
            score: f64,
        }

        #[inline]
        fn log_sum_exp(a: f64, b: f64) -> f64 {
            let m = a.max(b);
            m + (1.0 + (-(a - b).abs()).exp()).ln()
        }

        let h0 = Tensor::zeros([2, 1, 640], (Kind::Float, self.device));
        let c0 = Tensor::zeros([2, 1, 640], (Kind::Float, self.device));
        let blank_tok = Tensor::from_slice(&[self.blank_id])
            .unsqueeze(0)
            .to_kind(Kind::Int64)
            .to_device(self.device);
        let (init_dec_out, init_h, init_c) = self.step_decoder(&blank_tok, &h0, &c0)?;

        let mut beams = vec![BeamState {
            tokens: Vec::new(),
            dec_out: init_dec_out,
            h: init_h,
            c: init_c,
            score: 0.0,
        }];

        for t in 0..num_frames {
            let frame = enc_out.slice(1, t, t + 1, 1); // [1, 1, 640]
            let b_len = beams.len() as i64;

            // Evaluate Joint Network across all active beams in a single batched GPU kernel call
            let dec_refs: Vec<&Tensor> = beams.iter().map(|b| &b.dec_out).collect();
            let batch_dec = Tensor::cat(&dec_refs, 0); // [B, 1, 640]
            let batch_frame = frame.expand([b_len, 1, 640], true); // [B, 1, 640]
            let batch_logits = self.joint.forward_ts(&[&batch_dec, &batch_frame])?; // [B, Vocab]
            let batch_log_probs = batch_logits.log_softmax(-1, Kind::Float); // [B, Vocab]

            let mut next_beams: Vec<BeamState> = Vec::with_capacity(BEAM_WIDTH * 4);

            for (b_idx, beam) in beams.iter().enumerate() {
                let log_probs = batch_log_probs.select(0, b_idx as i64); // [Vocab]
                let blank_logp = log_probs.double_value(&[self.blank_id]);

                // 1. Blank transition: advance to frame t+1 keeping current decoder state
                next_beams.push(BeamState {
                    tokens: beam.tokens.clone(),
                    dec_out: beam.dec_out.shallow_clone(),
                    h: beam.h.shallow_clone(),
                    c: beam.c.shallow_clone(),
                    score: beam.score + blank_logp,
                });

                // 2. Non-blank transitions: inspect top-4 tokens
                let (top_vals, top_idxs) = log_probs.topk(4, -1, true, true);
                let mut expanded = 0usize;

                for k in 0..4 {
                    let tok = top_idxs.int64_value(&[k]);
                    if tok == self.blank_id {
                        continue;
                    }
                    let tok_logp = top_vals.double_value(&[k]);
                    if tok_logp < blank_logp - PRUNE_MARGIN {
                        break;
                    }

                    // Guard against >2 consecutive identical subword tokens
                    let tok_u32 = tok as u32;
                    if beam.tokens.len() >= 2
                        && beam.tokens[beam.tokens.len() - 1] == tok_u32
                        && beam.tokens[beam.tokens.len() - 2] == tok_u32
                    {
                        continue;
                    }

                    let tok_tensor = Tensor::from_slice(&[tok])
                        .unsqueeze(0)
                        .to_kind(Kind::Int64)
                        .to_device(self.device);
                    let (next_dec_out, next_h, next_c) = self.step_decoder(&tok_tensor, &beam.h, &beam.c)?;

                    // Score blank transition after emitting tok on frame t
                    let logits2 = self.joint.forward_ts(&[&next_dec_out, &frame])?;
                    let log_probs2 = logits2.log_softmax(-1, Kind::Float).squeeze_dim(0);
                    let blank2_logp = log_probs2.double_value(&[self.blank_id]);

                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(tok_u32);
                    let score1 = beam.score + tok_logp + LENGTH_BONUS + blank2_logp;

                    next_beams.push(BeamState {
                        tokens: new_tokens.clone(),
                        dec_out: next_dec_out.shallow_clone(),
                        h: next_h.shallow_clone(),
                        c: next_c.shallow_clone(),
                        score: score1,
                    });

                    // Allow a 2nd subword emission on the same 80ms frame if a non-blank token beats blank2_logp
                    if expanded == 0 {
                        let (top2_vals, top2_idxs) = log_probs2.topk(2, -1, true, true);
                        for k2 in 0..2 {
                            let tok2 = top2_idxs.int64_value(&[k2]);
                            if tok2 == self.blank_id || tok2 == tok {
                                continue;
                            }
                            let tok2_logp = top2_vals.double_value(&[k2]);
                            if tok2_logp > blank2_logp {
                                let tok2_tensor = Tensor::from_slice(&[tok2])
                                    .unsqueeze(0)
                                    .to_kind(Kind::Int64)
                                    .to_device(self.device);
                                let (dec_out3, h3, c3) = self.step_decoder(&tok2_tensor, &next_h, &next_c)?;
                                let logits3 = self.joint.forward_ts(&[&dec_out3, &frame])?;
                                let log_probs3 = logits3.log_softmax(-1, Kind::Float).squeeze_dim(0);
                                let blank3_logp = log_probs3.double_value(&[self.blank_id]);

                                let mut tokens2 = new_tokens.clone();
                                tokens2.push(tok2 as u32);
                                next_beams.push(BeamState {
                                    tokens: tokens2,
                                    dec_out: dec_out3,
                                    h: h3,
                                    c: c3,
                                    score: beam.score + tok_logp + tok2_logp + (2.0 * LENGTH_BONUS) + blank3_logp,
                                });
                            }
                        }
                    }

                    expanded += 1;
                    if expanded >= 3 {
                        break;
                    }
                }
            }

            // Recombine identical token sequences via log-sum-exp and keep top BEAM_WIDTH hypotheses
            let mut merged: Vec<BeamState> = Vec::with_capacity(next_beams.len());
            for cand in next_beams {
                if let Some(existing) = merged.iter_mut().find(|m| m.tokens == cand.tokens) {
                    existing.score = log_sum_exp(existing.score, cand.score);
                } else {
                    merged.push(cand);
                }
            }
            merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            merged.truncate(BEAM_WIDTH);
            beams = merged;
        }

        let best_tokens = beams
            .into_iter()
            .next()
            .map(|b| b.tokens)
            .unwrap_or_default();

        let transcript = self.tokenizer.decode(&best_tokens, true)
            .map_err(|e| format!("Tokenizer decode error: {e}"))?;
        Ok(transcript)
    }

    /// Default transcription entry point (uses Pass 2 Full-Lookahead + Beam Search)
    pub fn transcribe(
        &self,
        raw_audio: &[f32],
        orig_sr: u32,
        language: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.transcribe_final(raw_audio, orig_sr, language)
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
