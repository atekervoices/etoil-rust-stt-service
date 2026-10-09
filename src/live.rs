use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parakeet_rs::{Nemotron, NemotronMode};
use std::env;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

mod language;
mod vad_implementation;

use language::{CHUNK_SIZE, SupportedLanguage};
use vad_implementation::{normalize_punctuation_spacing, EarshotVAD, VADResult};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let lang_input = args.get(1).map(|s| s.as_str()).unwrap_or("auto");
    let model_dir = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("crane-nemo-asr-1.1-onnx");

    let language = match SupportedLanguage::parse(lang_input) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    println!("============================================================");
    println!(" Crane Nemotron 3.5 Live Streaming ASR (Earshot Neural VAD)");
    println!(" Language: {} ({})", language.name(), language.code());
    println!(" Model directory: {model_dir}");
    println!(" Chunk size: 560ms (8960 samples @ 16kHz)");
    println!(" VAD Engine: Earshot (pyke.io, 16ms neural detector)");
    println!(" Note: Shona ('sn') is excluded.");
    println!("============================================================");

    println!("Loading Crane Nemotron model from '{model_dir}'...");
    let mut model = Nemotron::from_pretrained(model_dir, None)?;
    if model.mode() == NemotronMode::Multilingual {
        model.set_target_lang(language.to_nemotron_lang())?;
    }
    println!("Model loaded successfully! Initializing audio capture...");

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No audio input device (microphone) found")?;
    println!("Using input device: {}", device.name()?);

    let default_config = device.default_input_config()?;
    let sample_rate = default_config.sample_rate().0 as usize;
    let channels = default_config.channels() as usize;

    println!("Microphone format: {} Hz, {} channel(s)", sample_rate, channels);

    let audio_buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let running = Arc::new(AtomicBool::new(true));

    let audio_buf_clone = audio_buffer.clone();
    let running_clone = running.clone();

    // CPAL input stream
    let stream = match default_config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &default_config.into(),
            move |data: &[f32], _: &_| {
                if !running_clone.load(Ordering::SeqCst) {
                    return;
                }
                let mut buf = audio_buf_clone.lock().unwrap();
                if channels > 1 {
                    for frame in data.chunks(channels) {
                        let mono: f32 = frame.iter().sum::<f32>() / channels as f32;
                        buf.push(mono);
                    }
                } else {
                    buf.extend_from_slice(data);
                }
            },
            |err| eprintln!("Audio stream error: {err}"),
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &default_config.into(),
            move |data: &[i16], _: &_| {
                if !running_clone.load(Ordering::SeqCst) {
                    return;
                }
                let mut buf = audio_buf_clone.lock().unwrap();
                if channels > 1 {
                    for frame in data.chunks(channels) {
                        let mono: f32 = frame.iter().map(|&s| s as f32 / 32768.0).sum::<f32>()
                            / channels as f32;
                        buf.push(mono);
                    }
                } else {
                    for &s in data {
                        buf.push(s as f32 / 32768.0);
                    }
                }
            },
            |err| eprintln!("Audio stream error: {err}"),
            None,
        )?,
        _ => return Err("Unsupported audio sample format".into()),
    };

    stream.play()?;

    println!("\n Microphones listening... Speak now (Press Ctrl+C to stop)");
    println!("------------------------------------------------------------");

    let mut vad = EarshotVAD::new();
    let mut current_utterance = String::new();

    // Streaming processing loop
    while running.load(Ordering::SeqCst) {
        let chunk_opt = {
            let mut buf = audio_buffer.lock().unwrap();
            if buf.len() >= CHUNK_SIZE {
                Some(buf.drain(..CHUNK_SIZE).collect::<Vec<f32>>())
            } else {
                None
            }
        };

        if let Some(chunk) = chunk_opt {
            let vad_res = vad.process_chunk(&chunk);
            match vad_res {
                VADResult::SpeechStarted => {
                    print!("\n[Speech Detected] ");
                    std::io::stdout().flush()?;
                }
                VADResult::EndOfUtterance { should_finalize, .. } => {
                    if should_finalize && !current_utterance.trim().is_empty() {
                        println!(
                            "\n[Final Utterance]: {}\n",
                            normalize_punctuation_spacing(current_utterance.trim())
                        );
                        model.reset();
                        vad.reset();
                        current_utterance.clear();
                    }
                }
                _ => {}
            }

            match model.transcribe_chunk(&chunk) {
                Ok(delta) => {
                    if !delta.is_empty() {
                        print!("{delta}");
                        std::io::stdout().flush()?;
                        current_utterance.push_str(&delta);
                    }
                }
                Err(e) => {
                    eprintln!("\nInference error: {e}");
                }
            }
        } else {
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    Ok(())
}
