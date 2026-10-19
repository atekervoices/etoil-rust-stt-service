use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use etoil_rust_stt_service::language::SupportedLanguage;
use etoil_rust_stt_service::model_torch::NemotronTorchModel;
use std::env;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let lang_input = args.get(1).map(|s| s.as_str()).unwrap_or("lug");
    let model_dir = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("torchscript_model");

    let language = match SupportedLanguage::parse(lang_input) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    println!("============================================================");
    println!(" NVIDIA Nemotron 3.5 Live Microphone ASR (tch-rs)");
    println!(" Language: {} ({})", language.name(), language.code());
    println!(" Model directory: {model_dir}");
    println!("============================================================");

    println!("Loading TorchScript model from '{model_dir}'...");
    let model = NemotronTorchModel::load(model_dir)?;
    println!("Model loaded successfully! Initializing audio capture...");

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No audio input device (microphone) found")?;
    println!("Using input device: {}", device.name()?);

    let default_config = device.default_input_config()?;
    let sample_rate = default_config.sample_rate();
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

    let mut last_transcript = String::new();
    let target_lang = language.to_nemotron_lang();

    // Live processing loop
    while running.load(Ordering::SeqCst) {
        let (audio_chunk, should_transcribe) = {
            let buf = audio_buffer.lock().unwrap();
            if buf.len() >= 8000 { // at least 0.5s of audio
                (buf.clone(), true)
            } else {
                (Vec::new(), false)
            }
        };

        if should_transcribe {
            match model.transcribe(&audio_chunk, sample_rate, target_lang) {
                Ok(text) => {
                    let clean = text.trim();
                    if !clean.is_empty() && clean != last_transcript {
                        print!("\rTranscript: {clean}");
                        std::io::stdout().flush()?;
                        last_transcript = clean.to_string();
                    }
                }
                Err(e) => {
                    eprintln!("\nInference error: {e}");
                }
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    Ok(())
}
