use parakeet_rs::{Nemotron, NemotronMode};
use std::env;
use std::path::Path;
use std::time::Instant;

mod language;
mod vad_implementation;

use language::{CHUNK_SIZE, SAMPLE_RATE, SupportedLanguage};
use vad_implementation::normalize_punctuation_spacing;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage: {} <audio.wav> [language: lg|en|auto] [model_dir]", args[0]);
        println!("Example: {} sample.wav lg ./crane-nemo-asr-1.1-onnx", args[0]);
        println!();
        println!("Supported languages: lg (Luganda), en (English), auto (Auto-detect)");
        println!("Note: Shona ('sn') is excluded in this service configuration.");
        return Ok(());
    }

    let audio_path = &args[1];
    let lang_input = args.get(2).map(|s| s.as_str()).unwrap_or("auto");
    let model_dir = args
        .get(3)
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
    println!(" Crane Nemotron 3.5 Cache-Aware File Transcription");
    println!(" Audio file: {audio_path}");
    println!(" Target language: {} ({})", language.name(), language.code());
    println!(" Model directory: {model_dir}");
    println!(" Chunk size: 560ms (8960 samples @ 16kHz)");
    println!("============================================================");

    if !Path::new(model_dir).exists() {
        eprintln!(
            "Model directory '{model_dir}' not found. Please ensure the ONNX model files are in place."
        );
    }

    println!("Loading Nemotron model from '{model_dir}'...");
    let load_start = Instant::now();
    let mut model = Nemotron::from_pretrained(model_dir, None)?;
    println!("Model loaded in {:.2}s (Mode: {:?})", load_start.elapsed().as_secs_f64(), model.mode());

    if model.mode() == NemotronMode::Multilingual {
        println!(
            "Setting target language to '{}' (Nemotron prompt: '{}')...",
            language.name(),
            language.to_nemotron_lang()
        );
        model.set_target_lang(language.to_nemotron_lang())?;
    }

    // Read WAV audio
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();
    println!("Audio spec: {} Hz, {} channels, format: {:?}", spec.sample_rate, spec.channels, spec.sample_format);

    let mut samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
    };

    // Downmix to mono if stereo
    if spec.channels > 1 {
        samples = samples
            .chunks(spec.channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
            .collect();
    }

    let audio_duration = samples.len() as f64 / spec.sample_rate as f64;
    println!("Total audio duration: {:.2}s ({} samples)", audio_duration, samples.len());
    println!("\n--- Streaming Transcription Output ---");

    let transcribe_start = Instant::now();
    let mut full_transcript = String::new();

    // Stream through cache-aware 560ms chunks
    for (chunk_idx, chunk) in samples.chunks(CHUNK_SIZE).enumerate() {
        let chunk_data: Vec<f32> = if chunk.len() < CHUNK_SIZE {
            let mut padded = chunk.to_vec();
            padded.resize(CHUNK_SIZE, 0.0);
            padded
        } else {
            chunk.to_vec()
        };

        let text = model.transcribe_chunk(&chunk_data)?;
        if !text.trim().is_empty() {
            let norm = normalize_punctuation_spacing(text.trim());
            print!("{norm} ");
            std::io::Write::flush(&mut std::io::stdout())?;

            if !full_transcript.is_empty() && !full_transcript.ends_with(' ') {
                full_transcript.push(' ');
            }
            full_transcript.push_str(&norm);
        }
    }

    model.reset();
    println!("\n--------------------------------------");

    let elapsed = transcribe_start.elapsed().as_secs_f64();
    let rtf = if audio_duration > 0.0 { elapsed / audio_duration } else { 0.0 };

    println!("\n=== Transcription Complete ===");
    println!("Final Text: {}", normalize_punctuation_spacing(full_transcript.trim()));
    println!("Processing Time: {:.3}s", elapsed);
    println!("Audio Duration:  {:.3}s", audio_duration);
    println!("Real-Time Factor (RTF): {:.4}x (lower is faster)", rtf);

    Ok(())
}
