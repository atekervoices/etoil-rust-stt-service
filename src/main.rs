use std::env;
use std::path::Path;
use std::time::Instant;

use etoil_rust_stt_service::model_torch::NemotronTorchModel;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage: {} <audio.wav> [language: lug|en|sw|auto] [model_dir]", args[0]);
        println!("Example: {} sample.wav lug ./torchscript_model", args[0]);
        return Ok(());
    }

    let audio_path = &args[1];
    let lang_input = args.get(2).map(|s| s.as_str()).unwrap_or("lug");
    let model_dir = args.get(3).map(|s| s.as_str()).unwrap_or("torchscript_model");

    println!("============================================================");
    println!(" NVIDIA Nemotron 3.5 High-Speed Torch-RS Transcription");
    println!(" Audio file: {audio_path}");
    println!(" Language:   {lang_input}");
    println!(" Model dir:  {model_dir}");
    println!("============================================================");

    if !Path::new(model_dir).exists() {
        eprintln!("Warning: Model directory '{model_dir}' not found locally.");
    }

    let load_start = Instant::now();
    let model = NemotronTorchModel::load(model_dir)?;
    println!("Model loaded in {:.2}s", load_start.elapsed().as_secs_f64());

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

    let transcribe_start = Instant::now();
    let transcript = model.transcribe(&samples, spec.sample_rate, lang_input)?;
    let elapsed = transcribe_start.elapsed().as_secs_f64();
    let rtf = if audio_duration > 0.0 { elapsed / audio_duration } else { 0.0 };

    println!("\n=== Transcription Complete ===");
    println!("Final Text: {}", transcript.trim());
    println!("Processing Time: {:.3}s", elapsed);
    println!("Audio Duration:  {:.3}s", audio_duration);
    println!("Real-Time Factor (RTF): {:.4}x (lower is faster)", rtf);

    Ok(())
}
