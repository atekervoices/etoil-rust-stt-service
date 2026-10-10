use std::sync::Arc;

mod language;
mod server;
mod websocket;

use server::{run_server, NemotronService};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("============================================================");
    println!(" NVIDIA Nemotron 3.5 Cache-Aware Streaming ASR Service");
    println!(" Official Multilingual Streaming FastConformer RNN-Transducer");
    println!(" Model: nvidia/nemotron-3.5-asr-streaming-0.6b (via parakeet-rs)");
    println!("============================================================");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "nemotron-3.5-asr-streaming-0.6b-onnx".to_string());

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8099);

    let service = Arc::new(NemotronService::new(&model_path).await?);

    run_server(service, port).await;

    Ok(())
}
