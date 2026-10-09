use std::sync::Arc;

mod language;
mod server;
mod vad_implementation;
mod websocket;

use server::{run_server, NemotronService};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("============================================================");
    println!(" Crane Nemotron 3.5 Cache-Aware Streaming ASR Service");
    println!(" Fine-tuned for Luganda ('lg') & English ('en')");
    println!(" Model: CraneAILabs/crane-nemo-asr-1.1 (via parakeet-rs)");
    println!("============================================================");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "crane-nemo-asr-1.1-onnx".to_string());

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let service = Arc::new(NemotronService::new(&model_path).await?);

    run_server(service, port).await;

    Ok(())
}
