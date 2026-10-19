use std::sync::Arc;
use etoil_rust_stt_service::server::{run_server, NemotronService};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("============================================================");
    println!(" NVIDIA Nemotron 3.5 TorchScript ASR Service (tch-rs)");
    println!(" High-performance native C++/Rust multi-threaded inference");
    println!(" Model: TorchScript FastConformer RNN-Transducer");
    println!("============================================================");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "torchscript_model".to_string());

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8099);

    let service = Arc::new(NemotronService::new(&model_path).await?);

    run_server(service, port).await;

    Ok(())
}
