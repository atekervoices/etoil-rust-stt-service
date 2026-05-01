use std::sync::Arc;

mod server;
mod websocket;
use server::{CanaryService, run_server};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Canary STT API Server");
    println!("========================");

    // Load the model
    let service = Arc::new(CanaryService::new("canary-180m-flash-int8").await?);

    // Start the server
    let port = 8080;
    run_server(service, port).await;

    Ok(())
}
