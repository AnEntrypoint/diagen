use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,diagen=debug")))
        .init();

    tracing::info!(version = env!("CARGO_PKG_VERSION"), "diagen starting");
    tracing::info!(state = diagen_gate::State::Listening.name(), "gate initial state");
    tracing::info!(port = diagen_server::DEFAULT_PORT, "default server port");

    Ok(())
}
