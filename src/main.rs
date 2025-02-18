mod embed;
mod errors;
mod health;

use actix_cors::Cors;
use actix_web::web::Data;
use actix_web::{middleware, App, HttpServer};
use candle_core::Device;
use dotenvy::dotenv;
use std::env;
use tracing::info;

use errors::{Error, Result};

const NUM_WEB_WORKERS: usize = 10;

#[actix_web::main]
async fn main() -> Result<()> {
    dotenv().ok();

    tracing_subscriber::fmt::init();

    let device = Device::Cpu;

    let model_id =
        env::var("EMBED_MODEL_ID").unwrap_or("sentence-transformers/all-MiniLM-L6-v2".to_string());
    let model_revision = env::var("EMBED_MODEL_REVISION").unwrap_or("main".to_string());

    let config =
        embed::load_configurations(model_id.clone(), model_revision.clone(), device.clone())?;

    let host = env::var("HOSTNAME").unwrap_or("localhost".to_string());
    let port = env::var("PORT").unwrap_or("8080".to_string());
    let address = format!("{}:{}", host, port);

    info!("Device: {:?}", device);
    info!("Model ID: {:?}", model_id);
    info!("Model Revision: {:?}", model_revision);
    info!("Server Address: {:?}", address);

    HttpServer::new(move || {
        App::new()
            .wrap(middleware::Logger::default())
            .wrap(Cors::permissive())
            .app_data(Data::new(config.clone()))
            .service(embed::post_embeddings)
            .service(health::up)
    })
    .workers(NUM_WEB_WORKERS)
    .bind(address)
    .map_err(|err| Error::Server(err.to_string()))?
    .run()
    .await
    .map_err(|err| Error::Server(err.to_string()))
}
