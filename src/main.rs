mod embed;
mod errors;
mod health;

use actix_cors::Cors;
use actix_web::web::Data;
use actix_web::{middleware, App, HttpServer};
use std::env;
use tracing::info;
use tracing_subscriber::{fmt, layer::SubscriberExt, EnvFilter, Registry};

use errors::{Error, Result};

const NUM_WEB_WORKERS: usize = 10;

#[actix_web::main]
async fn main() -> Result<()> {
    tracing::subscriber::set_global_default(
        Registry::default().with(fmt::layer().compact()).with(
            EnvFilter::try_from_default_env()
                .or_else(|_| EnvFilter::try_new("debug"))
                .expect("failed to initialize tracing filter layer, exiting."),
        ),
    )
    .map_err(|err| Error::Server(err.to_string()))?;

    let config = embed::load_configurations()?;
    let host = env::var("HOSTNAME").unwrap_or("localhost".to_string());
    let port = env::var("PORT").unwrap_or("8080".to_string());
    let address = format!("{}:{}", host, port);

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
