use crate::errors::{Error, Result};
use actix_web::{post, web, HttpResponse};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};
use tracing::{error, info};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EmbedRequest {
    pub(crate) texts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EmbedResponse {
    pub(crate) embeddings: Vec<Vec<f32>>,
    pub(crate) response_type: String,
    pub(crate) id: String,
    pub(crate) texts: Vec<String>,
    pub(crate) meta: HashMap<String, usize>,
}

#[derive(Clone)]
pub(crate) struct EmbedModelConfig<'a> {
    pub(crate) vb: Arc<VarBuilder<'a>>,
    pub(crate) config: Config,
    pub(crate) device: Device,
    pub(crate) tokenizer: Tokenizer,
}

#[post("/v1/embed")]
pub(crate) async fn post_embeddings(
    config: web::Data<EmbedModelConfig<'_>>,
    request: web::Json<EmbedRequest>,
) -> HttpResponse {
    let start = chrono::Utc::now();
    let config = config.into_inner().as_ref().clone();
    let request = request.into_inner();
    let response_type = "embeddings_floats".to_string();
    let id = Uuid::new_v4().to_string();
    let texts = request.texts.clone();
    let mut meta = HashMap::new();
    meta.insert("api_version".to_string(), 1);
    meta.insert("input_tokens".to_string(), texts.len());

    match generate_response(config, request).await {
        Ok(embeddings) => {
            let elapsed = (chrono::Utc::now() - start).num_milliseconds();
            let count = embeddings.len();
            let rate = elapsed / max(count as i64, 1);
            info!("generated {count} embeddings in {elapsed}ms (avg {rate}ms)",);
            HttpResponse::Ok().json(EmbedResponse {
                embeddings,
                response_type,
                id,
                texts,
                meta,
            })
        }
        Err(error) => {
            error!("failed to generate response: {error}");
            HttpResponse::InternalServerError().body(error.to_string())
        }
    }
}

pub(crate) fn load_configurations<'a>(
    model_id: String,
    model_revision: String,
    device: Device,
) -> Result<EmbedModelConfig<'a>> {
    let start = chrono::Local::now();
    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .map_err(Error::from)?;
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        model_revision,
    ));
    let (config_filename, tokenizer_filename, weights_filename) = {
        let config = repo.get("config.json").map_err(Error::from)?;
        let tokenizer = repo.get("tokenizer.json").map_err(Error::from)?;
        let weights = repo.get("model.safetensors").map_err(Error::from)?;
        (config, tokenizer, weights)
    };
    let config =
        std::fs::read_to_string(config_filename).map_err(|e| Error::Embed(e.to_string()))?;
    let config: Config = serde_json::from_str(&config).map_err(|e| Error::Embed(e.to_string()))?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(Error::from)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)
            .map_err(Error::from)?
    };
    let elapsed = (chrono::Local::now() - start).num_milliseconds();
    info!("loaded embed model configurations in {elapsed} ms",);
    Ok(EmbedModelConfig {
        vb: Arc::new(vb),
        config,
        tokenizer,
        device,
    })
}

async fn generate_response(
    config: EmbedModelConfig<'_>,
    request: EmbedRequest,
) -> Result<Vec<Vec<f32>>> {
    let model = BertModel::load(config.vb.as_ref().clone(), &config.config).map_err(Error::from)?;
    let mut tokenizer = config.tokenizer;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        ..Default::default()
    }));
    let tokens = tokenizer
        .encode_batch(request.texts, true)
        .map_err(Error::from)?;
    let token_ids = tokens
        .iter()
        .map(|t| Tensor::new(t.get_ids().to_vec(), &config.device).map_err(Error::from))
        .collect::<Result<Vec<_>>>()?;
    let token_ids = Tensor::stack(&token_ids, 0).map_err(Error::from)?;
    let token_type_ids = token_ids.zeros_like().map_err(Error::from)?;
    let embeddings = model
        .forward(&token_ids, &token_type_ids, None)
        .map_err(Error::from)?;
    let n_contents = embeddings.dims3().map_err(Error::from)?.0;
    let mut results = Vec::with_capacity(n_contents);
    for i in 0..n_contents {
        let embedding = embeddings.get(i).map_err(Error::from)?;
        let embedding = embedding.to_vec2::<f32>().unwrap().first().unwrap().clone();
        results.push(embedding);
    }
    Ok(results)
}
