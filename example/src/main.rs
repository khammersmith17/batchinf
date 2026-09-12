//! Image embedding service using CLIP ViT-B/32 and batchinf.
//!
//! Accepts raw image bytes at `POST /embed` and returns a 512-dimensional
//! embedding vector. Requests are accumulated and dispatched as a batch to
//! the model, improving throughput under concurrent load.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release
//! curl -X POST http://localhost:3000/embed \
//!     --data-binary @image.jpg \
//!     -H "Content-Type: application/octet-stream"
//! ```

use anyhow::{Context, Result};
use axum::{
    Router,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::post,
};
use batchinf::{BatcherConfig, BatchinfError, Predictor, get_batcher};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{ClipConfig, ClipModel};
use hf_hub::{Repo, RepoType, api::sync::Api};
use image::{DynamicImage, imageops::FilterType};
use serde::Serialize;
use std::{num::NonZeroU32, sync::Arc};

const IMAGE_SIZE: usize = 224;

// CLIP normalisation constants.
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

#[derive(Clone)]
struct ClipPredictor {
    model: Arc<ClipModel>,
    device: Device,
}

#[derive(Debug, Clone)]
struct EmbedError(String);

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for EmbedError {}

impl From<candle_core::Error> for EmbedError {
    fn from(e: candle_core::Error) -> Self {
        EmbedError(e.to_string())
    }
}

impl Predictor for ClipPredictor {
    type Input = Vec<f32>;
    type Output = Vec<f32>;
    type Error = EmbedError;

    // Example implementation of Predictor.
    fn predict_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let batch = inputs.len();
        let flat: Vec<f32> = inputs.iter().flat_map(|v| v.iter().copied()).collect();
        let pixel_values =
            Tensor::from_vec(flat, (batch, 3, IMAGE_SIZE, IMAGE_SIZE), &self.device)?;
        let embeddings = self.model.get_image_features(&pixel_values)?;
        Ok(embeddings.to_vec2::<f32>()?)
    }
}

// Resize to 224×224, convert to RGB, scale to [0, 1], and apply CLIP normalisation.
// Returns a CHW-ordered flat Vec<f32>.
fn preprocess(img: DynamicImage) -> Vec<f32> {
    let img = img
        .resize_exact(IMAGE_SIZE as u32, IMAGE_SIZE as u32, FilterType::Lanczos3)
        .to_rgb8();

    let mut data = Vec::with_capacity(3 * IMAGE_SIZE * IMAGE_SIZE);
    for c in 0..3 {
        for pixel in img.pixels() {
            let val = pixel[c] as f32 / 255.0;
            data.push((val - MEAN[c]) / STD[c]);
        }
    }
    data
}

#[derive(Serialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

type Batcher = batchinf::Batchinf<Vec<f32>, Vec<f32>, EmbedError>;

async fn embed(State(batcher): State<Batcher>, body: Bytes) -> impl IntoResponse {
    let img = match image::load_from_memory(&body) {
        Ok(img) => img,
        Err(_) => return (StatusCode::BAD_REQUEST, "failed to decode image").into_response(),
    };

    match batcher.predict(preprocess(img)).await {
        Ok(embedding) => Json(EmbedResponse { embedding }).into_response(),
        Err(BatchinfError::InferenceError(e)) => {
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
        Err(e) => (StatusCode::SERVICE_UNAVAILABLE, e.to_string()).into_response(),
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let device = Device::Cpu;

    println!("Downloading openai/clip-vit-base-patch32 from Hugging Face...");
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        "openai/clip-vit-base-patch32".to_string(),
        RepoType::Model,
        "refs/pr/15".to_string(),
    ));
    let weights = repo.get("model.safetensors").context("model weights")?;

    let config = ClipConfig::vit_base_patch32();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
    let model = ClipModel::new(vb, &config)?;

    let predictor = ClipPredictor {
        model: Arc::new(model),
        device,
    };

    let batcher = get_batcher(
        predictor,
        BatcherConfig {
            batch_size: NonZeroU32::new(32).unwrap(),
            batch_timeout: NonZeroU32::new(10).unwrap(),
            pool_size: NonZeroU32::new(1).unwrap(),
        },
        None,
    );

    let app = Router::new()
        .route("/embed", post(embed))
        .with_state(batcher);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("Listening on http://0.0.0.0:3000");
    axum::serve(listener, app).await?;

    Ok(())
}
