//! HTTP throughput benchmark server: a simulated model served by axum, with and without batchinf.
//!
//! Two endpoints serve the same model. Each has its own simulated device, so they do not compete
//! for it, but they should be benchmarked one at a time so they do not compete for CPU:
//!
//! - `POST /predict` sends the request through [batchinf], which batches concurrent requests.
//! - `POST /predict/unbatched` runs every request as its own one-item call on the device. This is
//!   the naive server that shares one model across requests.
//!
//! A device call costs a large fixed overhead plus a small per-item cost, the shape that makes
//! batching pay off on a GPU/TPU. The device is guarded by a mutex, so neither endpoint can gain
//! throughput by running calls in parallel.
//!
//! Configured through environment variables (defaults in brackets): `PORT` [3000],
//! `BATCH_SIZE` [32], `BATCH_TIMEOUT_MS` [2], `POOL_SIZE` [1], `CALL_OVERHEAD_MS` [5],
//! `PER_ITEM_US` [50].

use anyhow::{Context, Result};
use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::post,
};
use batchinf::{BatcherConfig, Batchinf, BatchinfError, Predictor, get_batcher};
use serde::{Deserialize, Serialize};
use std::{
    convert::Infallible,
    num::{NonZeroU8, NonZeroU32},
    str::FromStr,
    sync::{Arc, Mutex},
    time::Duration,
};

/// Cost model of one call on the simulated device.
#[derive(Clone, Copy)]
struct Cost {
    call_overhead: Duration,
    per_item: Duration,
}

/// A single accelerator shared by every caller. Holding the lock means owning the device.
#[derive(Clone)]
struct Device {
    lock: Arc<Mutex<()>>,
    cost: Cost,
}

impl Device {
    fn new(cost: Cost) -> Self {
        Self {
            lock: Arc::default(),
            cost,
        }
    }

    /// Scores each input by summing its features, taking as long as a real call would.
    fn run(&self, inputs: &[Vec<f32>]) -> Vec<f32> {
        let _guard = self.lock.lock().unwrap();
        // Sleep rather than spin: the host thread waits on the device, it does not compute.
        std::thread::sleep(self.cost.call_overhead + self.cost.per_item * inputs.len() as u32);
        inputs.iter().map(|f| f.iter().sum()).collect()
    }
}

struct SimulatedModel {
    device: Device,
}

impl Predictor for SimulatedModel {
    type Input = Vec<f32>;
    type Output = f32;
    type Error = Infallible;

    fn predict_batch(&mut self, inputs: Vec<Vec<f32>>) -> Result<Vec<f32>, Infallible> {
        Ok(self.device.run(&inputs))
    }
}

#[derive(Deserialize)]
struct PredictRequest {
    features: Vec<f32>,
}

#[derive(Serialize)]
struct PredictResponse {
    score: f32,
}

#[derive(Clone)]
struct AppState {
    batcher: Batchinf<Vec<f32>, f32, Infallible>,
    unbatched: Device,
}

// Baseline unbatched endpoint.
async fn predict(State(state): State<AppState>, Json(req): Json<PredictRequest>) -> Response {
    match state.batcher.predict(req.features).await {
        Ok(score) => Json(PredictResponse { score }).into_response(),
        Err(BatchinfError::InferenceError(e)) => {
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
        Err(e) => (StatusCode::SERVICE_UNAVAILABLE, e.to_string()).into_response(),
    }
}

// Candidate endpoint wired with `Batchinf` harness.
async fn predict_unbatched(
    State(state): State<AppState>,
    Json(req): Json<PredictRequest>,
) -> Response {
    let device = state.unbatched.clone();
    match tokio::task::spawn_blocking(move || device.run(&[req.features])).await {
        Ok(scores) => Json(PredictResponse { score: scores[0] }).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

fn env_or<T: FromStr>(name: &str, default: T) -> Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    match std::env::var(name) {
        Ok(v) => v.parse().with_context(|| format!("invalid {name}: {v:?}")),
        Err(_) => Ok(default),
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let port: u16 = env_or("PORT", 3000)?;
    let batch_size: NonZeroU32 = env_or("BATCH_SIZE", NonZeroU32::new(32).unwrap())?;
    let batch_timeout = Duration::from_millis(env_or("BATCH_TIMEOUT_MS", 2)?);
    let pool_size: NonZeroU8 = env_or("POOL_SIZE", NonZeroU8::MIN)?;
    let cost = Cost {
        call_overhead: Duration::from_millis(env_or("CALL_OVERHEAD_MS", 5)?),
        per_item: Duration::from_micros(env_or("PER_ITEM_US", 50)?),
    };

    // Every connection holds a socket, and the default soft limit (often 256 on macOS) is too low
    // for a few hundred clients. Raise it as far as the hard limit allows.
    let nofile = rlimit::increase_nofile_limit(u64::MAX).context("raising open file limit")?;

    // Every worker's predictor shares the one device, so extra workers cannot add device time.
    let device = Device::new(cost);
    let factory = Arc::new(move |_worker_id: u8| {
        Ok::<_, Infallible>(SimulatedModel {
            device: device.clone(),
        })
    });
    let config = BatcherConfig::builder(batch_size, batch_timeout)
        .pool_size(pool_size)
        // Room for the next batch to accumulate while the current one is on the device.
        .queue_size(batch_size.saturating_mul(NonZeroU32::new(2).unwrap()))
        .build();
    let batcher = get_batcher(factory, config)
        .await
        .map_err(|e| anyhow::anyhow!("failed to start batcher: {e}"))?;

    let waiter = batcher.shutdown_waiter();

    let state = AppState {
        batcher,
        unbatched: Device::new(cost),
    };
    let app = Router::new()
        .route("/predict", post(predict))
        .route("/predict/unbatched", post(predict_unbatched))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    println!(
        "listening on http://{}  device: {:?} per call + {:?} per item  \
         batch_size={batch_size} batch_timeout={batch_timeout:?} pool_size={pool_size}  \
         open file limit={nofile}",
        listener.local_addr()?,
        cost.call_overhead,
        cost.per_item,
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;

    waiter.wait().await;

    Ok(())
}
