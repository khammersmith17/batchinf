//! This crate provides a harness for framework agnostic static inference batching. Static
//! inference batching is useful for increasing utilization of an external device and increasing
//! throughput of expensive ML inference. Any framework can be used in this harness.
//!
//! Static inference batching is the technique of queueing up many requests, and performing
//! inference across many examples at once, rather than per example. This is useful for stateless
//! inference, things like image models, generating embeddings, etc. This crate does not currently
//! support continuous batching.
//!
//! For high throughput ML services, single example inference can lead to lower throughput and low
//! device utilization.
//!
//! This crate requires the multi-thread `tokio` runtime, and is built upon `tokio`
//! primitives. It handles graceful termination and restarts workers that panic during inference.
//!
//! To use, implement a type to hold inference state, such as model weights or an instance of a
//! model type in the framework the model is implemented in. Implement the [`Predictor`] trait on
//! that type. This trait requires a defined input type, output type, and an error type that can
//! bubble up when inference fails. Implement [`Predictor::predict_batch`], which takes a shared
//! reference to the buffered inference inputs and returns a `Result<Vec<Output>, Error>`.
//!
//! Within a service endpoint handler, [`Batchinf`] can be used as service state. It is cheap to
//! clone. This type handles all inference input dispatch and returns the inference result back to
//! the caller as a `Result<Output, `[`BatchinfError`]`>`.
//!
//! To add observability, implement the [`BatcherMetrics`] trait.
//!
//! # Quick Start
//!
//! ```no_run
//! use batchinf::{BatcherConfig, Predictor, get_batcher};
//! use std::num::NonZeroU32;
//!
//! #[derive(Clone)]
//! struct EchoModel;
//!
//! #[derive(Debug, Clone)]
//! struct ModelError;
//!
//! impl std::fmt::Display for ModelError {
//!     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//!         write!(f, "model error")
//!     }
//! }
//!
//! impl std::error::Error for ModelError {}
//!
//! impl Predictor for EchoModel {
//!     type Input = Vec<f32>;
//!     type Output = Vec<f32>;
//!     type Error = ModelError;
//!
//!     fn predict_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, ModelError> {
//!         // Run your model here. Each output must correspond to the input at the same index.
//!         Ok(inputs.to_vec())
//!     }
//! }
//!
//! #[tokio::main(flavor = "multi_thread")]
//! async fn main() {
//!     let batcher = get_batcher(
//!         EchoModel,
//!         BatcherConfig {
//!             batch_size: NonZeroU32::new(32).unwrap(),
//!             batch_timeout: NonZeroU32::new(10).unwrap(), // 10ms
//!             pool_size: NonZeroU32::new(1).unwrap(),
//!         },
//!         None,
//!     );
//!
//!     match batcher.predict(vec![1.0_f32, 2.0, 3.0]).await {
//!         Ok(embedding) => println!("{embedding:?}"),
//!         Err(e) => eprintln!("Error: {e}"),
//!     }
//! }
//! ```

pub(crate) mod batcher;
pub(crate) mod config;
pub(crate) mod control_plane;
pub(crate) mod error;
pub(crate) mod observability;
pub(crate) mod pool;
pub(crate) mod predictor;
pub(crate) mod setup;
pub(crate) mod state;
pub(crate) mod worker;

pub use batcher::Batchinf;
pub use config::BatcherConfig;
pub use error::BatchinfError;
pub use observability::{BatchTrigger, BatcherMetrics};
pub use predictor::Predictor;
pub use setup::get_batcher;
pub use state::{WorkerSnapshot, WorkerStatus};
