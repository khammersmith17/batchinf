//! This crate provides a harness for framework agonostic static inference batching. Static
//! inference batching is useful for incresing utilization of an external device and increasing
//! throughput of expensive ML inference.
//!
//! Static inference batching is the technique of queueing up many requests, and performing
//! inference requests across many examples, rather than per example.
//!
//! For high throughput ML services, single example inference can lead to lower throughput and low
//! device utilization.
//!
//! This crate requires using of the multi thread `tokio` runtime, and is built upon tokio
//! primitives. This crate also handles graceful termination, and panics during inference
//! computations by restarting a worker that has panicked.
//!
//! To use, implement a type to hold inference state, such as model weights or an instance of a
//! model type in the framework the model is implemented in. Then implemented the
//! [`Predictor`]('crate::predictor::Predictor`), trait on that type. This trait requires a defined
//! input type, defined output type, and an error type that could bubble up when a forward or
//! `predict` method errors. Then implemented the
//! [`predict_batch`](`crate::predictor::Predictor::predict_batch`], which takes in an exclusive
//! reference to the buffered inference input, and returns a Result<Output, Error>.
//!
//! Within a service endpoint handler [`Batchinf`](`crate::batcher::Batchinf`) can be used as
//! service state. This is cheap to clone throughout. This is how the inference workers are
//! interfaced with at an application level. This type handles all inference input dispatch, and
//! returning the inference result back to the caller. This will return a `Result<Predictor::Output,
//! BatchinfError>`.
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
