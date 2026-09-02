pub mod batcher;
pub mod config;
pub mod error;
pub mod observability;
pub(crate) mod pool;
pub mod predictor;
pub(crate) mod state;
pub(crate) mod worker;

use pool::WorkerPool;

use batcher::Batchinf;
pub use config::BatcherConfig;
use config::InnerConfig;
pub use predictor::Predictor;
pub use state::{WorkerSnapshot, WorkerStatus};
use state::{WorkerRef, WorkerState};
use std::sync::Arc;
use tokio::sync::mpsc::channel;
use worker::InferenceWorker;

pub(crate) type FunnelMessage<Input, Output, Error> =
    (Input, tokio::sync::oneshot::Sender<Result<Output, Error>>);

/*
* Implementation:
*   User implements the Predicter trait on a model type wrapper.
*
* The model runs in a dedicated tokio task.
* Inference inputs are buffered in the task, and inference is performed across all examples that
* are buffered. Inference occurs at either max buffer size, or timeout.
*
* Data is passed to this background thread through a channel and gets an associated oneshot::Sender.
* The calling thread enqueues the inference input and the oneshot::Sender, and waits on the
* oneshot::Receiver.
* */

fn init_worker_ref_pairs<P: Predictor + Send + Sync + 'static>(
    predictor: P,
    state: &[WorkerState],
    channel_size: usize,
    obs: Option<Arc<dyn observability::BatcherMetrics>>,
) -> (
    Vec<InferenceWorker<P>>,
    Vec<WorkerRef<P::Input, P::Output, P::Error>>,
) {
    let mut inf_workers = Vec::with_capacity(state.len());
    let mut worker_refs = Vec::with_capacity(state.len());
    for worker in state {
        let (tx, rx) = channel::<FunnelMessage<P::Input, P::Output, P::Error>>(channel_size);
        let inf_worker = InferenceWorker::new(worker.clone(), predictor.clone(), rx, obs.clone());
        let worker_ref = WorkerRef::new(worker.clone(), tx);
        inf_workers.push(inf_worker);
        worker_refs.push(worker_ref);
    }
    (inf_workers, worker_refs)
}

/// Creates a [`Batchinf`] handle backed by the provided [`Predictor`].
///
/// Inference requests submitted via [`Batchinf::predict`] are accumulated and dispatched as a
/// batch when either `config.batch_size` is reached or `config.batch_timeout` elapses, whichever
/// comes first. Requests are distributed across `config.pool_size` workers using load-aware
/// random-start routing.
///
/// # Runtime requirement
///
/// Workers use [`tokio::task::block_in_place`] for inference and require the multi-thread Tokio
/// runtime. Using `current_thread` will panic at inference time.
///
/// ```rust,ignore
/// #[tokio::main(flavor = "multi_thread")]
/// async fn main() { ... }
/// ```
///
/// # Parameters
///
/// - `predictor`: The inference backend. Cloned once per pool worker at startup.
/// - `config`: Batching and pool configuration. See [`BatcherConfig`].
/// - `observability`: Optional metrics hook. Pass `None` to disable. See [`observability::BatcherMetrics`].
pub fn get_batcher<P: Predictor + Send + Sync + 'static>(
    predictor: P,
    config: BatcherConfig,
    observability: Option<Arc<dyn observability::BatcherMetrics>>,
) -> Batchinf<P::Input, P::Output, P::Error> {
    let batch_size = config.batch_size.get();
    let pool_size = config.pool_size.get();

    let conf: InnerConfig = config.into();

    let workers: Vec<WorkerState> = (0..pool_size).map(|_| WorkerState::new(conf)).collect();

    let (inf_workers, worker_refs) = init_worker_ref_pairs(
        predictor,
        &workers,
        batch_size as usize,
        observability.clone(),
    );
    let pool = WorkerPool::new(worker_refs);

    for worker in inf_workers.into_iter() {
        worker.start();
    }

    Batchinf::new(pool, observability)
}
