use crate::{
    batcher::Batchinf,
    config::BatcherConfig,
    config::InnerConfig,
    observability,
    pool::{FunnelMessage, WorkerPool},
    predictor::Predictor,
    state::{WorkerRef, WorkerState},
    worker::{InferenceWorker, InputReceiver, run_worker},
};
use std::sync::Arc;
use tokio::sync::mpsc::channel;

pub type PredictorFactory<P> = Arc<dyn Fn(usize) -> P + Send + Sync>;
type WorkerRefPairs<P> = (
    Vec<(InferenceWorker<P>, InputReceiver<P>)>,
    Vec<WorkerRef<<P as Predictor>::Input, <P as Predictor>::Output, <P as Predictor>::Error>>,
);

/*
* Implementation:
*   User implements the Predictor trait on a model type wrapper.
*
* The model runs in a dedicated tokio task.
* Inference inputs are buffered in the task, and inference is performed across all examples that
* are buffered. Inference occurs at either max buffer size, or timeout.
*
* Data is passed to this background thread through a channel and gets an associated oneshot::Sender.
* The calling thread enqueues the inference input and the oneshot::Sender, and waits on the
* oneshot::Receiver.
* */

fn init_worker_ref_pairs<P: Predictor + 'static>(
    predictor_factory: PredictorFactory<P>,
    state: &[WorkerState],
    channel_size: usize,
    obs: Option<Arc<dyn observability::BatcherMetrics>>,
) -> WorkerRefPairs<P> {
    let mut inf_workers = Vec::with_capacity(state.len());
    let mut worker_refs = Vec::with_capacity(state.len());
    for (i, worker) in state.iter().enumerate() {
        let (tx, rx) = channel::<FunnelMessage<P::Input, P::Output, P::Error>>(channel_size);
        let obs = obs.clone();
        let inf_worker =
            InferenceWorker::new(worker.clone(), Arc::clone(&predictor_factory), obs, i);
        let worker_ref = WorkerRef::new(worker.clone(), tx);
        inf_workers.push((inf_worker, rx));
        worker_refs.push(worker_ref);
    }
    (inf_workers, worker_refs)
}

/// Creates a [`Batchinf`] handle backed by the provided [`Predictor`].
///
/// Inference requests submitted via [`Batchinf::predict`] are accumulated and dispatched as a
/// batch when either `config.batch_size` is reached or `config.batch_timeout` elapses, whichever
/// comes first. Requests are distributed across `config.pool_size` workers using load-aware
/// round-robin routing.
///
/// # Threads
///
/// Each worker runs batch accumulation as a `tokio` task and inference on its own dedicated OS
/// thread, named `batchinf-worker-{id}`. Inference never blocks the async runtime, so both the
/// multi-thread and current-thread runtimes are supported.
///
/// # Parameters
///
/// - `predictor_factory`: Builds the predictor for a worker, given the worker's index
///   (`0..pool_size`). Called on that worker's inference thread at startup, and again to rebuild
///   the predictor after [`Predictor::predict_batch`] panics. Load model weights once outside the
///   factory and capture them in the closure, rather than loading inside it. Signal failure by
///   panicking: a factory that keeps failing marks its worker
///   [`PredictorStatus::Dead`](`crate::PredictorStatus::Dead`). This function still returns, the
///   router skips that worker, and the failure is visible in
///   [`Batchinf::pool_status`](`crate::Batchinf::pool_status`).
/// - `config`: Batching and pool configuration. See [`BatcherConfig`].
/// - `observability`: Optional metrics hook. Pass `None` to disable. See [`observability::BatcherMetrics`].
pub fn get_batcher<P: Predictor + 'static>(
    predictor_factory: PredictorFactory<P>,
    config: BatcherConfig,
    observability: Option<Arc<dyn observability::BatcherMetrics>>,
) -> Batchinf<P::Input, P::Output, P::Error> {
    let pool_size = config.pool_size.get();
    let channel_size = config.queue_size.get();

    let conf: InnerConfig = config.clone().into();

    let workers: Vec<WorkerState> = (0..pool_size).map(|_| WorkerState::new(conf)).collect();

    let (inf_workers, worker_refs) = init_worker_ref_pairs(
        Arc::clone(&predictor_factory),
        &workers,
        channel_size as usize,
        observability.clone(),
    );
    let pool: WorkerPool<P::Input, P::Output, P::Error> = WorkerPool::new(worker_refs);

    for (w, rx) in inf_workers {
        run_worker(w, rx);
    }

    Batchinf::new(pool, observability)
}
