use crate::{
    batcher::Batchinf,
    config::BatcherConfig,
    config::InnerConfig,
    error::StartupError,
    observability,
    pool::{FunnelMessage, WorkerPool},
    predictor::{Predictor, PredictorFactory},
    shutdown::shutdown_channel,
    state::{WorkerRef, WorkerState},
    worker::{InferenceWorker, InputReceiver, PredictorStartConfig, RunnerHandoff, run_worker},
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    select,
    sync::{
        Notify,
        mpsc::{Receiver, channel},
    },
    time::sleep,
};

type WorkerRefPairs<P> = (
    Vec<(InferenceWorker, InputReceiver<P>)>,
    Vec<WorkerRef<<P as Predictor>::Input, <P as Predictor>::Output, <P as Predictor>::Error>>,
);

/*
* Implementation:
*   User implements the Predictor trait on a model type wrapper, and passes a factory that builds
*   one predictor per worker.
*
* Each worker has two parts:
* - An accumulator tokio task, which pulls requests from the worker's channel into a batch until
*   the batch is full or its timeout fires, then hands the batch to the inference thread.
* - A dedicated OS inference thread, which builds the predictor from the factory, runs
*   predict_batch on each batch, sends each result back, and rebuilds the predictor after a panic.
*
* The caller enqueues the input together with a oneshot::Sender on a worker's channel, chosen by
* the router in `pool`, and waits on the matching oneshot::Receiver.
* */

fn init_worker_ref_pairs<P: Predictor + 'static>(
    state: &[WorkerState],
    channel_size: usize,
    obs: Option<Arc<dyn observability::BatcherMetrics>>,
    signal: &Arc<Notify>,
) -> WorkerRefPairs<P> {
    let mut inf_workers = Vec::with_capacity(state.len());
    let mut worker_refs = Vec::with_capacity(state.len());
    for (i, worker) in state.iter().enumerate() {
        let (tx, rx) = channel::<FunnelMessage<P::Input, P::Output, P::Error>>(channel_size);
        let obs = obs.clone();
        let inf_worker = InferenceWorker::new(worker.clone(), obs, i as u8, Arc::clone(signal));
        let worker_ref = WorkerRef::new(worker.clone(), tx);
        inf_workers.push((inf_worker, rx));
        worker_refs.push(worker_ref);
    }
    (inf_workers, worker_refs)
}

// Waits for every worker to report the outcome of its first predictor build.
async fn poll_acks<E>(
    mut rxs: Receiver<Result<(), StartupError<E>>>,
    pool_size: u8,
) -> Result<(), StartupError<E>> {
    let mut acked = 0_u8;
    while let Some(ack) = rxs.recv().await {
        // Fail on the first worker that could not start.
        ack?;
        acked += 1;
        // Done once every worker has reported. No need to wait for the channel to close.
        if acked == pool_size {
            return Ok(());
        }
    }
    // The channel closed before every worker reported, so a worker stopped without acking.
    Err(StartupError::InternalError)
}

async fn collect_acks<E>(
    rxs: Receiver<Result<(), StartupError<E>>>,
    pool_size: u8,
    timeout: Duration,
) -> Result<(), StartupError<E>> {
    select!(
         res = poll_acks(rxs, pool_size) => {
            res
        }
        _ = sleep(timeout) => {
            Err(StartupError::Timeout)
        }
    )
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
///   returning `Err`; a panic also counts as a failure. A failed call is retried as configured by
///   [`BatcherConfig::worker_start_retries`]. If the factory keeps failing at startup, this
///   function returns an error. If it keeps failing during a later rebuild, that worker is
///   marked [`PredictorStatus::Dead`](`crate::PredictorStatus::Dead`), the router skips it, and
///   the failure is visible in [`Batchinf::pool_status`](`crate::Batchinf::pool_status`).
/// - `config`: Batching and pool configuration, including the optional metrics hook
///   ([`BatcherConfig::metrics`]). See [`BatcherConfig`].
///
/// # Errors
///
/// Returns once every worker has built its predictor, or as soon as startup fails:
///
/// - [`StartupError::WorkerStartFailure`] — a worker's factory returned `Err` on its last
///   attempt. Carries that error.
/// - [`StartupError::WorkerStartPanic`] — a worker's factory panicked on its last attempt.
/// - [`StartupError::Timeout`] — not every worker was ready within
///   [`BatcherConfig::startup_timeout`].
/// - [`StartupError::InternalError`] — a worker stopped before reporting, for example because
///   its inference thread could not be spawned.
///
/// On error the workers that did start are shut down.
pub async fn get_batcher<P: Predictor + 'static, E: Send + 'static>(
    predictor_factory: PredictorFactory<P, E>,
    config: BatcherConfig,
) -> Result<Batchinf<P::Input, P::Output, P::Error>, StartupError<E>> {
    // Assert that input and output are not ZST at compile time. This will break in the accumulator and predictor
    // threads.
    const {
        assert!(
            std::mem::size_of::<P::Input>() != 0,
            "Predictor::Input cannot be zero sized"
        );
    }
    let start_ts = Instant::now();
    let pool_size = config.pool_size.get();
    let channel_size = config.queue_size.get();
    let observability = config.metrics.clone();

    let conf: InnerConfig = config.clone().into();

    let workers: Vec<WorkerState> = (0..pool_size).map(|_| WorkerState::new(conf)).collect();

    let signal = Arc::new(Notify::new());

    let (inf_workers, worker_refs) = init_worker_ref_pairs::<P>(
        &workers,
        channel_size as usize,
        observability.clone(),
        &signal,
    );

    let pool: WorkerPool<P::Input, P::Output, P::Error> =
        WorkerPool::new(worker_refs, Arc::clone(&signal));

    let start = PredictorStartConfig {
        retries: config.worker_start_retries,
        backoff: config.worker_start_backoff_start,
        max_wait: config.max_worker_retry_wait,
    };

    let (start_ack_tx, start_ack_rx) =
        channel::<Result<(), StartupError<E>>>(usize::from(pool_size));
    let (shutdown_tx, shutdown_rx) = shutdown_channel();
    for (w, rx) in inf_workers {
        let handoff = RunnerHandoff {
            factory: Arc::clone(&predictor_factory),
            start,
            ready_tx: start_ack_tx.clone(),
            shutdown_tx: Arc::clone(&shutdown_tx),
        };
        run_worker(w, rx, handoff);
    }

    // Drop my copies so each channel closes once the workers drop theirs.
    drop(start_ack_tx);
    drop(shutdown_tx);
    let consumed = Instant::now().duration_since(start_ts);

    collect_acks(
        start_ack_rx,
        pool_size,
        config.startup_timeout.saturating_sub(consumed),
    )
    .await?;
    Ok(Batchinf::new(pool, observability, shutdown_rx))
}
