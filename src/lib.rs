pub mod batcher;
pub mod config;
pub(crate) mod pool;
pub mod predictor;
pub(crate) mod worker;

use pool::{FunnelMessage, WorkerPool, WorkerRef};

use batcher::Batchinf;
pub use config::BatcherConfig;
use config::InnerConfig;
pub use predictor::Predictor;
use tokio::sync::mpsc::channel;
use tokio::sync::oneshot::Sender as OneshotSender;
use worker::{InferenceWorker, WorkerState};

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
) -> (
    Vec<InferenceWorker<P>>,
    Vec<WorkerRef<P::Input, P::Output, P::Error>>,
) {
    let mut inf_workers = Vec::with_capacity(state.len());
    let mut worker_refs = Vec::with_capacity(state.len());
    for worker in state {
        let (tx, rx) =
            channel::<(P::Input, OneshotSender<Result<P::Output, P::Error>>)>(channel_size);
        let inf_worker = InferenceWorker::new(worker.clone(), predictor.clone(), rx);
        let worker_ref = WorkerRef::new(worker.clone(), tx);
        inf_workers.push(inf_worker);
        worker_refs.push(worker_ref);
    }
    (inf_workers, worker_refs)
}

pub fn get_batcher<P: Predictor + Send + Sync + 'static>(
    predictor: P,
    config: BatcherConfig,
) -> Batchinf<P::Input, P::Output, P::Error> {
    let batch_size = config.batch_size.get();
    let pool_size = config.pool_size.get();
    let total_channel_size = (pool_size * batch_size) as usize;
    let (funnel_tx, funnel_rx) =
        channel::<FunnelMessage<P::Input, P::Output, P::Error>>(total_channel_size);

    let conf: InnerConfig = config.into();

    let workers: Vec<WorkerState> = (0..pool_size)
        .into_iter()
        .map(|_| WorkerState::new(conf))
        .collect();

    let (inf_workers, worker_refs) =
        init_worker_ref_pairs(predictor, &workers, batch_size as usize);
    let pool = WorkerPool::new(worker_refs, funnel_rx);

    pool.start();
    for worker in inf_workers.into_iter() {
        worker.start();
    }

    Batchinf::new(funnel_tx)
}
