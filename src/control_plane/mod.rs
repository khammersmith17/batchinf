use crate::{
    config::BatcherConfig,
    observability::BatcherMetrics,
    pool::FunnelMessage,
    predictor::Predictor,
    state::{WorkerRef, WorkerState, WorkerStatus},
    worker::{InferenceWorker, run_worker},
};
use std::sync::{Arc, Weak};
use tokio::sync::mpsc::{Receiver, Sender, channel};

pub(crate) struct ControlPlane<P: Predictor + Send + Sync + 'static> {
    pub(crate) predictor: P,
    pub(crate) pool_weak: Weak<[WorkerRef<P::Input, P::Output, P::Error>]>,
    pub(crate) obs: Option<Arc<dyn BatcherMetrics>>,
    pub(crate) config: BatcherConfig,
    pub(crate) panic_rx: Receiver<u8>,
}

pub(crate) fn run_control_plane<P: Predictor + Send + Sync + 'static>(
    control_plane: ControlPlane<P>,
) {
    tokio::task::spawn(async { supervisor_loop(control_plane).await });
}

async fn supervisor_loop<P: Predictor + Send + Sync + 'static>(control_plane: ControlPlane<P>) {
    let ControlPlane {
        predictor,
        pool_weak,
        obs,
        config,
        mut panic_rx,
    } = control_plane;

    loop {
        // Wake the control plane when a worker signals crash.
        let Some(worker_id) = panic_rx.recv().await else {
            break;
        };

        // When there are no more strong references to the pool, no more requests will be forwarded
        // to workers. Workers shutdown gracefully on their own.
        let Some(pool) = pool_weak.upgrade() else {
            return;
        };

        let status = pool[worker_id as usize].snapshot().status;
        // Avoid double restart if poll loop already handled crash restart.
        if !matches!(status, WorkerStatus::Crashed) {
            continue;
        }

        let handle = &pool[usize::from(worker_id)];
        crash_handler(predictor.clone(), handle, obs.clone(), &config);
    }
}

fn crash_handler<P: Predictor + Send + Sync + 'static>(
    predictor: P,
    handle: &WorkerRef<P::Input, P::Output, P::Error>,
    obs: Option<Arc<dyn BatcherMetrics>>,
    config: &BatcherConfig,
) {
    let tx = restart_worker(predictor, handle.clone_worker_state(), obs, config);
    handle.replace_queue(tx);
}

/// Restart a worker that has crashed.
fn restart_worker<P: Predictor + Send + Sync + 'static>(
    predictor: P,
    state: WorkerState<P::Input, P::Output, P::Error>,
    obs: Option<Arc<dyn BatcherMetrics>>,
    config: &BatcherConfig,
) -> Sender<FunnelMessage<P::Input, P::Output, P::Error>> {
    // Emit metrics around panic.
    emit_worker_panic(obs.clone());

    // Create fresh channel.
    let (tx, rx) = channel(config.batch_size.get() as usize);
    // Drain DL messages and enqueue them in the new channel.
    let dl_funnel = state.drain_dl_queue();
    fill_queue_with_orphaned(dl_funnel, &tx);

    // Clean state.
    state.reset_queue_len();
    let worker = InferenceWorker::new(state, predictor, obs);
    run_worker(worker, rx);
    tx
}

// Takes the drained DL queue and writes all messages to the newly defined sender.
fn fill_queue_with_orphaned<Input, Output, Error>(
    dl: Vec<FunnelMessage<Input, Output, Error>>,
    tx: &Sender<FunnelMessage<Input, Output, Error>>,
) where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    debug_assert!(tx.capacity() >= dl.len());
    for msg in dl.into_iter() {
        // Channel is open (just created), no workers hold a copy of a sender yet.
        let _ = tx.try_send(msg);
    }
}

/// Emit worker panicked.
fn emit_worker_panic(obs: Option<Arc<dyn BatcherMetrics>>) {
    if let Some(obs) = obs {
        obs.on_worker_panic()
    }
}
