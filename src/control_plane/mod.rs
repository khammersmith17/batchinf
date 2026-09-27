use crate::{
    config::BatcherConfig,
    observability::BatcherMetrics,
    pool::FunnelMessage,
    predictor::Predictor,
    state::{WorkerRef, WorkerSnapshot, WorkerState, WorkerStatus},
    worker::{InferenceWorker, run_worker},
};
use std::sync::{
    Arc, Weak,
    atomic::{AtomicBool, Ordering},
};
use tokio::{
    select,
    sync::mpsc::{Receiver, Sender, channel},
    time::{Duration, sleep},
};

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
    let shutdown_signal = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&shutdown_signal);

    // Spawns shutdown signal handler.
    tokio::task::spawn(async move { wait_for_shutdown(signal).await });

    loop {
        if shutdown_signal.load(Ordering::Acquire) {
            break;
        }

        let Some(pool) = pool_weak.upgrade() else {
            return;
        };

        select! {
            // Wake the control plane loop when a worker signals a crash during panic.
            signal = panic_rx.recv() => {
                let Some(worker_id) = signal else {break};

                let handle = &pool[usize::from(worker_id)];
                crash_handler(predictor.clone(),  handle, obs.clone(), &config);
            },
            _ = sleep(Duration::from_millis(250)) => {}
        }

        let pool_size = pool.len();
        // If all workers have exited, then the control plane also exits.
        let mut exited = 0_usize;
        let mut total_queue_depth = 0_usize;
        for i in 0..pool_size {
            let worker_snapshot = pool[i].snapshot();

            let WorkerSnapshot { status, queue_len } = worker_snapshot;
            total_queue_depth += queue_len as usize;

            match status {
                WorkerStatus::Crashed => {
                    let handle = &pool[i];
                    if !matches!(handle.snapshot().status, WorkerStatus::Crashed) {
                        continue;
                    }

                    crash_handler(predictor.clone(), &handle, obs.clone(), &config);
                }
                WorkerStatus::Exit => exited += 1,
                _ => {}
            }
        }

        if exited == pool_size {
            return;
        }

        if let Some(ref obs) = obs {
            obs.on_queue_depth(total_queue_depth)
        }
    }

    shutdown_workers(pool_weak).await;
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

#[cfg(unix)]
async fn wait_for_shutdown(flag: Arc<AtomicBool>) {
    use tokio::signal::unix::{SignalKind, signal};
    let Ok(mut sig) = signal(SignalKind::terminate()) else {
        flag.store(true, Ordering::Release);
        return;
    };
    sig.recv().await;
    flag.store(true, Ordering::Release);
}

#[cfg(windows)]
async fn wait_for_shutdown(flag: Arc<AtomicBool>) {
    use tokio::signal::windows::ctrl_shutdown;
    let Ok(mut sig) = ctrl_shutdown() else {
        flag.store(true, Ordering::Release);
        return;
    };
    sig.recv().await;
    flag.store(true, Ordering::Release);
}

// On platforms where no signal API is available, graceful shutdown via signal is unsupported.
// The handler suspends indefinitely without consuming CPU.
#[cfg(not(any(unix, windows)))]
async fn wait_for_shutdown(_flag: Arc<AtomicBool>) {
    std::future::pending::<()>().await;
}

// Set the state of each worker to Exit.
// Poll the states to ensure they are not overwritten until the reference count of the worker pool
// is 0.
async fn shutdown_workers<Input, Output, Error>(pool_weak: Weak<[WorkerRef<Input, Output, Error>]>)
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    loop {
        let mut exited = 0_usize;
        let Some(pool) = pool_weak.upgrade() else {
            return;
        };

        let pool_size = pool.len();

        for worker in pool.iter() {
            let snapshot = { worker.snapshot() };

            if !matches!(snapshot.status, WorkerStatus::Exit) {
                worker.set_exit();
            } else {
                exited += 1;
            }
        }

        if exited == pool_size {
            break;
        }

        // Sleep for half a second inbetween poll loops.
        sleep(Duration::from_millis(500)).await;
    }
}
