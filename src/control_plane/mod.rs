use crate::config::BatcherConfig;
use crate::observability::BatcherMetrics;
use crate::pool::FunnelMessage;
use crate::predictor::Predictor;
use crate::state::{WorkerRef, WorkerSnapshot, WorkerState, WorkerStatus};
use crate::worker::{InferenceWorker, run_worker};
use std::sync::{
    Arc, Weak,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::RwLock;
use tokio::sync::mpsc::{Sender, channel};
use tokio::time::{Duration, sleep};

pub(crate) struct ControlPlane<P: Predictor + Send + Sync + 'static> {
    pub(crate) predictor: P,
    pub(crate) pool_weak: Weak<[RwLock<WorkerRef<P::Input, P::Output, P::Error>>]>,
    pub(crate) obs: Option<Arc<dyn BatcherMetrics>>,
    pub(crate) config: BatcherConfig,
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

        let pool_size = pool.len();
        // If all workers have exited, then the control plane also exits.
        let mut exited = 0_usize;
        let mut total_queue_depth = 0_usize;
        for i in 0..pool_size {
            let worker_snapshot = {
                let handle = pool[i].read().await;
                handle.snapshot()
            };

            let WorkerSnapshot { status, queue_len } = worker_snapshot;
            total_queue_depth += queue_len as usize;

            match status {
                WorkerStatus::Crashed => {
                    let mut handle = pool[i].write().await;
                    // Ensure the worker has not changed state, should not in practice.
                    if !matches!(handle.snapshot().status, WorkerStatus::Crashed) {
                        continue;
                    }

                    let tx = restart_worker(
                        predictor.clone(),
                        handle.clone_worker_state(),
                        obs.clone(),
                        &config,
                    );

                    handle.replace_queue(tx);
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

        sleep(Duration::from_millis(250)).await;
    }

    shutdown_workers(pool_weak).await;
}

/// Restart a worker that has crashed.
fn restart_worker<P: Predictor + Send + Sync + 'static>(
    predictor: P,
    state: WorkerState,
    obs: Option<Arc<dyn BatcherMetrics>>,
    config: &BatcherConfig,
) -> Sender<FunnelMessage<P::Input, P::Output, P::Error>> {
    let (tx, rx) = channel(config.batch_size.get() as usize);
    state.reset_queue_len();
    let worker = InferenceWorker::new(state, predictor, obs);
    run_worker(worker, rx);
    tx
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
async fn shutdown_workers<Input, Output, Error>(
    pool_weak: Weak<[RwLock<WorkerRef<Input, Output, Error>>]>,
) where
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
            let snapshot = { worker.read().await.snapshot() };

            if !matches!(snapshot.status, WorkerStatus::Exit) {
                let mut handle = worker.write().await;
                handle.set_exit();
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
