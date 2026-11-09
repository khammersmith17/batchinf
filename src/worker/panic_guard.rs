use crate::state::WorkerState;
use std::{sync::Arc, thread::panicking};
use tokio::sync::Notify;
use tracing::error;

/// Marks a worker's predictor `Dead` if the accumulator task or the inference thread holding it
/// panics.
///
/// Neither should panic on its own; this covers the user's metrics callbacks, which run inline on
/// both. Only the predictor status is written. It is a plain store, so either side can write it
/// safely, unlike the worker status, which only the accumulator may write. `Dead` alone stops the
/// router from sending this worker requests.
pub(super) struct PanicGuard {
    state: WorkerState,
    signal: Arc<Notify>,
}

impl PanicGuard {
    pub(super) fn new(state: WorkerState, signal: Arc<Notify>) -> PanicGuard {
        PanicGuard { state, signal }
    }
}

impl Drop for PanicGuard {
    fn drop(&mut self) {
        if panicking() {
            error!("worker panicked, marking predictor as PredictorStatus::Dead");
            self.state.set_dead();
            // Notify waiters in the case this is the only worker and messages can no longer be serviced.
            self.signal.notify_waiters();
        }
    }
}
