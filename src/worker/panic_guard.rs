use crate::state::WorkerState;
use std::{sync::Arc, thread::panicking};
use tokio::sync::Notify;
use tracing::error;

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
            // Setting the state to crashed, if there is a panic
            // This should not happen. There should not be anything that can panic here, but to be
            // safe this guard will make sure no requests get routed.
            error!("batching task panicked, marking worker as WorkerStatus::Crashed");
            self.state.set_dead();
            // Notify waiters in the case this is the only worker and messages can no longer be serviced.
            self.signal.notify_waiters();
        }
    }
}
