use crate::FunnelMessage;
use crate::config::InnerConfig;
use crate::error::BatchinfError;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc::Sender;

// A worker can be in one the three following states.
// Waiting if when inference requests are being queued, Running trigger an inference
// run, and Exit defines when resources are being cleaned up.
/// The operational state of an inference worker.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerStatus {
    /// Accumulating requests into the next batch.
    Waiting,
    /// The worker has exited and is no longer accepting requests.
    Exit,
    /// Currently executing [`Predictor::predict_batch`].
    Running,
}

impl From<WorkerStatus> for u8 {
    fn from(state: WorkerStatus) -> u8 {
        match state {
            WorkerStatus::Waiting => worker_states::WAITING,
            WorkerStatus::Exit => worker_states::EXIT,
            WorkerStatus::Running => worker_states::RUNNING_INFERENCE,
        }
    }
}

impl From<u8> for WorkerStatus {
    fn from(state: u8) -> WorkerStatus {
        match state {
            worker_states::WAITING => Self::Waiting,
            worker_states::EXIT => Self::Exit,
            worker_states::RUNNING_INFERENCE => Self::Running,
            _ => unreachable!("Invalid state value"),
        }
    }
}

// Mappings out u8 key and enum variant.
// These mappings allow for the state and queue size to be stored in a single atomic.
// The 2 most significant bits store the state.
// These u8 values are never stored.
pub(crate) mod worker_states {
    pub(crate) const WAITING: u8 = 0_u8;
    pub(crate) const EXIT: u8 = 1_u8;
    pub(crate) const RUNNING_INFERENCE: u8 = 2_u8;
    // Mask the state bits to get the queue len.
    pub(crate) const QUEUE_MASK: u64 = !(0b11_u64 << 62);
}

/// A point-in-time snapshot of a worker's state.
///
/// Returned by [`Batchinf::pool_status`] and [`Batchinf::worker_status`]. Reflects the state
/// at the moment of the atomic load; the worker may have advanced by the time it is read.
pub struct WorkerSnapshot {
    /// The worker's current operational status.
    pub status: WorkerStatus,
    /// Number of requests accumulated in the current batch, up to `batch_size`.
    pub queue_len: u64,
}

#[derive(Debug)]
pub(crate) struct WorkerState_ {
    // State is stored in the 2 MSB here atomic load/store.
    // The other 62 bits store the queue length.
    state: AtomicU64,
    config: InnerConfig,
}

impl WorkerState_ {
    fn new(config: InnerConfig) -> WorkerState_ {
        WorkerState_ {
            state: AtomicU64::new(0_u64),
            config,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct WorkerState {
    inner: Arc<WorkerState_>,
}

impl WorkerState {
    pub(crate) fn new(config: InnerConfig) -> WorkerState {
        let inner = Arc::new(WorkerState_::new(config));

        WorkerState { inner }
    }
}

impl WorkerState {
    pub(crate) fn capacity(&self) -> u64 {
        self.inner.config.size
    }

    pub(crate) fn timeout(&self) -> u64 {
        self.inner.config.timeout
    }

    pub(crate) fn increment_len(&self) {
        self.inner.state.fetch_add(1_u64, Ordering::Relaxed);
    }

    pub(crate) fn set_state(&self, state: WorkerStatus) {
        let state_key: u8 = state.into();
        let state = u64::from(state_key) << 62;

        // The 2 MSB need to be cleared here, and then ORed with the state value.
        // So we need a CAS loop.
        let mut current = self.inner.state.load(Ordering::Relaxed);

        loop {
            let new = (current & worker_states::QUEUE_MASK) | state;
            match self.inner.state.compare_exchange_weak(
                current,
                new,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    pub(crate) fn get_state(&self) -> WorkerStatus {
        let state_key = self.inner.state.load(Ordering::Acquire);
        ((state_key >> 62) as u8).into()
    }

    pub(crate) fn snapshot(&self) -> WorkerSnapshot {
        let state = self.inner.state.load(Ordering::Acquire);
        let status: WorkerStatus = ((state >> 62) as u8).into();
        let queue_len = state & worker_states::QUEUE_MASK;

        WorkerSnapshot { status, queue_len }
    }

    // Reseting the queue len to 0 also puts the worker state back to Waiting.
    pub(crate) fn reset_queue_len(&self) {
        self.inner.state.store(0_u64, Ordering::Release);
    }
}

#[derive(Debug)]
pub(crate) struct WorkerRef<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: Send + Sync + 'static,
{
    state: WorkerState,
    worker_queue: Sender<FunnelMessage<Input, Output, Error>>,
}

impl<Input, Output, Error> WorkerRef<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: Send + Sync + 'static,
{
    pub(crate) fn new(
        state: WorkerState,
        worker_queue: Sender<FunnelMessage<Input, Output, Error>>,
    ) -> Self {
        Self {
            state,
            worker_queue,
        }
    }

    pub(crate) fn snapshot(&self) -> WorkerSnapshot {
        self.state.snapshot()
    }

    pub(crate) fn capacity(&self) -> u64 {
        self.state.capacity()
    }

    pub(crate) async fn push<E: Clone + std::error::Error + Send + Sync + 'static>(
        &self,
        msg: FunnelMessage<Input, Output, Error>,
    ) -> Result<(), BatchinfError<E>> {
        if matches!(self.snapshot().status, WorkerStatus::Exit) {
            return Err(BatchinfError::InternalError);
        }
        // If the worker channel is closed (worker exited), the send error is dropped here.
        // The caller's oneshot receiver will return Err, which maps to BatchinfError::InternalError.
        let _ = self.worker_queue.send(msg).await;
        Ok(())
    }
}
