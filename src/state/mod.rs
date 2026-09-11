use crate::config::InnerConfig;
use crate::pool::FunnelMessage;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc::{Sender, channel, error::TrySendError};

pub(crate) enum QueuePushResult<Input, Output, Error>
where
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    Success,
    QueueFull(FunnelMessage<Input, Output, Error>),
    QueueClosed(FunnelMessage<Input, Output, Error>),
}

/// The operational state of an inference worker.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerStatus {
    /// Accumulating requests into the next batch.
    Waiting,
    /// The worker has exited and is no longer accepting requests.
    Exit,
    /// Currently executing [`Predictor::predict_batch`](`crate::Predictor`).
    Running,
    /// [`Predictor::predict_batch`](`crate::Predictor`) panicked. The control plane will restart the worker.
    Crashed,
}

impl From<WorkerStatus> for u8 {
    fn from(state: WorkerStatus) -> u8 {
        match state {
            WorkerStatus::Waiting => worker_states::WAITING,
            WorkerStatus::Exit => worker_states::EXIT,
            WorkerStatus::Running => worker_states::RUNNING_INFERENCE,
            WorkerStatus::Crashed => worker_states::CRASHED,
        }
    }
}

impl From<u8> for WorkerStatus {
    fn from(state: u8) -> WorkerStatus {
        match state {
            worker_states::WAITING => Self::Waiting,
            worker_states::EXIT => Self::Exit,
            worker_states::RUNNING_INFERENCE => Self::Running,
            worker_states::CRASHED => Self::Crashed,
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
    pub(crate) const CRASHED: u8 = 3_u8;
    // Mask the state bits to get the queue len.
    pub(crate) const QUEUE_MASK: u64 = !(0b11_u64 << 62);
}

/// A point-in-time snapshot of a worker's state.
///
/// Returned by [`Batchinf::pool_status`](`crate::Batchinf`) and [`Batchinf::worker_status`](`crate::Batchinf`). Reflects the state
/// at the moment of the atomic load; the worker may have advanced by the time it is read.
pub struct WorkerSnapshot {
    /// The worker's current operational status.
    pub status: WorkerStatus,
    /// Number of requests accumulated in the current batch, up to `batch_size`.
    pub queue_len: u64,
}

#[derive(Debug)]
pub(crate) struct WorkerStateInner {
    // State is stored in the 2 MSB here atomic load/store.
    // The other 62 bits store the queue length.
    state: AtomicU64,
    config: InnerConfig,
}

impl WorkerStateInner {
    fn new(config: InnerConfig) -> WorkerStateInner {
        WorkerStateInner {
            state: AtomicU64::new(0_u64),
            config,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct WorkerState {
    inner: Arc<WorkerStateInner>,
}

impl WorkerState {
    pub(crate) fn new(config: InnerConfig) -> WorkerState {
        let inner = Arc::new(WorkerStateInner::new(config));

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
        // Batch size will never overwrite state bits in practice.
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

    /// Get a snapshot of the worker state.
    ///
    /// Provides a `WorkerSnapshot`, which provides the current state and the size of the queue.
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
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    state: WorkerState,
    worker_queue: Sender<FunnelMessage<Input, Output, Error>>,
}

impl<Input, Output, Error> WorkerRef<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
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

    /// Get a snapshot of the worker state.
    pub(crate) fn snapshot(&self) -> WorkerSnapshot {
        self.state.snapshot()
    }

    /// Provides the capacity of the worker queue, describing the maximum number of inference
    /// requests that can be queued on the worker.
    pub(crate) fn capacity(&self) -> u64 {
        self.state.capacity()
    }

    pub(crate) fn push(
        &self,
        msg: FunnelMessage<Input, Output, Error>,
    ) -> QueuePushResult<Input, Output, Error> {
        if matches!(
            self.snapshot().status,
            WorkerStatus::Exit | WorkerStatus::Crashed
        ) {
            return QueuePushResult::QueueClosed(msg);
        }
        // If the worker channel is closed (worker exited), the send error is dropped here.
        // The caller's oneshot receiver will return Err, which maps to BatchinfError::InternalError.
        match self.worker_queue.try_send(msg) {
            Ok(_) => QueuePushResult::Success,
            Err(TrySendError::Full(m)) => QueuePushResult::QueueFull(m),
            Err(TrySendError::Closed(m)) => QueuePushResult::QueueClosed(m),
        }
    }

    /// Swap in a new sender channel.
    ///
    /// This is called when a crashed worker is restarted, or during shutdown signal handling.
    pub(crate) fn replace_queue(
        &mut self,
        worker_queue: Sender<FunnelMessage<Input, Output, Error>>,
    ) {
        self.worker_queue = worker_queue;
    }

    /// Pull a clone of the worker state.
    pub(crate) fn clone_worker_state(&self) -> WorkerState {
        self.state.clone()
    }

    /// On SIGTERM, set the worker state to exit.
    /// Replace the sender with a dummy channel, to drop the sender channel.
    /// When the channel drops, the inference worker receiver channel also closes.
    pub(crate) fn set_exit(&mut self) {
        self.state.set_state(WorkerStatus::Exit);
        let (tx, _) = channel(1);
        self.replace_queue(tx);
    }
}
