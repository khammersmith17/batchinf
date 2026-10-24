use crate::{config::InnerConfig, pool::FunnelMessage};
use std::{
    sync::Arc,
    sync::atomic::{AtomicU8, AtomicU64, Ordering},
    time::Duration,
};
use tokio::sync::mpsc::{Sender, error::TrySendError};

// Mask the state bits to get the batch len.
const BATCH_LEN_MASK: u64 = !(0b11_u64 << 62);

/// Result type to signal to caller if pushing a work unit on the queue was successful.
pub(crate) enum QueuePushResult<Input, Output, Error>
where
    Error: std::error::Error + Send + Sync + 'static,
{
    Success,
    QueueFull(FunnelMessage<Input, Output, Error>),
    QueueClosed(FunnelMessage<Input, Output, Error>),
}

mod worker_codes {
    // These mappings allow for the state and batch length to be stored in a single atomic.
    // The 2 most significant bits store the status. Remaining 62 bits store the batch len.
    // [state bits] [remaining 62 bits]
    // These u8 values are never stored.

    // 0b00
    pub(super) const WAITING: u8 = 0_u8;
    // 0b01
    pub(super) const EXIT: u8 = 1_u8;
    // 0b10
    pub(super) const RUNNING_INFERENCE: u8 = 2_u8;
    // 0b11
    pub(super) const CRASHED: u8 = 3_u8;
}

/// The operational state of an inference worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WorkerStatus {
    /// Accumulating requests into the next batch.
    Waiting = 0,
    /// The worker has exited and is no longer accepting requests.
    Exit = 1,
    /// A full batch is ready and is being handed to the inference thread. The worker stays in this
    /// state while the inference thread is busy and already has a batch queued.
    Running = 2,
    /// The worker's inference thread is gone, so batches can no longer be handed off. The worker
    /// has stopped and no longer accepts requests. This is caused by the predictor crashing and no
    /// recovering, or a panic in the observability callbacks.
    Crashed = 3,
}

impl WorkerStatus {
    fn from_u8(state: u8) -> WorkerStatus {
        match state {
            worker_codes::WAITING => WorkerStatus::Waiting,
            worker_codes::EXIT => WorkerStatus::Exit,
            worker_codes::RUNNING_INFERENCE => WorkerStatus::Running,
            worker_codes::CRASHED => WorkerStatus::Crashed,
            _ => unreachable!("Invalid state code"),
        }
    }

    fn to_u8(self) -> u8 {
        match self {
            Self::Waiting => worker_codes::WAITING,
            Self::Exit => worker_codes::EXIT,
            Self::Running => worker_codes::RUNNING_INFERENCE,
            Self::Crashed => worker_codes::CRASHED,
        }
    }
}

mod predictor_codes {
    // Predictor is healthy.
    pub(super) const ALIVE: u8 = 0_u8;
    // Predictor crashed but is being rebuilt.
    pub(super) const REBUILDING: u8 = 1_u8;
    // Rebuild of a predictor was unsuccessful, and the predictor no longer services work.
    pub(super) const DEAD: u8 = 2_u8;
}

/// The state of a worker's predictor, owned by the worker's inference thread.
///
/// Reported alongside [`WorkerStatus`] in [`WorkerSnapshot`]. The router only sends new requests
/// to workers whose predictor is [`PredictorStatus::Alive`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PredictorStatus {
    /// The predictor is built and serving batches.
    Alive = 0,
    /// [`Predictor::predict_batch`](`crate::Predictor::predict_batch`) panicked and the predictor
    /// is being rebuilt from the factory. Requests already queued on this worker wait for the
    /// rebuild; new requests are routed elsewhere, or get
    /// [`BatchinfError::QueueFullError`](`crate::BatchinfError::QueueFullError`) if no other
    /// worker can take them.
    Rebuilding = 1,
    /// The factory failed repeatedly, at startup or during a rebuild. The inference thread has
    /// exited and this worker will not serve any more requests.
    Dead = 2,
}

impl PredictorStatus {
    fn from_u8(state: u8) -> PredictorStatus {
        match state {
            predictor_codes::ALIVE => Self::Alive,
            predictor_codes::REBUILDING => Self::Rebuilding,
            predictor_codes::DEAD => Self::Dead,
            _ => unreachable!(),
        }
    }

    fn to_u8(self) -> u8 {
        self as u8
    }
}

/// A point-in-time snapshot of a worker's state.
///
/// Returned by [`Batchinf::pool_status`](`crate::Batchinf`) and [`Batchinf::worker_status`](`crate::Batchinf`). Reflects the state
/// at the moment of the atomic load; the worker may have advanced by the time it is read.
#[derive(Clone, Debug)]
pub struct WorkerSnapshot {
    /// The worker's current operational status.
    pub worker_status: WorkerStatus,
    /// The state of the worker's predictor on its inference thread.
    pub predictor_status: PredictorStatus,
    /// Number of requests accumulated in the current batch, up to `batch_size`.
    pub batch_len: u32,
}

/// Struct to store inner worker state.
/// State that will be shared and wrapped in Arc<T>.
#[derive(Debug)]
struct WorkerStateInner {
    // State is stored in the 2 MSB here atomic load/store.
    // The other 62 bits store the batch length.
    // Fused into a single atomic given both attributes may need to be updated in a single store.
    worker_state: AtomicU64,
    // State of the predictor represented as a single byte.
    // Maps to PredictorStatus codes.
    predictor_state: AtomicU8,
    config: InnerConfig,
}

impl WorkerStateInner {
    fn new(config: InnerConfig) -> WorkerStateInner {
        WorkerStateInner {
            worker_state: AtomicU64::new(0_u64),
            predictor_state: AtomicU8::new(0_u8),
            config,
        }
    }

    /// Set the state of a worker.
    /// Only the accumulator for that worker will ever set the state of the worker,
    /// so a load then a store is OK here, as nobody else is mutating this value.
    ///
    /// If another write is ever added, this needs to be a CAS loop.
    fn set_worker_state(&self, state: WorkerStatus) {
        let state_key: u8 = state.to_u8();
        let state_bits = u64::from(state_key) << 62;

        // Clear the current state bits, preserving the length bits.
        let current = self.worker_state.load(Ordering::Acquire);
        let new_state = (current & BATCH_LEN_MASK) | state_bits;

        self.worker_state.store(new_state, Ordering::Release);
    }

    fn get_worker_state(&self) -> WorkerStatus {
        let state_key = self.worker_state.load(Ordering::Acquire);
        // Ignore length bits to evaluate state.
        WorkerStatus::from_u8((state_key >> 62) as u8)
    }

    fn set_predictor_state(&self, state: PredictorStatus) {
        let state = state.to_u8();
        self.predictor_state.store(state, Ordering::Release);
    }

    fn get_predictor_state(&self) -> PredictorStatus {
        let state_key = self.predictor_state.load(Ordering::Acquire);
        // Ignore length bits to evaluate state.
        PredictorStatus::from_u8(state_key)
    }

    /// Get a snapshot of the current workers state.
    fn snapshot(&self) -> WorkerSnapshot {
        let worker_state = self.worker_state.load(Ordering::Acquire);
        let worker_status: WorkerStatus = WorkerStatus::from_u8((worker_state >> 62) as u8);
        let batch_len = (worker_state & BATCH_LEN_MASK) as u32;
        let predictor_status = self.get_predictor_state();

        WorkerSnapshot {
            worker_status,
            predictor_status,
            batch_len,
        }
    }

    fn reset_batch_len(&self) {
        self.worker_state.store(0_u64, Ordering::Release);
    }

    /// Determines if the worker can accept work now.
    /// The accumulator is waiting for a batch to complete, the predictor is healthy,
    /// and there is room in the batch.
    ///
    /// The room-in-the-batch check covers an accumulator whose batch is full but that has not yet
    /// signaled it by setting `Running`.
    pub(crate) fn can_accept_primary(&self) -> bool {
        let snapshot = self.snapshot();
        snapshot.worker_status == WorkerStatus::Waiting
            && snapshot.predictor_status == PredictorStatus::Alive
            && snapshot.batch_len < self.config.batch_size
    }

    /// Determines if a worker can accept work soon, as a fallback.
    /// True when both the accumulator and the predictor are healthy.
    pub(crate) fn can_accept_secondary(&self) -> bool {
        let snapshot = self.snapshot();
        !matches!(
            snapshot.worker_status,
            WorkerStatus::Exit | WorkerStatus::Crashed
        ) && snapshot.predictor_status == PredictorStatus::Alive
    }

    /// Signals if the predictor is currently being rebuilt.
    pub(crate) fn is_rebuilding(&self) -> bool {
        self.snapshot().predictor_status == PredictorStatus::Rebuilding
    }
}

/// Crate public wrapper type for [WorkerStateInner].
#[derive(Debug)]
pub(crate) struct WorkerState {
    inner: Arc<WorkerStateInner>,
}

impl Clone for WorkerState {
    fn clone(&self) -> Self {
        let inner = Arc::clone(&self.inner);
        Self { inner }
    }
}

// Delegate to the inner methods.
impl WorkerState {
    pub(crate) fn new(config: InnerConfig) -> WorkerState {
        let inner = Arc::new(WorkerStateInner::new(config));

        WorkerState { inner }
    }

    pub(crate) fn batch_size(&self) -> u32 {
        self.inner.config.batch_size
    }

    pub(crate) fn timeout(&self) -> Duration {
        self.inner.config.timeout
    }

    pub(crate) fn increment_len(&self) {
        // Stricter ordering given moving parts.
        self.inner.worker_state.fetch_add(1_u64, Ordering::Release);
    }

    pub(crate) fn set_dead(&self) {
        self.set_predictor_state(PredictorStatus::Dead);
    }

    pub(crate) fn set_worker_state(&self, state: WorkerStatus) {
        self.inner.set_worker_state(state);
    }

    pub(crate) fn get_worker_state(&self) -> WorkerStatus {
        self.inner.get_worker_state()
    }

    pub(crate) fn set_predictor_state(&self, state: PredictorStatus) {
        self.inner.set_predictor_state(state);
    }

    pub(crate) fn can_accept_primary(&self) -> bool {
        self.inner.can_accept_primary()
    }

    pub(crate) fn can_accept_secondary(&self) -> bool {
        self.inner.can_accept_secondary()
    }

    pub(crate) fn is_rebuilding(&self) -> bool {
        self.inner.is_rebuilding()
    }

    /// Get a snapshot of the worker state.
    ///
    /// Provides a `WorkerSnapshot`, which provides the current state and the length of the batch being built.
    pub(crate) fn snapshot(&self) -> WorkerSnapshot {
        self.inner.snapshot()
    }

    // Resetting the batch len to 0 also puts the worker state back to Waiting.
    pub(crate) fn reset_batch_len(&self) {
        self.inner.reset_batch_len()
    }
}

#[derive(Debug)]
pub(crate) struct WorkerRef<Input, Output, Error>
where
    Input: Send + 'static,
    Output: Send + 'static,
    Error: std::error::Error + Send + Sync + 'static,
{
    state: WorkerState,
    // Fixed for the worker's lifetime: predictor rebuilds happen on the inference thread, so the
    // request channel is never replaced.
    worker_queue: Sender<FunnelMessage<Input, Output, Error>>,
}

impl<Input, Output, Error> WorkerRef<Input, Output, Error>
where
    Input: Send + 'static,
    Output: Send + 'static,
    Error: std::error::Error + Send + Sync + 'static,
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

    pub(crate) fn can_accept_primary(&self) -> bool {
        self.state.can_accept_primary()
    }

    pub(crate) fn can_accept_secondary(&self) -> bool {
        self.state.can_accept_secondary()
    }

    pub(crate) fn is_rebuilding(&self) -> bool {
        self.state.is_rebuilding()
    }

    /// Get a snapshot of the worker state.
    pub(crate) fn snapshot(&self) -> WorkerSnapshot {
        self.state.snapshot()
    }

    /// Attempt to push work onto a worker's queue.
    ///
    /// If the work cannot be forwarded, the work unit is handed back to the caller.
    pub(crate) fn push(
        &self,
        msg: FunnelMessage<Input, Output, Error>,
    ) -> QueuePushResult<Input, Output, Error> {
        if matches!(
            self.snapshot().worker_status,
            WorkerStatus::Exit | WorkerStatus::Crashed
        ) {
            return QueuePushResult::QueueClosed(msg);
        }

        match self.worker_queue.try_send(msg) {
            Ok(_) => QueuePushResult::Success,
            Err(TrySendError::Full(m)) => QueuePushResult::QueueFull(m),
            Err(TrySendError::Closed(m)) => QueuePushResult::QueueClosed(m),
        }
    }
}

/// Test that repr(u8) on [WorkerStatus] maintains correct behavior.
#[cfg(test)]
mod state_tests {
    use super::WorkerStatus;

    #[test]
    fn waiting_status_from_u8() {
        assert_eq!(
            WorkerStatus::from_u8(super::worker_codes::WAITING),
            WorkerStatus::Waiting
        )
    }

    #[test]
    fn exit_status_from_u8() {
        assert_eq!(
            WorkerStatus::from_u8(super::worker_codes::EXIT),
            WorkerStatus::Exit
        )
    }

    #[test]
    fn running_status_from_u8() {
        assert_eq!(
            WorkerStatus::from_u8(super::worker_codes::RUNNING_INFERENCE),
            WorkerStatus::Running
        )
    }

    #[test]
    fn crashed_status_from_u8() {
        assert_eq!(
            WorkerStatus::from_u8(super::worker_codes::CRASHED),
            WorkerStatus::Crashed
        )
    }

    #[test]
    fn waiting_status_to_u8() {
        assert_eq!(WorkerStatus::Waiting.to_u8(), super::worker_codes::WAITING)
    }

    #[test]
    fn exit_status_to_u8() {
        assert_eq!(WorkerStatus::Exit.to_u8(), super::worker_codes::EXIT)
    }

    #[test]
    fn running_status_to_u8() {
        assert_eq!(
            WorkerStatus::Running.to_u8(),
            super::worker_codes::RUNNING_INFERENCE
        )
    }

    #[test]
    fn crashed_status_to_u8() {
        assert_eq!(WorkerStatus::Crashed.to_u8(), super::worker_codes::CRASHED)
    }
}
