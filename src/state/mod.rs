use crate::{config::InnerConfig, pool::FunnelMessage};
use std::{
    collections::VecDeque,
    sync::atomic::{AtomicU64, Ordering},
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::mpsc::{Receiver, Sender, error::TrySendError};

/// Exclusive access around the DL Queue buffer to recover orphaned requests on worker crash.
type WorkerDlQueue<Input, Output, Error> = Mutex<VecDeque<FunnelMessage<Input, Output, Error>>>;

// Mask the state bits to get the queue len.
const QUEUE_MASK: u64 = !(0b11_u64 << 62);

pub(crate) enum QueuePushResult<Input, Output, Error>
where
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    Success,
    QueueFull(FunnelMessage<Input, Output, Error>),
    QueueClosed(FunnelMessage<Input, Output, Error>),
}

// These mappings allow for the state and queue size to be stored in a single atomic.
// The 2 most significant bits store the status. Remaining 62 bits store the queue len.
// [state bits] [remaining 62 bits]
// These u8 values are never stored.
// 0b00
const WAITING: u8 = 0_u8;
// 0b01
const EXIT: u8 = 1_u8;
// 0b10
const RUNNING_INFERENCE: u8 = 2_u8;
// 0b11
const CRASHED: u8 = 3_u8;

/// The operational state of an inference worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WorkerStatus {
    /// Accumulating requests into the next batch.
    Waiting = 0,
    /// The worker has exited and is no longer accepting requests.
    Exit = 1,
    /// Currently executing [`Predictor::predict_batch`](`crate::Predictor`).
    Running = 2,
    /// [`Predictor::predict_batch`](`crate::Predictor`) panicked. The control plane will restart the worker.
    Crashed = 3,
}

impl WorkerStatus {
    fn from_u8(state: u8) -> WorkerStatus {
        match state {
            WAITING => WorkerStatus::Waiting,
            EXIT => WorkerStatus::Exit,
            RUNNING_INFERENCE => WorkerStatus::Running,
            CRASHED => WorkerStatus::Crashed,
            _ => unreachable!("Invalid state code"),
        }
    }

    fn to_u8(self) -> u8 {
        match self {
            Self::Waiting => WAITING,
            Self::Exit => EXIT,
            Self::Running => RUNNING_INFERENCE,
            Self::Crashed => CRASHED,
        }
    }
}

/// A point-in-time snapshot of a worker's state.
///
/// Returned by [`Batchinf::pool_status`](`crate::Batchinf`) and [`Batchinf::worker_status`](`crate::Batchinf`). Reflects the state
/// at the moment of the atomic load; the worker may have advanced by the time it is read.
#[derive(Clone, Debug)]
pub struct WorkerSnapshot {
    /// The worker's current operational status.
    pub status: WorkerStatus,
    /// Number of requests accumulated in the current batch, up to `batch_size`.
    pub queue_len: u32,
}

/// Struct to store inner worker state.
/// State that will be shared and wrapped in Arc<T>.
#[derive(Debug)]
struct WorkerStateInner<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    // State is stored in the 2 MSB here atomic load/store.
    // The other 62 bits store the queue length.
    // Fused into a single atomic given both attributes may need to be updated in a single store.
    state: AtomicU64,
    config: InnerConfig,
    // Collection to handle orphaned requests on crash.
    dl_queue: WorkerDlQueue<Input, Output, Error>,
    // Tx to signal crash to control loop with `worker_id`.
    panic_tx: Sender<u8>,
    // ID for random access in control plane during crash handling.
    worker_id: u8,
}

impl<Input, Output, Error> WorkerStateInner<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    fn new(
        config: InnerConfig,
        panic_tx: Sender<u8>,
        worker_id: u8,
    ) -> WorkerStateInner<Input, Output, Error> {
        let dl_queue = Mutex::new(VecDeque::new());
        WorkerStateInner {
            state: AtomicU64::new(0_u64),
            config,
            dl_queue,
            panic_tx,
            worker_id,
        }
    }

    fn set_state(&self, state: WorkerStatus) {
        let state_key: u8 = state.to_u8();
        let state = u64::from(state_key) << 62;

        // The 2 MSB need to be cleared here, and then ORed with the state value.
        // So we need a CAS loop.
        let mut current = self.state.load(Ordering::Relaxed);

        // Update the state bits using a CAS loop.
        loop {
            let new = (current & QUEUE_MASK) | state;
            match self.state.compare_exchange_weak(
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

    /// On worker crash, transmit sentinel signal, unit type, to the control plane loop to wake it
    /// up.
    fn signal_crash(&self) {
        // Try send here.
        // If this errors, the poll loop in the control plane will read the crash state.
        let _ = self.panic_tx.try_send(self.worker_id);
    }

    /// On worker crash, buffer orphaned messages for the restarted worker and set the state to
    /// [WorkerStatus::Crashed].
    ///
    /// The state is set while holding the DL queue lock. The control plane only restarts a worker
    /// after observing [WorkerStatus::Crashed], and [Self::drain_dl_queue] blocks on the same lock,
    /// so a restart always sees the complete set of orphaned messages.
    fn mark_crashed(&self, recv: &mut Receiver<FunnelMessage<Input, Output, Error>>) {
        let mut handle = self.dl_queue.lock().unwrap();
        self.set_state(WorkerStatus::Crashed);
        while let Ok(inp) = recv.try_recv() {
            handle.push_back(inp);
        }
    }

    /// Drain all orphaned messages to queue them for the restarted worker.
    fn drain_dl_queue(&self) -> Vec<FunnelMessage<Input, Output, Error>> {
        let mut handle = self.dl_queue.lock().unwrap();
        handle.drain(..).collect()
    }
}

/// Crate public wrapper type for [WorkerStateInner].
#[derive(Debug)]
pub(crate) struct WorkerState<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    inner: Arc<WorkerStateInner<Input, Output, Error>>,
}

impl<Input, Output, Error> Clone for WorkerState<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        let inner = Arc::clone(&self.inner);
        Self { inner }
    }
}

impl<Input, Output, Error> WorkerState<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    pub(crate) fn new(
        config: InnerConfig,
        crash_tx: Sender<u8>,
        worker_id: u8,
    ) -> WorkerState<Input, Output, Error> {
        let inner = Arc::new(WorkerStateInner::new(config, crash_tx, worker_id));

        WorkerState { inner }
    }

    pub(crate) fn capacity(&self) -> u32 {
        self.inner.config.size
    }

    pub(crate) fn timeout(&self) -> Duration {
        self.inner.config.timeout
    }

    pub(crate) fn increment_len(&self) {
        // Stricter ordering given moving parts.
        self.inner.state.fetch_add(1_u64, Ordering::Release);
    }

    pub(crate) fn set_state(&self, state: WorkerStatus) {
        self.inner.set_state(state);
    }

    pub(crate) fn get_state(&self) -> WorkerStatus {
        let state_key = self.inner.state.load(Ordering::Acquire);
        // Ignore length bits to evaluate state.
        WorkerStatus::from_u8((state_key >> 62) as u8)
    }

    /// Get a snapshot of the worker state.
    ///
    /// Provides a `WorkerSnapshot`, which provides the current state and the size of the queue.
    pub(crate) fn snapshot(&self) -> WorkerSnapshot {
        let state = self.inner.state.load(Ordering::Acquire);
        let status: WorkerStatus = WorkerStatus::from_u8((state >> 62) as u8);
        let queue_len = (state & QUEUE_MASK) as u32;

        WorkerSnapshot { status, queue_len }
    }

    // Resetting the queue len to 0 also puts the worker state back to Waiting.
    pub(crate) fn reset_queue_len(&self) {
        self.inner.state.store(0_u64, Ordering::Release);
    }

    pub(crate) fn signal_crash(&self) {
        self.inner.signal_crash()
    }

    /// On worker crash, buffer orphaned messages for the restarted worker and set the state to
    /// [WorkerStatus::Crashed]. See [WorkerStateInner::mark_crashed] for the locking invariant.
    pub(crate) fn mark_crashed(&self, recv: &mut Receiver<FunnelMessage<Input, Output, Error>>) {
        self.inner.mark_crashed(recv);
    }

    /// Drain all orphaned messages to queue them for the restarted worker.
    pub(crate) fn drain_dl_queue(&self) -> Vec<FunnelMessage<Input, Output, Error>> {
        self.inner.drain_dl_queue()
    }
}

#[derive(Debug)]
pub(crate) struct WorkerRef<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    state: WorkerState<Input, Output, Error>,
    // This type is already wrapped in Arc, no need for the extra indirection.
    worker_queue: Mutex<Sender<FunnelMessage<Input, Output, Error>>>,
}

impl<Input, Output, Error> WorkerRef<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    pub(crate) fn new(
        state: WorkerState<Input, Output, Error>,
        worker_queue: Sender<FunnelMessage<Input, Output, Error>>,
    ) -> Self {
        Self {
            state,
            worker_queue: Mutex::new(worker_queue),
        }
    }

    /// Get a snapshot of the worker state.
    pub(crate) fn snapshot(&self) -> WorkerSnapshot {
        self.state.snapshot()
    }

    /// Provides the capacity of the worker queue, describing the maximum number of inference
    /// requests that can be queued on the worker.
    pub(crate) fn capacity(&self) -> u32 {
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

        // Only hold the lock to clone the tx pointer.
        let queue_handle = self.acquire_queue_handle();

        match queue_handle.try_send(msg) {
            Ok(_) => QueuePushResult::Success,
            Err(TrySendError::Full(m)) => QueuePushResult::QueueFull(m),
            Err(TrySendError::Closed(m)) => QueuePushResult::QueueClosed(m),
        }
    }

    /// Acquire a pointer to the queue channel sender.
    /// The critical section is only a pointer copy.
    #[inline]
    fn acquire_queue_handle(&self) -> Sender<FunnelMessage<Input, Output, Error>> {
        self.worker_queue.lock().unwrap().clone()
    }

    /// Swap in a new sender channel.
    /// This is called when a crashed worker is restarted, or during shutdown signal handling.
    pub(crate) fn replace_queue(&self, worker_queue: Sender<FunnelMessage<Input, Output, Error>>) {
        let mut handle = self.worker_queue.lock().unwrap();
        let _ = std::mem::replace(&mut *handle, worker_queue);
    }

    /// Pull a clone of the worker state.
    pub(crate) fn clone_worker_state(&self) -> WorkerState<Input, Output, Error> {
        self.state.clone()
    }
}

/// Test that repr(u8) on [WorkerStatus] maintains correct behavior.
#[cfg(test)]
mod state_tests {
    use super::WorkerStatus;

    #[test]
    fn waiting_status_from_u8() {
        assert_eq!(WorkerStatus::from_u8(super::WAITING), WorkerStatus::Waiting)
    }

    #[test]
    fn exit_status_from_u8() {
        assert_eq!(WorkerStatus::from_u8(super::EXIT), WorkerStatus::Exit)
    }

    #[test]
    fn running_status_from_u8() {
        assert_eq!(
            WorkerStatus::from_u8(super::RUNNING_INFERENCE),
            WorkerStatus::Running
        )
    }

    #[test]
    fn crashed_status_from_u8() {
        assert_eq!(WorkerStatus::from_u8(super::CRASHED), WorkerStatus::Crashed)
    }

    #[test]
    fn waiting_status_to_u8() {
        assert_eq!(WorkerStatus::Waiting.to_u8(), super::WAITING)
    }

    #[test]
    fn exit_status_to_u8() {
        assert_eq!(WorkerStatus::Exit.to_u8(), super::EXIT)
    }

    #[test]
    fn running_status_to_u8() {
        assert_eq!(WorkerStatus::Running.to_u8(), super::RUNNING_INFERENCE)
    }

    #[test]
    fn crashed_status_to_u8() {
        assert_eq!(WorkerStatus::Crashed.to_u8(), super::CRASHED)
    }
}
