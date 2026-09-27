use crate::config::InnerConfig;
use crate::pool::FunnelMessage;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc::{Sender, channel, error::TrySendError};

type WorkerDlQueue<Input, Output, Error> = Mutex<VecDeque<FunnelMessage<Input, Output, Error>>>;

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
// [state bits] [remaining 62 bits]
// These u8 values are never stored.
pub(crate) mod worker_states {
    // 0b00
    pub(crate) const WAITING: u8 = 0_u8;
    // 0b01
    pub(crate) const EXIT: u8 = 1_u8;
    // 0b10
    pub(crate) const RUNNING_INFERENCE: u8 = 2_u8;
    // 0b11
    pub(crate) const CRASHED: u8 = 3_u8;
    // Mask the state bits to get the queue len.
    pub(crate) const QUEUE_MASK: u64 = !(0b11_u64 << 62);
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

#[derive(Debug)]
pub(crate) struct WorkerStateInner<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    // State is stored in the 2 MSB here atomic load/store.
    // The other 62 bits store the queue length.
    state: AtomicU64,
    config: InnerConfig,
    dl_queue: WorkerDlQueue<Input, Output, Error>,
    panic_tx: Sender<u8>,
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

    /// On worker crash, transmit sentinel signal, unit type, to the control plane loop to wake it
    /// up.
    fn signal_crash(&self) {
        // Try send here.
        // If this errors, the poll loop in the control plane will read the crash state.
        let _ = self.panic_tx.try_send(self.worker_id);
    }

    /// On worker crash, the workers buffer may have messages in its buffer that can no longer be
    /// serviced by the worker given a crash.
    ///
    /// Those orphaned messages are buffered here while the worker is restarting.
    fn queue_orphaned_messages(&self, mut buffer: Vec<FunnelMessage<Input, Output, Error>>) {
        let items = buffer.drain(..);
        let mut handle = self.dl_queue.lock().unwrap();
        for item in items {
            handle.push_back(item);
        }
    }

    /// Drain all orphaned messages to queue them for the restarted worker.
    fn drain_dl_queue(&self) -> Vec<FunnelMessage<Input, Output, Error>> {
        let mut handle = self.dl_queue.lock().unwrap();
        handle.drain(..).collect()
    }
}

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
        self.inner.state.fetch_add(1_u64, Ordering::Release);
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
        let queue_len = (state & worker_states::QUEUE_MASK) as u32;

        WorkerSnapshot { status, queue_len }
    }

    // Resetting the queue len to 0 also puts the worker state back to Waiting.
    pub(crate) fn reset_queue_len(&self) {
        self.inner.state.store(0_u64, Ordering::Release);
    }

    pub(crate) fn signal_crash(&self) {
        self.inner.signal_crash()
    }

    /// On worker crash, the workers buffer may have messages in its buffer that can no longer be
    /// serviced by the worker given a crash.
    ///
    /// Those orphaned messages are buffered here while the worker is restarting.
    pub(crate) fn queue_orphaned_messages(&self, buffer: Vec<FunnelMessage<Input, Output, Error>>) {
        self.inner.queue_orphaned_messages(buffer);
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
    // This type is already wrapped in Arc, no need to the extra indirection.
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
        let queue_handle = {
            let handle = self.worker_queue.lock().unwrap();
            handle.clone()
        };
        match queue_handle.try_send(msg) {
            Ok(_) => QueuePushResult::Success,
            Err(TrySendError::Full(m)) => QueuePushResult::QueueFull(m),
            Err(TrySendError::Closed(m)) => QueuePushResult::QueueClosed(m),
        }
    }

    /// Swap in a new sender channel.
    ///
    /// This is called when a crashed worker is restarted, or during shutdown signal handling.
    pub(crate) fn replace_queue(&self, worker_queue: Sender<FunnelMessage<Input, Output, Error>>) {
        let mut handle = self.worker_queue.lock().unwrap();
        *handle = worker_queue;
    }

    /// Pull a clone of the worker state.
    pub(crate) fn clone_worker_state(&self) -> WorkerState<Input, Output, Error> {
        self.state.clone()
    }

    /// On shutdown, set the worker state to exit.
    /// Replace the sender with a dummy channel, to drop the sender channel.
    /// When the channel drops, the inference worker receiver channel also closes.
    pub(crate) fn set_exit(&self) {
        self.state.set_state(WorkerStatus::Exit);
        let (tx, _) = channel(1);
        self.replace_queue(tx);
    }
}
