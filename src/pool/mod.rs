use crate::{
    error::BatchinfError,
    state::{QueuePushResult, WorkerRef, WorkerSnapshot},
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

type PoolSlab<Input, Output, Error> = Arc<[WorkerRef<Input, Output, Error>]>;

pub(crate) type FunnelMessage<Input, Output, Error> = (
    Input,
    tokio::sync::oneshot::Sender<Result<Output, BatchinfError<Error>>>,
);

#[derive(Debug)]
pub(crate) struct WorkerPool<Input, Output, Error>
where
    Input: Send + 'static,
    Output: Send + 'static,
    Error: std::error::Error + Send + Sync + 'static,
{
    // Arc over a fixed-size slice — pool slots are never added or removed. Predictor rebuilds
    // happen on each worker's inference thread, so a slot's channel sender never changes.
    pool: PoolSlab<Input, Output, Error>,
    size: u8,
    start: Arc<AtomicUsize>,
}

impl<Input, Output, Error> Clone for WorkerPool<Input, Output, Error>
where
    Input: Send + 'static,
    Output: Send + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        let pool = Arc::clone(&self.pool);
        let start = Arc::clone(&self.start);
        Self {
            pool,
            size: self.size,
            start,
        }
    }
}

impl<Input, Output, Error> WorkerPool<Input, Output, Error>
where
    Input: Send + 'static,
    Output: Send + 'static,
    Error: std::error::Error + Send + Sync + 'static,
{
    pub(crate) fn new(
        pool: Vec<WorkerRef<Input, Output, Error>>,
    ) -> WorkerPool<Input, Output, Error> {
        let size = pool.len() as u8;
        let start = AtomicUsize::new(0_usize);
        WorkerPool {
            pool: pool.into(),
            size,
            start: start.into(),
        }
    }

    /// Load-aware round-robin routing.
    ///
    /// Selects a start point via global round-robin, then makes two passes over the pool from that
    /// point. The first pass only tries [WorkerStatus::Waiting] workers. The second tries every
    /// worker, so a busy worker with room in its channel can still take the request.
    ///
    /// Returns [`BatchinfError::QueueFullError`] if at least one worker is alive but every channel
    /// is full, and [`BatchinfError::NoAvailableWorkersError`] if every worker has exited or
    /// crashed.
    pub(crate) fn push(
        &self,
        mut msg: FunnelMessage<Input, Output, Error>,
    ) -> Result<(), BatchinfError<Error>> {
        // If the pool only has a single worker, then it is just dispatched.
        if self.is_single_worker() {
            let handle = &self.pool[0];
            if !handle.can_accept_secondary() {
                if handle.is_rebuilding() {
                    return Err(BatchinfError::QueueFullError);
                } else {
                    return Err(BatchinfError::NoAvailableWorkersError);
                };
            }
            match handle.push(msg) {
                QueuePushResult::Success => return Ok(()),
                QueuePushResult::QueueFull(_) => return Err(BatchinfError::QueueFullError),
                QueuePushResult::QueueClosed(_) => {
                    return Err(BatchinfError::NoAvailableWorkersError);
                }
            }
        }

        // First try uses global round robin.
        let mut sink = self.get_and_increment();

        let size = self.pool_size();

        // Pass 1: select the first worker in the waiting state. Exhaust all workers.
        for _ in 0..size {
            let handle = &self.pool[sink];
            if handle.can_accept_primary() {
                match handle.push(msg) {
                    QueuePushResult::Success => return Ok(()),
                    // When the queue is closed due to either a worker crash or exit, messages get
                    // handed back and retried.
                    QueuePushResult::QueueClosed(m) => msg = m,
                    // The resolved queue is full. Hand back the message; it is retried in pass 2.
                    QueuePushResult::QueueFull(m) => {
                        msg = m;
                    }
                }
            }

            // Subsequent attempts to resolve worker is thread local walk the worker search space
            // linearly.
            sink = (sink + 1) % usize::from(self.size);
        }

        let mut has_live_worker = false;
        let mut has_rebuilding_worker = false;

        // Try all workers if none are waiting.
        for _ in 0..size {
            let handle = &self.pool[sink];
            if handle.can_accept_secondary() {
                match handle.push(msg) {
                    QueuePushResult::Success => return Ok(()),
                    QueuePushResult::QueueFull(m) => {
                        msg = m;
                        has_live_worker = true;
                    }
                    QueuePushResult::QueueClosed(m) => {
                        msg = m;
                    }
                }
            } else if handle.is_rebuilding() {
                has_rebuilding_worker = true;
            }

            sink = (sink + 1) % usize::from(self.size);
        }

        // Resolve error to user.
        if has_live_worker || has_rebuilding_worker {
            Err(BatchinfError::QueueFullError)
        } else {
            Err(BatchinfError::NoAvailableWorkersError)
        }
    }

    /// Query the status of all workers in the pool.
    pub(crate) fn pool_status(&self) -> Vec<WorkerSnapshot> {
        let size = self.pool_size();
        let mut result = Vec::with_capacity(size);

        for i in 0..size {
            result.push(self.pool[i].snapshot());
        }
        result
    }

    /// Query the status of a single worker in the pool.
    pub(crate) fn worker_status(&self, idx: usize) -> Option<WorkerSnapshot> {
        if idx >= self.pool_size() {
            return None;
        }
        Some(self.pool[idx].snapshot())
    }

    fn is_single_worker(&self) -> bool {
        self.pool_size() == 1_usize
    }

    fn pool_size(&self) -> usize {
        self.pool.len()
    }

    /// Get thread start position for worker resolution.
    ///
    /// Search space start is round robin among threads.
    #[inline]
    fn get_and_increment(&self) -> usize {
        self.start.fetch_add(1, Ordering::AcqRel) % usize::from(self.size)
    }
}
