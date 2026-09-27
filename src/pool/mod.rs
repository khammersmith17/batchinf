use crate::{
    error::BatchinfError,
    state::{QueuePushResult, WorkerRef, WorkerSnapshot, WorkerStatus},
};
use std::sync::{
    Arc, Weak,
    atomic::{AtomicUsize, Ordering},
};

pub(crate) type FunnelMessage<Input, Output, Error> = (
    Input,
    tokio::sync::oneshot::Sender<Result<Output, BatchinfError<Error>>>,
);

#[derive(Debug)]
pub(crate) struct WorkerPool<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    // Arc over a fixed-size slice — pool slots are never added or removed. Crashed workers are
    // restarted in-place by the control plane, which swaps the channel sender within the slot.
    pool: Arc<[WorkerRef<Input, Output, Error>]>,
    size: u8,
    start: Arc<AtomicUsize>,
}

impl<Input, Output, Error> Clone for WorkerPool<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
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
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
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

    pub(crate) fn get_weak_ref(&self) -> Weak<[WorkerRef<Input, Output, Error>]> {
        Arc::downgrade(&self.pool)
    }

    /// Use a load aware uniform random routing.  
    ///
    /// Select an initial start point in the pool, and traversing to pool until all workers are
    /// exhausted. If a worker that can accept work is not found, then the first observed worker
    /// who is still alive, not in [WorkerStatus::Exit] or [WorkerStatus::Crashed] state, is
    /// selected as the fallback destination.
    pub(crate) fn push(
        &self,
        mut msg: FunnelMessage<Input, Output, Error>,
    ) -> Result<(), BatchinfError<Error>> {
        // If the pool only has a single worker, then it is just dispatched.
        if self.is_single_worker() {
            let handle = &self.pool[0];
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

        // Fallback is the first worker not in Exit or Crashed state that we observe when looking
        // for a worker that can accept work.
        let mut fallback: Option<usize> = None;
        let size = self.pool_size();

        // Select the first worker in the waiting state. Exhaust all workers.
        for _ in 0..size {
            let handle = &self.pool[sink];
            let WorkerSnapshot { status, queue_len } = handle.snapshot();
            let capacity = handle.capacity();
            match status {
                WorkerStatus::Exit | WorkerStatus::Crashed => {}
                // If worker is waiting and has capacity, route to it.
                // Additional capacity check is for the case where a worker is full, but has yet to
                // update state.
                WorkerStatus::Waiting if queue_len < capacity => match handle.push(msg) {
                    QueuePushResult::Success => return Ok(()),
                    // When the queue is closed due to either a worker crash or exit, messages get
                    // handed back and retried.
                    QueuePushResult::QueueClosed(m) => msg = m,
                    // The resolved queue is full. Hand back message and set fallback sink.
                    // The first live worker is set to be the fallback sink.
                    QueuePushResult::QueueFull(m) => {
                        msg = m;
                        if fallback.is_none() {
                            fallback = Some(sink)
                        }
                    }
                },
                _ => {
                    if fallback.is_none() {
                        fallback = Some(sink)
                    }
                }
            }

            // Subsequent attempts to resolve worker is thread local walk the worker search space
            // linearly.
            sink = (sink + 1) % usize::from(self.size);
        }

        // If no fallback workers were identified, then no workers are available.
        let Some(fallback_sink) = fallback else {
            return Err(BatchinfError::NoAvailableWorkersError);
        };

        // If we are unable to find an available worker, we dispatch to the first worker we found
        // that has not exited or crashed.
        let handle = &self.pool[fallback_sink];
        match handle.push(msg) {
            QueuePushResult::Success => Ok(()),
            // If the resolved sink cannot accept the request, error to user code.
            QueuePushResult::QueueFull(_) => Err(BatchinfError::QueueFullError),
            QueuePushResult::QueueClosed(_) => Err(BatchinfError::NoAvailableWorkersError),
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
    /// Search space start is round robin amoong threads.
    #[inline]
    fn get_and_increment(&self) -> usize {
        self.start.fetch_add(1, Ordering::AcqRel) % usize::from(self.size)
    }
}
