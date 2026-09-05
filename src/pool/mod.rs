use crate::error::BatchinfError;
use crate::state::{QueuePushResult, WorkerRef, WorkerSnapshot, WorkerStatus};
use std::sync::{Arc, Weak};
use tokio::sync::RwLock;

pub(crate) type FunnelMessage<Input, Output, Error> =
    (Input, tokio::sync::oneshot::Sender<Result<Output, Error>>);

#[derive(Debug)]
pub(crate) struct WorkerPool<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: Send + Sync + 'static,
{
    // Arc over a fixed-size slice — pool slots are never added or removed. Crashed workers are
    // restarted in-place by the control plane, which swaps the channel sender within the slot.
    pool: Arc<[RwLock<WorkerRef<Input, Output, Error>>]>,
    size: usize,
}

impl<Input, Output, Error> Clone for WorkerPool<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            pool: Arc::clone(&self.pool),
            size: self.size,
        }
    }
}

impl<Input, Output, Error> WorkerPool<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: Send + Sync + 'static,
{
    pub(crate) fn new(
        pool: Vec<WorkerRef<Input, Output, Error>>,
    ) -> WorkerPool<Input, Output, Error> {
        let size = pool.len();
        let pool: Vec<RwLock<WorkerRef<Input, Output, Error>>> =
            pool.into_iter().map(|p| RwLock::new(p)).collect();
        WorkerPool {
            pool: pool.into(),
            size,
        }
    }

    pub(crate) fn get_weak_ref(&self) -> Weak<[RwLock<WorkerRef<Input, Output, Error>>]> {
        Arc::downgrade(&self.pool)
    }

    /// Use a load aware round robin starting at a random index.
    ///
    /// Select an initial start point in the pool, and traversing to pool until all workers are
    /// exhausted. If a worker that can accept work is not found, then the first observed worker
    /// who is still alive, not [WorkerStatus::Exit] state, is selected as the fallback
    /// destination.
    pub(crate) async fn push<E: Clone + std::error::Error + Send + Sync + 'static>(
        &self,
        mut msg: FunnelMessage<Input, Output, Error>,
    ) -> Result<(), BatchinfError<E>> {
        // If the pool only has a single worker, then it is just dispatched.
        if self.size == 1 {
            let handle = self.pool[0].read().await;
            match handle.push(msg) {
                QueuePushResult::Success => return Ok(()),
                QueuePushResult::QueueFull(_) => return Err(BatchinfError::QueueFullError),
                QueuePushResult::QueueClosed(_) => {
                    return Err(BatchinfError::NoAvailableWorkersError);
                }
            }
        }

        // Select random place to start in the pool. This position is where we start from.
        let mut sink = self.get_search_start();

        // Fallback is the first non exited worker that we observe when looking for a worker that
        // can accept work.
        let mut fallback: Option<usize> = None;

        // Select the first worker in the waiting state. Exhaust all workers.
        for _ in 0..self.size {
            let handle = self.pool[sink].read().await;
            let WorkerSnapshot { status, queue_len } = handle.snapshot();
            let capacity = handle.capacity();
            match status {
                WorkerStatus::Exit | WorkerStatus::Crashed => {}
                // If worker is waiting and has capacity, route to it.
                // Additional capacity check is for the case where a worker is full, but has yet to
                // update state.
                WorkerStatus::Waiting if queue_len < capacity => match handle.push(msg) {
                    QueuePushResult::Success => return Ok(()),
                    QueuePushResult::QueueClosed(m) => msg = m,
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

            sink = (sink + 1) % self.size;
        }

        // If not fallback workers where identified, then no workers are available.
        let Some(fallback_sink) = fallback else {
            return Err(BatchinfError::NoAvailableWorkersError);
        };

        // If we are unable to find an available worker, we dispatch to the first worker we find
        // that has not/is exited.
        let handle = self.pool[fallback_sink].read().await;
        match handle.push(msg) {
            QueuePushResult::Success => Ok(()),
            QueuePushResult::QueueFull(_) => Err(BatchinfError::QueueFullError),
            QueuePushResult::QueueClosed(_) => Err(BatchinfError::NoAvailableWorkersError),
        }
    }

    /// Query the status of all pools.
    pub(crate) async fn pool_status(&self) -> Vec<WorkerSnapshot> {
        let mut result = Vec::with_capacity(self.size);

        for i in 0..self.size {
            let handle = self.pool[i].read().await;
            result.push(handle.snapshot());
        }
        result
    }

    /// Query the status of a single worker in the pool.
    pub(crate) async fn worker_status(&self, idx: usize) -> Option<WorkerSnapshot> {
        if idx >= self.size {
            return None;
        }
        let handle = self.pool[idx].read().await;
        Some(handle.snapshot())
    }

    // Select a random start position in the pool, rather than maintaining a round robin count.
    fn get_search_start(&self) -> usize {
        fastrand::usize(..self.pool.len())
    }
}
