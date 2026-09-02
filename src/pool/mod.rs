use crate::FunnelMessage;
use crate::error::BatchinfError;
use crate::state::{WorkerRef, WorkerSnapshot, WorkerStatus};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) struct WorkerPool<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: Send + Sync + 'static,
{
    // Arc to a boxed slice implies that workers are never evicted from the pool if they ever are
    // in a bad state.
    pool: Arc<[WorkerRef<Input, Output, Error>]>,
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
        WorkerPool { pool: pool.into() }
    }

    /// Use a load aware round robin starting at a random index.
    ///
    /// Select an initial start point in the pool, and traversing to pool until all workers are
    /// exhausted. If a worker that can accept work is not found, then the first observed worker
    /// who is still alive, not [WorkerStatus::Exit] state, is selected as the fallback
    /// destination.
    pub(crate) async fn push<E: Clone + std::error::Error + Send + Sync + 'static>(
        &self,
        msg: FunnelMessage<Input, Output, Error>,
    ) -> Result<(), BatchinfError<E>> {
        let pool_size = self.pool.len();

        // If the pool only has a single worker, then it is just dispatched.
        if pool_size == 1 {
            self.pool[0].push(msg).await?;
            return Ok(());
        }

        // Select random place to start in the pool. This position is where we start from.
        let mut sink = self.get_search_start();

        // Fallback is the first non exited worker that we observe when looking for a worker that
        // can accept work.
        let mut fallback: Option<usize> = None;

        // Select the first worker in the waiting state. Exhaust all workers.
        for _ in 0..pool_size {
            let WorkerSnapshot { status, queue_len } = self.pool[sink].snapshot();
            let capacity = self.pool[sink].capacity();
            match status {
                WorkerStatus::Exit => {}
                // If worker is waiting and has capacity, route to it.
                // Additional capacity check is for the case where a worker is full, but has yet to
                // update state.
                WorkerStatus::Waiting if queue_len < capacity => {
                    self.pool[sink].push(msg).await?;
                    return Ok(());
                }
                _ => {
                    if fallback.is_none() {
                        fallback = Some(sink)
                    }
                }
            }

            sink = (sink + 1) % pool_size;
        }

        // If we are unable to find an available worker, we dispatch to the first worker we find
        // that has not/is exited.
        if let Some(fallback_sink) = fallback {
            self.pool[fallback_sink].push(msg).await?;
            return Ok(());
        }

        Err(BatchinfError::NoAvailableWorkersError)
    }

    /// Query the status of all pools.
    pub(crate) fn pool_status(&self) -> Vec<WorkerSnapshot> {
        self.pool.iter().map(|w| w.snapshot()).collect()
    }

    /// Query the status of a single worker in the pool.
    pub(crate) fn worker_status(&self, idx: usize) -> Option<WorkerSnapshot> {
        self.pool.get(idx).map(|w| w.snapshot())
    }

    // Select a random start position in the pool, rather than maintaining a round robin count.
    fn get_search_start(&self) -> usize {
        fastrand::usize(..self.pool.len())
    }
}
