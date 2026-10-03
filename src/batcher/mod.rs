use crate::{
    error::BatchinfError, observability::BatcherMetrics, pool::WorkerPool, state::WorkerSnapshot,
};
use std::sync::Arc;
use tokio::{
    select,
    sync::oneshot::channel as oneshot_channel,
    time::{Duration, sleep},
};

/// The public handle for submitting inference requests.
///
/// [`Batchinf`] is cheap to clone — all clones share the same underlying worker pool via [`Arc`].
/// Each clone can independently submit requests and query worker status.
///
/// Dropping all `Batchinf` clones triggers graceful shutdown: the pool stops accepting new
/// requests and each worker flushes its in-progress batch before exiting.
#[derive(Debug)]
pub struct Batchinf<Input, Output, Error>
where
    Input: Send + 'static,
    Output: Send + 'static,
    Error: std::error::Error + Clone + Send + 'static,
{
    pool: WorkerPool<Input, Output, Error>,
    obs: Option<Arc<dyn BatcherMetrics>>,
}

impl<Input, Output, Error> Clone for Batchinf<Input, Output, Error>
where
    Input: Send + 'static,
    Output: Send + 'static,
    Error: std::error::Error + Clone + Send + 'static,
{
    fn clone(&self) -> Self {
        let obs = self.obs.clone();
        let pool = self.pool.clone();
        Batchinf { pool, obs }
    }
}

impl<Input, Output, Error> Batchinf<Input, Output, Error>
where
    Input: Send + 'static,
    Output: Send + 'static,
    Error: std::error::Error + Clone + Send + 'static,
{
    pub(crate) fn new(
        pool: WorkerPool<Input, Output, Error>,
        obs: Option<Arc<dyn BatcherMetrics>>,
    ) -> Self {
        Self { pool, obs }
    }

    /// Submits an inference request and awaits the result.
    ///
    /// The input is queued and dispatched as part of a batch with other concurrent requests.
    /// Returns when inference completes or the worker exits.
    ///
    /// # Errors
    ///
    /// - [`BatchinfError::InferenceError`] — [`Predictor::predict_batch`](`crate::Predictor`) returned an error.
    /// - [`BatchinfError::InvalidPredictorOutput`] — [`Predictor::predict_batch`](`crate::Predictor`) returned a different number of outputs than inputs.
    /// - [`BatchinfError::InternalError`] — the worker exited before returning a result (e.g. after a panic).
    /// - [`BatchinfError::QueueFullError`] — all worker queues are full; the caller should retry or apply backpressure.
    /// - [`BatchinfError::NoAvailableWorkersError`] — all workers have exited or crashed.
    pub async fn predict(&self, input: Input) -> Result<Output, BatchinfError<Error>> {
        let (tx, rx) = oneshot_channel::<Result<Output, BatchinfError<Error>>>();
        self.pool.push((input, tx))?;
        rx.await?
    }

    /// Submits an inference request with a caller-side deadline.
    ///
    /// Equivalent to [`predict`](Batchinf::predict) but returns [`BatchinfError::TimeoutError`]
    /// if `timeout` elapses before the result arrives. The request may still be processed by the
    /// worker after the timeout — the result is simply discarded. All errors from [`predict`](Batchinf::predict)
    /// may also be returned if inference fails before the timeout.
    pub async fn predict_with_timeout(
        &self,
        input: Input,
        timeout: Duration,
    ) -> Result<Output, BatchinfError<Error>> {
        select! {
            res = self.predict(input) => res,
            _ = sleep(timeout) => {
                self.emit_timeout();
                Err(BatchinfError::TimeoutError)
            }
        }
    }

    /// Returns a snapshot of every worker in the pool, indexed by worker position.
    pub fn pool_status(&self) -> Vec<WorkerSnapshot> {
        self.pool.pool_status()
    }

    /// Returns a snapshot of the worker at `idx`, or `None` if out of bounds.
    pub fn worker_status(&self, idx: usize) -> Option<WorkerSnapshot> {
        self.pool.worker_status(idx)
    }

    fn emit_timeout(&self) {
        if let Some(ref obs) = self.obs {
            obs.on_request_timeout();
        }
    }
}
