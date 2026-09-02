use crate::error::BatchinfError;
use crate::observability::BatcherMetrics;
use crate::pool::WorkerPool;
use crate::state::WorkerSnapshot;
use std::sync::Arc;
use tokio::select;
use tokio::sync::oneshot::{Receiver as OneshotReceiver, channel as oneshot_channel};
use tokio::time::{Duration, sleep};

/// The public handle for submitting inference requests.
///
/// `Batchinf` is cheap to clone — all clones share the same underlying worker pool via [`Arc`].
/// Each clone can independently submit requests and query worker status.
///
/// Dropping all `Batchinf` clones triggers graceful shutdown: the pool stops accepting new
/// requests and each worker flushes its in-progress batch before exiting.
#[derive(Debug, Clone)]
pub struct Batchinf<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: Send + Sync + 'static,
{
    pool: WorkerPool<Input, Output, Error>,
    obs: Option<Arc<dyn BatcherMetrics>>,
}

impl<Input, Output, Error> Batchinf<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
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
    /// - [`BatchinfError::InferenceError`] — [`Predictor::predict_batch`] returned an error.
    /// - [`BatchinfError::InternalError`] — the worker exited before returning a result.
    pub async fn predict(&self, input: Input) -> Result<Output, BatchinfError<Error>> {
        let (tx, rx) = oneshot_channel::<Result<Output, Error>>();
        self.pool.push((input, tx)).await?;
        handle_recv(rx).await
    }

    /// Submits an inference request with a caller-side deadline.
    ///
    /// Equivalent to [`predict`](Batchinf::predict) but returns [`BatchinfError::TimeoutError`]
    /// if `timeout` elapses before the result arrives. The request may still be processed by the
    /// worker after the timeout — the result is simply discarded.
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

async fn handle_recv<Output, Error>(
    rx: OneshotReceiver<Result<Output, Error>>,
) -> Result<Output, BatchinfError<Error>>
where
    Output: Send + Sync + 'static,
    Error: std::error::Error + Clone + Send + Sync + 'static,
{
    match rx.await {
        Ok(result) => match result {
            Ok(r) => Ok(r),
            Err(e) => Err(BatchinfError::InferenceError(e)),
        },
        Err(_) => Err(BatchinfError::InternalError),
    }
}
