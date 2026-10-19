use crate::{
    error::BatchinfError, observability::BatcherMetrics, pool::WorkerPool,
    shutdown::ShutdownWaiter, state::WorkerSnapshot,
};
use std::sync::Arc;
use tokio::{
    select,
    sync::{oneshot::channel as oneshot_channel, watch},
    time::{Duration, sleep},
};

/// The public handle for submitting inference requests.
///
/// [`Batchinf`] is cheap to clone — all clones share the same underlying worker pool via [`Arc`].
/// Each clone can independently submit requests and query worker status.
///
/// # Shutdown
///
/// Dropping all `Batchinf` clones triggers graceful shutdown: each worker's inference thread
/// finishes the batch it is running and any batch already handed to it, then drops its predictor
/// and exits.
///
/// Requests still being accumulated into a batch are dropped without running inference. No
/// caller can be waiting on them: every caller waiting for a result holds a handle, so while one
/// is waiting the last handle has not been dropped.
///
/// Shutdown runs in the background. To wait for it to complete, use
/// [`shutdown_waiter`](Batchinf::shutdown_waiter).
///
/// Shutdown starts only when the *last* handle is dropped. A clone held elsewhere, such as in a
/// spawned task or in a web framework's state, keeps every worker running:
///
/// ```
/// # use batchinf::{BatcherConfig, Predictor, get_batcher};
/// # use std::{num::NonZeroU32, sync::Arc, time::Duration};
/// # #[derive(Debug)]
/// # struct ModelError;
/// # impl std::fmt::Display for ModelError {
/// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "model error") }
/// # }
/// # impl std::error::Error for ModelError {}
/// # struct Doubler;
/// # impl Predictor for Doubler {
/// #     type Input = u32;
/// #     type Output = u32;
/// #     type Error = ModelError;
/// #     fn predict_batch(&mut self, inputs: &[u32]) -> Result<Vec<u32>, ModelError> {
/// #         Ok(inputs.iter().map(|x| x * 2).collect())
/// #     }
/// # }
/// # #[tokio::main(flavor = "multi_thread")]
/// # async fn main() {
/// let batcher = get_batcher(
///     Arc::new(|_worker_id| Ok::<_, ModelError>(Doubler)),
///     BatcherConfig::builder(NonZeroU32::new(8).unwrap(), Duration::from_millis(5)).build(),
/// )
/// .await
/// .unwrap();
///
/// // Get the waiter first: once every handle is gone there is nothing left to ask.
/// let waiter = batcher.shutdown_waiter();
///
/// // A spawned task owns a clone of the handle.
/// let handle = batcher.clone();
/// let task = tokio::spawn(async move { handle.predict(21).await });
///
/// // Will not shutdown handle is the task is dropped.
/// drop(batcher);
/// assert_eq!(task.await.unwrap().unwrap(), 42);
///
/// // The task has finished and dropped its clone, the last handle.
/// // Shutdown is now under way, and the waiter resolves once every predictor has been dropped.
/// waiter.wait().await;
/// # }
/// ```
#[derive(Debug)]
pub struct Batchinf<Input, Output, Error>
where
    Input: Send + 'static,
    Output: Send + 'static,
    Error: std::error::Error + Send + Sync + 'static,
{
    inner: Arc<BatchinfInner<Input, Output, Error>>,
}

/// State shared by every clone of a [`Batchinf`]. Dropping the last clone drops the pool, which
/// closes every worker's request channel and starts shutdown.
#[derive(Debug)]
struct BatchinfInner<Input, Output, Error>
where
    Input: Send + 'static,
    Output: Send + 'static,
    Error: std::error::Error + Send + Sync + 'static,
{
    pool: WorkerPool<Input, Output, Error>,
    obs: Option<Arc<dyn BatcherMetrics>>,
    // Only the receiver is kept here, so a `ShutdownWaiter` cloned from it never keeps the pool
    // alive.
    shutdown_rx: watch::Receiver<()>,
}

impl<Input, Output, Error> Clone for Batchinf<Input, Output, Error>
where
    Input: Send + 'static,
    Output: Send + 'static,
    Error: std::error::Error + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Batchinf {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<Input, Output, Error> Batchinf<Input, Output, Error>
where
    Input: Send + 'static,
    Output: Send + 'static,
    Error: std::error::Error + Send + Sync + 'static,
{
    pub(crate) fn new(
        pool: WorkerPool<Input, Output, Error>,
        obs: Option<Arc<dyn BatcherMetrics>>,
        shutdown_rx: watch::Receiver<()>,
    ) -> Self {
        let inner = BatchinfInner {
            pool,
            obs,
            shutdown_rx,
        };
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Submits an inference request and awaits the result, failing immediately if no worker has
    /// room for it.
    ///
    /// The fail-fast counterpart of [`predict`](Batchinf::predict). If every worker's queue is
    /// full, this returns [`BatchinfError::QueueFullError`] straight away instead of waiting, so
    /// the caller can shed load, for example by answering with a 503. Otherwise the input is
    /// queued and dispatched as part of a batch with other concurrent requests, and this returns
    /// when inference completes or the worker exits.
    ///
    /// # Errors
    ///
    /// - [`BatchinfError::QueueFullError`] — every available worker's queue is full, or the only
    ///   workers left are rebuilding their predictor. The request was not queued.
    /// - [`BatchinfError::NoAvailableWorkersError`] — all workers have exited or crashed.
    /// - [`BatchinfError::InferenceError`] — [`Predictor::predict_batch`](`crate::Predictor`) returned an error.
    /// - [`BatchinfError::InvalidPredictorOutput`] — [`Predictor::predict_batch`](`crate::Predictor`) returned a different number of outputs than inputs.
    /// - [`BatchinfError::InternalError`] — the worker exited before returning a result (e.g. after a panic).
    pub async fn try_predict(&self, input: Input) -> Result<Output, BatchinfError<Error>> {
        let (tx, rx) = oneshot_channel::<Result<Output, BatchinfError<Error>>>();
        self.inner.pool.push((input, tx))?;
        rx.await?
    }

    /// Submits an inference request and awaits the result, waiting for room if every worker's
    /// queue is full.
    ///
    /// The input is queued and dispatched as part of a batch with other concurrent requests.
    /// If no worker has room, this waits until one does: a queued request is picked up, or a
    /// worker finishes rebuilding its predictor. That wait is the backpressure, so this never
    /// returns [`BatchinfError::QueueFullError`]. Waiting callers are not served in any
    /// particular order. To bound the wait, use
    /// [`predict_with_timeout`](Batchinf::predict_with_timeout); to fail immediately instead of
    /// waiting, use [`try_predict`](Batchinf::try_predict).
    ///
    /// Dropping the returned future is safe at any point. If it is dropped while waiting for
    /// room, the request was never queued. If it is dropped after being queued, the request is
    /// skipped unless its batch has already started.
    ///
    /// # Errors
    ///
    /// - [`BatchinfError::NoAvailableWorkersError`] — all workers have exited or crashed. Returned
    ///   immediately, and also to callers already waiting for room when the last worker stops.
    /// - [`BatchinfError::InferenceError`] — [`Predictor::predict_batch`](`crate::Predictor`) returned an error.
    /// - [`BatchinfError::InvalidPredictorOutput`] — [`Predictor::predict_batch`](`crate::Predictor`) returned a different number of outputs than inputs.
    /// - [`BatchinfError::InternalError`] — the worker exited before returning a result (e.g. after a panic).
    pub async fn predict(&self, mut input: Input) -> Result<Output, BatchinfError<Error>> {
        let (mut tx, rx) = oneshot_channel::<Result<Output, BatchinfError<Error>>>();
        (input, tx) = match self.inner.pool.try_push((input, tx)) {
            Ok(_) => return rx.await?,
            Err((msg, BatchinfError::QueueFullError)) => msg,
            Err((_, err)) => return Err(err),
        };

        loop {
            let notification = self.inner.pool.get_notified();
            tokio::pin!(notification);

            notification.as_mut().enable();

            (input, tx) = match self.inner.pool.try_push((input, tx)) {
                Ok(_) => return rx.await?,
                Err((msg, BatchinfError::QueueFullError)) => msg,
                Err((_, err)) => return Err(err),
            };

            notification.await;
        }
    }

    /// Submits an inference request and blocks the current thread until the result arrives.
    ///
    /// The blocking counterpart of [`try_predict`](Batchinf::try_predict), for callers that are
    /// not in async code, such as a plain OS thread. Like `try_predict`, it does not wait for
    /// room: if every worker's queue is full it returns [`BatchinfError::QueueFullError`]
    /// immediately. It returns the same errors as [`try_predict`](Batchinf::try_predict).
    ///
    /// # Panics
    ///
    /// Panics if called from within an asynchronous execution context, such as inside an async
    /// function running on a Tokio runtime. Use [`predict`](Batchinf::predict) or
    /// [`try_predict`](Batchinf::try_predict) there instead.
    pub fn blocking_try_predict(&self, input: Input) -> Result<Output, BatchinfError<Error>> {
        let (tx, rx) = oneshot_channel::<Result<Output, BatchinfError<Error>>>();
        self.inner.pool.push((input, tx))?;
        rx.blocking_recv()?
    }

    /// Submits an inference request with a caller-side deadline.
    ///
    /// Equivalent to [`predict`](Batchinf::predict) but returns [`BatchinfError::TimeoutError`]
    /// if `timeout` elapses before the result arrives. The deadline covers the whole call: the
    /// wait for room in a worker's queue, the time spent queued, and inference itself.
    ///
    /// A request that times out while waiting for room was never queued. One that times out
    /// after being queued is skipped if its batch has not started yet; if inference is already
    /// running, it completes and the result is discarded. All errors from
    /// [`predict`](Batchinf::predict) may also be returned if the request fails before the
    /// timeout.
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

    /// Returns a [`ShutdownWaiter`] that resolves once this batcher has fully shut down.
    /// Workers exit when all associated [Batchinf] handles are dropped.
    ///
    /// Thus, the waiter does not keep the batcher alive and depends on all instances being droped.
    /// Get it before dropping the last handle.
    pub fn shutdown_waiter(&self) -> ShutdownWaiter {
        ShutdownWaiter::new(self.inner.shutdown_rx.clone())
    }

    /// Returns a snapshot of every worker in the pool, indexed by worker position.
    pub fn pool_status(&self) -> Vec<WorkerSnapshot> {
        self.inner.pool.pool_status()
    }

    /// Returns a snapshot of the worker at `idx`, or `None` if out of bounds.
    pub fn worker_status(&self, idx: usize) -> Option<WorkerSnapshot> {
        self.inner.pool.worker_status(idx)
    }

    fn emit_timeout(&self) {
        if let Some(ref obs) = self.inner.obs {
            obs.on_request_timeout();
        }
    }
}
