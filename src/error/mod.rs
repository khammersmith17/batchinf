use std::sync::Arc;
use thiserror::Error;
use tokio::sync::oneshot::error::RecvError;

/// Error returned by the request methods on [`Batchinf`](`crate::Batchinf`):
/// [`predict`](`crate::Batchinf::predict`), [`try_predict`](`crate::Batchinf::try_predict`),
/// [`predict_with_timeout`](`crate::Batchinf::predict_with_timeout`) and
/// [`blocking_try_predict`](`crate::Batchinf::blocking_try_predict`).
#[derive(Debug, Error)]
pub enum BatchinfError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    /// [`Predictor::predict_batch`](`crate::Predictor`) returned an error. The error is propagated to every
    /// caller whose request was part of the failed batch.
    ///
    /// The predictor's error is this error's [`source`](std::error::Error::source), not part of
    /// its message, so error reporters that print the chain show it once. Match on the variant to
    /// get the error itself.
    #[error("Unable to perform inference")]
    InferenceError(#[source] Arc<E>),
    /// The worker exited before returning a result, typically caused by a panic inside
    /// [`Predictor::predict_batch`](`crate::Predictor`).
    #[error("Internal Error")]
    InternalError,
    /// The deadline passed before a result arrived. Only returned by
    /// [`Batchinf::predict_with_timeout`](`crate::Batchinf::predict_with_timeout`). The deadline
    /// covers waiting for room, queueing and inference. If inference on the request had already
    /// started it still completes, and the result is discarded.
    #[error("Inference request timed out")]
    TimeoutError,
    /// All workers have exited or crashed and no request could be dispatched.
    #[error("No available workers")]
    NoAvailableWorkersError,
    /// No worker has room for the request right now: every available worker's queue is full, or
    /// the only workers left are rebuilding their predictor. The request was not queued.
    ///
    /// Only returned by the fail-fast methods,
    /// [`Batchinf::try_predict`](`crate::Batchinf::try_predict`) and
    /// [`Batchinf::blocking_try_predict`](`crate::Batchinf::blocking_try_predict`). The caller should
    /// retry later or shed load. [`Batchinf::predict`](`crate::Batchinf::predict`) waits for
    /// room instead and never returns this.
    #[error("All worker queues are full")]
    QueueFullError,
    /// [`Predictor::predict_batch`](`crate::Predictor`) returned a `Vec` whose length does not
    /// match the number of inputs in the batch. The error is propagated to every caller whose
    /// request was part of the affected batch.
    #[error("Predictor produced an invalid number of outputs for the given input")]
    InvalidPredictorOutput,
}

impl<E> Clone for BatchinfError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn clone(&self) -> BatchinfError<E> {
        match self {
            Self::InferenceError(e) => {
                let err = Arc::clone(e);
                Self::InferenceError(err)
            }
            Self::InternalError => Self::InternalError,
            Self::TimeoutError => Self::TimeoutError,
            Self::NoAvailableWorkersError => Self::NoAvailableWorkersError,
            Self::QueueFullError => Self::QueueFullError,
            Self::InvalidPredictorOutput => Self::InvalidPredictorOutput,
        }
    }
}

impl<E> From<RecvError> for BatchinfError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn from(_err: RecvError) -> BatchinfError<E> {
        BatchinfError::InternalError
    }
}

/// Error returned by [`get_batcher`](`crate::get_batcher`) when the worker pool fails to start.
///
/// When startup fails, every worker that did start is shut down again. A predictor factory call
/// that is already running cannot be interrupted, so it runs to completion in the background and
/// its predictor is then dropped.
#[derive(Debug, Error, Copy, Clone, PartialEq, Eq)]
pub enum StartupError<E> {
    /// Not every worker built its predictor within
    /// [`BatcherConfig::startup_timeout`](`crate::BatcherConfig::startup_timeout`).
    #[error("Workers did not start within timeout bounds.")]
    Timeout,
    /// A worker's predictor factory failed on every attempt, returning `Err` on the last one.
    /// See [`BatcherConfig::worker_start_retries`](`crate::BatcherConfig::worker_start_retries`).
    #[error("Worker {worker_id} failed to start: {err}")]
    WorkerStartFailure {
        /// Index of the worker whose factory failed, in `0..pool_size`.
        worker_id: u8,
        /// The error the factory returned on its last attempt.
        err: E,
    },
    /// A worker's predictor factory failed on every attempt, panicking on the last one.
    #[error("Worker {worker_id} failed to start: the predictor factory panicked")]
    WorkerStartPanic {
        /// Index of the worker whose factory panicked, in `0..pool_size`.
        worker_id: u8,
    },
    /// A worker stopped before reporting whether its predictor was built, for example because
    /// its inference thread could not be spawned.
    #[error("Worker unable to start due to internal error")]
    InternalError,
}
