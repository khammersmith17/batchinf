use std::sync::Arc;
use thiserror::Error;
use tokio::sync::oneshot::error::RecvError;

/// Error returned by [`Batchinf::predict`](`crate::Batchinf`) and [`Batchinf::predict_with_timeout`](`crate::Batchinf`).
#[derive(Debug, Error)]
pub enum BatchinfError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    /// [`Predictor::predict_batch`](`crate::Predictor`) returned an error. The error is propagated to every
    /// caller whose request was part of the failed batch.
    #[error("Unable to perform inference: {0}")]
    InferenceError(Arc<E>),
    /// The worker exited before returning a result, typically caused by a panic inside
    /// [`Predictor::predict_batch`](`crate::Predictor`).
    #[error("Internal Error")]
    InternalError,
    /// The request timed out before inference completed. Only returned by
    /// [`Batchinf::predict_with_timeout`](`crate::Batchinf`). The worker may still process the request; the
    /// result is discarded.
    #[error("Inference request timed out")]
    TimeoutError,
    /// All workers have exited or crashed and no request could be dispatched.
    #[error("No available workers")]
    NoAvailableWorkersError,
    /// Every worker's channel is full. The caller should retry or apply backpressure.
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
pub enum StartupError {
    /// Not every worker built its predictor within
    /// [`BatcherConfig::startup_timeout`](`crate::BatcherConfig::startup_timeout`).
    #[error("Workers did not start within timeout bounds.")]
    Timeout,
    /// A worker's predictor factory failed on every attempt. See
    /// [`BatcherConfig::worker_start_retries`](`crate::BatcherConfig::worker_start_retries`).
    #[error("Worker {worker_id} failed to start.")]
    WorkerStartFailure {
        /// Index of the worker whose factory failed, in `0..pool_size`.
        worker_id: u8,
    },
    /// A worker stopped before reporting whether its predictor was built, for example because
    /// its inference thread could not be spawned.
    #[error("Worker unable to start due to internal error")]
    InternalError,
}
