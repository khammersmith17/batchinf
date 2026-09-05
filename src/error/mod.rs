use thiserror::Error;
use tokio::sync::oneshot::error::RecvError;

/// Error returned by [`Batchinf::predict`](`crate::Batchinf`) and [`Batchinf::predict_with_timeout`](`crate::Batchinf`).
#[derive(Debug, Error, Clone)]
pub enum BatchinfError<E>
where
    E: std::error::Error + Clone + Sync + Send + 'static,
{
    /// [`Predictor::predict_batch`](`crate::Predictor`) returned an error. The error is propagated to every
    /// caller whose request was part of the failed batch.
    #[error("Unable to perform inference: {0}")]
    InferenceError(E),
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
    #[error("Predictor produced an invalid number of outputs for the given input")]
    InvalidPredictorOutput,
}

impl<E> From<RecvError> for BatchinfError<E>
where
    E: std::error::Error + Clone + Sync + Send + 'static,
{
    fn from(_err: RecvError) -> BatchinfError<E> {
        BatchinfError::InternalError
    }
}
