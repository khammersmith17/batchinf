use thiserror::Error;

/// Error returned by [`Batchinf::predict`] and [`Batchinf::predict_with_timeout`].
#[derive(Debug, Error, Clone)]
pub enum BatchinfError<E>
where
    E: std::error::Error + Clone + Sync + Send + 'static,
{
    /// [`Predictor::predict_batch`] returned an error. The error is propagated to every
    /// caller whose request was part of the failed batch.
    #[error("Unable to perform inference: {0}")]
    InferenceError(E),
    /// The worker exited before returning a result, typically caused by a panic inside
    /// [`Predictor::predict_batch`].
    #[error("Internal Error")]
    InternalError,
    /// The request timed out before inference completed. Only returned by
    /// [`Batchinf::predict_with_timeout`]. The worker may still process the request; the
    /// result is discarded.
    #[error("Inference request timed out")]
    TimeoutError,
    #[error("No available workers")]
    NoAvailableWorkersError,
}
