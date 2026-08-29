use thiserror::Error;

#[derive(Debug, Error, Clone)]
pub enum BatchinfError<E>
where
    E: std::error::Error + Clone + Sync + Send + 'static,
{
    #[error("Unable to perform inference: {0}")]
    InferenceError(E),
    #[error("Internal Error")]
    InternalError,
}
