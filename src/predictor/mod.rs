pub trait Predictor: Clone {
    type Input: Send + Sync + 'static;
    type Output: Send + Sync + 'static;
    type Error: std::error::Error + Clone + Sync + Send + 'static;

    /// This is where the magic happens.
    ///
    /// This method is called on each inference batch and is a required implementation. It will be
    /// called once per batch. When configured with multiple workers, it will called once per batch
    /// local worker.
    ///
    /// Implement this method to take in a slice of queued inference data, and perform inference on
    /// the batch of examples.
    ///
    /// Each inference result is its own result, and all inference results are returned from this
    /// method as a Vec.
    ///
    /// This trait requires clone, for the case when multiple workers are configured in the pool.
    /// The type will be cloned n - 1 times to provide each pool with isolated resources to perform
    /// inference.
    ///
    /// A panic here is caught by the Tokio task executor. Callers whose requests were in the
    /// panicking batch will receive [`crate::error::BatchinfError::InternalError`]. To make
    /// panics visible, install a panic hook via [`std::panic::set_hook`] before starting the
    /// batcher.
    fn predict_batch(&self, inp: &[Self::Input]) -> Result<Vec<Self::Output>, Self::Error>;
}
