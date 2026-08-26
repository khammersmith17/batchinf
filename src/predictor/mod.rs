pub trait Predictor: Clone {
    type Input: Send + Sync + 'static;
    type Output: Send + Sync + 'static;
    type Error: std::error::Error + Clone + Sync + Send + 'static;

    /// This is where the magic happens.
    ///
    /// This method is called on each inference batch and is a required implementation. It will be
    /// called once per patch. When configured with multiple workers, it will called once per batch
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
    fn predict_batch(&self, inp: &[Self::Input]) -> Result<Vec<Self::Output>, Self::Error>;
}
