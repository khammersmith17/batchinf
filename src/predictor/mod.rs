/// Each worker builds its own predictor on its dedicated inference thread, via the factory passed
/// to [`get_batcher`](`crate::get_batcher`). The predictor is created, used and dropped on that
/// thread, so it does not need to be `Clone`, `Send` or `Sync`.
pub trait Predictor {
    /// A single inference request, as passed to [`Batchinf::predict`](`crate::Batchinf::predict`).
    /// Sent from the caller to the worker's inference thread.
    type Input: Send + 'static;
    /// The result for a single request. Sent from the inference thread back to the caller.
    type Output: Send + 'static;
    /// The error [`predict_batch`](Predictor::predict_batch) returns when a whole batch fails.
    /// One instance is shared by every caller in the failed batch, as
    /// [`BatchinfError::InferenceError`](`crate::BatchinfError::InferenceError`), which is why it
    /// must be `Sync`.
    type Error: std::error::Error + Send + Sync + 'static;

    /// This is where the magic happens.
    ///
    /// This method is called on each inference batch and is a required implementation. It will be
    /// called once per batch. When configured with multiple workers, it will be called once per batch
    /// local worker.
    ///
    /// Implement this method to take in a slice of queued inference data, and perform inference on
    /// the batch of examples.
    ///
    /// Each inference result is its own result, and all inference results are returned from this
    /// method as a Vec.
    ///
    /// It is assumed under this contract that the output and input vectors are element wise pairs;
    /// the order of outputs must match the order of inputs.
    ///
    /// A panic here fails the current batch: the oneshot senders for all requests in the batch are
    /// dropped, so callers receive [`crate::error::BatchinfError::InternalError`]. The worker then
    /// discards this predictor and builds a fresh one from the factory. Requests queued behind the
    /// panicking batch are not affected. To make panics visible, install a panic hook via
    /// [`std::panic::set_hook`] before starting the batcher.
    ///
    /// This recovery relies on unwinding. In a binary built with `panic = "abort"`, a panic here
    /// terminates the whole process instead.
    fn predict_batch(&mut self, inp: &[Self::Input]) -> Result<Vec<Self::Output>, Self::Error>;
}
