use std::sync::Arc;

/// Builds the [`Predictor`] for one worker. Passed to [`get_batcher`](`crate::get_batcher`).
///
/// A factory gives the user more flexibility on how a model is constructed and orchestrated with
/// a particular device.
///
/// The argument is the worker's index, in `0..pool_size`, which a factory can use to bind each
/// worker to its own device (for example `worker_id % n_gpus` for pinning device index when
/// configuring ExecutionProvider in `ort`).
///
/// The factory is called on that worker's dedicated inference thread, and the predictor it
/// returns stays on that thread, so the predictor does not need to be `Send`, `Sync` or `Clone`.
/// The factory itself is shared by every worker, so it must be `Send + Sync`.
///
/// It is called once per worker at startup, and again each time a worker rebuilds its predictor
/// after [`Predictor::predict_batch`] panics.
///
/// Load or download model weights once, outside the factory, and capture them in the closure
/// (for example as an `Arc`) rather than fetching them on every call.
///
/// Signal failure by returning `Err`. A panic is also treated as a failure. A failed call is
/// retried as configured by
/// [`BatcherConfig::worker_start_retries`](`crate::BatcherConfig::worker_start_retries`). If
/// every attempt fails at startup, the error from the last attempt is returned by
/// [`get_batcher`](`crate::get_batcher`) as
/// [`StartupError::WorkerStartFailure`](`crate::StartupError::WorkerStartFailure`). Errors from a
/// rebuild after a panic have no caller to return to, so they are discarded.
/// ```
/// # use batchinf::PredictorFactory;
/// # use std::sync::Arc;
/// # struct MyModel { device: u8 }
/// # #[derive(Debug)] struct LoadError;
/// let n_gpus = 4;
/// let factory: PredictorFactory<MyModel, LoadError> = Arc::new(move |worker_id| {
///     Ok(MyModel { device: worker_id % n_gpus })
/// });
/// ```
pub type PredictorFactory<P, E> = Arc<dyn Fn(u8) -> Result<P, E> + Send + Sync>;

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
    /// This method is called once per batch and is a required implementation. With several
    /// workers, each worker calls its own predictor for the batches it builds.
    ///
    /// Implement this method to take the queued inference inputs and perform inference on the
    /// batch of examples. The method owns the inputs, so it can move them into whatever the model
    /// needs (a tokenizer, a tensor) without cloning them.
    ///
    /// Each inference result is its own result, and all inference results are returned from this
    /// method as a Vec.
    ///
    /// It is assumed under this contract that the output and input vectors are element wise pairs:
    /// output `i` must answer the `i`-th input in the order it was received. The inputs may be
    /// consumed or reordered internally, as long as the outputs come back in that original order.
    ///
    /// A panic here fails the current batch: every caller in it receives
    /// [`BatchinfError::InternalError`](`crate::BatchinfError::InternalError`). The worker then
    /// drops this predictor and builds a fresh one from the factory. Requests already queued on
    /// this worker wait for the rebuild and are then served by the new predictor; if the rebuild
    /// fails, they also receive `InternalError`. To make panics visible, install a panic hook via
    /// [`std::panic::set_hook`] before starting the batcher.
    ///
    /// This recovery relies on unwinding. In a binary built with `panic = "abort"`, a panic here
    /// terminates the whole process instead.
    ///
    /// Note that the input buffer will be at most `batch_size` in size.
    fn predict_batch(&mut self, inp: Vec<Self::Input>) -> Result<Vec<Self::Output>, Self::Error>;
}
