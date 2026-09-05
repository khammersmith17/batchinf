/// Enum describing what triggered a batch to fire.
#[derive(Debug, Clone, Copy)]
pub enum BatchTrigger {
    Capacity,
    Timeout,
}

/// This crate provides the contract for emitting observability metrics. Implement the backend.
pub trait BatcherMetrics: std::fmt::Debug + Send + Sync + 'static {
    /// Fires at the point a batch is triggered to fire. Takes in the trigger type, [BatchTrigger]
    ///. This may provide some information that can help tune the batch size and the timeout for a
    ///batch.
    fn on_batch_trigger(&self, batch_size: usize, trigger: BatchTrigger);

    /// Fires after a batch is complete and successful. Clocks how fast the
    /// [crate::predictor::Predictor::predict_batch] runs on a batch size.
    fn on_batch_complete_ok(&self, batch_size: usize, latency: tokio::time::Duration);

    /// Fires after a batch is complete and not successful. Clocks how fast the
    /// [crate::predictor::Predictor::predict_batch] runs on a batch size.
    fn on_batch_complete_err(&self, batch_size: usize);

    /// Fires when an inference request queued through
    /// [crate::batcher::Batchinf::predict_with_timeout] times out.
    fn on_request_timeout(&self);

    /// Fires in a background supervisor control plane and emits the total queue depth, which is
    /// the total number of inference requests waiting to be serviced.
    fn on_queue_depth(&self, queue_depth: usize);
}

pub(crate) struct InfBatchMetrics {
    pub(crate) size: usize,
    pub(crate) latency: tokio::time::Duration,
}
