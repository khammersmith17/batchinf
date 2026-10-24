/// Enum describing what triggered a batch to fire.
#[derive(Debug, Clone, Copy)]
pub enum BatchTrigger {
    /// The batch reached `batch_size` requests.
    Capacity,
    /// `batch_timeout` elapsed since the first request of the batch, before it filled up.
    Timeout,
}

/// Hooks for reporting batcher metrics to any backend. Implement this trait and pass it to
/// [`BatcherConfigBuilder::metrics`](`crate::BatcherConfigBuilder::metrics`).
///
/// Every method defaults to a no-op, so implement only the hooks you need.
///
/// # Contract
///
/// Hooks are called inline: [`on_batch_trigger`](Self::on_batch_trigger) and
/// [`on_queue_depth`](Self::on_queue_depth) on the worker's batching task (a tokio task),
/// [`on_request_timeout`](Self::on_request_timeout) on the caller's task, and the others on the
/// worker's inference thread. Implementations must be cheap and must not block: record into
/// atomics or a lock-free metrics registry, or forward to a channel, and do expensive work such
/// as network export elsewhere. A slow hook delays batches and reduces throughput.
///
/// Implementations must not panic. Unlike a panic in
/// [`Predictor::predict_batch`](`crate::Predictor::predict_batch`), which the worker recovers
/// from, a panic in a hook is not caught and stops the worker.
pub trait BatcherMetrics: Send + Sync + 'static {
    /// Fires at the point a batch is triggered to fire. Takes in the trigger type, [BatchTrigger].
    /// This may provide some information that can help tune the batch size and the timeout for a
    /// batch.
    ///
    /// `batch_size` is the number of requests accumulated when the batch fired. Requests that
    /// were cancelled while waiting are removed just before inference, so the size reported by
    /// [`on_batch_complete_ok`](BatcherMetrics::on_batch_complete_ok) and
    /// [`on_batch_complete_err`](BatcherMetrics::on_batch_complete_err) can be smaller, and a
    /// batch whose requests were all cancelled reports no completion at all.
    #[allow(unused_variables)]
    fn on_batch_trigger(&self, batch_size: usize, trigger: BatchTrigger) {}

    /// Fires after a batch completes successfully.
    ///
    /// `latency` is the wall-clock time spent inside
    /// [`Predictor::predict_batch`](`crate::predictor::Predictor::predict_batch`) only. It excludes
    /// time queued in the batcher and result delivery, so it reflects the cost of the predictor
    /// implementation at this batch size. Use it to profile and tune the predictor; measure
    /// per-request latency around [`Batchinf::predict`](`crate::batcher::Batchinf::predict`)
    /// instead.
    #[allow(unused_variables)]
    fn on_batch_complete_ok(&self, batch_size: usize, latency: tokio::time::Duration) {}

    /// Fires after a batch completes with an error: either
    /// [`Predictor::predict_batch`](`crate::predictor::Predictor::predict_batch`) returned `Err`,
    /// or it returned a different number of outputs than inputs.
    ///
    /// `latency` is measured the same way as in
    /// [`on_batch_complete_ok`](BatcherMetrics::on_batch_complete_ok): time spent inside
    /// `predict_batch` only. Slow failures, such as device timeouts or out-of-memory errors at
    /// large batch sizes, show up here.
    #[allow(unused_variables)]
    fn on_batch_complete_err(&self, batch_size: usize, latency: tokio::time::Duration) {}

    /// Fires when a call to
    /// [`Batchinf::predict_with_timeout`](`crate::batcher::Batchinf::predict_with_timeout`) times
    /// out, whether the request was still waiting for room in a worker's queue, was queued, or was
    /// already running.
    #[allow(unused_variables)]
    fn on_request_timeout(&self) {}

    /// Reports the number of requests waiting in one worker's queue, not yet pulled into a batch.
    ///
    /// Emitted per worker at batch boundaries: when a batch is handed to the inference thread,
    /// and again once the hand-off completes, which captures any backlog that built up while the
    /// inference thread was busy or rebuilding. Nothing is emitted while a worker is idle, so
    /// treat this as a gauge that keeps its last value. Aggregate across workers as needed: the
    /// maximum shows whether any single worker is backed up, the sum shows total backlog.
    #[allow(unused_variables)]
    fn on_queue_depth(&self, worker_id: u8, queue_depth: usize) {}

    /// Fires on the worker's inference thread when
    /// [`Predictor::predict_batch`](`crate::predictor::Predictor::predict_batch`) panics, before
    /// the predictor is rebuilt from the factory.
    fn on_worker_panic(&self) {}
}

// Implementations are not required to be `Debug`, since metrics backends often hold handles that
// are not. Implementing it on the trait object instead keeps `#[derive(Debug)]` working on every
// type that holds an `Arc<dyn BatcherMetrics>`, such as `BatcherConfig`.
impl std::fmt::Debug for dyn BatcherMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("dyn BatcherMetrics")
    }
}

pub(crate) mod emitters {
    use super::{BatchTrigger, BatcherMetrics, InfBatchMetrics};
    use std::sync::Arc;

    pub(crate) fn emit_batch_start(
        obs: Option<Arc<dyn BatcherMetrics>>,
        trigger_type: BatchTrigger,
        size: usize,
    ) {
        if let Some(obs) = obs {
            obs.on_batch_trigger(size, trigger_type);
        }
    }

    pub(crate) fn emit_inference_ok(
        obs: Option<Arc<dyn BatcherMetrics>>,
        metrics: InfBatchMetrics,
    ) {
        if let Some(obs) = obs {
            let InfBatchMetrics { size, latency } = metrics;
            obs.on_batch_complete_ok(size, latency)
        }
    }

    pub(crate) fn emit_inference_err(
        obs: Option<Arc<dyn BatcherMetrics>>,
        metrics: InfBatchMetrics,
    ) {
        if let Some(obs) = obs {
            let InfBatchMetrics { size, latency } = metrics;
            obs.on_batch_complete_err(size, latency)
        }
    }

    pub(crate) fn emit_worker_panic(obs: Option<Arc<dyn BatcherMetrics>>) {
        if let Some(obs) = obs {
            obs.on_worker_panic();
        }
    }

    pub(crate) fn emit_queue_depth(
        obs: Option<Arc<dyn BatcherMetrics>>,
        worker_id: u8,
        depth: usize,
    ) {
        if let Some(obs) = obs {
            obs.on_queue_depth(worker_id, depth);
        }
    }
}

pub(crate) struct InfBatchMetrics {
    pub(crate) size: usize,
    pub(crate) latency: tokio::time::Duration,
}
