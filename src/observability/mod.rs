/// Enum describing what triggered a batch to fire.
#[derive(Debug, Clone, Copy)]
pub enum BatchTrigger {
    Capacity,
    Timeout,
}

/// This trait provides the contract for emitting observability metrics. Implement the backend.
pub trait BatcherMetrics: std::fmt::Debug + Send + Sync + 'static {
    /// Fires at the point a batch is triggered to fire. Takes in the trigger type, [BatchTrigger].
    /// This may provide some information that can help tune the batch size and the timeout for a
    /// batch.
    fn on_batch_trigger(&self, batch_size: usize, trigger: BatchTrigger);

    /// Fires after a batch is complete and successful. Clocks how fast the
    /// [`Predictor::predict_batch`](`crate::predictor::Predictor::predict_batch`) runs on a batch size.
    fn on_batch_complete_ok(&self, batch_size: usize, latency: tokio::time::Duration);

    /// Fires after a batch completes with an error.
    fn on_batch_complete_err(&self, batch_size: usize);

    /// Fires when an inference request queued through
    /// [`Batchinf::predict_with_timeout`](`crate::batcher::Batchinf::predict_with_timeout`) times out.
    fn on_request_timeout(&self);

    /// Intended to report the total number of inference requests waiting to be serviced.
    ///
    /// Not currently emitted: queue-depth reporting is being reworked.
    fn on_queue_depth(&self, queue_depth: usize);

    /// Fires on the worker's inference thread when
    /// [`Predictor::predict_batch`](`crate::predictor::Predictor::predict_batch`) panics, before
    /// the predictor is rebuilt from the factory.
    fn on_worker_panic(&self);

    fn emit_batch_start(&self, trigger_type: BatchTrigger, size: usize) {
        self.on_batch_trigger(size, trigger_type)
    }

    // Emit a succesful inference.
    fn emit_inference_ok(&self, metrics: InfBatchMetrics) {
        let InfBatchMetrics { size, latency } = metrics;
        self.on_batch_complete_ok(size, latency)
    }

    // Emit and unsuccesful inference.
    fn emit_inference_err(&self, size: usize) {
        self.on_batch_complete_err(size)
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
            obs.emit_batch_start(trigger_type, size);
        }
    }

    pub(crate) fn emit_inference_ok(
        obs: Option<Arc<dyn BatcherMetrics>>,
        metrics: InfBatchMetrics,
    ) {
        if let Some(obs) = obs {
            obs.emit_inference_ok(metrics);
        }
    }

    pub(crate) fn emit_inference_err(obs: Option<Arc<dyn BatcherMetrics>>, size: usize) {
        if let Some(obs) = obs {
            obs.emit_inference_err(size);
        }
    }

    pub(crate) fn emit_worker_panic(obs: Option<Arc<dyn BatcherMetrics>>) {
        if let Some(obs) = obs {
            obs.on_worker_panic();
        }
    }
}

pub struct InfBatchMetrics {
    pub(crate) size: usize,
    pub(crate) latency: tokio::time::Duration,
}
