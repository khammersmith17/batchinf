use std::{
    num::{NonZeroU8, NonZeroU32},
    time::Duration,
};

/// Config that defines the batching semantics. An inference batch will fire at the first occurrence
/// of either the batch size being reached or the timeout.
///
/// Batch size is defined per worker in the pool. Multiple pool workers may be beneficial when the
/// machine you are running your server on has multiple devices, to maximize utilization.
///
/// Pool size defines the number of workers that inference requests will be distributed across.
///
/// Inference requests use a load aware round robin algorithm. It is a best effort round robin
/// implementation to try and distribute load and not send inference requests to a busy worker if
/// possible.
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Duration to wait before firing a batch that has not reached `batch_size`.
    pub batch_timeout: Duration,
    /// Batch size per inference run.
    pub batch_size: NonZeroU32,
    /// The number of workers defined in the pool.
    pub pool_size: NonZeroU8,
    /// Capacity of each worker's request queue: the number of requests that can wait for a
    /// worker before it is considered full. When every live worker's queue is full,
    /// [`Batchinf::predict`](`crate::Batchinf::predict`) returns
    /// [`BatchinfError::QueueFullError`](`crate::BatchinfError::QueueFullError`).
    pub queue_size: NonZeroU32,
    /// Number of times a worker retries the predictor factory after a failed call, so the
    /// factory is called at most `worker_start_retries + 1` times. `0` means a single attempt.
    /// A factory call fails when it panics. Applies both at startup and when rebuilding the
    /// predictor after [`Predictor::predict_batch`](`crate::Predictor::predict_batch`) panics.
    pub worker_start_retries: u8,
    /// Time a worker waits after the first failed factory call before retrying. The wait
    /// doubles after each further failure, up to
    /// [`max_worker_retry_wait`](BatcherConfig::max_worker_retry_wait).
    pub worker_start_backoff_start: Duration,
    /// Upper bound on the wait between two factory calls. A
    /// [`worker_start_backoff_start`](BatcherConfig::worker_start_backoff_start) above this
    /// value is clamped to it.
    pub max_worker_retry_wait: Duration,
    /// How long [`get_batcher`](`crate::get_batcher`) waits for every worker to build its
    /// predictor before returning [`StartupError::Timeout`](`crate::StartupError::Timeout`).
    /// Set it above the worst case for one worker: the factory's run time multiplied by
    /// `worker_start_retries + 1`, plus the waits in between. Otherwise a slow start that would
    /// have succeeded is cut off.
    pub startup_timeout: Duration,
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct InnerConfig {
    pub(crate) timeout: Duration,
    pub(crate) batch_size: u32,
}

impl From<BatcherConfig> for InnerConfig {
    fn from(conf: BatcherConfig) -> InnerConfig {
        let BatcherConfig {
            batch_timeout: timeout,
            batch_size,
            ..
        } = conf;
        InnerConfig {
            timeout,
            batch_size: batch_size.into(),
        }
    }
}
