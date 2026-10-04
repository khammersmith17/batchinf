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
