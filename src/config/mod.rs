use std::num::NonZeroU64;

/// Config that defines the batching semantics. An inference batch will fire at the first occurance
/// of either the batch size being reached or the timeout.
///
/// Batch size is defined per worker in the pool. Multiple pool workers may be beneficial when the
/// machine you are running your server on has multiple devices, to maximize utilization.
///
/// Pool size defines the number of workers that inference requests will be distributed across.
///
/// Inference requests use a load aware round robin algorithm. It is a best effort round robin
/// implementation to try and distrubte load and not send inference requests to a busy worker if
/// possible.
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Timeout defined in milliseconds.
    pub batch_timeout: NonZeroU64,
    /// Batch size per inference run.
    pub batch_size: NonZeroU64,
    /// The number of workers defined in the pool.
    pub pool_size: NonZeroU64,
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct InnerConfig {
    pub(crate) timeout: u64,
    pub(crate) size: u64,
}

impl From<BatcherConfig> for InnerConfig {
    fn from(conf: BatcherConfig) -> InnerConfig {
        let BatcherConfig {
            batch_timeout: timeout,
            batch_size: size,
            ..
        } = conf;
        InnerConfig {
            timeout: timeout.into(),
            size: size.into(),
        }
    }
}
