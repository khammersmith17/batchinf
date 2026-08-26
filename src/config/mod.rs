use std::num::NonZeroU64;

#[derive(Debug, Clone)]
pub struct BatcherConfig {
    pub batch_timeout: NonZeroU64,
    pub batch_size: NonZeroU64,
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
