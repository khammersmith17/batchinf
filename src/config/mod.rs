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
///
/// # Construction
///
/// Create a config with [`BatcherConfig::builder`], which takes the two settings every batcher
/// needs and supplies defaults for the rest:
///
/// ```
/// use batchinf::BatcherConfig;
/// use std::num::{NonZeroU8, NonZeroU32};
/// use std::time::Duration;
///
/// let config = BatcherConfig::builder(NonZeroU32::new(32).unwrap(), Duration::from_millis(10))
///     .pool_size(NonZeroU8::new(2).unwrap())
///     .startup_timeout(Duration::from_secs(120))
///     .build();
///
/// assert_eq!(config.pool_size.get(), 2);
/// // Unset options keep their defaults.
/// assert_eq!(config.queue_size.get(), 32);
/// ```
///
/// The struct is `#[non_exhaustive]`: it cannot be built with a struct literal outside this
/// crate, so new options can be added without breaking existing code. The fields are public, so
/// a built config can still be read and adjusted, for example `config.pool_size = ...`.
#[derive(Debug, Clone)]
#[non_exhaustive]
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

impl BatcherConfig {
    /// Default for [`pool_size`](BatcherConfig::pool_size): a single worker.
    pub const DEFAULT_POOL_SIZE: NonZeroU8 = NonZeroU8::MIN;
    /// Default for [`worker_start_retries`](BatcherConfig::worker_start_retries).
    pub const DEFAULT_WORKER_START_RETRIES: u8 = 3;
    /// Default for [`worker_start_backoff_start`](BatcherConfig::worker_start_backoff_start).
    pub const DEFAULT_WORKER_START_BACKOFF_START: Duration = Duration::from_millis(100);
    /// Default for [`max_worker_retry_wait`](BatcherConfig::max_worker_retry_wait).
    pub const DEFAULT_MAX_WORKER_RETRY_WAIT: Duration = Duration::from_secs(5);
    /// Default for [`startup_timeout`](BatcherConfig::startup_timeout).
    pub const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(60);

    /// Starts building a config from the two settings that have no sensible default: how many
    /// requests go into a batch, and how long a partial batch waits before it fires.
    ///
    /// Every other option starts at its default and can be changed with the builder's methods:
    ///
    /// | Option | Default |
    /// |---|---|
    /// | [`pool_size`](BatcherConfigBuilder::pool_size) | 1 worker |
    /// | [`queue_size`](BatcherConfigBuilder::queue_size) | the same as `batch_size`, one batch of backlog per worker |
    /// | [`worker_start_retries`](BatcherConfigBuilder::worker_start_retries) | 3 |
    /// | [`worker_start_backoff_start`](BatcherConfigBuilder::worker_start_backoff_start) | 100 ms |
    /// | [`max_worker_retry_wait`](BatcherConfigBuilder::max_worker_retry_wait) | 5 s |
    /// | [`startup_timeout`](BatcherConfigBuilder::startup_timeout) | 60 s |
    pub fn builder(batch_size: NonZeroU32, batch_timeout: Duration) -> BatcherConfigBuilder {
        BatcherConfigBuilder {
            batch_size,
            batch_timeout,
            pool_size: Self::DEFAULT_POOL_SIZE,
            queue_size: None,
            worker_start_retries: Self::DEFAULT_WORKER_START_RETRIES,
            worker_start_backoff_start: Self::DEFAULT_WORKER_START_BACKOFF_START,
            max_worker_retry_wait: Self::DEFAULT_MAX_WORKER_RETRY_WAIT,
            startup_timeout: Self::DEFAULT_STARTUP_TIMEOUT,
        }
    }
}

/// Builder for [`BatcherConfig`]. Created by [`BatcherConfig::builder`].
///
/// Each method sets one option on the builder in place and returns `&mut Self`, so calls can be
/// chained in one expression or made one at a time on a builder held in a variable. Finish with
/// [`build`](BatcherConfigBuilder::build), which does not consume the builder.
///
/// ```
/// use batchinf::BatcherConfig;
/// use std::num::{NonZeroU8, NonZeroU32};
/// use std::time::Duration;
///
/// let mut builder =
///     BatcherConfig::builder(NonZeroU32::new(32).unwrap(), Duration::from_millis(10));
/// if std::env::var("MULTI_GPU").is_ok() {
///     builder.pool_size(NonZeroU8::new(4).unwrap());
/// }
/// let config = builder.build();
/// # let _ = config;
/// ```
#[derive(Debug, Clone)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct BatcherConfigBuilder {
    batch_size: NonZeroU32,
    batch_timeout: Duration,
    pool_size: NonZeroU8,
    // `None` means "the same as `batch_size`", resolved in `build`.
    queue_size: Option<NonZeroU32>,
    worker_start_retries: u8,
    worker_start_backoff_start: Duration,
    max_worker_retry_wait: Duration,
    startup_timeout: Duration,
}

impl BatcherConfigBuilder {
    /// Sets the number of workers in the pool. See [`BatcherConfig::pool_size`].
    pub fn pool_size(&mut self, pool_size: NonZeroU8) -> &mut Self {
        self.pool_size = pool_size;
        self
    }

    /// Sets the capacity of each worker's request queue. See [`BatcherConfig::queue_size`].
    ///
    /// If this is not called, the queue holds as many requests as one batch.
    pub fn queue_size(&mut self, queue_size: NonZeroU32) -> &mut Self {
        self.queue_size = Some(queue_size);
        self
    }

    /// Sets how many times a worker retries a failed predictor factory call. `0` means a single
    /// attempt. See [`BatcherConfig::worker_start_retries`].
    pub fn worker_start_retries(&mut self, retries: u8) -> &mut Self {
        self.worker_start_retries = retries;
        self
    }

    /// Sets the wait after the first failed factory call. See
    /// [`BatcherConfig::worker_start_backoff_start`].
    pub fn worker_start_backoff_start(&mut self, backoff: Duration) -> &mut Self {
        self.worker_start_backoff_start = backoff;
        self
    }

    /// Sets the upper bound on the wait between two factory calls. See
    /// [`BatcherConfig::max_worker_retry_wait`].
    pub fn max_worker_retry_wait(&mut self, max_wait: Duration) -> &mut Self {
        self.max_worker_retry_wait = max_wait;
        self
    }

    /// Sets how long startup waits for every worker to build its predictor. See
    /// [`BatcherConfig::startup_timeout`].
    pub fn startup_timeout(&mut self, timeout: Duration) -> &mut Self {
        self.startup_timeout = timeout;
        self
    }

    /// Builds the config from the current settings. The builder is left unchanged, so it can
    /// be adjusted and used again.
    pub fn build(&self) -> BatcherConfig {
        BatcherConfig {
            batch_timeout: self.batch_timeout,
            batch_size: self.batch_size,
            pool_size: self.pool_size,
            queue_size: self.queue_size.unwrap_or(self.batch_size),
            worker_start_retries: self.worker_start_retries,
            worker_start_backoff_start: self.worker_start_backoff_start,
            max_worker_retry_wait: self.max_worker_retry_wait,
            startup_timeout: self.startup_timeout,
        }
    }
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
