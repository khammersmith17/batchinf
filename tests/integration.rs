use batchinf::{
    BatchTrigger, BatcherConfig, BatcherConfigBuilder, BatcherMetrics, Batchinf, BatchinfError,
    Predictor, PredictorStatus, StartupError, WorkerSnapshot, WorkerStatus, get_batcher,
};
use std::{
    num::{NonZeroU8, NonZeroU32},
    sync::{
        Arc,
        atomic::AtomicBool,
        atomic::{AtomicU32, Ordering},
    },
};
use tokio::time::{Duration, Instant};

#[derive(Clone)]
struct EchoPredictor;

#[derive(Debug, Clone)]
struct TestError;

impl std::fmt::Display for TestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "test error")
    }
}

impl std::error::Error for TestError {}

impl Predictor for EchoPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        Ok(inp.to_vec())
    }
}

#[derive(Clone)]
struct MismatchPredictor;

impl Predictor for MismatchPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, _inp: &[u64]) -> Result<Vec<u64>, TestError> {
        Ok(vec![]) // always returns wrong number of outputs
    }
}

#[derive(Clone)]
struct FailPredictor;

impl Predictor for FailPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, _inp: &[u64]) -> Result<Vec<u64>, TestError> {
        Err(TestError)
    }
}

#[derive(Clone)]
struct CountingPredictor {
    call_count: Arc<AtomicU32>,
}

impl CountingPredictor {
    fn new() -> Self {
        Self {
            call_count: Arc::new(AtomicU32::new(0)),
        }
    }

    fn call_count(&self) -> u32 {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl Predictor for CountingPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(inp.to_vec())
    }
}

#[derive(Debug)]
struct TestMetrics {
    capacity_triggers: AtomicU32,
    timeout_triggers: AtomicU32,
    ok_completions: AtomicU32,
    err_completions: AtomicU32,
    request_timeouts: AtomicU32,
    worker_panics: AtomicU32,
    last_trigger_size: AtomicU32,
    queue_depth_calls: AtomicU32,
    max_queue_depth: AtomicU32,
}

impl TestMetrics {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            capacity_triggers: AtomicU32::new(0),
            timeout_triggers: AtomicU32::new(0),
            ok_completions: AtomicU32::new(0),
            err_completions: AtomicU32::new(0),
            request_timeouts: AtomicU32::new(0),
            worker_panics: AtomicU32::new(0),
            last_trigger_size: AtomicU32::new(0),
            queue_depth_calls: AtomicU32::new(0),
            max_queue_depth: AtomicU32::new(0),
        })
    }

    fn capacity_triggers(&self) -> u32 {
        self.capacity_triggers.load(Ordering::SeqCst)
    }
    fn timeout_triggers(&self) -> u32 {
        self.timeout_triggers.load(Ordering::SeqCst)
    }
    fn ok_completions(&self) -> u32 {
        self.ok_completions.load(Ordering::SeqCst)
    }
    fn err_completions(&self) -> u32 {
        self.err_completions.load(Ordering::SeqCst)
    }
    fn request_timeouts(&self) -> u32 {
        self.request_timeouts.load(Ordering::SeqCst)
    }
    fn worker_panics(&self) -> u32 {
        self.worker_panics.load(Ordering::SeqCst)
    }
    fn last_trigger_size(&self) -> u32 {
        self.last_trigger_size.load(Ordering::SeqCst)
    }
    fn queue_depth_calls(&self) -> u32 {
        self.queue_depth_calls.load(Ordering::SeqCst)
    }
    fn max_queue_depth(&self) -> u32 {
        self.max_queue_depth.load(Ordering::SeqCst)
    }
}

impl BatcherMetrics for TestMetrics {
    fn on_batch_trigger(&self, batch_size: usize, trigger: BatchTrigger) {
        self.last_trigger_size
            .store(batch_size as u32, Ordering::SeqCst);
        match trigger {
            BatchTrigger::Capacity => self.capacity_triggers.fetch_add(1, Ordering::SeqCst),
            BatchTrigger::Timeout => self.timeout_triggers.fetch_add(1, Ordering::SeqCst),
        };
    }

    fn on_batch_complete_ok(&self, _batch_size: usize, _latency: Duration) {
        self.ok_completions.fetch_add(1, Ordering::SeqCst);
    }

    fn on_batch_complete_err(&self, _batch_size: usize, _latency: Duration) {
        self.err_completions.fetch_add(1, Ordering::SeqCst);
    }

    fn on_request_timeout(&self) {
        self.request_timeouts.fetch_add(1, Ordering::SeqCst);
    }

    fn on_queue_depth(&self, _worker_id: u8, queue_depth: usize) {
        self.queue_depth_calls.fetch_add(1, Ordering::SeqCst);
        self.max_queue_depth
            .fetch_max(queue_depth as u32, Ordering::SeqCst);
    }

    fn on_worker_panic(&self) {
        self.worker_panics.fetch_add(1, Ordering::SeqCst);
    }
}

/// Builder with the settings most tests share. `queue_size` is left at its default, which is the
/// same as `batch_size`: one batch of backlog per worker.
fn config_builder(batch_size: u32, timeout_ms: u64, pool_size: u8) -> BatcherConfigBuilder {
    let mut builder = BatcherConfig::builder(
        NonZeroU32::new(batch_size).unwrap(),
        Duration::from_millis(timeout_ms),
    );
    builder
        .pool_size(NonZeroU8::new(pool_size).unwrap())
        // Short and few, so tests with a failing factory finish quickly.
        .worker_start_retries(3)
        .worker_start_backoff_start(Duration::from_millis(1))
        .max_worker_retry_wait(Duration::from_millis(100))
        // Generous: only tests that set this themselves expect startup to time out.
        .startup_timeout(Duration::from_secs(5));
    builder
}

fn config(batch_size: u32, timeout_ms: u64, pool_size: u8) -> BatcherConfig {
    config_builder(batch_size, timeout_ms, pool_size).build()
}

/// [`config`] with `m` installed as the metrics hook.
fn metrics_config(
    batch_size: u32,
    timeout_ms: u64,
    pool_size: u8,
    m: &Arc<TestMetrics>,
) -> BatcherConfig {
    config_builder(batch_size, timeout_ms, pool_size)
        .metrics(m.clone())
        .build()
}

/// Factory that hands each worker (and each rebuild) a clone of `predictor`. Clones share any
/// `Arc` state, so counters and panic flags are observed across workers and rebuilds.
fn factory<P: Clone + Send + Sync + 'static>(
    predictor: P,
) -> Arc<dyn Fn(u8) -> Result<P, TestError> + Send + Sync> {
    Arc::new(move |_worker_id| Ok(predictor.clone()))
}

async fn join<T: Send + 'static>(handles: Vec<tokio::task::JoinHandle<T>>) -> Vec<T> {
    let mut out = Vec::with_capacity(handles.len());
    for h in handles {
        out.push(h.await.unwrap());
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn test_single_request() {
    let batcher = get_batcher(factory(EchoPredictor), config(8, 100, 1))
        .await
        .unwrap();
    assert_eq!(batcher.predict(42).await.unwrap(), 42);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_results_match_inputs() {
    let batcher = get_batcher(factory(EchoPredictor), config(8, 500, 1))
        .await
        .unwrap();

    let handles: Vec<_> = (0..8u64)
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(i).await.unwrap() })
        })
        .collect();

    let mut results = join(handles).await;
    results.sort_unstable();
    assert_eq!(results, (0..8u64).collect::<Vec<_>>());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_fires_at_capacity() {
    let predictor = CountingPredictor::new();
    let batcher = get_batcher(factory(predictor.clone()), config(4, 10_000, 1))
        .await
        .unwrap();

    let start = Instant::now();
    let handles: Vec<_> = (0..4u64)
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(i).await.unwrap() })
        })
        .collect();
    join(handles).await;

    assert!(
        start.elapsed() < Duration::from_secs(5),
        "batch should have fired at capacity, not waited for 10s timeout"
    );
    assert_eq!(
        predictor.call_count(),
        1,
        "exactly one predict_batch call expected"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_fires_at_timeout() {
    let timeout_ms = 50u64;
    let batcher = get_batcher(factory(EchoPredictor), config(8, timeout_ms, 1))
        .await
        .unwrap();

    let start = Instant::now();
    let result = batcher.predict(99).await.unwrap();
    let elapsed = start.elapsed();

    assert_eq!(result, 99);
    assert!(
        elapsed >= Duration::from_millis(timeout_ms),
        "should have waited at least {}ms for timeout, took {:?}",
        timeout_ms,
        elapsed
    );
}

// Regression test: `batch_timeout = Duration::MAX` means "only fire on capacity". Computing the
// batch deadline used to overflow `Instant + Duration` and panic the worker's accumulator task on
// the first request, leaving every later request with `NoAvailableWorkersError`.
#[tokio::test(flavor = "multi_thread")]
async fn test_max_batch_timeout_fires_only_at_capacity() {
    let predictor = CountingPredictor::new();
    let batcher = get_batcher(
        factory(predictor.clone()),
        BatcherConfig::builder(NonZeroU32::new(2).unwrap(), Duration::MAX).build(),
    )
    .await
    .unwrap();

    // A lone request never fills the batch, and no timer fires it, so the caller's own deadline
    // is what ends the wait.
    let lone = batcher
        .predict_with_timeout(1, Duration::from_millis(100))
        .await;
    assert!(
        matches!(lone, Err(BatchinfError::TimeoutError)),
        "a partial batch must not fire on a timer, got {lone:?}"
    );

    // The accumulator survived computing the deadline and still holds the (now cancelled)
    // request.
    let status = batcher.worker_status(0).unwrap();
    assert_eq!(status.worker_status, WorkerStatus::Waiting);
    assert_eq!(status.predictor_status, PredictorStatus::Alive);
    assert_eq!(status.batch_len, 1);

    // The next request fills the batch, which fires on capacity. The cancelled request is
    // filtered out before inference.
    assert_eq!(batcher.predict(2).await.unwrap(), 2);

    // Two concurrent requests fill the next batch together.
    let handles: Vec<_> = [3u64, 4]
        .into_iter()
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(i).await.unwrap() })
        })
        .collect();
    let mut results = tokio::time::timeout(Duration::from_secs(2), join(handles))
        .await
        .expect("a full batch should fire on capacity");
    results.sort_unstable();
    assert_eq!(results, vec![3, 4]);

    assert_eq!(
        predictor.call_count(),
        2,
        "only the two full batches should have reached the predictor"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_error_propagates_to_caller() {
    let batcher = get_batcher(factory(FailPredictor), config(1, 50, 1))
        .await
        .unwrap();
    assert!(batcher.predict(0).await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_error_propagates_to_all_callers_in_batch() {
    let batcher = get_batcher(factory(FailPredictor), config(4, 500, 1))
        .await
        .unwrap();

    let handles: Vec<_> = (0..4u64)
        .map(|_| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(0).await })
        })
        .collect();

    let results = join(handles).await;
    assert!(
        results.iter().all(|r| r.is_err()),
        "all callers should receive the error"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_requests_all_complete() {
    let n = 100u64;
    let batcher = get_batcher(factory(EchoPredictor), config(100, 50, 1))
        .await
        .unwrap();

    let handles: Vec<_> = (0..n)
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(i).await.unwrap() })
        })
        .collect();

    let results = join(handles).await;
    assert_eq!(results.len(), n as usize);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_multi_worker_correct_results() {
    let n = 64u64;
    let batcher = get_batcher(factory(EchoPredictor), config(16, 50, 4))
        .await
        .unwrap();

    let handles: Vec<_> = (0..n)
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(i).await.unwrap() })
        })
        .collect();

    let mut results = join(handles).await;
    results.sort_unstable();
    assert_eq!(results, (0..n).collect::<Vec<_>>());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_multiple_sequential_batches() {
    let predictor = CountingPredictor::new();
    let batcher = get_batcher(factory(predictor.clone()), config(4, 500, 1))
        .await
        .unwrap();

    let handles: Vec<_> = (0..4u64)
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(i).await })
        })
        .collect();
    join(handles).await;

    let handles: Vec<_> = (4..8u64)
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(i).await })
        })
        .collect();
    join(handles).await;

    batcher.predict(8).await.unwrap();

    assert_eq!(predictor.call_count(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pool_status_length_matches_pool_size() {
    let batcher = get_batcher(factory(EchoPredictor), config(4, 100, 3))
        .await
        .unwrap();
    assert_eq!(batcher.pool_status().len(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_workers_initially_waiting() {
    let batcher = get_batcher(factory(EchoPredictor), config(4, 100, 2))
        .await
        .unwrap();
    for WorkerSnapshot {
        worker_status,
        predictor_status,
        batch_len,
    } in batcher.pool_status()
    {
        assert_eq!(worker_status, WorkerStatus::Waiting);
        assert_eq!(predictor_status, PredictorStatus::Alive);
        assert_eq!(batch_len, 0);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_worker_status_valid_index() {
    let batcher = get_batcher(factory(EchoPredictor), config(4, 100, 2))
        .await
        .unwrap();
    assert!(batcher.worker_status(0).is_some());
    assert!(batcher.worker_status(1).is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_worker_status_out_of_bounds() {
    let batcher = get_batcher(factory(EchoPredictor), config(4, 100, 2))
        .await
        .unwrap();
    assert!(batcher.worker_status(2).is_none());
}

/// Implements only one hook and, like many metrics backends, is not `Debug`.
struct NoDebugMetrics {
    batches: Arc<AtomicU32>,
}

impl BatcherMetrics for NoDebugMetrics {
    fn on_batch_complete_ok(&self, _batch_size: usize, _latency: Duration) {
        self.batches.fetch_add(1, Ordering::SeqCst);
    }
}

// A metrics type needs neither `Debug` nor every hook, and a config holding it is still `Debug`.
#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_without_debug_or_all_hooks() {
    let batches = Arc::new(AtomicU32::new(0));
    let config = config_builder(1, 50, 1)
        .metrics(Arc::new(NoDebugMetrics {
            batches: Arc::clone(&batches),
        }))
        .build();
    assert!(
        format!("{config:?}").contains("metrics: Some(dyn BatcherMetrics)"),
        "unexpected Debug output: {config:?}"
    );

    let batcher = get_batcher(factory(EchoPredictor), config).await.unwrap();
    assert_eq!(batcher.predict(7).await.unwrap(), 7);
    assert_eq!(batches.load(Ordering::SeqCst), 1);
    // `Batchinf` holds the hook too and stays `Debug`.
    let _ = format!("{batcher:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_capacity_trigger() {
    let metrics = TestMetrics::new();
    let batcher = get_batcher(
        factory(EchoPredictor),
        metrics_config(4, 10_000, 1, &metrics),
    )
    .await
    .unwrap();

    let handles: Vec<_> = (0..4u64)
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(i).await.unwrap() })
        })
        .collect();
    join(handles).await;

    assert_eq!(metrics.capacity_triggers(), 1);
    assert_eq!(metrics.timeout_triggers(), 0);
    assert_eq!(metrics.ok_completions(), 1);
    assert_eq!(metrics.err_completions(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_timeout_trigger() {
    let metrics = TestMetrics::new();
    let batcher = get_batcher(factory(EchoPredictor), metrics_config(8, 50, 1, &metrics))
        .await
        .unwrap();

    batcher.predict(1).await.unwrap();

    assert_eq!(metrics.timeout_triggers(), 1);
    assert_eq!(metrics.capacity_triggers(), 0);
    assert_eq!(metrics.ok_completions(), 1);
    assert_eq!(metrics.err_completions(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_err_completion() {
    let metrics = TestMetrics::new();
    let batcher = get_batcher(factory(FailPredictor), metrics_config(1, 50, 1, &metrics))
        .await
        .unwrap();

    let _ = batcher.predict(0).await;

    assert_eq!(metrics.err_completions(), 1);
    assert_eq!(metrics.ok_completions(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_request_timeout() {
    let metrics = TestMetrics::new();
    // Large batch_size and batch_timeout so the worker won't fire on its own.
    let batcher = get_batcher(
        factory(EchoPredictor),
        metrics_config(8, 10_000, 1, &metrics),
    )
    .await
    .unwrap();

    let result = batcher
        .predict_with_timeout(0, Duration::from_millis(20))
        .await;

    assert!(
        matches!(result, Err(BatchinfError::TimeoutError)),
        "expected TimeoutError, got {result:?}"
    );
    assert_eq!(metrics.request_timeouts(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_predictor_output_propagates_to_all_callers() {
    let batcher = get_batcher(factory(MismatchPredictor), config(4, 500, 1))
        .await
        .unwrap();

    let handles: Vec<_> = (0..4u64)
        .map(|_| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(0).await })
        })
        .collect();

    let results = join(handles).await;
    assert!(
        results
            .iter()
            .all(|r| matches!(r, Err(BatchinfError::InvalidPredictorOutput))),
        "all callers should receive InvalidPredictorOutput"
    );
}

#[derive(Clone)]
struct PanicOncePredictor {
    has_panicked: Arc<AtomicBool>,
}

impl PanicOncePredictor {
    fn new() -> Self {
        Self {
            has_panicked: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Predictor for PanicOncePredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        if !self.has_panicked.swap(true, Ordering::SeqCst) {
            panic!("intentional panic for crash recovery test");
        }
        Ok(inp.to_vec())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_worker_restarts_after_panic() {
    let predictor = PanicOncePredictor::new();
    let batcher = get_batcher(factory(predictor), config(1, 50, 1))
        .await
        .unwrap();

    // First request triggers the panic; callers in that batch receive InternalError.
    let first = batcher.predict(1).await;
    assert!(
        matches!(first, Err(BatchinfError::InternalError)),
        "expected InternalError from panicking batch, got {first:?}"
    );

    // Allow the control plane time to detect the crash and restart the worker.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Subsequent requests should succeed on the restarted worker.
    let second = batcher.predict(2).await;
    assert_eq!(
        second.unwrap(),
        2,
        "restarted worker should process requests"
    );
}

#[derive(Clone)]
struct AlwaysPanicPredictor;

impl Predictor for AlwaysPanicPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, _inp: &[u64]) -> Result<Vec<u64>, TestError> {
        panic!("intentional panic");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_worker_panic() {
    let metrics = TestMetrics::new();
    let predictor = PanicOncePredictor::new();
    let batcher = get_batcher(factory(predictor), metrics_config(1, 50, 1, &metrics))
        .await
        .unwrap();

    // Trigger the panic.
    let _ = batcher.predict(1).await;

    // Allow the control plane time to detect the crash and emit on_worker_panic.
    tokio::time::sleep(Duration::from_millis(500)).await;

    assert_eq!(metrics.worker_panics(), 1);
}

// --- predict_with_timeout success path ---

#[tokio::test(flavor = "multi_thread")]
async fn test_predict_with_timeout_succeeds() {
    let batcher = get_batcher(factory(EchoPredictor), config(1, 50, 1))
        .await
        .unwrap();
    let result = batcher
        .predict_with_timeout(42, Duration::from_millis(500))
        .await;
    assert_eq!(result.unwrap(), 42);
}

// When a caller times out, its oneshot sender remains in the worker's batch buffer. The worker
// should continue accumulating and processing normally. Once the batch fires (here via capacity),
// the timed-out slot is silently skipped and the remaining callers receive their results.
#[tokio::test(flavor = "multi_thread")]
async fn test_timed_out_request_does_not_block_subsequent() {
    // Large timeout so the batch only fires at capacity, not by time.
    let batcher = get_batcher(factory(EchoPredictor), config(8, 10_000, 1))
        .await
        .unwrap();

    // Queue one request and let it time out. Its input stays in the worker buffer.
    let timed_out = batcher
        .predict_with_timeout(0, Duration::from_millis(20))
        .await;
    assert!(matches!(timed_out, Err(BatchinfError::TimeoutError)));

    // Fill the remaining 7 slots to trigger capacity. These should all succeed.
    let handles: Vec<_> = (1..8u64)
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(i).await })
        })
        .collect();
    let results = join(handles).await;
    assert!(
        results.iter().all(|r| r.is_ok()),
        "subsequent requests should succeed after a timed-out slot in the batch"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_multi_worker_one_panic_others_serve() {
    // 4-worker pool; predictor panics exactly once.
    let predictor = PanicOncePredictor::new();
    let batcher = get_batcher(factory(predictor), config(1, 50, 4))
        .await
        .unwrap();

    let handles: Vec<_> = (0..8u64)
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(i).await })
        })
        .collect();
    let results = join(handles).await;

    // All concurrent requests may have been queued to the panicking worker before its state
    // was set to Crashed. The recovery guarantee is that the worker restarts, not that every
    // in-flight request succeeds. Assert that at least some requests were served by other workers.
    let successes = results.iter().filter(|r| r.is_ok()).count();
    assert!(
        successes > 0,
        "at least one other worker should have served a request"
    );

    // After recovery the pool is fully functional again.
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(batcher.predict(99).await.unwrap(), 99);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_batch_trigger_size() {
    let metrics = TestMetrics::new();
    let batcher = get_batcher(
        factory(EchoPredictor),
        metrics_config(4, 10_000, 1, &metrics),
    )
    .await
    .unwrap();

    let handles: Vec<_> = (0..4u64)
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(i).await.unwrap() })
        })
        .collect();
    join(handles).await;

    assert_eq!(metrics.capacity_triggers(), 1);
    assert_eq!(
        metrics.last_trigger_size(),
        4,
        "on_batch_trigger should report the correct batch size"
    );
}

// Slow enough that a burst of requests backs up behind the inference thread.
#[derive(Clone)]
struct SlowEchoPredictor;

impl Predictor for SlowEchoPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        std::thread::sleep(Duration::from_millis(30));
        Ok(inp.to_vec())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_queue_depth_emitted() {
    let metrics = TestMetrics::new();
    let queue_size = 64;
    let batcher = get_batcher(
        factory(SlowEchoPredictor),
        config_builder(2, 5, 1)
            .queue_size(NonZeroU32::new(queue_size).unwrap())
            .metrics(metrics.clone())
            .build(),
    )
    .await
    .unwrap();

    // A lone request leaves nothing waiting behind it.
    batcher.predict(0).await.unwrap();
    assert!(
        metrics.queue_depth_calls() >= 1,
        "on_queue_depth should fire when a batch is dispatched"
    );
    assert_eq!(
        metrics.max_queue_depth(),
        0,
        "an idle worker should report an empty queue"
    );

    // A burst against a slow predictor backs up in the worker's queue.
    let handles: Vec<_> = (1..=20u64)
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(i).await })
        })
        .collect();
    let results = join(handles).await;
    assert!(results.iter().all(|r| r.is_ok()), "{results:?}");

    let max = metrics.max_queue_depth();
    assert!(max > 0, "a backlog should be reported under a burst");
    assert!(
        max <= queue_size,
        "reported depth {max} exceeds the queue capacity {queue_size}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_no_available_workers_error() {
    // Single worker that always panics → crashes immediately.
    let batcher = get_batcher(factory(AlwaysPanicPredictor), config(1, 50, 1))
        .await
        .unwrap();

    // Crash the worker.
    let _ = batcher.predict(0).await;

    // The sentinel channel wakes the control plane immediately on crash, so the worker may
    // already be restarted by the time the next predict is submitted. Either way, an
    // always-panicking worker returns an error — no hang, no silent success.
    let result = batcher.predict(1).await;
    assert!(
        result.is_err(),
        "expected an error from an always-panicking worker, got Ok"
    );
}

#[derive(Clone)]
struct MarkedPredictor {
    // Every clone of the predictor holds a strong reference to this marker.
    _marker: Arc<()>,
}

impl Predictor for MarkedPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        Ok(inp.to_vec())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_drop_releases_all_predictor_clones() {
    let marker = Arc::new(());
    let predictor = MarkedPredictor {
        _marker: Arc::clone(&marker),
    };
    let batcher = get_batcher(factory(predictor), config(4, 10, 3))
        .await
        .unwrap();

    assert_eq!(batcher.predict(1).await.unwrap(), 1);
    let waiter = batcher.shutdown_waiter();
    drop(batcher);

    // Workers exit asynchronously once the last handle is dropped.
    tokio::time::timeout(Duration::from_secs(2), waiter.wait())
        .await
        .expect("shutdown should complete once the last handle is dropped");

    assert_eq!(
        Arc::strong_count(&marker),
        1,
        "workers still hold predictor clones after shutdown completed"
    );
}

// The waiter must not resolve while any handle is alive, and must not itself keep the batcher
// alive.
#[tokio::test(flavor = "multi_thread")]
async fn test_shutdown_waiter_waits_for_every_handle() {
    let marker = Arc::new(());
    let predictor = MarkedPredictor {
        _marker: Arc::clone(&marker),
    };
    let batcher = get_batcher(factory(predictor), config(4, 10, 2))
        .await
        .unwrap();
    let clone = batcher.clone();
    let waiter = batcher.shutdown_waiter();

    drop(batcher);
    assert!(
        tokio::time::timeout(Duration::from_millis(100), waiter.clone().wait())
            .await
            .is_err(),
        "shutdown must not complete while a clone is alive"
    );
    assert!(!waiter.is_shutdown());
    assert_eq!(clone.predict(5).await.unwrap(), 5, "the clone still serves");

    drop(clone);
    tokio::time::timeout(Duration::from_secs(2), waiter.clone().wait())
        .await
        .expect("shutdown should complete once the last handle is dropped");
    assert!(waiter.is_shutdown());
    assert_eq!(Arc::strong_count(&marker), 1);
}

/// Sleeps in `predict_batch` and holds `marker` until dropped.
struct SlowMarkedPredictor {
    _marker: Arc<()>,
    delay: Duration,
}

impl Predictor for SlowMarkedPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        std::thread::sleep(self.delay);
        Ok(inp.to_vec())
    }
}

// A batch already running when the last handle is dropped finishes first; the waiter resolves
// only after that, once the predictor has been dropped.
#[tokio::test(flavor = "multi_thread")]
async fn test_shutdown_waiter_waits_for_running_batch() {
    let marker = Arc::new(());
    let factory_marker = Arc::clone(&marker);
    let batcher = get_batcher(
        Arc::new(move |_worker_id| {
            Ok::<_, TestError>(SlowMarkedPredictor {
                _marker: Arc::clone(&factory_marker),
                delay: Duration::from_millis(300),
            })
        }),
        config(1, 10, 1),
    )
    .await
    .unwrap();
    let waiter = batcher.shutdown_waiter();

    // Batch size 1, so the request goes to the inference thread right away. The caller gives up
    // while inference is still running.
    let start = Instant::now();
    let res = batcher
        .predict_with_timeout(1, Duration::from_millis(50))
        .await;
    assert!(matches!(res, Err(BatchinfError::TimeoutError)));
    drop(batcher);

    tokio::time::timeout(Duration::from_secs(2), waiter.wait())
        .await
        .expect("shutdown should complete after the running batch");
    assert!(
        start.elapsed() >= Duration::from_millis(250),
        "shutdown completed before the running batch finished, after {:?}",
        start.elapsed()
    );
    // Holds only if the factory closure (which holds a clone) is gone too.
    assert_eq!(
        Arc::strong_count(&marker),
        1,
        "the predictor or factory outlived shutdown"
    );
}

// Neither `Clone` nor `Send`: `Rc` is thread-local. This only compiles because each predictor is
// built, used and dropped on its own inference thread.
struct ThreadLocalPredictor {
    calls: std::rc::Rc<std::cell::Cell<u64>>,
}

impl Predictor for ThreadLocalPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        self.calls.set(self.calls.get() + 1);
        Ok(inp.to_vec())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_non_send_non_clone_predictor() {
    let batcher = get_batcher(
        Arc::new(|_worker_id| {
            Ok::<_, TestError>(ThreadLocalPredictor {
                calls: std::rc::Rc::new(std::cell::Cell::new(0)),
            })
        }),
        config(4, 10, 2),
    )
    .await
    .unwrap();
    assert_eq!(batcher.predict(7).await.unwrap(), 7);
}

/// Polls `worker_status(idx)` until its predictor reaches `want`, or gives up after `limit`.
async fn wait_for_predictor_status<I, O, E>(
    batcher: &Batchinf<I, O, E>,
    idx: usize,
    want: PredictorStatus,
    limit: Duration,
) -> bool
where
    I: Send + 'static,
    O: Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if batcher.worker_status(idx).unwrap().predictor_status == want {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    false
}

// Echoes its input, or panics on every call when `always_panics` is set.
struct MaybePanicPredictor {
    always_panics: bool,
}

impl Predictor for MaybePanicPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        if self.always_panics {
            panic!("intentional panic during inference");
        }
        Ok(inp.to_vec())
    }
}

// Worker 0 starts fine, but its predictor panics on its first batch and its factory then fails on
// every rebuild attempt, so it is marked Dead. The router must skip it and serve every request
// from workers 1 and 2.
#[tokio::test(flavor = "multi_thread")]
async fn test_dead_worker_is_skipped_by_router() {
    let worker_zero_builds = Arc::new(AtomicU32::new(0));
    let batcher = get_batcher(
        Arc::new(move |worker_id: u8| {
            if worker_id == 0 && worker_zero_builds.fetch_add(1, Ordering::SeqCst) > 0 {
                panic!("intentional factory failure rebuilding worker 0");
            }
            Ok::<_, TestError>(MaybePanicPredictor {
                always_panics: worker_id == 0,
            })
        }),
        // Room for the whole burst on the two healthy workers, so any rejection is a routing bug
        // rather than backpressure.
        config_builder(4, 10, 3)
            .queue_size(NonZeroU32::new(64).unwrap())
            .build(),
    )
    .await
    .unwrap();

    // Round-robin routing sends one of the first few requests to worker 0, which kills it.
    for i in 0..10u64 {
        let _ = batcher.predict(i).await;
        if batcher.worker_status(0).unwrap().predictor_status != PredictorStatus::Alive {
            break;
        }
    }
    assert!(
        wait_for_predictor_status(&batcher, 0, PredictorStatus::Dead, Duration::from_secs(2)).await,
        "worker 0 should be marked Dead after its rebuilds keep failing"
    );

    let handles: Vec<_> = (0..30u64)
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(i).await })
        })
        .collect();
    let results = join(handles).await;
    assert!(
        results.iter().all(|r| r.is_ok()),
        "every request should be served by the healthy workers: {results:?}"
    );
}

// Sleeps, then panics on its first call only. The panic flag is shared across rebuilds.
struct SlowPanicOncePredictor {
    has_panicked: Arc<AtomicBool>,
    delay: Duration,
}

impl Predictor for SlowPanicOncePredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        if !self.has_panicked.swap(true, Ordering::SeqCst) {
            std::thread::sleep(self.delay);
            panic!("intentional panic during inference");
        }
        Ok(inp.to_vec())
    }
}

/// Factory for [`SlowPanicOncePredictor`] that sleeps for `rebuild_delay` on every build after
/// the first, so the predictor stays in its rebuilding state long enough to observe.
fn slow_rebuild_factory(
    rebuild_delay: Duration,
    panic_delay: Duration,
) -> Arc<dyn Fn(u8) -> Result<SlowPanicOncePredictor, TestError> + Send + Sync> {
    let has_panicked = Arc::new(AtomicBool::new(false));
    let built_once = Arc::new(AtomicBool::new(false));
    Arc::new(move |_worker_id| {
        if built_once.swap(true, Ordering::SeqCst) {
            std::thread::sleep(rebuild_delay);
        }
        Ok(SlowPanicOncePredictor {
            has_panicked: Arc::clone(&has_panicked),
            delay: panic_delay,
        })
    })
}

// Regression test: a batch whose timeout fires while the predictor is being rebuilt used to hit
// `unreachable!` in the worker loop. It must instead be dispatched once the rebuild finishes.
//
// Timeline (ms): req 1 at 0 → its batch times out at 100 → predict_batch sleeps until ~180 and
// panics → rebuild runs ~180–380. Req 2 is sent at 130 while the predictor is still alive, and its
// batch timer fires at ~230, during the rebuild.
#[tokio::test(flavor = "multi_thread")]
async fn test_timeout_batch_during_rebuild_is_served() {
    let batcher = get_batcher(
        slow_rebuild_factory(Duration::from_millis(200), Duration::from_millis(80)),
        config(8, 100, 1),
    )
    .await
    .unwrap();

    let b = batcher.clone();
    let first = tokio::spawn(async move { b.predict(1).await });

    tokio::time::sleep(Duration::from_millis(130)).await;
    let second = batcher.predict(2).await;

    assert!(
        matches!(first.await.unwrap(), Err(BatchinfError::InternalError)),
        "the batch that panicked should fail with InternalError"
    );
    assert_eq!(
        second.unwrap(),
        2,
        "a batch that times out during the rebuild should run on the rebuilt predictor"
    );
}

// While a single worker's predictor is rebuilding, `try_predict` rejects new requests as
// temporarily full, not as permanently unavailable. `predict` waits for the rebuild instead and
// then succeeds.
#[tokio::test(flavor = "multi_thread")]
async fn test_rebuilding_single_worker_returns_queue_full() {
    let batcher = get_batcher(
        slow_rebuild_factory(Duration::from_millis(300), Duration::ZERO),
        config(1, 10, 1),
    )
    .await
    .unwrap();

    // Trigger the panic and the rebuild.
    let first = batcher.predict(1).await;
    assert!(
        matches!(first, Err(BatchinfError::InternalError)),
        "{first:?}"
    );

    assert!(
        wait_for_predictor_status(
            &batcher,
            0,
            PredictorStatus::Rebuilding,
            Duration::from_millis(200)
        )
        .await,
        "predictor should be rebuilding after the panic"
    );

    // Fail-fast variant: rejected immediately while the rebuild is in progress.
    let rejected = batcher.try_predict(2).await;
    assert!(
        matches!(rejected, Err(BatchinfError::QueueFullError)),
        "expected QueueFullError from try_predict while rebuilding, got {rejected:?}"
    );

    // Waiting variant: issued during the rebuild, completes once the predictor is back. The
    // timeout turns a missed wake-up into a failure instead of a hang.
    let waited = tokio::time::timeout(Duration::from_secs(2), batcher.predict(3))
        .await
        .expect("predict should be woken when the rebuild finishes");
    assert_eq!(waited.unwrap(), 3);
    assert_eq!(
        batcher.worker_status(0).unwrap().predictor_status,
        PredictorStatus::Alive
    );
}

// With a slow single worker and room for one queued request, a burst overflows the queue.
// `try_predict` reports that as QueueFullError; `predict` waits for room and every request
// completes. Runs with batch sizes 1 and 2: with a batch size of 1, the only dequeue that can
// wake a waiter is the first request of each batch.
#[tokio::test(flavor = "multi_thread")]
async fn test_predict_waits_for_queue_capacity() {
    for batch_size in [1u32, 2] {
        let batcher = get_batcher(
            factory(SlowEchoPredictor),
            config_builder(batch_size, 5, 1)
                .queue_size(NonZeroU32::new(1).unwrap())
                .build(),
        )
        .await
        .unwrap();

        // Premise: this burst really does overflow the queue.
        let handles: Vec<_> = (0..20u64)
            .map(|i| {
                let b = batcher.clone();
                tokio::spawn(async move { b.try_predict(i).await })
            })
            .collect();
        let rejected = join(handles)
            .await
            .iter()
            .filter(|r| matches!(r, Err(BatchinfError::QueueFullError)))
            .count();
        assert!(
            rejected > 0,
            "batch_size {batch_size}: the burst should overflow the queue for try_predict"
        );

        // The same burst through `predict` waits for room instead of failing.
        let handles: Vec<_> = (0..20u64)
            .map(|i| {
                let b = batcher.clone();
                tokio::spawn(async move { (i, b.predict(i).await) })
            })
            .collect();
        let results = tokio::time::timeout(Duration::from_secs(10), join(handles))
            .await
            .unwrap_or_else(|_| {
                panic!("batch_size {batch_size}: a waiting predict was never woken")
            });
        for (input, result) in results {
            assert_eq!(
                result.unwrap(),
                input,
                "batch_size {batch_size}: every waiting request should complete"
            );
        }
    }
}

// Every caller waiting in `predict` must be woken when the last worker dies, not just one. Once the
// last worker is gone nothing is dequeued again, so a waiter that misses this wakeup would hang
// forever. Guards the `notify_waiters` on the dead path against becoming `notify_one`.
#[tokio::test(flavor = "multi_thread")]
async fn test_all_waiting_predicts_fail_when_last_worker_dies() {
    const WAITERS: u64 = 8;

    let builds = Arc::new(AtomicU32::new(0));
    let batcher = get_batcher(
        Arc::new(move |_worker_id| {
            // The first build succeeds; every rebuild attempt fails.
            if builds.fetch_add(1, Ordering::SeqCst) > 0 {
                panic!("intentional factory failure during rebuild");
            }
            Ok::<_, TestError>(MaybePanicPredictor {
                always_panics: true,
            })
        }),
        // Rebuild attempts at 0ms, 150ms and 450ms after the panic, then the worker is Dead.
        retry_config(2, 150, 1_000),
    )
    .await
    .unwrap();

    // Trigger the panic and the (failing) rebuild.
    let first = batcher.predict(1).await;
    assert!(
        matches!(first, Err(BatchinfError::InternalError)),
        "{first:?}"
    );

    // While the only worker is rebuilding, every request is told to wait.
    let handles: Vec<_> = (0..WAITERS)
        .map(|i| {
            let batcher = batcher.clone();
            tokio::spawn(async move { batcher.predict(i).await })
        })
        .collect();

    // Make sure the callers are actually blocked waiting, not already finished, before the worker
    // dies.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        batcher.worker_status(0).unwrap().predictor_status,
        PredictorStatus::Rebuilding,
        "the rebuild should still be running, so the callers are waiting"
    );
    assert!(
        handles.iter().all(|h| !h.is_finished()),
        "every caller should be waiting for the rebuild"
    );

    let results = tokio::time::timeout(Duration::from_secs(2), join(handles))
        .await
        .expect("every waiting predict should be woken when the last worker dies");
    for result in results {
        assert!(
            matches!(result, Err(BatchinfError::NoAvailableWorkersError)),
            "expected NoAvailableWorkersError, got {result:?}"
        );
    }
    assert_eq!(
        batcher.worker_status(0).unwrap().predictor_status,
        PredictorStatus::Dead
    );
}

// A caller waiting in `predict` for a rebuilding worker must be woken when the rebuild gives up,
// and get NoAvailableWorkersError rather than waiting forever.
#[tokio::test(flavor = "multi_thread")]
async fn test_waiting_predict_fails_when_last_worker_dies() {
    let builds = Arc::new(AtomicU32::new(0));
    let batcher = get_batcher(
        Arc::new(move |_worker_id| {
            // The first build succeeds; every rebuild attempt fails.
            if builds.fetch_add(1, Ordering::SeqCst) > 0 {
                panic!("intentional factory failure during rebuild");
            }
            Ok::<_, TestError>(MaybePanicPredictor {
                always_panics: true,
            })
        }),
        // Rebuild attempts at 0ms, 50ms and 150ms after the panic, then the worker is Dead.
        retry_config(2, 50, 1_000),
    )
    .await
    .unwrap();

    // Trigger the panic and the (failing) rebuild.
    let first = batcher.predict(1).await;
    assert!(
        matches!(first, Err(BatchinfError::InternalError)),
        "{first:?}"
    );

    // Issued while the worker is rebuilding (or just after it died): waits, then is told that
    // no worker is left.
    let waited = tokio::time::timeout(Duration::from_secs(2), batcher.predict(2))
        .await
        .expect("a waiting predict should be woken when the last worker dies");
    assert!(
        matches!(waited, Err(BatchinfError::NoAvailableWorkersError)),
        "expected NoAvailableWorkersError, got {waited:?}"
    );
    assert_eq!(
        batcher.worker_status(0).unwrap().predictor_status,
        PredictorStatus::Dead
    );
}

// `predict_with_timeout` bounds the wait for room as well as the wait for the result.
#[tokio::test(flavor = "multi_thread")]
async fn test_predict_with_timeout_bounds_wait_for_capacity() {
    let batcher = get_batcher(
        slow_rebuild_factory(Duration::from_millis(500), Duration::ZERO),
        config(1, 10, 1),
    )
    .await
    .unwrap();

    // Trigger the panic and a 500ms rebuild.
    let first = batcher.predict(1).await;
    assert!(
        matches!(first, Err(BatchinfError::InternalError)),
        "{first:?}"
    );
    assert!(
        wait_for_predictor_status(
            &batcher,
            0,
            PredictorStatus::Rebuilding,
            Duration::from_millis(200)
        )
        .await
    );

    // There is no room until the rebuild finishes, so this waits and then times out.
    let start = Instant::now();
    let result = batcher
        .predict_with_timeout(2, Duration::from_millis(50))
        .await;
    let elapsed = start.elapsed();

    assert!(
        matches!(result, Err(BatchinfError::TimeoutError)),
        "expected TimeoutError while waiting for room, got {result:?}"
    );
    assert!(
        elapsed < Duration::from_millis(300),
        "the timeout should interrupt the wait for room, took {elapsed:?}"
    );
}

// Records which worker ran each batch, and takes long enough that workers overlap.
struct RecordingPredictor {
    worker_id: u8,
    batches_per_worker: Arc<[AtomicU32]>,
}

impl Predictor for RecordingPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        self.batches_per_worker[self.worker_id as usize].fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(20));
        Ok(inp.to_vec())
    }
}

// Under a burst of concurrent requests with a slow predictor, every worker should take a share
// of the load and no request should be rejected while channels have room.
#[tokio::test(flavor = "multi_thread")]
async fn test_saturation_spreads_load_across_workers() {
    let pool_size = 3;
    let batches_per_worker: Arc<[AtomicU32]> = (0..pool_size)
        .map(|_| AtomicU32::new(0))
        .collect::<Vec<_>>()
        .into();
    let counts = Arc::clone(&batches_per_worker);

    let batcher = get_batcher(
        Arc::new(move |worker_id| {
            Ok::<_, TestError>(RecordingPredictor {
                worker_id,
                batches_per_worker: Arc::clone(&counts),
            })
        }),
        // Room for the whole burst, so any rejection would be a routing bug.
        config_builder(4, 5, pool_size as u8)
            .queue_size(NonZeroU32::new(64).unwrap())
            .build(),
    )
    .await
    .unwrap();

    let handles: Vec<_> = (0..60u64)
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(i).await })
        })
        .collect();
    let results = join(handles).await;

    assert!(
        results.iter().all(|r| r.is_ok()),
        "no request should be rejected while channels have room: {results:?}"
    );
    for (worker_id, count) in batches_per_worker.iter().enumerate() {
        assert!(
            count.load(Ordering::SeqCst) > 0,
            "worker {worker_id} ran no batches; load was not spread"
        );
    }
}

/// Config for a single worker with the given factory retry settings.
fn retry_config(retries: u8, backoff_ms: u64, max_wait_ms: u64) -> BatcherConfig {
    config_builder(4, 10, 1)
        .worker_start_retries(retries)
        .worker_start_backoff_start(Duration::from_millis(backoff_ms))
        .max_worker_retry_wait(Duration::from_millis(max_wait_ms))
        .build()
}

/// Factory that returns an error on every call and counts how often it was called.
fn always_failing_factory(
    calls: &Arc<AtomicU32>,
) -> Arc<dyn Fn(u8) -> Result<EchoPredictor, TestError> + Send + Sync> {
    let calls = Arc::clone(calls);
    Arc::new(move |_worker_id| {
        calls.fetch_add(1, Ordering::SeqCst);
        Err(TestError)
    })
}

// With 2 retries the factory is called 3 times, with waits of 50ms and 100ms between calls and no
// wait after the last one. Too fast means the backoff is skipped; too slow means it also sleeps
// after the final attempt (which would add another 200ms).
#[tokio::test(flavor = "multi_thread")]
async fn test_factory_retries_with_doubling_backoff() {
    let calls = Arc::new(AtomicU32::new(0));
    let start = Instant::now();
    let result = get_batcher(always_failing_factory(&calls), retry_config(2, 50, 10_000)).await;
    let elapsed = start.elapsed();

    assert!(
        matches!(
            result,
            Err(StartupError::WorkerStartFailure {
                worker_id: 0,
                err: TestError
            })
        ),
        "startup should fail once the factory's retries are used up"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3, "1 attempt + 2 retries");
    assert!(
        elapsed >= Duration::from_millis(150),
        "expected waits of 50ms + 100ms between attempts, took {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_millis(300),
        "should not wait after the final attempt, took {elapsed:?}"
    );
}

// The wait doubles from 20ms but is capped at 30ms: 20 + 30 + 30 + 30 = 110ms across 4 retries.
// Uncapped it would be 20 + 40 + 80 + 160 = 300ms.
#[tokio::test(flavor = "multi_thread")]
async fn test_factory_backoff_is_capped() {
    let calls = Arc::new(AtomicU32::new(0));
    let start = Instant::now();
    let result = get_batcher(always_failing_factory(&calls), retry_config(4, 20, 30)).await;
    let elapsed = start.elapsed();

    assert!(matches!(
        result,
        Err(StartupError::WorkerStartFailure {
            worker_id: 0,
            err: TestError
        })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 5, "1 attempt + 4 retries");
    assert!(
        elapsed >= Duration::from_millis(110),
        "expected capped waits totalling 110ms, took {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_millis(250),
        "the wait should be capped by max_worker_retry_wait, took {elapsed:?}"
    );
}

// Zero retries means a single attempt and no waiting, even with a long backoff configured.
#[tokio::test(flavor = "multi_thread")]
async fn test_zero_retries_is_a_single_attempt() {
    let calls = Arc::new(AtomicU32::new(0));
    let start = Instant::now();
    let result = get_batcher(
        always_failing_factory(&calls),
        retry_config(0, 1_000, 10_000),
    )
    .await;
    let elapsed = start.elapsed();

    assert!(matches!(
        result,
        Err(StartupError::WorkerStartFailure {
            worker_id: 0,
            err: TestError
        })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        elapsed < Duration::from_millis(300),
        "with no retries startup should fail without waiting for the backoff, took {elapsed:?}"
    );
}

// One failing worker fails the whole startup, even when the others build fine.
#[tokio::test(flavor = "multi_thread")]
async fn test_startup_fails_when_one_worker_cannot_start() {
    let result = get_batcher(
        Arc::new(|worker_id: u8| {
            if worker_id == 1 {
                return Err(TestError);
            }
            Ok(EchoPredictor)
        }),
        config(4, 10, 3),
    )
    .await;

    assert!(
        matches!(
            result,
            Err(StartupError::WorkerStartFailure {
                worker_id: 1,
                err: TestError
            })
        ),
        "startup should fail and name the worker that could not build its predictor"
    );
}

// A factory that takes longer than `startup_timeout` fails startup at the deadline, without
// waiting for the factory to finish.
#[tokio::test(flavor = "multi_thread")]
async fn test_startup_times_out_with_slow_factory() {
    let start = Instant::now();
    let result = get_batcher(
        Arc::new(|_worker_id| {
            std::thread::sleep(Duration::from_millis(400));
            Ok::<_, TestError>(EchoPredictor)
        }),
        config_builder(4, 10, 1)
            .startup_timeout(Duration::from_millis(50))
            .build(),
    )
    .await;
    let elapsed = start.elapsed();

    assert!(
        matches!(result, Err(StartupError::Timeout)),
        "startup should time out"
    );
    assert!(
        elapsed < Duration::from_millis(300),
        "startup should return at the deadline, not when the factory finishes, took {elapsed:?}"
    );
}

// A factory that fails twice and then succeeds leaves the worker healthy.
#[tokio::test(flavor = "multi_thread")]
async fn test_factory_succeeds_on_a_retry() {
    let calls = Arc::new(AtomicU32::new(0));
    let counter = Arc::clone(&calls);
    let batcher = get_batcher(
        Arc::new(move |_worker_id| {
            if counter.fetch_add(1, Ordering::SeqCst) < 2 {
                panic!("intentional factory failure");
            }
            Ok::<_, TestError>(EchoPredictor)
        }),
        retry_config(3, 5, 100),
    )
    .await
    .unwrap();

    assert_eq!(batcher.predict(7).await.unwrap(), 7);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "two failures, then success"
    );
    assert_eq!(
        batcher.worker_status(0).unwrap().predictor_status,
        PredictorStatus::Alive
    );
}

// Rebuilding after a predict_batch panic uses the same retry settings: here the first rebuild
// attempt fails and the second succeeds.
#[tokio::test(flavor = "multi_thread")]
async fn test_rebuild_after_panic_retries_factory() {
    let calls = Arc::new(AtomicU32::new(0));
    let counter = Arc::clone(&calls);
    let has_panicked = Arc::new(AtomicBool::new(false));
    let batcher = get_batcher(
        Arc::new(move |_worker_id| {
            // Call 0 builds the first predictor, call 1 is the failed rebuild attempt.
            if counter.fetch_add(1, Ordering::SeqCst) == 1 {
                panic!("intentional factory failure during rebuild");
            }
            Ok::<_, TestError>(SlowPanicOncePredictor {
                has_panicked: Arc::clone(&has_panicked),
                delay: Duration::ZERO,
            })
        }),
        {
            // The fields are public, so a built config can still be adjusted.
            let mut config = retry_config(2, 5, 100);
            config.batch_size = NonZeroU32::new(1).unwrap();
            config
        },
    )
    .await
    .unwrap();

    let first = batcher.predict(1).await;
    assert!(
        matches!(first, Err(BatchinfError::InternalError)),
        "{first:?}"
    );

    assert!(
        wait_for_predictor_status(&batcher, 0, PredictorStatus::Alive, Duration::from_secs(2))
            .await,
        "the predictor should be rebuilt on the second attempt"
    );
    // Give the Rebuilding → Alive transition a moment in case the poll saw the initial Alive.
    let deadline = Instant::now() + Duration::from_secs(2);
    while calls.load(Ordering::SeqCst) < 3 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "build, failed rebuild, rebuild"
    );

    assert!(
        wait_for_predictor_status(&batcher, 0, PredictorStatus::Alive, Duration::from_secs(2))
            .await
    );
    assert_eq!(batcher.predict(2).await.unwrap(), 2);
}

// Dropping the batcher while a rebuild is waiting between factory attempts cuts the wait short:
// shutdown completes quickly instead of sitting out the backoff, and no further attempts are made.
#[tokio::test(flavor = "multi_thread")]
async fn test_shutdown_cancels_rebuild_backoff() {
    const BACKOFF: Duration = Duration::from_secs(3);

    let calls = Arc::new(AtomicU32::new(0));
    let counter = Arc::clone(&calls);
    let batcher = get_batcher(
        Arc::new(move |_worker_id| {
            // Call 0 builds the first predictor; every rebuild attempt fails.
            if counter.fetch_add(1, Ordering::SeqCst) > 0 {
                return Err(TestError);
            }
            Ok(MaybePanicPredictor {
                always_panics: true,
            })
        }),
        {
            let mut config =
                retry_config(3, BACKOFF.as_millis() as u64, BACKOFF.as_millis() as u64);
            config.batch_size = NonZeroU32::new(1).unwrap();
            config
        },
    )
    .await
    .unwrap();

    // Trigger the panic and the rebuild.
    let first = batcher.predict(1).await;
    assert!(
        matches!(first, Err(BatchinfError::InternalError)),
        "{first:?}"
    );

    // Wait until the first rebuild attempt has failed: the thread is now in its backoff wait.
    let deadline = Instant::now() + Duration::from_secs(2);
    while calls.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2, "build, failed rebuild");
    assert_eq!(
        batcher.worker_status(0).unwrap().predictor_status,
        PredictorStatus::Rebuilding
    );

    let waiter = batcher.shutdown_waiter();
    let start = Instant::now();
    drop(batcher);
    tokio::time::timeout(BACKOFF, waiter.wait())
        .await
        .expect("shutdown should not wait out the rebuild backoff");
    assert!(
        start.elapsed() < Duration::from_millis(500),
        "shutdown should cut the backoff short, took {:?}",
        start.elapsed()
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "no factory attempt should be made after shutdown"
    );
}

// Echoes its input and records every batch it was asked to run.
struct BatchRecordingPredictor {
    batches: Arc<std::sync::Mutex<Vec<Vec<u64>>>>,
}

impl Predictor for BatchRecordingPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        self.batches.lock().unwrap().push(inp.to_vec());
        Ok(inp.to_vec())
    }
}

/// Factory for [`BatchRecordingPredictor`] plus a handle to the batches it records.
#[allow(clippy::type_complexity)]
fn batch_recording_factory() -> (
    Arc<dyn Fn(u8) -> Result<BatchRecordingPredictor, TestError> + Send + Sync>,
    Arc<std::sync::Mutex<Vec<Vec<u64>>>>,
) {
    let batches = Arc::new(std::sync::Mutex::new(Vec::new()));
    let recorded = Arc::clone(&batches);
    let factory = Arc::new(move |_worker_id| {
        Ok(BatchRecordingPredictor {
            batches: Arc::clone(&recorded),
        })
    });
    (factory, batches)
}

// Regression test: a batch in which every request was cancelled used to panic the inference
// thread while filtering, leaving the worker permanently dead. The worker must keep serving.
#[tokio::test(flavor = "multi_thread")]
async fn test_fully_cancelled_batch_does_not_kill_worker() {
    let (factory, _batches) = batch_recording_factory();
    let batcher = get_batcher(factory, config(8, 100, 1)).await.unwrap();

    // Times out at 20ms. Its batch fires at 100ms, by which time its only request is cancelled.
    let cancelled = batcher
        .predict_with_timeout(1, Duration::from_millis(20))
        .await;
    assert!(matches!(cancelled, Err(BatchinfError::TimeoutError)));

    // Let the cancelled batch fire.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let WorkerSnapshot {
        worker_status,
        predictor_status,
        ..
    } = batcher.worker_status(0).unwrap();
    assert_eq!(predictor_status, PredictorStatus::Alive);
    assert_ne!(worker_status, WorkerStatus::Crashed);

    // The worker still serves requests, more than once.
    assert_eq!(batcher.predict(2).await.unwrap(), 2);
    assert_eq!(batcher.predict(3).await.unwrap(), 3);
}

// A batch whose requests were all cancelled should be skipped, not passed to the predictor as an
// empty slice.
#[tokio::test(flavor = "multi_thread")]
async fn test_fully_cancelled_batch_is_not_sent_to_predictor() {
    let (factory, batches) = batch_recording_factory();
    let batcher = get_batcher(factory, config(8, 100, 1)).await.unwrap();

    let cancelled = batcher
        .predict_with_timeout(1, Duration::from_millis(20))
        .await;
    assert!(matches!(cancelled, Err(BatchinfError::TimeoutError)));
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert_eq!(batcher.predict(2).await.unwrap(), 2);

    let batches = batches.lock().unwrap().clone();
    assert!(
        batches.iter().all(|batch| !batch.is_empty()),
        "predict_batch was called with an empty batch: {batches:?}"
    );
    assert_eq!(batches, vec![vec![2]], "only the live request should run");
}

// In a mixed batch, cancelled requests are removed before inference and every live caller still
// gets its own result back, even though filtering reorders the batch.
#[tokio::test(flavor = "multi_thread")]
async fn test_cancelled_requests_are_filtered_from_mixed_batch() {
    let (factory, batches) = batch_recording_factory();
    // Long batch timeout, so the batch only fires once it reaches 6 requests.
    let batcher = get_batcher(factory, config(6, 10_000, 1)).await.unwrap();

    // Three requests that time out while waiting in the batch.
    for i in [100u64, 101, 102] {
        let cancelled = batcher
            .predict_with_timeout(i, Duration::from_millis(20))
            .await;
        assert!(matches!(cancelled, Err(BatchinfError::TimeoutError)));
    }

    // Three live requests fill the batch to capacity and trigger it.
    let handles: Vec<_> = [1u64, 2, 3]
        .into_iter()
        .map(|i| {
            let b = batcher.clone();
            tokio::spawn(async move { (i, b.predict(i).await) })
        })
        .collect();
    for (input, result) in join(handles).await {
        assert_eq!(
            result.unwrap(),
            input,
            "caller got another request's result"
        );
    }

    let batches = batches.lock().unwrap().clone();
    assert_eq!(batches.len(), 1, "expected a single batch: {batches:?}");
    let mut ran = batches[0].clone();
    ran.sort_unstable();
    assert_eq!(
        ran,
        vec![1, 2, 3],
        "cancelled requests should not reach the predictor"
    );
}

// `blocking_try_predict` is for callers outside async code. Called from a plain OS thread, it queues
// the request and blocks that thread until the result arrives.
#[tokio::test(flavor = "multi_thread")]
async fn test_blocking_predict_from_plain_thread() {
    let batcher = get_batcher(factory(EchoPredictor), config(4, 10, 1))
        .await
        .unwrap();

    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let b = batcher.clone();
    std::thread::spawn(move || {
        let _ = done_tx.send(b.blocking_try_predict(7));
    });

    let result = tokio::time::timeout(Duration::from_secs(2), done_rx)
        .await
        .expect("blocking_try_predict should return once inference completes")
        .expect("the calling thread should not panic");
    assert_eq!(result.unwrap(), 7);
}

// Several plain threads can call `blocking_try_predict` at once, and each gets its own result.
#[tokio::test(flavor = "multi_thread")]
async fn test_blocking_predict_from_many_threads() {
    let batcher = get_batcher(
        factory(EchoPredictor),
        // Room for every request, so none is rejected as full.
        config_builder(4, 10, 2)
            .queue_size(NonZeroU32::new(64).unwrap())
            .build(),
    )
    .await
    .unwrap();

    let threads: Vec<_> = (0..16u64)
        .map(|i| {
            let b = batcher.clone();
            std::thread::spawn(move || (i, b.blocking_try_predict(i)))
        })
        .collect();

    // Joining blocks, so do it off the async runtime.
    let results = tokio::task::spawn_blocking(move || {
        threads
            .into_iter()
            .map(|t| t.join().expect("a calling thread panicked"))
            .collect::<Vec<_>>()
    })
    .await
    .unwrap();

    for (input, result) in results {
        assert_eq!(result.unwrap(), input);
    }
}

// `blocking_try_predict` fails fast like `try_predict`: while the only worker is rebuilding, it
// returns QueueFullError immediately instead of blocking until the rebuild finishes.
#[tokio::test(flavor = "multi_thread")]
async fn test_blocking_predict_fails_fast_when_no_room() {
    let batcher = get_batcher(
        slow_rebuild_factory(Duration::from_millis(500), Duration::ZERO),
        config(1, 10, 1),
    )
    .await
    .unwrap();

    // Trigger the panic and a 500ms rebuild.
    let first = batcher.predict(1).await;
    assert!(
        matches!(first, Err(BatchinfError::InternalError)),
        "{first:?}"
    );
    assert!(
        wait_for_predictor_status(
            &batcher,
            0,
            PredictorStatus::Rebuilding,
            Duration::from_millis(200)
        )
        .await
    );

    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let b = batcher.clone();
    std::thread::spawn(move || {
        let start = std::time::Instant::now();
        let result = b.blocking_try_predict(2);
        let _ = done_tx.send((result, start.elapsed()));
    });

    let (result, elapsed) = tokio::time::timeout(Duration::from_secs(2), done_rx)
        .await
        .expect("blocking_try_predict should not block while there is no room")
        .expect("the calling thread should not panic");
    assert!(
        matches!(result, Err(BatchinfError::QueueFullError)),
        "expected QueueFullError, got {result:?}"
    );
    assert!(
        elapsed < Duration::from_millis(200),
        "blocking_try_predict should return without waiting for the rebuild, took {elapsed:?}"
    );
}

// The documented contract: calling `blocking_try_predict` from async code panics rather than
// blocking a runtime thread.
#[tokio::test(flavor = "multi_thread")]
#[should_panic]
async fn test_blocking_predict_panics_in_async_context() {
    let batcher = get_batcher(factory(EchoPredictor), config(4, 10, 1))
        .await
        .unwrap();
    let _ = batcher.blocking_try_predict(1);
}

// The builder takes the two required settings and fills in documented defaults for the rest.
#[test]
fn test_config_builder_defaults() {
    let config =
        BatcherConfig::builder(NonZeroU32::new(16).unwrap(), Duration::from_millis(25)).build();

    assert_eq!(config.batch_size.get(), 16);
    assert_eq!(config.batch_timeout, Duration::from_millis(25));
    assert_eq!(config.pool_size, BatcherConfig::DEFAULT_POOL_SIZE);
    assert_eq!(config.pool_size.get(), 1);
    assert_eq!(
        config.queue_size.get(),
        16,
        "queue_size should default to batch_size"
    );
    assert_eq!(
        config.worker_start_retries,
        BatcherConfig::DEFAULT_WORKER_START_RETRIES
    );
    assert_eq!(
        config.worker_start_backoff_start,
        BatcherConfig::DEFAULT_WORKER_START_BACKOFF_START
    );
    assert_eq!(
        config.max_worker_retry_wait,
        BatcherConfig::DEFAULT_MAX_WORKER_RETRY_WAIT
    );
    assert_eq!(
        config.startup_timeout,
        BatcherConfig::DEFAULT_STARTUP_TIMEOUT
    );
}

// Every option can be overridden in one chained expression, and an explicit queue_size is not
// replaced by the default.
#[test]
fn test_config_builder_overrides() {
    let config = BatcherConfig::builder(NonZeroU32::new(8).unwrap(), Duration::from_millis(5))
        .pool_size(NonZeroU8::new(4).unwrap())
        .queue_size(NonZeroU32::new(100).unwrap())
        .worker_start_retries(0)
        .worker_start_backoff_start(Duration::from_millis(7))
        .max_worker_retry_wait(Duration::from_millis(70))
        .startup_timeout(Duration::from_secs(9))
        .build();

    assert_eq!(config.batch_size.get(), 8);
    assert_eq!(config.batch_timeout, Duration::from_millis(5));
    assert_eq!(config.pool_size.get(), 4);
    assert_eq!(config.queue_size.get(), 100);
    assert_eq!(config.worker_start_retries, 0);
    assert_eq!(config.worker_start_backoff_start, Duration::from_millis(7));
    assert_eq!(config.max_worker_retry_wait, Duration::from_millis(70));
    assert_eq!(config.startup_timeout, Duration::from_secs(9));
}

// The builder's methods do not consume it: it can be configured step by step from a variable,
// and `build` can be called more than once.
#[test]
fn test_config_builder_is_reusable() {
    let mut builder = BatcherConfig::builder(NonZeroU32::new(8).unwrap(), Duration::from_millis(5));
    builder.pool_size(NonZeroU8::new(2).unwrap());

    let first = builder.build();
    assert_eq!(first.pool_size.get(), 2);
    assert_eq!(first.queue_size.get(), 8);

    // Later changes affect only configs built afterwards.
    builder.queue_size(NonZeroU32::new(64).unwrap());
    let second = builder.build();
    assert_eq!(second.pool_size.get(), 2);
    assert_eq!(second.queue_size.get(), 64);
    assert_eq!(first.queue_size.get(), 8);
}

// A config built with only the required settings is enough to start a working batcher.
#[tokio::test(flavor = "multi_thread")]
async fn test_default_built_config_starts_a_batcher() {
    let config =
        BatcherConfig::builder(NonZeroU32::new(4).unwrap(), Duration::from_millis(10)).build();
    let batcher = get_batcher(factory(EchoPredictor), config).await.unwrap();

    assert_eq!(batcher.pool_status().len(), 1);
    assert_eq!(batcher.predict(5).await.unwrap(), 5);
}

// A factory that panics on every attempt is reported as a panic, not as a returned error.
#[tokio::test(flavor = "multi_thread")]
async fn test_factory_panic_reports_worker_start_panic() {
    let calls = Arc::new(AtomicU32::new(0));
    let counter = Arc::clone(&calls);
    let result = get_batcher(
        Arc::new(move |_worker_id| -> Result<EchoPredictor, TestError> {
            counter.fetch_add(1, Ordering::SeqCst);
            panic!("intentional factory panic");
        }),
        retry_config(1, 0, 0),
    )
    .await;

    assert!(matches!(
        result,
        Err(StartupError::WorkerStartPanic { worker_id: 0 })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 2, "1 attempt + 1 retry");
}

// The error the factory returns on its last attempt reaches the caller of `get_batcher`, and the
// factory's error type does not need to implement `Clone`.
#[tokio::test(flavor = "multi_thread")]
async fn test_factory_error_is_returned_to_caller() {
    #[derive(Debug, PartialEq)]
    struct LoadError(u32);

    let attempts = Arc::new(AtomicU32::new(0));
    let counter = Arc::clone(&attempts);
    let result = get_batcher(
        Arc::new(move |_worker_id| -> Result<EchoPredictor, LoadError> {
            Err(LoadError(counter.fetch_add(1, Ordering::SeqCst)))
        }),
        retry_config(2, 0, 0),
    )
    .await;

    match result {
        Err(StartupError::WorkerStartFailure { worker_id, err }) => {
            assert_eq!(worker_id, 0);
            assert_eq!(
                err,
                LoadError(2),
                "the error from the last attempt is returned"
            );
        }
        Err(other) => panic!("unexpected startup error: {other:?}"),
        Ok(_) => panic!("startup should fail"),
    }
}
