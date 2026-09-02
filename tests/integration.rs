use batchinf::observability::{BatchTrigger, BatcherMetrics};
use batchinf::{BatcherConfig, Predictor, WorkerSnapshot, WorkerStatus, get_batcher};
use std::num::NonZeroU64;
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};
use tokio::time::{Duration, Instant};

// --- Test predictors ---

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

    fn predict_batch(&self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        Ok(inp.to_vec())
    }
}

#[derive(Clone)]
struct FailPredictor;

impl Predictor for FailPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&self, _inp: &[u64]) -> Result<Vec<u64>, TestError> {
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

    fn predict_batch(&self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(inp.to_vec())
    }
}

// --- Test metrics ---

#[derive(Debug)]
struct TestMetrics {
    capacity_triggers: AtomicU32,
    timeout_triggers: AtomicU32,
    ok_completions: AtomicU32,
    err_completions: AtomicU32,
    request_timeouts: AtomicU32,
}

impl TestMetrics {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            capacity_triggers: AtomicU32::new(0),
            timeout_triggers: AtomicU32::new(0),
            ok_completions: AtomicU32::new(0),
            err_completions: AtomicU32::new(0),
            request_timeouts: AtomicU32::new(0),
        })
    }

    fn capacity_triggers(&self) -> u32 { self.capacity_triggers.load(Ordering::SeqCst) }
    fn timeout_triggers(&self) -> u32 { self.timeout_triggers.load(Ordering::SeqCst) }
    fn ok_completions(&self) -> u32 { self.ok_completions.load(Ordering::SeqCst) }
    fn err_completions(&self) -> u32 { self.err_completions.load(Ordering::SeqCst) }
    fn request_timeouts(&self) -> u32 { self.request_timeouts.load(Ordering::SeqCst) }
}

impl BatcherMetrics for TestMetrics {
    fn on_batch_trigger(&self, _batch_size: usize, trigger: BatchTrigger) {
        match trigger {
            BatchTrigger::Capacity => self.capacity_triggers.fetch_add(1, Ordering::SeqCst),
            BatchTrigger::Timeout => self.timeout_triggers.fetch_add(1, Ordering::SeqCst),
        };
    }

    fn on_batch_complete_ok(&self, _batch_size: usize, _latency: Duration) {
        self.ok_completions.fetch_add(1, Ordering::SeqCst);
    }

    fn on_batch_complete_err(&self, _batch_size: usize) {
        self.err_completions.fetch_add(1, Ordering::SeqCst);
    }

    fn on_request_timeout(&self) {
        self.request_timeouts.fetch_add(1, Ordering::SeqCst);
    }
}

// --- Helpers ---

fn config(batch_size: u64, timeout_ms: u64, pool_size: u64) -> BatcherConfig {
    BatcherConfig {
        batch_size: NonZeroU64::new(batch_size).unwrap(),
        batch_timeout: NonZeroU64::new(timeout_ms).unwrap(),
        pool_size: NonZeroU64::new(pool_size).unwrap(),
    }
}

fn no_obs() -> Option<Arc<dyn BatcherMetrics>> {
    None
}

fn with_obs(m: &Arc<TestMetrics>) -> Option<Arc<dyn BatcherMetrics>> {
    Some(m.clone())
}

async fn join<T: Send + 'static>(handles: Vec<tokio::task::JoinHandle<T>>) -> Vec<T> {
    let mut out = Vec::with_capacity(handles.len());
    for h in handles {
        out.push(h.await.unwrap());
    }
    out
}

// --- Tests ---

#[tokio::test(flavor = "multi_thread")]
async fn test_single_request() {
    let batcher = get_batcher(EchoPredictor, config(8, 100, 1), no_obs());
    assert_eq!(batcher.predict(42).await.unwrap(), 42);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_results_match_inputs() {
    let batcher = Arc::new(get_batcher(EchoPredictor, config(8, 500, 1), no_obs()));

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
    let batcher = Arc::new(get_batcher(predictor.clone(), config(4, 10_000, 1), no_obs()));

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
    assert_eq!(predictor.call_count(), 1, "exactly one predict_batch call expected");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_fires_at_timeout() {
    let timeout_ms = 50u64;
    let batcher = get_batcher(EchoPredictor, config(8, timeout_ms, 1), no_obs());

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

#[tokio::test(flavor = "multi_thread")]
async fn test_error_propagates_to_caller() {
    let batcher = get_batcher(FailPredictor, config(1, 50, 1), no_obs());
    assert!(batcher.predict(0).await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_error_propagates_to_all_callers_in_batch() {
    let batcher = Arc::new(get_batcher(FailPredictor, config(4, 500, 1), no_obs()));

    let handles: Vec<_> = (0..4u64)
        .map(|_| {
            let b = batcher.clone();
            tokio::spawn(async move { b.predict(0).await })
        })
        .collect();

    let results = join(handles).await;
    assert!(results.iter().all(|r| r.is_err()), "all callers should receive the error");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_requests_all_complete() {
    let n = 100u64;
    let batcher = Arc::new(get_batcher(EchoPredictor, config(16, 50, 1), no_obs()));

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
    let batcher = Arc::new(get_batcher(EchoPredictor, config(4, 50, 4), no_obs()));

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
    let batcher = Arc::new(get_batcher(predictor.clone(), config(4, 500, 1), no_obs()));

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

// --- Status tests ---

#[tokio::test(flavor = "multi_thread")]
async fn test_pool_status_length_matches_pool_size() {
    let batcher = get_batcher(EchoPredictor, config(4, 100, 3), no_obs());
    assert_eq!(batcher.pool_status().len(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_workers_initially_waiting() {
    let batcher = get_batcher(EchoPredictor, config(4, 100, 2), no_obs());
    for WorkerSnapshot { status, queue_len } in batcher.pool_status() {
        assert_eq!(status, WorkerStatus::Waiting);
        assert_eq!(queue_len, 0);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_worker_status_valid_index() {
    let batcher = get_batcher(EchoPredictor, config(4, 100, 2), no_obs());
    assert!(batcher.worker_status(0).is_some());
    assert!(batcher.worker_status(1).is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_worker_status_out_of_bounds() {
    let batcher = get_batcher(EchoPredictor, config(4, 100, 2), no_obs());
    assert!(batcher.worker_status(2).is_none());
}

// --- Observability tests ---

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_capacity_trigger() {
    let metrics = TestMetrics::new();
    let batcher = Arc::new(get_batcher(EchoPredictor, config(4, 10_000, 1), with_obs(&metrics)));

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
    let batcher = get_batcher(EchoPredictor, config(8, 50, 1), with_obs(&metrics));

    batcher.predict(1).await.unwrap();

    assert_eq!(metrics.timeout_triggers(), 1);
    assert_eq!(metrics.capacity_triggers(), 0);
    assert_eq!(metrics.ok_completions(), 1);
    assert_eq!(metrics.err_completions(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_err_completion() {
    let metrics = TestMetrics::new();
    let batcher = get_batcher(FailPredictor, config(1, 50, 1), with_obs(&metrics));

    let _ = batcher.predict(0).await;

    assert_eq!(metrics.err_completions(), 1);
    assert_eq!(metrics.ok_completions(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_request_timeout() {
    let metrics = TestMetrics::new();
    // Large batch_size and batch_timeout so the worker won't fire on its own.
    let batcher = get_batcher(EchoPredictor, config(8, 10_000, 1), with_obs(&metrics));

    let _ = batcher
        .predict_with_timeout(0, Duration::from_millis(20))
        .await;

    assert_eq!(metrics.request_timeouts(), 1);
}
