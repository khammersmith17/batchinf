use batchinf::{
    BatchTrigger, BatcherConfig, BatcherMetrics, BatchinfError, Predictor, WorkerSnapshot,
    WorkerStatus, get_batcher,
};
use std::num::{NonZeroU8, NonZeroU32};
use std::sync::atomic::AtomicBool;
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
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

    fn predict_batch(&self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        Ok(inp.to_vec())
    }
}

#[derive(Clone)]
struct MismatchPredictor;

impl Predictor for MismatchPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&self, _inp: &[u64]) -> Result<Vec<u64>, TestError> {
        Ok(vec![]) // always returns wrong number of outputs
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

    fn on_batch_complete_err(&self, _batch_size: usize) {
        self.err_completions.fetch_add(1, Ordering::SeqCst);
    }

    fn on_request_timeout(&self) {
        self.request_timeouts.fetch_add(1, Ordering::SeqCst);
    }

    fn on_queue_depth(&self, _queue_depth: usize) {
        self.queue_depth_calls.fetch_add(1, Ordering::SeqCst);
    }

    fn on_worker_panic(&self) {
        self.worker_panics.fetch_add(1, Ordering::SeqCst);
    }
}

fn config(batch_size: u32, timeout_ms: u64, pool_size: u8) -> BatcherConfig {
    BatcherConfig {
        batch_size: NonZeroU32::new(batch_size).unwrap(),
        batch_timeout: Duration::from_millis(timeout_ms),
        pool_size: NonZeroU8::new(pool_size).unwrap(),
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

#[tokio::test(flavor = "multi_thread")]
async fn test_single_request() {
    let batcher = get_batcher(EchoPredictor, config(8, 100, 1), no_obs());
    assert_eq!(batcher.predict(42).await.unwrap(), 42);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_results_match_inputs() {
    let batcher = get_batcher(EchoPredictor, config(8, 500, 1), no_obs());

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
    let batcher = get_batcher(predictor.clone(), config(4, 10_000, 1), no_obs());

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
    let batcher = get_batcher(EchoPredictor, config(8, timeout_ms, 1), no_obs());

    let start = Instant::now();
    let result = batcher.predict(99).await.unwrap();
    let elapsed = start.elapsed();

    assert_eq!(result, 99);
    assert!(
        elapsed >= Duration::from_millis(u64::from(timeout_ms)),
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
    let batcher = get_batcher(FailPredictor, config(4, 500, 1), no_obs());

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
    let batcher = get_batcher(EchoPredictor, config(100, 50, 1), no_obs());

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
    let batcher = get_batcher(EchoPredictor, config(16, 50, 4), no_obs());

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
    let batcher = get_batcher(predictor.clone(), config(4, 500, 1), no_obs());

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

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_capacity_trigger() {
    let metrics = TestMetrics::new();
    let batcher = get_batcher(EchoPredictor, config(4, 10_000, 1), with_obs(&metrics));

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
    let batcher = get_batcher(MismatchPredictor, config(4, 500, 1), no_obs());

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

    fn predict_batch(&self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        if !self.has_panicked.swap(true, Ordering::SeqCst) {
            panic!("intentional panic for crash recovery test");
        }
        Ok(inp.to_vec())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_worker_restarts_after_panic() {
    let predictor = PanicOncePredictor::new();
    let batcher = get_batcher(predictor, config(1, 50, 1), no_obs());

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

    fn predict_batch(&self, _inp: &[u64]) -> Result<Vec<u64>, TestError> {
        panic!("intentional panic");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_worker_panic() {
    let metrics = TestMetrics::new();
    let predictor = PanicOncePredictor::new();
    let batcher = get_batcher(predictor, config(1, 50, 1), with_obs(&metrics));

    // Trigger the panic.
    let _ = batcher.predict(1).await;

    // Allow the control plane time to detect the crash and emit on_worker_panic.
    tokio::time::sleep(Duration::from_millis(500)).await;

    assert_eq!(metrics.worker_panics(), 1);
}

// --- predict_with_timeout success path ---

#[tokio::test(flavor = "multi_thread")]
async fn test_predict_with_timeout_succeeds() {
    let batcher = get_batcher(EchoPredictor, config(1, 50, 1), no_obs());
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
    let batcher = get_batcher(EchoPredictor, config(8, 10_000, 1), no_obs());

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
    let batcher = get_batcher(predictor, config(1, 50, 4), no_obs());

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
    let batcher = get_batcher(EchoPredictor, config(4, 10_000, 1), with_obs(&metrics));

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

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_queue_depth_emitted() {
    let metrics = TestMetrics::new();
    let batcher = get_batcher(EchoPredictor, config(8, 500, 1), with_obs(&metrics));

    // Keep the batcher alive past one control plane poll cycle (250ms).
    tokio::time::sleep(Duration::from_millis(350)).await;
    drop(batcher);

    assert!(
        metrics.queue_depth_calls() >= 1,
        "on_queue_depth should fire at least once per control plane poll cycle"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_no_available_workers_error() {
    // Single worker that always panics → crashes immediately.
    let batcher = get_batcher(AlwaysPanicPredictor, config(1, 50, 1), no_obs());

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
