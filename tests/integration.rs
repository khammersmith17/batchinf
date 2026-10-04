use batchinf::{
    BatchTrigger, BatcherConfig, BatcherMetrics, Batchinf, BatchinfError, Predictor, PredictorStatus,
    WorkerSnapshot, WorkerStatus, get_batcher,
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

    fn on_batch_complete_err(&self, _batch_size: usize, _latency: Duration) {
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
        // Same as batch_size: one batch of backlog per worker.
        queue_size: NonZeroU32::new(batch_size).unwrap(),
    }
}

fn no_obs() -> Option<Arc<dyn BatcherMetrics>> {
    None
}

fn with_obs(m: &Arc<TestMetrics>) -> Option<Arc<dyn BatcherMetrics>> {
    Some(m.clone())
}

/// Factory that hands each worker (and each rebuild) a clone of `predictor`. Clones share any
/// `Arc` state, so counters and panic flags are observed across workers and rebuilds.
fn factory<P: Clone + Send + Sync + 'static>(
    predictor: P,
) -> Arc<dyn Fn(usize) -> P + Send + Sync> {
    Arc::new(move |_worker_id| predictor.clone())
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
    let batcher = get_batcher(factory(EchoPredictor), config(8, 100, 1), no_obs());
    assert_eq!(batcher.predict(42).await.unwrap(), 42);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_results_match_inputs() {
    let batcher = get_batcher(factory(EchoPredictor), config(8, 500, 1), no_obs());

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
    let batcher = get_batcher(factory(predictor.clone()), config(4, 10_000, 1), no_obs());

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
    let batcher = get_batcher(factory(EchoPredictor), config(8, timeout_ms, 1), no_obs());

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
    let batcher = get_batcher(factory(FailPredictor), config(1, 50, 1), no_obs());
    assert!(batcher.predict(0).await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_error_propagates_to_all_callers_in_batch() {
    let batcher = get_batcher(factory(FailPredictor), config(4, 500, 1), no_obs());

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
    let batcher = get_batcher(factory(EchoPredictor), config(100, 50, 1), no_obs());

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
    let batcher = get_batcher(factory(EchoPredictor), config(16, 50, 4), no_obs());

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
    let batcher = get_batcher(factory(predictor.clone()), config(4, 500, 1), no_obs());

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
    let batcher = get_batcher(factory(EchoPredictor), config(4, 100, 3), no_obs());
    assert_eq!(batcher.pool_status().len(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_workers_initially_waiting() {
    let batcher = get_batcher(factory(EchoPredictor), config(4, 100, 2), no_obs());
    for WorkerSnapshot {
        worker_status,
        predictor_status,
        queue_len,
    } in batcher.pool_status()
    {
        assert_eq!(worker_status, WorkerStatus::Waiting);
        assert_eq!(predictor_status, PredictorStatus::Alive);
        assert_eq!(queue_len, 0);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_worker_status_valid_index() {
    let batcher = get_batcher(factory(EchoPredictor), config(4, 100, 2), no_obs());
    assert!(batcher.worker_status(0).is_some());
    assert!(batcher.worker_status(1).is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_worker_status_out_of_bounds() {
    let batcher = get_batcher(factory(EchoPredictor), config(4, 100, 2), no_obs());
    assert!(batcher.worker_status(2).is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_capacity_trigger() {
    let metrics = TestMetrics::new();
    let batcher = get_batcher(
        factory(EchoPredictor),
        config(4, 10_000, 1),
        with_obs(&metrics),
    );

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
    let batcher = get_batcher(factory(EchoPredictor), config(8, 50, 1), with_obs(&metrics));

    batcher.predict(1).await.unwrap();

    assert_eq!(metrics.timeout_triggers(), 1);
    assert_eq!(metrics.capacity_triggers(), 0);
    assert_eq!(metrics.ok_completions(), 1);
    assert_eq!(metrics.err_completions(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_err_completion() {
    let metrics = TestMetrics::new();
    let batcher = get_batcher(factory(FailPredictor), config(1, 50, 1), with_obs(&metrics));

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
        config(8, 10_000, 1),
        with_obs(&metrics),
    );

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
    let batcher = get_batcher(factory(MismatchPredictor), config(4, 500, 1), no_obs());

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
    let batcher = get_batcher(factory(predictor), config(1, 50, 1), no_obs());

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
    let batcher = get_batcher(factory(predictor), config(1, 50, 1), with_obs(&metrics));

    // Trigger the panic.
    let _ = batcher.predict(1).await;

    // Allow the control plane time to detect the crash and emit on_worker_panic.
    tokio::time::sleep(Duration::from_millis(500)).await;

    assert_eq!(metrics.worker_panics(), 1);
}

// --- predict_with_timeout success path ---

#[tokio::test(flavor = "multi_thread")]
async fn test_predict_with_timeout_succeeds() {
    let batcher = get_batcher(factory(EchoPredictor), config(1, 50, 1), no_obs());
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
    let batcher = get_batcher(factory(EchoPredictor), config(8, 10_000, 1), no_obs());

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
    let batcher = get_batcher(factory(predictor), config(1, 50, 4), no_obs());

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
        config(4, 10_000, 1),
        with_obs(&metrics),
    );

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

#[ignore]
#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_queue_depth_emitted() {
    let metrics = TestMetrics::new();
    let batcher = get_batcher(
        factory(EchoPredictor),
        config(8, 500, 1),
        with_obs(&metrics),
    );

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
    let batcher = get_batcher(factory(AlwaysPanicPredictor), config(1, 50, 1), no_obs());

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
    let batcher = get_batcher(factory(predictor), config(4, 10, 3), no_obs());

    assert_eq!(batcher.predict(1).await.unwrap(), 1);
    drop(batcher);

    // Workers and the control plane exit asynchronously once the last handle is dropped.
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(
        Arc::strong_count(&marker),
        1,
        "workers or control plane still hold predictor clones after the batcher was dropped"
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
        Arc::new(|_worker_id| ThreadLocalPredictor {
            calls: std::rc::Rc::new(std::cell::Cell::new(0)),
        }),
        config(4, 10, 2),
        no_obs(),
    );
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

// Worker 0's factory always panics, so after its retries it is marked Dead. The router must skip
// it and serve every request from workers 1 and 2.
#[tokio::test(flavor = "multi_thread")]
async fn test_dead_worker_is_skipped_by_router() {
    let batcher = get_batcher(
        Arc::new(|worker_id: usize| {
            if worker_id == 0 {
                panic!("intentional factory failure for worker 0");
            }
            EchoPredictor
        }),
        BatcherConfig {
            // Room for the whole burst on the two healthy workers, so any rejection is a routing
            // bug rather than backpressure.
            queue_size: NonZeroU32::new(64).unwrap(),
            ..config(4, 10, 3)
        },
        no_obs(),
    );

    assert!(
        wait_for_predictor_status(&batcher, 0, PredictorStatus::Dead, Duration::from_secs(2)).await,
        "worker 0 should be marked Dead after its factory keeps failing"
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
) -> Arc<dyn Fn(usize) -> SlowPanicOncePredictor + Send + Sync> {
    let has_panicked = Arc::new(AtomicBool::new(false));
    let built_once = Arc::new(AtomicBool::new(false));
    Arc::new(move |_worker_id| {
        if built_once.swap(true, Ordering::SeqCst) {
            std::thread::sleep(rebuild_delay);
        }
        SlowPanicOncePredictor {
            has_panicked: Arc::clone(&has_panicked),
            delay: panic_delay,
        }
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
        no_obs(),
    );

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

// While a single worker's predictor is rebuilding, new requests are rejected as temporarily full,
// not as permanently unavailable. Once the rebuild finishes, requests succeed again.
#[tokio::test(flavor = "multi_thread")]
async fn test_rebuilding_single_worker_returns_queue_full() {
    let batcher = get_batcher(
        slow_rebuild_factory(Duration::from_millis(300), Duration::ZERO),
        config(1, 10, 1),
        no_obs(),
    );

    // Trigger the panic and the rebuild.
    let first = batcher.predict(1).await;
    assert!(matches!(first, Err(BatchinfError::InternalError)), "{first:?}");

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
    let during = batcher.predict(2).await;
    assert!(
        matches!(during, Err(BatchinfError::QueueFullError)),
        "expected QueueFullError while rebuilding, got {during:?}"
    );

    assert!(
        wait_for_predictor_status(&batcher, 0, PredictorStatus::Alive, Duration::from_secs(2))
            .await,
        "predictor should come back after the rebuild"
    );
    assert_eq!(batcher.predict(3).await.unwrap(), 3);
}

// Records which worker ran each batch, and takes long enough that workers overlap.
struct RecordingPredictor {
    worker_id: usize,
    batches_per_worker: Arc<[AtomicU32]>,
}

impl Predictor for RecordingPredictor {
    type Input = u64;
    type Output = u64;
    type Error = TestError;

    fn predict_batch(&mut self, inp: &[u64]) -> Result<Vec<u64>, TestError> {
        self.batches_per_worker[self.worker_id].fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(20));
        Ok(inp.to_vec())
    }
}

// Under a burst of concurrent requests with a slow predictor, every worker should take a share
// of the load and no request should be rejected while channels have room.
#[tokio::test(flavor = "multi_thread")]
async fn test_saturation_spreads_load_across_workers() {
    let pool_size = 3;
    let batches_per_worker: Arc<[AtomicU32]> =
        (0..pool_size).map(|_| AtomicU32::new(0)).collect::<Vec<_>>().into();
    let counts = Arc::clone(&batches_per_worker);

    let batcher = get_batcher(
        Arc::new(move |worker_id| RecordingPredictor {
            worker_id,
            batches_per_worker: Arc::clone(&counts),
        }),
        BatcherConfig {
            batch_size: NonZeroU32::new(4).unwrap(),
            batch_timeout: Duration::from_millis(5),
            pool_size: NonZeroU8::new(pool_size as u8).unwrap(),
            // Room for the whole burst, so any rejection would be a routing bug.
            queue_size: NonZeroU32::new(64).unwrap(),
        },
        no_obs(),
    );

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
