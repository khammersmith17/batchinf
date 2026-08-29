use batchinf::{BatcherConfig, Predictor, get_batcher};
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

// --- Helpers ---

fn config(batch_size: u64, timeout_ms: u64, pool_size: u64) -> BatcherConfig {
    BatcherConfig {
        batch_size: NonZeroU64::new(batch_size).unwrap(),
        batch_timeout: NonZeroU64::new(timeout_ms).unwrap(),
        pool_size: NonZeroU64::new(pool_size).unwrap(),
    }
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
    let batcher = get_batcher(EchoPredictor, config(8, 100, 1));
    assert_eq!(batcher.predict(42).await.unwrap(), 42);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_results_match_inputs() {
    // Fill an entire batch; each result must equal the corresponding input.
    let batcher = Arc::new(get_batcher(EchoPredictor, config(8, 500, 1)));

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
    // Long timeout (10s): the only thing that can trigger inference is the batch being full.
    let predictor = CountingPredictor::new();
    let batcher = Arc::new(get_batcher(predictor.clone(), config(4, 10_000, 1)));

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
    // batch_size=8 but only 1 request is sent, so inference must wait for the timeout.
    let timeout_ms = 50u64;
    let batcher = get_batcher(EchoPredictor, config(8, timeout_ms, 1));

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
    let batcher = get_batcher(FailPredictor, config(1, 50, 1));
    assert!(batcher.predict(0).await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_error_propagates_to_all_callers_in_batch() {
    let batcher = Arc::new(get_batcher(FailPredictor, config(4, 500, 1)));

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
    let batcher = Arc::new(get_batcher(EchoPredictor, config(16, 50, 1)));

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
    let batcher = Arc::new(get_batcher(EchoPredictor, config(4, 50, 4)));

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
    // 9 requests with batch_size=4 → 3 separate predict_batch calls (4, 4, 1).
    let predictor = CountingPredictor::new();
    let batcher = Arc::new(get_batcher(predictor.clone(), config(4, 500, 1)));

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
