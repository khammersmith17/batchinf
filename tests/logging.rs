//! Tests for the crate's `tracing` output.
//!
//! Inference runs on dedicated OS threads, which a thread-local subscriber would not see. This
//! file is its own test binary, so it installs one global subscriber that records every event at
//! every level. The tests share it, so they run one at a time and each starts from an empty log.
//!
//! Assertions use levels, span names, `worker_id` and field names rather than message text, so
//! rewording a log message does not break them.

use batchinf::{
    BatcherConfig, BatchinfError, Predictor, PredictorStatus, WorkerStatus, get_batcher,
};
use std::{
    collections::BTreeSet,
    num::{NonZeroU8, NonZeroU32},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU32, Ordering},
    },
    time::{Duration, Instant},
};
use tracing::{
    Event, Level, Subscriber,
    field::{Field, Visit},
    span,
};
use tracing_subscriber::{
    layer::{Context, Layer, SubscriberExt},
    registry::LookupSpan,
};

// ---------------------------------------------------------------------------------------------
// Capturing subscriber
// ---------------------------------------------------------------------------------------------

/// One recorded event.
#[derive(Debug, Clone)]
struct Captured {
    level: Level,
    /// The innermost `batchinf*` span the event was emitted in, if any.
    span: Option<&'static str>,
    /// `worker_id` from the event itself, or else from an enclosing span.
    worker_id: Option<u64>,
    /// Every field of the event, formatted, including `message`.
    fields: Vec<(&'static str, String)>,
}

impl Captured {
    fn field(&self, name: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(field, _)| *field == name)
            .map(|(_, value)| value.as_str())
    }
}

/// Collects the fields of an event or span.
#[derive(Default)]
struct FieldRecorder {
    fields: Vec<(&'static str, String)>,
    worker_id: Option<u64>,
}

impl Visit for FieldRecorder {
    fn record_u64(&mut self, field: &Field, value: u64) {
        // `worker_id` is a `u8`, which tracing records as a `u64`.
        if field.name() == "worker_id" {
            self.worker_id = Some(value);
        }
        self.fields.push((field.name(), value.to_string()));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.fields.push((field.name(), value.to_string()));
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.fields.push((field.name(), format!("{value:?}")));
    }
}

/// The `worker_id` recorded on a span, stored in the span's extensions.
struct SpanWorkerId(Option<u64>);

struct CaptureLayer(Arc<Mutex<Vec<Captured>>>);

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        let mut recorder = FieldRecorder::default();
        attrs.record(&mut recorder);
        if let Some(span) = ctx.span(id) {
            span.extensions_mut()
                .insert(SpanWorkerId(recorder.worker_id));
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let mut recorder = FieldRecorder::default();
        event.record(&mut recorder);

        let mut worker_id = recorder.worker_id;
        let mut span_name = None;
        if let Some(scope) = ctx.event_scope(event) {
            // Innermost span first.
            for span in scope {
                if span_name.is_none() && span.name().starts_with("batchinf") {
                    span_name = Some(span.name());
                }
                if worker_id.is_none() {
                    if let Some(SpanWorkerId(Some(id))) = span.extensions().get::<SpanWorkerId>() {
                        worker_id = Some(*id);
                    }
                }
            }
        }

        self.0.lock().unwrap().push(Captured {
            level: *event.metadata().level(),
            span: span_name,
            worker_id,
            fields: recorder.fields,
        });
    }
}

static EVENTS: OnceLock<Arc<Mutex<Vec<Captured>>>> = OnceLock::new();
static SERIAL: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

/// The shared event log. Installs the global subscriber on first use.
fn events() -> &'static Arc<Mutex<Vec<Captured>>> {
    EVENTS.get_or_init(|| {
        let events = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&events)));
        tracing::subscriber::set_global_default(subscriber)
            .expect("no other global subscriber in this test binary");
        events
    })
}

/// Starts a test: waits for the previous one to finish, then clears the event log.
async fn start() -> tokio::sync::MutexGuard<'static, ()> {
    let guard = SERIAL
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await;
    events().lock().unwrap().clear();
    guard
}

fn captured() -> Vec<Captured> {
    events().lock().unwrap().clone()
}

/// Polls `condition` every 5ms until it holds or `timeout` passes.
async fn wait_until(timeout: Duration, condition: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if condition() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    condition()
}

// ---------------------------------------------------------------------------------------------
// Predictors
// ---------------------------------------------------------------------------------------------

#[derive(Debug)]
struct LogTestError(&'static str);

impl std::fmt::Display for LogTestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for LogTestError {}

struct Echo;

impl Predictor for Echo {
    type Input = u64;
    type Output = u64;
    type Error = LogTestError;

    fn predict_batch(&mut self, inputs: Vec<u64>) -> Result<Vec<u64>, LogTestError> {
        Ok(inputs)
    }
}

struct Exploding;

impl Predictor for Exploding {
    type Input = u64;
    type Output = u64;
    type Error = LogTestError;

    fn predict_batch(&mut self, _inputs: Vec<u64>) -> Result<Vec<u64>, LogTestError> {
        panic!("model exploded");
    }
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

// A `predict_batch` panic is logged as a warning with the panic message, each failed rebuild
// attempt as a warning with the factory's error, and giving up as an error, all tagged with the
// worker they happened on.
#[tokio::test(flavor = "multi_thread")]
async fn test_panic_and_failed_rebuild_are_logged() {
    let _serial = start().await;

    let builds = Arc::new(AtomicU32::new(0));
    let counter = Arc::clone(&builds);
    let batcher = get_batcher(
        Arc::new(move |_worker_id| {
            // The first build succeeds; every rebuild attempt fails.
            if counter.fetch_add(1, Ordering::SeqCst) > 0 {
                return Err(LogTestError("weights missing"));
            }
            Ok(Exploding)
        }),
        BatcherConfig::builder(NonZeroU32::new(1).unwrap(), Duration::from_millis(5))
            .worker_start_retries(1)
            .worker_start_backoff_start(Duration::ZERO)
            .build(),
    )
    .await
    .unwrap();
    let waiter = batcher.shutdown_waiter();

    let result = batcher.predict(1).await;
    assert!(
        matches!(result, Err(BatchinfError::InternalError)),
        "{result:?}"
    );
    assert!(
        wait_until(Duration::from_secs(2), || {
            batcher.worker_status(0).unwrap().predictor_status == PredictorStatus::Dead
        })
        .await,
        "the rebuild should give up"
    );
    drop(batcher);
    tokio::time::timeout(Duration::from_secs(2), waiter.wait())
        .await
        .expect("shutdown should complete");

    let events = captured();
    let inference: Vec<&Captured> = events
        .iter()
        .filter(|e| e.span == Some("batchinf_inference"))
        .collect();

    let panic_at = inference
        .iter()
        .position(|e| e.level == Level::WARN && e.field("panic") == Some("model exploded"))
        .unwrap_or_else(|| panic!("no warning with the panic message: {events:#?}"));
    assert_eq!(inference[panic_at].worker_id, Some(0));

    // One attempt plus one retry, each failing with the factory's error.
    let factory_errors: Vec<&&Captured> = inference
        .iter()
        .filter(|e| {
            e.level == Level::WARN
                && e.field("error")
                    .is_some_and(|err| err.contains("weights missing"))
        })
        .collect();
    assert_eq!(factory_errors.len(), 2, "{events:#?}");
    assert!(factory_errors.iter().all(|e| e.worker_id == Some(0)));

    let gave_up_at = inference
        .iter()
        .position(|e| e.level == Level::ERROR)
        .unwrap_or_else(|| panic!("no error when the rebuild gave up: {events:#?}"));
    assert!(gave_up_at > panic_at, "the error should follow the panic");
    assert_eq!(inference[gave_up_at].worker_id, Some(0));
}

// Every event from a worker's batching task and inference thread is emitted inside that worker's
// span, so it carries the right `worker_id`.
#[tokio::test(flavor = "multi_thread")]
async fn test_events_carry_their_worker_id() {
    let _serial = start().await;

    let batcher = get_batcher(
        Arc::new(|_worker_id| Ok::<_, LogTestError>(Echo)),
        BatcherConfig::builder(NonZeroU32::new(1).unwrap(), Duration::from_millis(5))
            .pool_size(NonZeroU8::new(2).unwrap())
            .build(),
    )
    .await
    .unwrap();
    let waiter = batcher.shutdown_waiter();

    let handles: Vec<_> = (0..16u64)
        .map(|i| {
            let batcher = batcher.clone();
            tokio::spawn(async move { batcher.predict(i).await })
        })
        .collect();
    for (i, handle) in handles.into_iter().enumerate() {
        assert_eq!(handle.await.unwrap().unwrap(), i as u64);
    }
    drop(batcher);
    tokio::time::timeout(Duration::from_secs(2), waiter.wait())
        .await
        .expect("shutdown should complete");

    let events = captured();
    for span in ["batchinf_accumulator", "batchinf_inference"] {
        let workers: BTreeSet<Option<u64>> = events
            .iter()
            .filter(|e| e.span == Some(span))
            .map(|e| e.worker_id)
            .collect();
        assert_eq!(
            workers,
            BTreeSet::from([Some(0), Some(1)]),
            "every `{span}` event should carry worker_id 0 or 1, and both workers should log: \
             {events:#?}"
        );
    }
}

// Regression test: with `batch_timeout = Duration::MAX` and the batching task's events enabled,
// logging how long a full batch took to fill used to overflow `Instant - Duration` and panic the
// batching task, crashing the worker. Events are only evaluated when a subscriber wants them,
// which is why this needs the capturing subscriber.
#[tokio::test(flavor = "multi_thread")]
async fn test_capacity_batch_with_max_timeout_while_logging() {
    let _serial = start().await;

    let batcher = get_batcher(
        Arc::new(|_worker_id| Ok::<_, LogTestError>(Echo)),
        BatcherConfig::builder(NonZeroU32::new(2).unwrap(), Duration::MAX).build(),
    )
    .await
    .unwrap();
    let waiter = batcher.shutdown_waiter();

    let handles: Vec<_> = [1u64, 2]
        .into_iter()
        .map(|i| {
            let batcher = batcher.clone();
            tokio::spawn(async move { batcher.predict(i).await })
        })
        .collect();
    let mut results = Vec::new();
    for handle in handles {
        results.push(
            tokio::time::timeout(Duration::from_secs(2), handle)
                .await
                .expect("a full batch should fire on capacity")
                .unwrap()
                .expect("the batch should succeed"),
        );
    }
    results.sort_unstable();
    assert_eq!(results, vec![1, 2]);

    let status = batcher.worker_status(0).unwrap();
    assert_ne!(status.worker_status, WorkerStatus::Crashed);
    assert_eq!(status.predictor_status, PredictorStatus::Alive);

    drop(batcher);
    tokio::time::timeout(Duration::from_secs(2), waiter.wait())
        .await
        .expect("shutdown should complete");

    let events = captured();
    assert!(
        events.iter().all(|e| e.level != Level::ERROR),
        "no errors expected: {events:#?}"
    );
}
