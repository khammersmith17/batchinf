use super::{StartAckSender, buffer::WorkerBuffer, dispatch};
use crate::{
    error::StartupError,
    observability::{BatcherMetrics, InfBatchMetrics, emitters},
    predictor::{Predictor, PredictorFactory},
    shutdown::ShutdownSender,
    state::{PredictorStatus, WorkerState},
};
use std::{
    sync::Arc,
    thread::Builder,
    time::{Duration, Instant},
};
use tokio::sync::{Notify, mpsc::Receiver};

// Type alias for the receiver where work is passed from the accumulator.
pub(super) type JobReceiver<P> = Receiver<WorkerBuffer<P>>;

// Fat struct that contains all that is needed to start/run the inference thread.
pub(super) struct InferenceRunnerArgs<P: Predictor, E> {
    pub(super) factory: PredictorFactory<P, E>,
    pub(super) batch_recv: JobReceiver<P>,
    pub(super) worker_state: WorkerState,
    pub(super) worker_id: u8,
    pub(super) retries: u8,
    pub(super) backoff: Duration,
    pub(super) max_wait: Duration,
    pub(super) obs: Option<Arc<dyn BatcherMetrics>>,
    pub(super) ready_tx: StartAckSender<E>,
    pub(super) signal: Arc<Notify>,
}

fn backoff_schedule(
    attempts: u8,
    base_backoff: Duration,
    max_wait: Duration,
) -> impl Iterator<Item = Duration> {
    let mut next = base_backoff.min(max_wait);
    (0..attempts).map(move |i| {
        let wait = next;
        next = next.saturating_mul(2).min(max_wait);
        if i == attempts - 1 {
            Duration::ZERO
        } else {
            wait
        }
    })
}

/// Why a predictor could not be built.
enum BuildFailure<E> {
    /// The factory returned an error on the last attempt.
    Error(E),
    /// The factory panicked on the last attempt.
    Panic,
    /// The exit condition was met before the next attempt; nobody is waiting for the result.
    Cancelled,
}

// Sleep that can be cancelled.
// Poll the exit condition every 20 ms to determine if the build has been cancelled.
// If the build has been cancelled, exit.
fn sleep_with_exit(mut wait: Duration, exit_cond: &impl Fn() -> bool) -> bool {
    let slice = Duration::from_millis(20);

    while wait > Duration::ZERO {
        std::thread::sleep(slice.min(wait));
        if exit_cond() {
            return false;
        }
        let Some(new_wait) = wait.checked_sub(slice) else {
            return true;
        };
        wait = new_wait
    }
    true
}

fn try_make_predictor<P: Predictor, E>(
    f: PredictorFactory<P, E>,
    worker_id: u8,
    wait: Duration,
    exit_cond: &impl Fn() -> bool,
) -> Result<P, BuildFailure<E>> {
    let mut failure =
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || f(worker_id))) {
            Ok(Ok(predictor)) => return Ok(predictor),
            Ok(Err(e)) => BuildFailure::Error(e),
            Err(_) => BuildFailure::Panic,
        };
    let cont = sleep_with_exit(wait, exit_cond);

    if !cont {
        failure = BuildFailure::Cancelled;
    }
    Err(failure)
}

fn spawn_predictor<P: Predictor, E>(
    factory: PredictorFactory<P, E>,
    worker_id: u8,
    total_attempts: u8,
    backoff: Duration,
    max_wait: Duration,
    exit_cond: impl Fn() -> bool,
) -> Result<P, BuildFailure<E>> {
    let mut last_failure = BuildFailure::Cancelled;
    for wait in backoff_schedule(total_attempts, backoff, max_wait) {
        if exit_cond() {
            return Err(BuildFailure::Cancelled);
        }
        match try_make_predictor(Arc::clone(&factory), worker_id, wait, &exit_cond) {
            Ok(predictor) => return Ok(predictor),
            Err(failure) => last_failure = failure,
        }
    }
    Err(last_failure)
}

pub(super) fn spawn_inference_runner<P: Predictor + 'static, E: Send + 'static>(
    args: InferenceRunnerArgs<P, E>,
    shutdown_tx: ShutdownSender,
) {
    let worker_id = args.worker_id;
    let name = format!("batchinf-worker-{worker_id}");
    let ready_tx = args.ready_tx.clone();

    if Builder::new()
        .name(name)
        .spawn(move || {
            inference_runner(args);
            // `inference_runner` owned the predictor, the factory (and any weights it captures)
            // and every channel end, so all of it has been dropped by now. Only then release the
            // shutdown sender, which lets a `ShutdownWaiter` resolve. If the runner panics,
            // unwinding drops its frame first and then this one, so the order still holds.
            drop(shutdown_tx);
        })
        .is_err()
    {
        let _ = ready_tx.try_send(Err(StartupError::InternalError));
    }
    drop(ready_tx);
}

fn inference_runner<P: Predictor, E>(args: InferenceRunnerArgs<P, E>) {
    let InferenceRunnerArgs {
        factory,
        mut batch_recv,
        worker_state,
        worker_id,
        retries,
        backoff,
        max_wait,
        obs,
        ready_tx,
        signal,
    } = args;

    // ****** Start up ******
    let total_attempts = retries.saturating_add(1);

    let tx_ref = &ready_tx;

    // Report the outcome of the first build to `get_batcher`. If the factory keeps failing,
    // startup as a whole fails.
    let mut predictor = match spawn_predictor(
        Arc::clone(&factory),
        worker_id,
        total_attempts,
        backoff,
        max_wait,
        move || tx_ref.is_closed(), // If the startup ack sender is closed, either another worker
                                    // failed, or the user's code did not await the startup, so we stop
    ) {
        Ok(predictor) => predictor,
        Err(failure) => {
            worker_state.set_dead();
            let err = match failure {
                BuildFailure::Error(err) => StartupError::WorkerStartFailure { worker_id, err },
                BuildFailure::Panic => StartupError::WorkerStartPanic { worker_id },
                // Startup already ended (timeout or another worker failed), so nobody is
                // waiting for this worker's result.
                BuildFailure::Cancelled => return,
            };
            let _ = ready_tx.blocking_send(Err(err));
            return;
        }
    };

    // If send failures, the startup timed out or another worker failed to start and we should exit.
    if ready_tx.blocking_send(Ok(())).is_err() {
        return;
    };
    drop(ready_tx);

    // ****** Start Inference Runner ******

    // `blocking_recv` is fine here: this is a dedicated OS thread, not a runtime thread.
    while let Some(mut buffer) = batch_recv.blocking_recv() {
        // Filter out any canceled (timed out) requests.
        buffer.filter_canceled();

        // All request might have been canceled.
        if buffer.is_empty() {
            continue;
        }
        // Capture panic signal to handle crash and restart.
        let panic_signal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            predict(&mut predictor, buffer.input())
        }));

        // Inspect the result and handle panic in the case the user defined prediction method panicked.
        let (metrics, inf_result) = match panic_signal {
            Ok((m, ir)) => (m, ir),
            Err(_) => {
                // Mark the predictor as rebuilding, so new requests are routed elsewhere.
                worker_state.set_predictor_state(PredictorStatus::Rebuilding);

                // Emit a worker panic.
                emitters::emit_worker_panic(obs.clone());

                // Dispatch the internal error. Requests should not wait on a worker to restart.
                // They will not be retried here.
                let senders = buffer.take_senders();
                dispatch::send_errors_on_crash::<P>(senders);

                // Drop predictor while new one is being rebuilt.
                // Otherwise, memory footprint on the predictor would 2x.
                drop(predictor);
                let Ok(new_predictor) = spawn_predictor(
                    Arc::clone(&factory),
                    worker_id,
                    total_attempts,
                    backoff,
                    max_wait,
                    // Cancel a rebuild if the application is shutting down.
                    || batch_recv.is_closed(),
                ) else {
                    worker_state.set_dead();
                    // Allows requesters to get a `NoAvailableWorkers`.
                    signal.notify_waiters();
                    return;
                };
                predictor = new_predictor;
                worker_state.set_predictor_state(PredictorStatus::Alive);
                // This worker can accept work again.
                signal.notify_waiters();
                continue;
            }
        };

        let senders = buffer.take_senders();
        dispatch::send_output::<P>(inf_result, senders, metrics, obs.clone());
    }
}

/// Perform inference on a batch by calling the user defined batch inference method.
///
/// Captures inference latency and batch size for metric emission.
fn predict<P: Predictor>(
    predictor: &mut P,
    input: &[P::Input],
) -> (InfBatchMetrics, Result<Vec<P::Output>, P::Error>) {
    let start = Instant::now();

    // Runs on this worker's dedicated inference thread, so blocking here does not affect the
    // async runtime.
    let inf_results = predictor.predict_batch(input);

    let latency = Instant::now().duration_since(start);
    let metrics = InfBatchMetrics {
        size: input.len(),
        latency,
    };

    (metrics, inf_results)
}

#[cfg(test)]
mod backoff_schedule_tests {
    use super::backoff_schedule;
    use std::time::Duration;

    fn ms(millis: u64) -> Duration {
        Duration::from_millis(millis)
    }

    /// The waits produced for the given settings, in milliseconds.
    fn waits_ms(attempts: u8, base_ms: u64, max_ms: u64) -> Vec<u128> {
        backoff_schedule(attempts, ms(base_ms), ms(max_ms))
            .map(|wait| wait.as_millis())
            .collect()
    }

    #[test]
    fn yields_one_entry_per_attempt_with_ascending_indices() {
        let attempts: Vec<Duration> = backoff_schedule(4, ms(10), ms(1_000)).collect();
        assert_eq!(attempts.len(), 4);
    }

    #[test]
    fn zero_attempts_yields_nothing() {
        assert!(waits_ms(0, 50, 1_000).is_empty());
    }

    #[test]
    fn single_attempt_never_waits() {
        // `worker_start_retries: 0` gives one attempt: try once, and do not wait afterwards,
        // whatever the configured backoff is.
        assert_eq!(waits_ms(1, 50, 1_000), vec![0]);
        assert_eq!(waits_ms(1, 60_000, 60_000), vec![0]);
    }

    #[test]
    fn last_wait_is_always_zero() {
        // There is nothing to wait for after the final attempt.
        for attempts in [1u8, 2, 3, 10, u8::MAX] {
            assert_eq!(
                waits_ms(attempts, 50, 1_000).last(),
                Some(&0),
                "the wait after the final attempt should be zero for {attempts} attempts"
            );
        }
    }

    #[test]
    fn first_wait_is_the_base_backoff() {
        assert_eq!(waits_ms(2, 50, 1_000), vec![50, 0]);
    }

    #[test]
    fn wait_doubles_after_each_attempt() {
        assert_eq!(waits_ms(4, 50, 10_000), vec![50, 100, 200, 0]);
    }

    #[test]
    fn wait_is_capped_at_max_wait() {
        assert_eq!(waits_ms(5, 20, 30), vec![20, 30, 30, 30, 0]);
    }

    #[test]
    fn wait_stops_growing_exactly_at_the_cap() {
        // 25 → 50 → 100 reaches the cap without overshooting it.
        assert_eq!(waits_ms(5, 25, 100), vec![25, 50, 100, 100, 0]);
    }

    #[test]
    fn base_above_the_cap_is_clamped_from_the_first_wait() {
        assert_eq!(waits_ms(3, 500, 100), vec![100, 100, 0]);
    }

    #[test]
    fn zero_base_backoff_never_waits() {
        assert_eq!(waits_ms(4, 0, 1_000), vec![0, 0, 0, 0]);
    }

    #[test]
    fn zero_max_wait_never_waits() {
        assert_eq!(waits_ms(4, 50, 0), vec![0, 0, 0, 0]);
    }

    #[test]
    fn waits_never_decrease_and_never_exceed_the_cap() {
        for (base, max) in [(1, 1_000), (7, 500), (50, 50), (300, 100), (0, 10)] {
            let mut waits = waits_ms(u8::MAX, base, max);
            // Ignore no-op wait.
            waits.pop();
            assert_eq!(waits.len(), usize::from(u8::MAX) - 1);
            assert!(
                waits.windows(2).all(|pair| pair[0] <= pair[1]),
                "waits decreased for base {base}ms, max {max}ms"
            );
            assert!(
                waits.iter().all(|wait| *wait <= u128::from(max)),
                "a wait exceeded the cap for base {base}ms, max {max}ms"
            );
        }
    }

    #[test]
    fn maximum_attempts_settle_at_the_cap() {
        let waits: Vec<Duration> =
            backoff_schedule(u8::MAX, Duration::from_secs(1), Duration::from_secs(30)).collect();

        let mut rev = waits.into_iter().rev();
        let _ = rev.next();
        let last = rev.next();
        assert_eq!(last, Some(Duration::from_secs(30)));
    }

    #[test]
    fn uncapped_backoff_does_not_overflow() {
        // `Duration::MAX` as the cap means "no cap". Doubling 255 times must saturate rather
        // than panic.
        let waits: Vec<Duration> =
            backoff_schedule(u8::MAX, Duration::from_secs(1), Duration::MAX).collect();
        assert_eq!(waits[0], Duration::from_secs(1));
        assert_eq!(waits[10], Duration::from_secs(1 << 10));
        let mut rev = waits.iter().rev();
        let _ = rev.next();
        let last = rev.next();
        assert_eq!(last, Some(&Duration::MAX));
    }
}
