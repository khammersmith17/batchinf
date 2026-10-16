use super::{WorkerBuffer, dispatch};
use crate::{
    observability::{BatcherMetrics, InfBatchMetrics, emitters},
    predictor::Predictor,
    setup::PredictorFactory,
    state::{PredictorStatus, WorkerState},
};
use std::{
    sync::Arc,
    thread::Builder,
    time::{Duration, Instant},
};
use tokio::sync::{
    Notify,
    mpsc::{Receiver, Sender},
};

pub(super) type JobReceiver<P> = Receiver<WorkerBuffer<P>>;

pub(super) struct InferenceRunnerArgs<P: Predictor> {
    pub(super) factory: PredictorFactory<P>,
    pub(super) recv: JobReceiver<P>,
    pub(super) worker_state: WorkerState,
    pub(super) worker_id: u8,
    pub(super) retries: u8,
    pub(super) backoff: Duration,
    pub(super) max_wait: Duration,
    pub(super) obs: Option<Arc<dyn BatcherMetrics>>,
    pub(super) ready_tx: Sender<(u8, bool)>,
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

fn try_make_predictor<P: Predictor>(
    f: PredictorFactory<P>,
    worker_id: u8,
    wait: Duration,
) -> Result<P, ()> {
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || f(worker_id)))
        .map_err(|_e| ());
    if res.is_err() {
        // Last wait is no-op.
        std::thread::sleep(wait)
    }
    res
}

fn spawn_predictor<P: Predictor>(
    factory: PredictorFactory<P>,
    worker_id: u8,
    total_attempts: u8,
    backoff: Duration,
    max_wait: Duration,
    exit_cond: impl Fn() -> bool,
) -> Result<P, ()> {
    for wait in backoff_schedule(total_attempts, backoff, max_wait) {
        if exit_cond() {
            return Err(());
        }
        let p = try_make_predictor(Arc::clone(&factory), worker_id, wait);

        if p.is_ok() {
            return p;
        }
    }
    Err(())
}

pub(super) fn spawn_inference_runner<P: Predictor + 'static>(args: InferenceRunnerArgs<P>) {
    let worker_id = args.worker_id;
    let name = format!("batchinf-worker-{worker_id}");
    let ready_tx = args.ready_tx.clone();

    if Builder::new()
        .name(name)
        .spawn(move || inference_runner(args))
        .is_err()
    {
        let _ = ready_tx.try_send((worker_id, false));
    }
    drop(ready_tx);
}

fn inference_runner<P: Predictor>(args: InferenceRunnerArgs<P>) {
    let InferenceRunnerArgs {
        factory,
        mut recv,
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
    let Ok(mut predictor) = spawn_predictor(
        Arc::clone(&factory),
        worker_id,
        total_attempts,
        backoff,
        max_wait,
        move || tx_ref.is_closed(),
    ) else {
        let _ = ready_tx.blocking_send((worker_id, false));
        worker_state.set_dead();
        return;
    };

    // If send failures, the startup timed out or another worker failed to start and we should exit.
    if ready_tx.blocking_send((worker_id, true)).is_err() {
        return;
    };
    drop(ready_tx);

    // ****** Start Inference Runner ******

    // Fn closure to generate a new predictor on panic.
    let generator = move || {
        spawn_predictor(
            Arc::clone(&factory),
            worker_id,
            total_attempts,
            backoff,
            max_wait,
            || false,
        )
    };

    // `blocking_recv` is fine here: this is a dedicated OS thread, not a runtime thread.
    while let Some(mut buffer) = recv.blocking_recv() {
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
                let senders = buffer.clear_and_take_senders();
                dispatch::send_errors_on_crash::<P>(senders);
                let Ok(new_predictor) = generator() else {
                    worker_state.set_dead();
                    // If this is the only worker, notify a caller that there are no longer workers
                    // available.
                    signal.notify_waiters();
                    return;
                };
                predictor = new_predictor;
                worker_state.set_predictor_state(PredictorStatus::Alive);
                // This worker can accept work again, in the case others are blocked.
                signal.notify_waiters();
                continue;
            }
        };

        let senders = buffer.clear_and_take_senders();
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
