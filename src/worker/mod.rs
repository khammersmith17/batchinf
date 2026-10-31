use crate::{
    error::{BatchinfError, StartupError},
    observability::{BatchTrigger, BatcherMetrics, emitters},
    predictor::{Predictor, PredictorFactory},
    shutdown::ShutdownSender,
    state::{PredictorStatus, WorkerState, WorkerStatus},
};
use std::sync::Arc;
use tokio::{
    select,
    sync::{
        Notify,
        mpsc::{Receiver, Sender, channel},
        oneshot::Sender as OneshotSender,
    },
    time::{Duration, Instant, sleep_until},
};

mod buffer;
mod dispatch;
mod inf;
mod panic_guard;
use buffer::WorkerBuffer;
use inf::{InferenceRunnerArgs, spawn_inference_runner};
use panic_guard::PanicGuard;
use tracing::{Instrument, debug, error, info, info_span};

pub(crate) type OutputSender<P> =
    OneshotSender<Result<<P as Predictor>::Output, BatchinfError<<P as Predictor>::Error>>>;
pub(crate) type InputReceiver<P> = Receiver<(<P as Predictor>::Input, OutputSender<P>)>;
pub(crate) type InferenceResult<P> = Result<Vec<<P as Predictor>::Output>, <P as Predictor>::Error>;
type JobSender<P> = Sender<WorkerBuffer<P>>;
/// Reports the outcome of a worker's first predictor build to `get_batcher`.
pub(crate) type StartAckSender<E> = Sender<Result<(), StartupError<E>>>;

// One batch queued while one runs on the inference thread. When both slots are taken, the
// accumulator waits on `send`, its request channel fills, and the router moves on to other workers.
const JOB_QUEUE_DEPTH: usize = 1;

// Far out timeout for when a user specifies Duration::MAX as a batch timeout.
// This is a valid design choice by a user: only perform inference once the batch size is reached.
const ONE_YEAR: Duration = Duration::from_secs(24 * 60 * 60 * 365);

/// Describes that status of an accumulated batch.
enum BatchStatus<P: Predictor> {
    /// Batch is ready for inference
    Ready {
        buffer: WorkerBuffer<P>,
        trigger: BatchTrigger,
    },
    /// The batch has been flushed and erros returned to the callers.
    /// This occurs when the predictor is no longer able to serve inference.
    Flushed,
    /// When the channel has been closed.
    Exit,
}

enum BatchHandoffResult {
    Sent,
    Flushed,
    Dead,
}

/// State the accumulator task holds for its whole lifetime.
pub(crate) struct InferenceWorker {
    state: WorkerState,
    obs: Option<Arc<dyn BatcherMetrics>>,
    worker_id: u8,
    signal: Arc<Notify>,
}

/// Core behavior of the inference worker.
impl InferenceWorker {
    pub(crate) fn new(
        state: WorkerState,
        obs: Option<Arc<dyn BatcherMetrics>>,
        worker_id: u8,
        signal: Arc<Notify>,
    ) -> InferenceWorker {
        InferenceWorker {
            state,
            obs,
            worker_id,
            signal,
        }
    }

    // Reset the inference timeout clock.
    // Checked add for the case where user defines [Duration::MAX], which is a valid configuration.
    fn reset_next_inf(&self, ts: &mut Instant) {
        let now = Instant::now();
        *ts = now
            .checked_add(self.state.timeout())
            .unwrap_or(now + ONE_YEAR);
    }

    fn emit_batch_start(&self, trigger_type: BatchTrigger, batch_size: usize) {
        self.state.set_worker_state(WorkerStatus::Running);
        emitters::emit_batch_start(self.obs.clone(), trigger_type, batch_size);
    }

    // Report the number of requests waiting in this worker's queue, not yet pulled into a batch.
    fn emit_queue_depth<T>(&self, input_receiver: &Receiver<T>) {
        emitters::emit_queue_depth(self.obs.clone(), self.worker_id, input_receiver.len());
    }

    fn emit_empty_queue(&self) {
        emitters::emit_queue_depth(self.obs.clone(), self.worker_id, 0_usize);
    }

    fn notify_one(&self) {
        self.signal.notify_one()
    }
}
/// Settings for building a worker's predictor. Only needed to spawn the inference thread, so
/// they are passed through to it by value rather than kept in the worker's shared state.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PredictorStartConfig {
    /// Number of retries after the first failed factory call.
    pub(crate) retries: u8,
    /// Wait after the first failed attempt; doubles after each further failure.
    pub(crate) backoff: Duration,
    /// Upper bound on the wait between two attempts.
    pub(crate) max_wait: Duration,
}

/// Everything a worker hands to its inference thread when spawning it. Passed by value from
/// `get_batcher` straight through to the thread, so the accumulator task never keeps any of it:
/// the factory (and any model weights it captures) and the shutdown sender live exactly as long
/// as the inference thread.
pub(crate) struct RunnerHandoff<P: Predictor, E> {
    pub(crate) factory: PredictorFactory<P, E>,
    pub(crate) start: PredictorStartConfig,
    // Acks readiness before returning handle to user.
    pub(crate) ready_tx: StartAckSender<E>,
    // Notifies on shutdown.
    pub(crate) shutdown_tx: ShutdownSender,
}

fn make_runner<P: Predictor + 'static, E: std::fmt::Debug + Send + 'static>(
    worker: &InferenceWorker,
    handoff: RunnerHandoff<P, E>,
) -> JobSender<P> {
    let RunnerHandoff {
        factory,
        start,
        ready_tx,
        shutdown_tx,
    } = handoff;
    let (job_tx, job_rx) = channel::<WorkerBuffer<P>>(JOB_QUEUE_DEPTH);
    spawn_inference_runner(
        InferenceRunnerArgs {
            factory,
            batch_recv: job_rx,
            worker_state: worker.state.clone(),
            worker_id: worker.worker_id,
            retries: start.retries,
            backoff: start.backoff,
            max_wait: start.max_wait,
            obs: worker.obs.clone(),
            ready_tx,
            signal: Arc::clone(&worker.signal),
        },
        shutdown_tx,
    );
    job_tx
}

// The main worker loop.
//
// Accumulate inference examples from request writers until batch conditions are satisfied,
// then dispatch inference.
//
// After inference, the state is evaluated to determine if the worker should continue.
async fn worker_loop<P: Predictor + 'static, E: std::fmt::Debug + Send + 'static>(
    worker: InferenceWorker,
    mut input_receiver: InputReceiver<P>,
    handoff: RunnerHandoff<P, E>,
) {
    info!("Starting accumulator");
    // Register the panic guard.
    // Panics could occur in the user defined observability callbacks.
    let _panic_guard = PanicGuard::new(worker.state.clone(), Arc::clone(&worker.signal));
    let cap = worker.state.batch_size() as usize;
    let mut next_inf = Instant::now();
    let mut buffer = WorkerBuffer::new(cap);
    let job_tx = make_runner(&worker, handoff);

    loop {
        let batch_result = accumulate_next_batch(
            &mut input_receiver,
            &worker,
            &mut buffer,
            &mut next_inf,
            &job_tx,
        )
        .await;
        match batch_result {
            BatchStatus::Ready { buffer, trigger } => {
                let buffer_len = buffer.len();
                worker.state.set_worker_state(WorkerStatus::Running);
                worker.emit_queue_depth(&input_receiver);
                // Waits here while the inference thread is busy and a batch is already queued.
                let handoff_result =
                    run_inference(buffer, &job_tx, &worker.state, &mut input_receiver).await;
                match handoff_result {
                    BatchHandoffResult::Dead => {
                        // The inference thread is gone. Returning drops the request receiver, so the
                        // router sees this worker as closed and queued requests get InternalError.
                        error!("inference thread is gone, marking worker as WorkerStatus::Crashed");
                        // The queue is now empty and this value will not emit again for this
                        // worker.
                        // Emitting an empty queue here to represent the state.
                        worker.emit_empty_queue();
                        worker.state.set_worker_state(WorkerStatus::Crashed);
                        return;
                    }
                    BatchHandoffResult::Sent => {
                        // Backlog that built up while waiting on a busy or rebuilding inference thread.
                        worker.emit_batch_start(trigger, buffer_len);
                    }
                    // Emit queue depth again to the value emitted is fresh.
                    // When the predictor is being rebuilt, it will not be emitted for some time.
                    BatchHandoffResult::Flushed => worker.emit_queue_depth(&input_receiver),
                }

                // Batch handed off or flushed. Signal that this worker can accept a new batch.
                worker.state.reset_batch_len();
            }
            BatchStatus::Flushed => continue,
            BatchStatus::Exit => return,
        }
    }
}

/// Accumulate the next batch of inference data.
/// Starts timer for the batch upon receiving the first record for the batch.
/// Sets state to Running when the batch is ready, or Exit when the channel closes.
///
/// The buffer is either non empty after this function completes, or the worker is being shut down.
#[must_use]
async fn accumulate_next_batch<P: Predictor + 'static>(
    receiver: &mut InputReceiver<P>,
    worker: &InferenceWorker,
    buffer: &mut WorkerBuffer<P>,
    next_inf: &mut Instant,
    job_tx: &JobSender<P>,
) -> BatchStatus<P> {
    // Wait for first item in batch to start batch timer.
    if let Some((request, sender)) = receiver.recv().await {
        debug!("Received first item in batch, starting timer");
        worker.notify_one();
        buffer.push(request, sender);
        worker.state.increment_len();
        worker.reset_next_inf(next_inf);
    } else {
        debug!("Receiver closed, setting state to WorkerStatus::Exit");
        // None indicates the channel is closed. In this case set the state to exit.
        worker.state.set_worker_state(WorkerStatus::Exit);
        return BatchStatus::Exit;
    }

    // Batch size 1 is satisfied on first read.
    if worker.state.batch_size() == 1 {
        let predictor_state = check_predictor_alive(&worker.state, receiver, buffer);
        return if matches!(predictor_state, PredictorStatus::Alive) {
            let live_buffer = buffer.swap_replace();
            BatchStatus::Ready {
                buffer: live_buffer,
                trigger: BatchTrigger::Capacity,
            }
        } else {
            BatchStatus::Flushed
        };
    }

    let batch_start = Instant::now();

    // Set the timer for a batch accumulation timeout.
    let timer = sleep_until(*next_inf);
    tokio::pin!(timer);

    let mut trigger = BatchTrigger::Capacity;

    loop {
        // Evaluate the winner between an inference request being queued and the batch timeout.
        select! {
            user_input = receiver.recv() => {
                if let Some((request, sender)) = user_input {
                    // Only one requester gets notified when a slot opens.
                    worker.notify_one();
                    buffer.push(request, sender);

                    worker.state.increment_len();

                    // Check the batch size condition to trigger inference.
                    if buffer.len() == (worker.state.batch_size() as usize) {

                        debug!("batch triggered by size in {} milliseconds",
                            batch_start.elapsed().as_millis());
                        // Set state and emit trigger.
                        break;
                    }

                } else {
                    worker.state.set_worker_state(WorkerStatus::Exit);
                    return BatchStatus::Exit;
                };

            }
            _ = &mut timer  => {
                    trigger = BatchTrigger::Timeout;
                    debug!("batch triggered by timeout with {} items", buffer.len());
                    break;
            }
            // Wake the accumulator when the predictor goes dead.
            _ = job_tx.closed() => break

        }
    }

    let predictor_state = check_predictor_alive(&worker.state, receiver, buffer);
    if matches!(predictor_state, PredictorStatus::Alive) {
        let live_buffer = buffer.swap_replace();
        BatchStatus::Ready {
            buffer: live_buffer,
            trigger,
        }
    } else {
        BatchStatus::Flushed
    }
}

fn check_predictor_alive<P: Predictor>(
    state: &WorkerState,
    receiver: &mut InputReceiver<P>,
    buffer: &mut WorkerBuffer<P>,
) -> PredictorStatus {
    let predictor_state = state.get_predictor_state();
    // If the predictor has crashed since the router read its state when resolving to this worker.
    if matches!(
        predictor_state,
        PredictorStatus::Rebuilding | PredictorStatus::Dead
    ) {
        debug!("Inference worker crashed draining the receiver and dispatching errors");
        handle_crashed_inf(receiver, buffer);
        state.reset_batch_len();
    }

    predictor_state
}

/// When the inference worker crashes, all requests queued in the current batch are failed as the
/// worker they were dispatch to is not currently available.
fn handle_crashed_inf<P: Predictor>(receiver: &mut InputReceiver<P>, buffer: &mut WorkerBuffer<P>) {
    // Drain the reciever. ALl currently queued requests fail as the worker is not
    // available.
    // This worker should also not be routed any new requests, as the resolver now will not forward
    // requests to this workers queue.
    while let Ok((inp, sender)) = receiver.try_recv() {
        buffer.push(inp, sender);
    }
    let batch = buffer.swap_replace();
    let (_, senders) = batch.split();
    dispatch::send_errors_on_crash::<P>(senders);
}

/// Forward the buffer to the inference thread.
async fn run_inference<P: Predictor>(
    mut buffer: WorkerBuffer<P>,
    job_sender: &JobSender<P>,
    state: &WorkerState,
    recv: &mut InputReceiver<P>,
) -> BatchHandoffResult {
    if buffer.is_empty() {
        return BatchHandoffResult::Sent;
    }

    let Ok(permit) = job_sender.reserve().await else {
        return BatchHandoffResult::Dead;
    };

    if state.get_predictor_state() != PredictorStatus::Alive {
        drop(permit);
        handle_crashed_inf(recv, &mut buffer);
        return BatchHandoffResult::Flushed;
    };

    permit.send(buffer);
    BatchHandoffResult::Sent
}

/// Entry point to worker lifecycle.
pub(crate) fn run_worker<P: Predictor + 'static, E: std::fmt::Debug + Send + 'static>(
    worker: InferenceWorker,
    receiver: InputReceiver<P>,
    handoff: RunnerHandoff<P, E>,
) {
    let span = info_span!("batchinf_accumulator", worker_id = worker.worker_id);
    tokio::task::spawn(
        async move { worker_loop(worker, receiver, handoff).await }.instrument(span),
    );
}
