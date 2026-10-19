use crate::{
    error::{BatchinfError, StartupError},
    observability::{BatchTrigger, BatcherMetrics, emitters},
    predictor::{Predictor, PredictorFactory},
    shutdown::ShutdownSender,
    state::{WorkerState, WorkerStatus},
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
use buffer::WorkerBuffer;
use inf::{InferenceRunnerArgs, spawn_inference_runner};

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
// This is valid design choice, only wanting to perform inference when batch size is satisfied.
const ONE_YEAR: Duration = Duration::from_secs(24 * 60 * 60 * 365);

struct DropGuard {
    state: WorkerState,
    signal: Arc<Notify>,
}

impl Drop for DropGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            // Setting the state to crashed, if there is a panic
            // This should not happen. There should not be anything that can panic here, but to be
            // safe this guard will make sure no requests get routed.
            self.state.set_worker_state(WorkerStatus::Crashed);
            // Notify waiters in the case this is the only worker and messages can no longer be serviced.
            self.signal.notify_waiters();
        }
    }
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

    fn set_running_and_emit(&self, trigger_type: BatchTrigger, batch_size: usize) {
        self.state.set_worker_state(WorkerStatus::Running);
        emitters::emit_batch_start(self.obs.clone(), trigger_type, batch_size);
    }

    // Report the number of requests waiting in this worker's queue, not yet pulled into a batch.
    fn emit_queue_depth<T>(&self, input_receiver: &Receiver<T>) {
        emitters::emit_queue_depth(self.obs.clone(), self.worker_id, input_receiver.len());
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

fn make_runner<P: Predictor + 'static, E: Send + 'static>(
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
async fn worker_loop<P: Predictor + 'static, E: Send + 'static>(
    worker: InferenceWorker,
    mut input_receiver: InputReceiver<P>,
    handoff: RunnerHandoff<P, E>,
) {
    let _drop_guard = DropGuard {
        state: worker.state.clone(),
        signal: worker.signal.clone(),
    };
    let cap = worker.state.batch_size() as usize;
    let mut next_inf = Instant::now();
    let mut buffer = WorkerBuffer::new(cap);
    let job_tx = make_runner(&worker, handoff);

    loop {
        accumulate_next_batch(&mut input_receiver, &worker, &mut buffer, &mut next_inf).await;
        let live_buffer = buffer.swap_replace();
        match worker.state.get_worker_state() {
            WorkerStatus::Waiting => unreachable!(),
            WorkerStatus::Running => {
                // Backlog behind the batch that is about to be handed off.
                worker.emit_queue_depth(&input_receiver);
                // Waits here while the inference thread is busy and a batch is already queued.
                let alive = run_inference(live_buffer, &job_tx).await;
                if alive.is_err() {
                    // The inference thread is gone. Returning drops the request receiver, so the
                    // router sees this worker as closed and queued requests get InternalError.
                    worker.state.set_worker_state(WorkerStatus::Crashed);
                    return;
                }
                // Backlog that built up while waiting on a busy or rebuilding inference thread.
                worker.emit_queue_depth(&input_receiver);
                // Batch handed off. Signal that this worker can accept a new batch.
                worker.state.reset_batch_len();
            }
            WorkerStatus::Exit => {
                // Ownership semantics ensure that no requests are still queued.
                // If there are any tasks waiting on a oneshot recv, all worker non-crashed channels will still be open.
                return;
            }
            // Only this loop sets Crashed, when the inference thread is gone, and it returns
            // immediately afterwards. A predict_batch panic is handled on the inference thread.
            WorkerStatus::Crashed => {
                unreachable!("Worker observed its own crashed state")
            }
        }
    }
}

/// Accumulate the next batch of inference data.
/// Starts timer for the batch upon receiving the first record for the batch.
/// Sets state to Running when the batch is ready, or Exit when the channel closes.
async fn accumulate_next_batch<P: Predictor + 'static>(
    receiver: &mut InputReceiver<P>,
    worker: &InferenceWorker,
    buffer: &mut WorkerBuffer<P>,
    next_inf: &mut Instant,
) {
    // Wait for first item in batch to start batch timer.
    if let Some((request, sender)) = receiver.recv().await {
        worker.notify_one();
        buffer.push(request, sender);
        worker.state.increment_len();
        worker.reset_next_inf(next_inf);
    } else {
        // None indicates the channel is closed. In this case set the state to exit.
        worker.state.set_worker_state(WorkerStatus::Exit);
        return;
    }

    // Batch size 1 is satisfied on first read.
    if worker.state.batch_size() == 1 {
        worker.set_running_and_emit(BatchTrigger::Capacity, buffer.len());
        return;
    }

    // Set the timer for a batch accumulation timeout.
    let timer = sleep_until(*next_inf);
    tokio::pin!(timer);

    loop {
        // Evaluate the winner between an inference request being queued and the batch timeout.
        select! {
            user_input = receiver.recv() => {
                if let Some((request, sender)) = user_input {
                    // Only one requester gets notified when a slot opens.
                    worker.notify_one();
                    buffer.push(request, sender);

                    worker.state.increment_len();
                    if buffer.len() == (worker.state.batch_size() as usize) {
                        // Set state and emit trigger.
                        worker.set_running_and_emit(BatchTrigger::Capacity, buffer.len());
                        break;
                    }

                } else {
                    worker.state.set_worker_state(WorkerStatus::Exit);
                    return;
                };

            }
            _ = &mut timer  => {
                    // Do not overwrite state to running when state has been set to Exit.
                    // Inference happens in an Exit state.
                    //
                    // Only log a timeout when the worker has not exited.
                    if matches!(worker.state.get_worker_state(), WorkerStatus::Waiting) && !buffer.is_empty() {
                        worker.set_running_and_emit(BatchTrigger::Timeout, buffer.len());
                    }

                    return;
            }

        }
    }
}

/// Inference runner.
/// Runs inference with buffered input and handle when the user implemented inference function
/// crashes.
async fn run_inference<P: Predictor>(
    buffer: WorkerBuffer<P>,
    job_sender: &JobSender<P>,
) -> Result<(), ()> {
    if buffer.is_empty() {
        return Ok(());
    }

    let res = job_sender.send(buffer).await;
    if res.is_ok() { Ok(()) } else { Err(()) }
}

/// Entry point to worker lifecycle.
pub(crate) fn run_worker<P: Predictor + 'static, E: Send + 'static>(
    worker: InferenceWorker,
    receiver: InputReceiver<P>,
    handoff: RunnerHandoff<P, E>,
) {
    tokio::task::spawn(async move { worker_loop(worker, receiver, handoff).await });
}
