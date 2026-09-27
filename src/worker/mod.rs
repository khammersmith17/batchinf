use crate::{
    error::BatchinfError,
    observability::{BatchTrigger, BatcherMetrics, InfBatchMetrics},
    predictor::Predictor,
    state::{WorkerState, WorkerStatus},
};
use std::sync::Arc;
use tokio::{
    select,
    sync::{mpsc::Receiver, oneshot::Sender as OneshotSender},
    time::{Duration, Instant, sleep},
};

pub(crate) type OutputSender<P> =
    OneshotSender<Result<<P as Predictor>::Output, BatchinfError<<P as Predictor>::Error>>>;
pub(crate) type InputReceiver<P> = Receiver<(<P as Predictor>::Input, OutputSender<P>)>;
pub(crate) type InferenceResult<P> = Result<Vec<<P as Predictor>::Output>, <P as Predictor>::Error>;

/// On crash, drain the receiver to capture inference requests that were queued while a worker was
/// running inferece.
///
/// This batch of inference requests was not in the group that lead to a panic, thus they are not
/// thrown away on the panic.
fn drain_receiver<P: Predictor + Send + Sync + 'static>(
    recv: &mut InputReceiver<P>,
) -> Vec<(P::Input, OutputSender<P>)> {
    let mut orphaned = Vec::new();
    while let Ok(inp) = recv.try_recv() {
        orphaned.push(inp);
    }
    orphaned
}

/// Handle the worker state on crash.
/// Drain messages currently buffered during inference + signal crash to the control plane.
fn handle_worker_panic<P: Predictor + Send + Sync + 'static>(
    state: &WorkerState<P::Input, P::Output, P::Error>,
    recv: &mut InputReceiver<P>,
) {
    // Block writers from queueing a crashed worker during restart.
    // Blocking writers ensures that new messages are not missed when the queue is drained.
    state.set_state(WorkerStatus::Crashed);

    // Drain requests that were queued since worker was dispatched for inference.
    let orphaned = drain_receiver::<P>(recv);
    state.queue_orphaned_messages(orphaned);

    // Wake the control loop.
    state.signal_crash();
}

/// Type that holds state required to orchestrate and perform inference.
pub(crate) struct InferenceWorker<P: Predictor + Send + Sync + 'static> {
    state: WorkerState<P::Input, P::Output, P::Error>,
    predictor: P,
    obs: Option<Arc<dyn BatcherMetrics>>,
}

/// Core behavior of the inference worker.
impl<P: Predictor + Send + Sync + 'static> InferenceWorker<P> {
    pub(crate) fn new(
        state: WorkerState<P::Input, P::Output, P::Error>,
        predictor: P,
        obs: Option<Arc<dyn BatcherMetrics>>,
    ) -> InferenceWorker<P> {
        InferenceWorker {
            state,
            predictor,
            obs,
        }
    }

    // The emit methods below are no-ops when obs is None. Option<Arc<dyn BatcherMetrics>> uses
    // the null pointer optimisation, so the None check is a single null pointer comparison.
    // The only overhead when obs is Some is the virtual dispatch through the fat pointer.
    fn emit_batch_start(&self, trigger_type: BatchTrigger, size: usize) {
        if let Some(ref obs) = self.obs {
            obs.on_batch_trigger(size, trigger_type)
        }
    }

    // Emit a succesful inference.
    fn emit_inference_ok(&self, metrics: InfBatchMetrics) {
        if let Some(ref obs) = self.obs {
            let InfBatchMetrics { size, latency } = metrics;
            obs.on_batch_complete_ok(size, latency)
        }
    }

    // Emit and unsuccesful inference.
    fn emit_inference_err(&self, size: usize) {
        if let Some(ref obs) = self.obs {
            obs.on_batch_complete_err(size)
        }
    }

    // Reset the inference timeout clock.
    fn reset_next_inf(&self, ts: &mut Instant) {
        *ts = Instant::now() + self.state.timeout();
    }
}

/// Buffer for inference input and the senders.
///
/// Inference requests and sender to hand back are element wise pairs.
struct WorkerBuffer<P: Predictor + Send + Sync + 'static> {
    sender_buffer: Vec<OutputSender<P>>,
    input_buffer: Vec<P::Input>,
}

impl<P: Predictor + Send + Sync + 'static> WorkerBuffer<P> {
    fn push(&mut self, request: P::Input, sender: OutputSender<P>) {
        self.sender_buffer.push(sender);
        self.input_buffer.push(request);
    }

    // Returns the number of items in the buffer.
    fn len(&self) -> usize {
        debug_assert_eq!(self.input_buffer.len(), self.sender_buffer.len());
        self.input_buffer.len()
    }

    // Returns true if the buffer is empty.
    fn is_empty(&self) -> bool {
        debug_assert_eq!(self.input_buffer.len(), self.sender_buffer.len());
        self.input_buffer.is_empty()
    }

    // Provides a reference to the input buffer.
    fn input(&self) -> &[P::Input] {
        &self.input_buffer
    }

    // Takes the held senders, and replaces the container with a fresh sender buffer. Sending
    // takes ownership, so this provides an own `Vec<Sender<T>>`, replaces it and clears the input
    // buffer.
    fn clear_and_take_senders(&mut self) -> Vec<OutputSender<P>> {
        let cap = self.sender_buffer.capacity();
        self.input_buffer.clear();
        std::mem::replace(&mut self.sender_buffer, Vec::with_capacity(cap))
    }
}

// The main worker loop.
//
// Accumulate inference examples from request writers until batch conditions are satisfied,
// then dispatch inference.
//
// After inference, the state is evaluated to determine if the worker should continue.
async fn worker_loop<P: Predictor + Send + Sync + 'static>(
    worker: InferenceWorker<P>,
    mut input_receiver: InputReceiver<P>,
) {
    let cap = worker.state.capacity() as usize;
    let sender_buffer = Vec::with_capacity(cap);
    let input_buffer = Vec::with_capacity(cap);
    let mut next_inf = Instant::now();
    let mut buffer = WorkerBuffer {
        sender_buffer,
        input_buffer,
    };
    loop {
        accumulate_next_batch(&mut input_receiver, &worker, &mut buffer, &mut next_inf).await;
        match worker.state.get_state() {
            WorkerStatus::Waiting => unreachable!(),
            WorkerStatus::Running => {
                run_inference(&worker, &mut buffer, &mut input_receiver);
                worker.state.reset_queue_len();
            }
            WorkerStatus::Exit => {
                run_inference(&worker, &mut buffer, &mut input_receiver);
                break;
            }
            // Crashed state is handled in the inference handler.
            // When a worker panics, the panic is resumed after saving state and signaling the
            // control plane.
            WorkerStatus::Crashed => {
                unreachable!("Worker observed its own crashed state")
            }
        }
    }
}

/// Accumulate the next batch of inference data.
/// Starts timer for the batch upon receiving the first record for the batch.
/// Sets state to Running when the batch is ready, or Exit when the channel closes.
async fn accumulate_next_batch<P: Predictor + Send + Sync + 'static>(
    receiver: &mut InputReceiver<P>,
    worker: &InferenceWorker<P>,
    buffer: &mut WorkerBuffer<P>,
    next_inf: &mut Instant,
) {
    // Wait for first item in batch to start batch timer.
    if let Some((request, sender)) = receiver.recv().await {
        buffer.push(request, sender);
        worker.state.increment_len();
        worker.reset_next_inf(next_inf);
    } else {
        // None indicates the channel is closed. In this case set the state to exit.
        worker.state.set_state(WorkerStatus::Exit);
        return;
    }

    loop {
        let timeout = time_until_timeout(next_inf);
        select! {
            user_input = receiver.recv() => {
                if let Some((request, sender)) = user_input {
                    buffer.push(request, sender);

                    worker.state.increment_len();
                    if buffer.len() == (worker.state.capacity() as usize){
                        // Set state and emit trigger.
                        worker.state.set_state(WorkerStatus::Running);
                        worker.emit_batch_start(BatchTrigger::Capacity, buffer.len());
                        break;
                    }

                } else {
                    worker.state.set_state(WorkerStatus::Exit);
                    return;
                };

            }
            _ = sleep(timeout) => {
                    // Do not overwrite state to running when state has been set to Exit.
                    // Inference happens in an Exit state.
                    //
                    // Only log a timeout when the worker has not exited.
                    if matches!(worker.state.get_state(), WorkerStatus::Waiting) && !buffer.is_empty() {
                        worker.state.set_state(WorkerStatus::Running);
                        worker.emit_batch_start(BatchTrigger::Timeout, buffer.len());
                    }

                    return;
            }

        }
    }
}

/// Inference runner.
/// Runs inference with buffered input and handle when the user implemented inference function
/// crashes.
fn run_inference<P: Predictor + Send + Sync + 'static>(
    worker: &InferenceWorker<P>,
    buffer: &mut WorkerBuffer<P>,
    recv: &mut InputReceiver<P>,
) {
    if buffer.is_empty() {
        return;
    }

    // Capture panic signal to handle crash and restart.
    let panic_signal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        predict(&worker.predictor, buffer.input())
    }));

    // Inspect the result and handle panic in the case the user defined prediction method panicked.
    let (metrics, inf_results) = match panic_signal {
        Ok((m, ir)) => (m, ir),
        Err(panic) => {
            handle_worker_panic::<P>(&worker.state, recv);
            // Resume panic after crash is handled to kill the task.
            std::panic::resume_unwind(panic);
        }
    };

    let senders = buffer.clear_and_take_senders();
    send_output(&worker, inf_results, senders, metrics);
}

/// Perform inference on a batch by calling the user defined batch inference method.
///
/// Captures inference latency and batch size for metric emission.
fn predict<P: Predictor + Send + Sync + 'static>(
    predictor: &P,
    input: &[P::Input],
) -> (InfBatchMetrics, Result<Vec<P::Output>, P::Error>) {
    let start = Instant::now();

    // `spawn_blocking` would be nice here, but not worth the additional allocation to pass an
    // owned input buffer.
    // This requires the multi thread runtime, reasonable trade off.
    let inf_results = tokio::task::block_in_place(|| predictor.predict_batch(input));

    let latency = Instant::now().duration_since(start);
    let metrics = InfBatchMetrics {
        size: input.len(),
        latency,
    };

    (metrics, inf_results)
}

/// On each wait for a message from the channel, compute the remaining timeout.
fn time_until_timeout(ts: &Instant) -> Duration {
    let now = Instant::now();

    // Calculates the difference or returns Duration::ZERO if 'now' has passed 'deadline'
    ts.saturating_duration_since(now)
}

/// Send the output back out through the oneshot senders.
fn send_output<P: Predictor + Send + Sync + 'static>(
    worker: &InferenceWorker<P>,
    output: InferenceResult<P>,
    senders: Vec<OutputSender<P>>,
    metrics: InfBatchMetrics,
) {
    // When user defined predict errors, dispatch to error handler.
    let batch = match output {
        // Ensure that the predictors output buffer matches the number of senders we have.
        // This is a user error, so the user defined error is overriden.
        Ok(b) if b.len() != senders.len() => {
            send_errors(
                worker,
                BatchinfError::InvalidPredictorOutput,
                senders,
                metrics,
            );
            return;
        }
        Ok(b) => b,
        // User defined error, so P::Error is propogated forward.
        Err(e) => {
            send_errors(worker, BatchinfError::InferenceError(e), senders, metrics);
            return;
        }
    };

    worker.emit_inference_ok(metrics);

    for (res, send) in batch.into_iter().zip(senders.into_iter()) {
        // Ignoring error here as receiver might have been closed due to timeout,
        // which is a valid state.
        let _ = send.send(Ok(res));
    }
}

/// Send `error` to all callers in the batch.
fn send_errors<P: Predictor + Send + Sync + 'static>(
    worker: &InferenceWorker<P>,
    error: BatchinfError<P::Error>,
    senders: Vec<OutputSender<P>>,
    metrics: InfBatchMetrics,
) {
    worker.emit_inference_err(metrics.size);
    for sender in senders.into_iter() {
        let e = Err(error.clone());
        // Ignoring error here as receiver might have been closed due to timeout,
        // which is a valid state.
        let _ = sender.send(e);
    }
}

/// Entry point to worker lifecycle.
pub(crate) fn run_worker<P: Predictor + Send + Sync + 'static>(
    worker: InferenceWorker<P>,
    receiver: InputReceiver<P>,
) {
    tokio::task::spawn(async { worker_loop(worker, receiver).await });
}
