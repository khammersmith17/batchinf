use crate::observability::{BatchTrigger, BatcherMetrics, InfBatchMetrics};
use crate::predictor::Predictor;
use crate::state::{WorkerState, WorkerStatus};
use std::sync::Arc;
use tokio::select;
use tokio::sync::mpsc::Receiver;
use tokio::sync::oneshot::Sender as OneshotSender;
use tokio::time::{Duration, Instant, sleep};

pub(crate) type OutputSender<P> =
    OneshotSender<Result<<P as Predictor>::Output, <P as Predictor>::Error>>;
pub(crate) type InputReceiver<P> = Receiver<(<P as Predictor>::Input, OutputSender<P>)>;
pub(crate) type InferenceResult<P> = Result<Vec<<P as Predictor>::Output>, <P as Predictor>::Error>;

/// When the user defined [Predictor::predict_batch] panics, the worker is marked as crashed
/// by setting its state to [WorkerStatus::Crashed].
///
/// The state is only set when a thread is panicking. The control plane detects the crashed state
/// and restarts the worker.
struct InferencePanicGuard(WorkerState);

impl Drop for InferencePanicGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let state = &self.0;
            state.set_state(WorkerStatus::Crashed)
        }
    }
}

pub(crate) struct InferenceWorker<P: Predictor + Send + Sync + 'static> {
    state: WorkerState,
    predictor: P,
    obs: Option<Arc<dyn BatcherMetrics>>,
}

impl<P: Predictor + Send + Sync + 'static> InferenceWorker<P> {
    pub(crate) fn new(
        state: WorkerState,
        predictor: P,
        obs: Option<Arc<dyn BatcherMetrics>>,
    ) -> InferenceWorker<P> {
        InferenceWorker {
            state,
            predictor,
            obs,
        }
    }

    /*
     * The following 3 methods emit metrics when an obserability handler with the proper callbacks
     * is defined, otherwise it is a no-op.
     *
     * This could get compiled away, given if there is no observability metrics handler defined,
     * then it can be statically proven all these methods are no-ops.
     * */
    fn emit_batch_start(&self, trigger_type: BatchTrigger, size: usize) {
        if let Some(ref obs) = self.obs {
            obs.on_batch_trigger(size, trigger_type)
        }
    }

    fn emit_inference_ok(&self, metrics: InfBatchMetrics) {
        if let Some(ref obs) = self.obs {
            let InfBatchMetrics { size, latency } = metrics;
            obs.on_batch_complete_ok(size, latency)
        }
    }

    fn emit_inference_err(&self, size: usize) {
        if let Some(ref obs) = self.obs {
            obs.on_batch_complete_err(size)
        }
    }
}

struct WorkerBuffer<P: Predictor + Send + Sync + 'static> {
    sender_buffer: Vec<OutputSender<P>>,
    input_buffer: Vec<P::Input>,
}

impl<P: Predictor + Send + Sync + 'static> WorkerBuffer<P> {
    fn push(&mut self, data: (P::Input, OutputSender<P>)) {
        let (inp, send) = data;
        self.sender_buffer.push(send);
        self.input_buffer.push(inp);
    }

    // Provides the length of the buffer. The
    fn len(&self) -> usize {
        debug_assert_eq!(self.input_buffer.len(), self.sender_buffer.len());
        self.input_buffer.len()
    }

    // Returns if the size is empty.
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
            // No inference data before timeout, restart accumulation phase.
            WorkerStatus::Waiting => unreachable!(
                "accumulate_next_batch only returns with an empty buffer on timeout, which cannot occur"
            ),
            WorkerStatus::Running => {
                run_inference(&worker, &mut buffer);
                // Reset the worker queue size and set to Waiting State.
                // Setting the queue len to 0, also sets the state bits to 0b00, which is
                // `WorkerStatus::Waiting`
                worker.state.reset_queue_len();
            }
            WorkerStatus::Exit => {
                run_inference(&worker, &mut buffer);
                // Inference worker exits.
                break;
            }
            WorkerStatus::Crashed => {
                unreachable!("Worker observed its own crashed state")
            }
        }
    }
}

// Accumulate the next batch of inference data.
// Starts timer for the batch upon receiving the first record for the batch.
// Rolls up state to perform inference, maintaining state when the buffer is empty.
//
// Sets state after accumulation phase, or on exit.
async fn accumulate_next_batch<P: Predictor + Send + Sync + 'static>(
    receiver: &mut InputReceiver<P>,
    worker: &InferenceWorker<P>,
    buffer: &mut WorkerBuffer<P>,
    next_inf: &mut Instant,
) {
    // Wait for first item in batch to start batch timer.
    if let Some(payload) = receiver.recv().await {
        buffer.push(payload);
        worker.state.increment_len();
        reset_next_inf(next_inf, worker.state.timeout());
    } else {
        worker.state.set_state(WorkerStatus::Exit);
        return;
    }

    loop {
        let timeout = time_until_timeout(next_inf);
        select! {
            user_input = receiver.recv() => {
                if let Some(payload) = user_input {
                    buffer.push(payload);

                    worker.state.increment_len();
                    if buffer.len() == (worker.state.capacity() as usize){
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

/// Perform inference on a batch by calling the user defined batch inference method.
fn run_inference<P: Predictor + Send + Sync + 'static>(
    worker: &InferenceWorker<P>,
    buffer: &mut WorkerBuffer<P>,
) {
    if buffer.is_empty() {
        return;
    };

    // Register the panic guard, to safely remove the worker on panic.
    let _guard = InferencePanicGuard(worker.state.clone());
    let size = buffer.len();

    let start = Instant::now();
    let inf_results =
        tokio::task::block_in_place(|| worker.predictor.predict_batch(&buffer.input()));

    let latency = Instant::now().duration_since(start);

    let senders = buffer.clear_and_take_senders();
    send_output(
        worker,
        inf_results,
        senders,
        InfBatchMetrics { size, latency },
    )
}

fn reset_next_inf(ts: &mut Instant, timeout: u64) {
    *ts = Instant::now() + Duration::from_millis(timeout)
}

/// On each wait for a message from the channel, compute the remaining timeout.
fn time_until_timeout(ts: &Instant) -> Duration {
    let now = Instant::now();

    // Calculates the difference or returns Duration::ZERO if 'now' has passed 'deadline'
    let duration_remaining = ts.saturating_duration_since(now);

    Duration::from_millis(duration_remaining.as_millis() as u64)
}

/// Send the output back out through the oneshot senders.
fn send_output<P: Predictor + Send + Sync + 'static>(
    worker: &InferenceWorker<P>,
    output: InferenceResult<P>,
    senders: Vec<OutputSender<P>>,
    metrics: InfBatchMetrics,
) {
    let batch = match output {
        Ok(b) => b,
        Err(e) => {
            send_errors(worker, e, senders, metrics);
            return;
        }
    };

    worker.emit_inference_ok(metrics);

    for (res, send) in batch.into_iter().zip(senders.into_iter()) {
        let _ = send.send(Ok(res));
    }
}

/// If the predict function result is Err, send all waiting the error.
fn send_errors<P: Predictor + Send + Sync + 'static>(
    worker: &InferenceWorker<P>,
    error: P::Error,
    senders: Vec<OutputSender<P>>,
    metrics: InfBatchMetrics,
) {
    worker.emit_inference_err(metrics.size);
    for sender in senders.into_iter() {
        let e = Err(error.clone());
        let _ = sender.send(e);
    }
}

pub(crate) fn run_worker<P: Predictor + Send + Sync + 'static>(
    worker: InferenceWorker<P>,
    receiver: InputReceiver<P>,
) {
    tokio::task::spawn(async { worker_loop(worker, receiver).await });
}
