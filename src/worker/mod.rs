use crate::{
    error::BatchinfError,
    observability::{BatchTrigger, BatcherMetrics, emitters},
    predictor::Predictor,
    setup::PredictorFactory,
    state::{WorkerState, WorkerStatus},
};
use std::sync::Arc;
use tokio::{
    select,
    sync::{
        mpsc::{Receiver, Sender, channel},
        oneshot::Sender as OneshotSender,
    },
    time::{Duration, Instant, sleep},
};

mod dispatch;
mod inf;
use inf::spawn_inference_runner;

pub(crate) type OutputSender<P> =
    OneshotSender<Result<<P as Predictor>::Output, BatchinfError<<P as Predictor>::Error>>>;
pub(crate) type InputReceiver<P> = Receiver<(<P as Predictor>::Input, OutputSender<P>)>;
pub(crate) type InferenceResult<P> = Result<Vec<<P as Predictor>::Output>, <P as Predictor>::Error>;
type JobSender<P> = Sender<WorkerBuffer<P>>;

// One batch queued while one runs on the inference thread. When both slots are taken, the
// accumulator waits on `send`, its request channel fills, and the router moves on to other workers.
const JOB_QUEUE_DEPTH: usize = 1;

/// Type that holds state required to orchestrate and perform inference.
pub(crate) struct InferenceWorker<P: Predictor> {
    state: WorkerState,
    factory: PredictorFactory<P>,
    obs: Option<Arc<dyn BatcherMetrics>>,
    worker_id: usize,
}

/// Core behavior of the inference worker.
impl<P: Predictor> InferenceWorker<P> {
    pub(crate) fn new(
        state: WorkerState,
        factory: PredictorFactory<P>,
        obs: Option<Arc<dyn BatcherMetrics>>,
        worker_id: usize,
    ) -> InferenceWorker<P> {
        InferenceWorker {
            state,
            factory,
            obs,
            worker_id,
        }
    }

    // The emit methods below are no-ops when obs is None. Option<Arc<dyn BatcherMetrics>> uses
    // the null pointer optimisation, so the None check is a single null pointer comparison.
    // The only overhead when obs is Some is the virtual dispatch through the fat pointer.

    // Reset the inference timeout clock.
    fn reset_next_inf(&self, ts: &mut Instant) {
        *ts = Instant::now() + self.state.timeout();
    }

    fn set_running_and_emit(&self, trigger_type: BatchTrigger, batch_size: usize) {
        self.state.set_worker_state(WorkerStatus::Running);
        emitters::emit_batch_start(self.obs.clone(), trigger_type, batch_size);
    }
}

/// Buffer for inference input and the senders.
///
/// Inference requests and sender to hand back are element wise pairs.
struct WorkerBuffer<P: Predictor> {
    sender_buffer: Vec<OutputSender<P>>,
    input_buffer: Vec<P::Input>,
}

impl<P: Predictor> WorkerBuffer<P> {
    fn new(cap: usize) -> WorkerBuffer<P> {
        WorkerBuffer {
            sender_buffer: Vec::with_capacity(cap),
            input_buffer: Vec::with_capacity(cap),
        }
    }

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

fn make_runner<P: Predictor + 'static>(
    factory: PredictorFactory<P>,
    worker_state: WorkerState,
    worker_id: usize,
    obs: Option<Arc<dyn BatcherMetrics>>,
) -> JobSender<P> {
    let (job_tx, job_rx) = channel::<WorkerBuffer<P>>(JOB_QUEUE_DEPTH);
    spawn_inference_runner(factory, job_rx, worker_state, worker_id, obs);
    job_tx
}

// The main worker loop.
//
// Accumulate inference examples from request writers until batch conditions are satisfied,
// then dispatch inference.
//
// After inference, the state is evaluated to determine if the worker should continue.
async fn worker_loop<P: Predictor + 'static>(
    worker: InferenceWorker<P>,
    mut input_receiver: InputReceiver<P>,
) {
    let cap = worker.state.batch_size() as usize;
    let mut next_inf = Instant::now();
    let mut buffer = WorkerBuffer::new(cap);
    let job_tx = make_runner(
        Arc::clone(&worker.factory),
        worker.state.clone(),
        worker.worker_id,
        worker.obs.clone(),
    );

    loop {
        accumulate_next_batch(&mut input_receiver, &worker, &mut buffer, &mut next_inf).await;
        let live_buffer = std::mem::replace(&mut buffer, WorkerBuffer::new(cap));
        match worker.state.get_worker_state() {
            WorkerStatus::Waiting => unreachable!(),
            WorkerStatus::Running => {
                // Waits here while the inference thread is busy and a batch is already queued.
                let alive = run_inference(live_buffer, &job_tx).await;
                if alive.is_err() {
                    // The inference thread is gone. Returning drops the request receiver, so the
                    // router sees this worker as closed and queued requests get InternalError.
                    worker.state.set_worker_state(WorkerStatus::Crashed);
                    return;
                }
                // Batch handed off. Signal that this worker can accept a new batch.
                worker.state.reset_queue_len();
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
        worker.state.set_worker_state(WorkerStatus::Exit);
        return;
    }

    // Batch size 1 is satisfied on first read.
    if worker.state.batch_size() == 1 {
        worker.set_running_and_emit(BatchTrigger::Capacity, buffer.len());
        return;
    }

    loop {
        let timeout = time_until_timeout(next_inf);
        select! {
            user_input = receiver.recv() => {
                if let Some((request, sender)) = user_input {
                    buffer.push(request, sender);

                    worker.state.increment_len();
                    if buffer.len() == (worker.state.batch_size() as usize){
                        // Set state and emit trigger.
                        worker.set_running_and_emit(BatchTrigger::Capacity, buffer.len());
                        break;
                    }

                } else {
                    worker.state.set_worker_state(WorkerStatus::Exit);
                    return;
                };

            }
            _ = sleep(timeout) => {
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

/// On each wait for a message from the channel, compute the remaining timeout.
fn time_until_timeout(ts: &Instant) -> Duration {
    let now = Instant::now();

    // Calculates the difference or returns Duration::ZERO if 'now' has passed 'deadline'
    ts.saturating_duration_since(now)
}

/// Entry point to worker lifecycle.
pub(crate) fn run_worker<P: Predictor + 'static>(
    worker: InferenceWorker<P>,
    receiver: InputReceiver<P>,
) {
    tokio::task::spawn(async { worker_loop(worker, receiver).await });
}
