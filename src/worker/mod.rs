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
        Notify,
        mpsc::{Receiver, Sender, channel},
        oneshot::Sender as OneshotSender,
    },
    time::{Duration, Instant, sleep},
};

mod dispatch;
mod inf;
use inf::{InferenceRunnerArgs, spawn_inference_runner};

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
    worker_id: u8,
    signal: Arc<Notify>,
}

/// Core behavior of the inference worker.
impl<P: Predictor> InferenceWorker<P> {
    pub(crate) fn new(
        state: WorkerState,
        factory: PredictorFactory<P>,
        obs: Option<Arc<dyn BatcherMetrics>>,
        worker_id: u8,
        signal: Arc<Notify>,
    ) -> InferenceWorker<P> {
        InferenceWorker {
            state,
            factory,
            obs,
            worker_id,
            signal,
        }
    }

    // Reset the inference timeout clock.
    fn reset_next_inf(&self, ts: &mut Instant) {
        *ts = Instant::now() + self.state.timeout();
    }

    fn set_running_and_emit(&self, trigger_type: BatchTrigger, batch_size: usize) {
        self.state.set_worker_state(WorkerStatus::Running);
        emitters::emit_batch_start(self.obs.clone(), trigger_type, batch_size);
    }

    // Report the number of requests waiting in this worker's queue, not yet pulled into a batch.
    fn emit_queue_depth(&self, input_receiver: &InputReceiver<P>) {
        emitters::emit_queue_depth(self.obs.clone(), self.worker_id, input_receiver.len());
    }

    fn notify_waiters(&self) {
        self.signal.notify_waiters()
    }
}

/// Buffer for inference input and the senders.
///
/// Inference requests and sender to hand back are element wise pairs.
pub(crate) struct WorkerBuffer<P: Predictor> {
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

    // Filter out the requests that have been canceled in place.
    // A request may be canceled when using the timeout api or a dropped predict future.
    #[inline]
    pub(crate) fn filter_canceled(&mut self) {
        // Single pass and swap remove.
        let mut i = 0_usize;
        while i < self.len() {
            if self.sender_buffer[i].is_closed() {
                self.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    // Swap remove both items in each pair.
    #[inline]
    fn swap_remove(&mut self, idx: usize) {
        self.input_buffer.swap_remove(idx);
        self.sender_buffer.swap_remove(idx);
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

fn make_runner<P: Predictor + 'static>(
    worker: &InferenceWorker<P>,
    start: PredictorStartConfig,
    ready_tx: Sender<(u8, bool)>,
) -> JobSender<P> {
    let (job_tx, job_rx) = channel::<WorkerBuffer<P>>(JOB_QUEUE_DEPTH);
    spawn_inference_runner(InferenceRunnerArgs {
        factory: Arc::clone(&worker.factory),
        recv: job_rx,
        worker_state: worker.state.clone(),
        worker_id: worker.worker_id,
        retries: start.retries,
        backoff: start.backoff,
        max_wait: start.max_wait,
        obs: worker.obs.clone(),
        ready_tx,
        signal: Arc::clone(&worker.signal),
    });
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
    start: PredictorStartConfig,
    ready_tx: Sender<(u8, bool)>,
) {
    let cap = worker.state.batch_size() as usize;
    let mut next_inf = Instant::now();
    let mut buffer = WorkerBuffer::new(cap);
    let job_tx = make_runner(&worker, start, ready_tx);

    loop {
        accumulate_next_batch(&mut input_receiver, &worker, &mut buffer, &mut next_inf).await;
        let live_buffer = std::mem::replace(&mut buffer, WorkerBuffer::new(cap));
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
        worker.notify_waiters();
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
                    worker.notify_waiters();
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
    start: PredictorStartConfig,
    ready_tx: Sender<(u8, bool)>,
) {
    tokio::task::spawn(async move { worker_loop(worker, receiver, start, ready_tx).await });
}

#[cfg(test)]
mod test_worker_buffer {
    use super::WorkerBuffer;
    use crate::predictor::Predictor;

    #[derive(Debug)]
    struct DummyError;

    impl std::fmt::Display for DummyError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Error")
        }
    }

    impl std::error::Error for DummyError {}

    struct Dummy;

    impl Predictor for Dummy {
        type Input = usize;
        type Output = usize;
        type Error = DummyError;

        fn predict_batch(
            &mut self,
            input: &[Self::Input],
        ) -> Result<Vec<Self::Output>, Self::Error> {
            Ok((0..input.len()).collect())
        }
    }

    #[test]
    fn test_filter_event_count() {
        let size = 24_usize;
        // only even idxs
        let mut buffer: WorkerBuffer<Dummy> = WorkerBuffer::new(size);

        let mut recv_vec = Vec::new();

        for i in 0..size {
            let (tx, rx) = tokio::sync::oneshot::channel::<
                Result<usize, crate::error::BatchinfError<DummyError>>,
            >();
            buffer.push(i, tx);
            if i % 2 == 0 {
                recv_vec.push(rx)
            }
        }
        buffer.filter_canceled();

        let WorkerBuffer {
            mut input_buffer, ..
        } = buffer;
        let base: Vec<usize> = (0..size).filter(|v| v % 2 == 0).collect();
        input_buffer.sort();
        assert_eq!(base, input_buffer);
    }

    #[test]
    fn test_filter_odd_count() {
        let size = 25_usize;
        // only even idxs
        let mut buffer: WorkerBuffer<Dummy> = WorkerBuffer::new(size);

        let mut recv_vec = Vec::new();

        for i in 0..size {
            let (tx, rx) = tokio::sync::oneshot::channel::<
                Result<usize, crate::error::BatchinfError<DummyError>>,
            >();
            buffer.push(i, tx);
            if i % 2 == 1 {
                recv_vec.push(rx)
            }
        }
        buffer.filter_canceled();

        let WorkerBuffer {
            mut input_buffer, ..
        } = buffer;
        let base: Vec<usize> = (0..size).filter(|v| v % 2 == 1).collect();
        input_buffer.sort();
        assert_eq!(base, input_buffer);
    }

    #[test]
    fn test_filter_uneven() {
        let size = 50_usize;
        // only even idxs
        let mut buffer: WorkerBuffer<Dummy> = WorkerBuffer::new(size);

        let mut recv_vec = Vec::new();

        for i in 0..size {
            let (tx, rx) = tokio::sync::oneshot::channel::<
                Result<usize, crate::error::BatchinfError<DummyError>>,
            >();
            buffer.push(i, tx);
            if i % 3 == 0 {
                recv_vec.push(rx)
            }
        }
        buffer.filter_canceled();

        let WorkerBuffer {
            mut input_buffer, ..
        } = buffer;
        let base: Vec<usize> = (0..size).filter(|v| v % 3 == 0).collect();
        input_buffer.sort();
        assert_eq!(base, input_buffer);
    }

    #[test]
    fn test_all_canceled() {
        let size = 50_usize;
        // only even idxs
        let mut buffer: WorkerBuffer<Dummy> = WorkerBuffer::new(size);

        for i in 0..size {
            let (tx, _) = tokio::sync::oneshot::channel::<
                Result<usize, crate::error::BatchinfError<DummyError>>,
            >();
            buffer.push(i, tx);
        }
        buffer.filter_canceled();

        let WorkerBuffer {
            mut input_buffer, ..
        } = buffer;
        let base: Vec<usize> = vec![];
        input_buffer.sort();

        assert_eq!(base, input_buffer);
    }
}
