use crate::config::InnerConfig;
use crate::predictor::Predictor;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::select;
use tokio::sync::mpsc::Receiver;
use tokio::sync::oneshot::Sender as OneshotSender;
use tokio::time::{Duration, Instant, sleep};

pub(crate) type OutputSender<P> =
    OneshotSender<Result<<P as Predictor>::Output, <P as Predictor>::Error>>;
pub(crate) type InputReceiver<P> = Receiver<(<P as Predictor>::Input, OutputSender<P>)>;
pub(crate) type InferenceResult<P> = Result<Vec<<P as Predictor>::Output>, <P as Predictor>::Error>;

// A worker can be in one the three following states.
// Waiting if when inference requests are being queued, Running trigger an inference
// run, and Exit defines when resources are being cleaned up.
#[derive(Debug, PartialEq)]
pub(crate) enum WorkerStatus {
    // Waiting to run inference.
    Waiting,
    // Exiting/cleaning up.
    Exit,
    // Running inference.
    Running,
}

impl From<WorkerStatus> for u8 {
    fn from(state: WorkerStatus) -> u8 {
        match state {
            WorkerStatus::Waiting => worker_states::WAITING,
            WorkerStatus::Exit => worker_states::EXIT,
            WorkerStatus::Running => worker_states::RUNNING_INFERENCE,
        }
    }
}

impl From<u8> for WorkerStatus {
    fn from(state: u8) -> WorkerStatus {
        match state {
            worker_states::WAITING => Self::Waiting,
            worker_states::EXIT => Self::Exit,
            worker_states::RUNNING_INFERENCE => Self::Running,
            _ => unreachable!("Invalid state value"),
        }
    }
}

// Mappings out u8 key and enum variant.
// These mappings allow for the state and queue size to be stored in a single atomic.
// The 2 most significant bits store the state.
// These u8 values are never stored.
pub(crate) mod worker_states {
    pub(crate) const WAITING: u8 = 0_u8;
    pub(crate) const EXIT: u8 = 1_u8;
    pub(crate) const RUNNING_INFERENCE: u8 = 2_u8;
    // Mask the state bits to get the queue len.
    pub(crate) const QUEUE_MASK: u64 = !(0b11_u64 << 62);
}

pub(crate) struct WorkerSnapshot {
    pub(crate) status: WorkerStatus,
    pub(crate) queue_len: u64,
}

#[derive(Debug)]
pub(crate) struct WorkerState_ {
    // State is stored in the 2 MSB here atomic load/store.
    // The other 62 bits store the queue length.
    state: AtomicU64,
    config: InnerConfig,
}

impl WorkerState_ {
    fn new(config: InnerConfig) -> WorkerState_ {
        WorkerState_ {
            state: AtomicU64::new(0_u64),
            config,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct WorkerState {
    inner: Arc<WorkerState_>,
}

impl WorkerState {
    pub(crate) fn new(config: InnerConfig) -> WorkerState {
        let inner = Arc::new(WorkerState_::new(config));

        WorkerState { inner }
    }
}

impl WorkerState {
    pub(crate) fn capacity(&self) -> u64 {
        self.inner.config.size
    }

    fn timeout(&self) -> u64 {
        self.inner.config.timeout
    }

    fn increment_len(&self) {
        self.inner.state.fetch_add(1_u64, Ordering::Relaxed);
    }

    pub(crate) fn set_state(&self, state: WorkerStatus) {
        let state_key: u8 = state.into();
        let state = u64::from(state_key) << 62;

        // The 2 MSB need to be cleared here, and then ORed with the state value.
        // So we need a CAS loop.
        let mut current = self.inner.state.load(Ordering::Relaxed);

        loop {
            let new = (current & worker_states::QUEUE_MASK) | state;
            match self.inner.state.compare_exchange_weak(
                current,
                new,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    pub(crate) fn get_state(&self) -> WorkerStatus {
        let state_key = self.inner.state.load(Ordering::Acquire);
        ((state_key >> 62) as u8).into()
    }

    pub(crate) fn snapshot(&self) -> WorkerSnapshot {
        let state = self.inner.state.load(Ordering::Acquire);
        let status: WorkerStatus = ((state >> 62) as u8).into();
        let queue_len = state & worker_states::QUEUE_MASK;

        WorkerSnapshot { status, queue_len }
    }

    fn reset_queue_len(&self) {
        self.inner.state.store(0_u64, Ordering::Release);
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

    fn len(&self) -> usize {
        debug_assert_eq!(self.input_buffer.len(), self.sender_buffer.len());
        self.input_buffer.len()
    }

    fn is_empty(&self) -> bool {
        debug_assert_eq!(self.input_buffer.len(), self.sender_buffer.len());
        self.input_buffer.is_empty()
    }

    fn input(&self) -> &[P::Input] {
        &self.input_buffer
    }

    fn clear_and_take_senders(&mut self) -> Vec<OutputSender<P>> {
        let cap = self.sender_buffer.capacity();
        self.input_buffer.clear();
        std::mem::replace(&mut self.sender_buffer, Vec::with_capacity(cap))
    }
}

pub(crate) struct InferenceWorker<P: Predictor + Send + Sync + 'static> {
    state: WorkerState,
    predictor: P,
    input_receiver: InputReceiver<P>,
    buffer: WorkerBuffer<P>,
    next_inf: Instant,
}

impl<P: Predictor + Send + Sync + 'static> InferenceWorker<P> {
    pub(crate) fn new(
        state: WorkerState,
        predictor: P,
        input_receiver: InputReceiver<P>,
    ) -> InferenceWorker<P> {
        let cap = state.capacity() as usize;
        let sender_buffer = Vec::with_capacity(cap);
        let input_buffer = Vec::with_capacity(cap);
        let next_inf = Instant::now();
        let buffer = WorkerBuffer {
            sender_buffer,
            input_buffer,
        };

        InferenceWorker {
            state,
            predictor,
            buffer,
            input_receiver,
            next_inf,
        }
    }

    // Start the worker on a long running thread.
    pub(crate) fn start(self) {
        tokio::spawn(async move { run_worker(self).await });
    }

    // The worker loop.
    //
    // Wait to the next inference request. Reading the next inference request off the channel and
    // the remaining time at the start of the poll race using tokio::select!.
    async fn worker_loop(&mut self) {
        loop {
            self.accumulate_next_batch().await;
            match self.state.get_state() {
                // No inference data before timeout, restart accumulation phase.
                WorkerStatus::Waiting => unreachable!("accumulate_next_batch only returns with an empty buffer on timeout, which cannot occur"),
                WorkerStatus::Running => {
                    self.run_inference();
                }
                WorkerStatus::Exit => {
                    self.run_inference();
                    // Pool worker exits.
                    break;
                }
            }
        }
    }

    // Accumulate the next batch of inference data.
    // Starts timer for the batch upon receiving the first record for the batch.
    // Rolls up state to perform inference, maintaining state when the buffer is empty.
    //
    // Sets state after accumulation phase, or on exit.
    async fn accumulate_next_batch(&mut self) {
        // Wait for first item in batch to start batch timer.
        if let Some(payload) = self.input_receiver.recv().await {
            self.buffer.push(payload);
            self.state.increment_len();
            self.reset_next_inf();
        } else {
            self.state.set_state(WorkerStatus::Exit);
            return;
        }

        loop {
            let timeout = self.time_until_timeout();
            select! {
                user_input = self.input_receiver.recv() => {
                    if let Some(payload) = user_input {
                        self.buffer.push(payload);

                        self.state.increment_len();
                        if self.buffer.len() == (self.state.capacity() as usize){
                            self.state.set_state(WorkerStatus::Running);
                            break;
                        }

                    } else {
                        self.state.set_state(WorkerStatus::Exit);
                        return;
                    };

                }
                _ = sleep(timeout) => {
                        // Do not overwrite state to running when state has been set to Exit.
                        // Inference happens in an Exit state.
                        if matches!(self.state.get_state(), WorkerStatus::Waiting) && !self.buffer.is_empty() {
                            self.state.set_state(WorkerStatus::Running)
                        }
                        return;
                }

            }
        }
    }

    fn reset_next_inf(&mut self) {
        self.next_inf = Instant::now() + Duration::from_millis(self.state.timeout())
    }

    fn time_until_timeout(&self) -> Duration {
        let now = Instant::now();

        // Calculates the difference or returns Duration::ZERO if 'now' has passed 'deadline'
        let duration_remaining = self.next_inf.saturating_duration_since(now);

        Duration::from_millis(duration_remaining.as_millis() as u64)
    }

    fn run_inference(&mut self) {
        if self.buffer.is_empty() {
            return;
        };
        let inf_results =
            tokio::task::block_in_place(|| self.predictor.predict_batch(&self.buffer.input()));

        let senders = self.buffer.clear_and_take_senders();
        self.state.reset_queue_len();
        self.send_output(inf_results, senders);
    }

    fn send_output(&mut self, output: InferenceResult<P>, senders: Vec<OutputSender<P>>) {
        let batch = match output {
            Ok(b) => b,
            Err(e) => {
                self.send_errors(e, senders);
                return;
            }
        };

        for (res, send) in batch.into_iter().zip(senders.into_iter()) {
            let _ = send.send(Ok(res));
        }
    }

    fn send_errors(&mut self, error: P::Error, senders: Vec<OutputSender<P>>) {
        for sender in senders.into_iter() {
            let e = Err(error.clone());
            let _ = sender.send(e);
        }
    }
}

async fn run_worker<P: Predictor + Send + Sync + 'static>(mut worker: InferenceWorker<P>) {
    worker.worker_loop().await;
}
