use crate::config::InnerConfig;
use crate::predictor::Predictor;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use tokio::select;
use tokio::sync::mpsc::Receiver;
use tokio::sync::oneshot::Sender as OneshotSender;
use tokio::time::{Duration, Instant, sleep};

pub(crate) type OutputSender<P> =
    OneshotSender<Result<<P as Predictor>::Output, <P as Predictor>::Error>>;
pub(crate) type InputReceiver<P> = Receiver<(<P as Predictor>::Input, OutputSender<P>)>;
pub(crate) type InferenceResult<P> = Result<Vec<<P as Predictor>::Output>, <P as Predictor>::Error>;

#[derive(Debug, PartialEq)]
pub(crate) enum WorkerStatus {
    Waiting,
    Exit,
    TimedOut,
    BufferFull,
}

impl From<WorkerStatus> for u8 {
    fn from(state: WorkerStatus) -> u8 {
        match state {
            WorkerStatus::Waiting => worker_states::WAITING,
            WorkerStatus::Exit => worker_states::EXIT,
            WorkerStatus::TimedOut => worker_states::TIMED_OUT,
            WorkerStatus::BufferFull => worker_states::BUFFER_FULL,
        }
    }
}

impl From<u8> for WorkerStatus {
    fn from(state: u8) -> WorkerStatus {
        match state {
            worker_states::WAITING => Self::Waiting,
            worker_states::EXIT => Self::Exit,
            worker_states::TIMED_OUT => Self::TimedOut,
            worker_states::BUFFER_FULL => Self::BufferFull,
            _ => unreachable!("Invalid state value"),
        }
    }
}

// Mappings out u8 key and enum variant.
pub(crate) mod worker_states {
    pub(crate) const WAITING: u8 = 0_u8;
    pub(crate) const EXIT: u8 = 1_u8;
    pub(crate) const TIMED_OUT: u8 = 2_u8;
    pub(crate) const BUFFER_FULL: u8 = 3_u8;
}

#[derive(Debug)]
pub(crate) struct WorkerState_ {
    queue_len: AtomicUsize,
    // State is stored as atomic u8 for atomic load/store.
    // Allows concurrent access under a shared reference.
    // All callers observe the enum, through getter/setter.
    state: AtomicU8,
    config: InnerConfig,
}

impl WorkerState_ {
    fn new(config: InnerConfig) -> WorkerState_ {
        WorkerState_ {
            queue_len: AtomicUsize::new(0_usize),
            state: AtomicU8::new(worker_states::WAITING),
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
    fn capacity(&self) -> usize {
        self.inner.config.size as usize
    }

    fn timeout(&self) -> u64 {
        self.inner.config.timeout
    }

    fn increment_len(&self) {
        self.inner.queue_len.fetch_add(1_usize, Ordering::Relaxed);
    }

    pub(crate) fn set_state(&self, state: WorkerStatus) {
        let state_key: u8 = state.into();
        self.inner.state.store(state_key, Ordering::Release);
    }

    pub(crate) fn get_state(&self) -> WorkerStatus {
        let state_key = self.inner.state.load(Ordering::Acquire);
        state_key.into()
    }

    fn reset_queue_len(&self) {
        self.inner.queue_len.store(0_usize, Ordering::Release);
    }
}

pub(crate) struct InferenceWorker<P: Predictor + Send + Sync + 'static> {
    state: WorkerState,
    predictor: P,
    sender_buffer: Vec<OutputSender<P>>,
    input_receiver: InputReceiver<P>,
    input_buffer: Vec<P::Input>,
    next_inf: Instant,
}

impl<P: Predictor + Send + Sync + 'static> InferenceWorker<P> {
    pub(crate) fn new(
        state: WorkerState,
        predictor: P,
        input_receiver: InputReceiver<P>,
    ) -> InferenceWorker<P> {
        let cap = state.capacity();
        let sender_buffer = Vec::with_capacity(cap);
        let input_buffer = Vec::with_capacity(cap);
        let next_inf = Instant::now() + Duration::from_millis(state.timeout());

        InferenceWorker {
            state,
            predictor,
            sender_buffer,
            input_receiver,
            input_buffer,
            next_inf,
        }
    }

    pub(crate) fn start(self) {
        tokio::spawn(async move { run_worker(self) });
    }

    async fn worker_loop(&mut self) {
        self.reset_next_inf();

        loop {
            let timeout = self.time_until_timeout();
            select! {
                user_input = self.input_receiver.recv() => {
                    let Some((inp, send)) = user_input else {
                        todo!("Perform last inference and exit")
                    };

                    self.input_buffer.push(inp);
                    self.sender_buffer.push(send);

                    if self.input_buffer.len() == self.state.capacity() {
                        self.state.set_state(WorkerStatus::BufferFull)
                    }
                    self.state.increment_len();

                }
                _ = sleep(timeout) => {
                        self.state.set_state(WorkerStatus::TimedOut)
                }

            }

            match self.state.get_state() {
                WorkerStatus::Waiting => continue,
                WorkerStatus::TimedOut | WorkerStatus::BufferFull => {
                    let inf_results =
                        tokio::task::block_in_place(|| self.run_inference(&self.input_buffer));
                    self.state.reset_queue_len();
                    self.input_buffer.clear();
                    self.send_output(inf_results);
                    self.reset_next_inf();
                    self.state.set_state(WorkerStatus::Waiting);
                }
                WorkerStatus::Exit => {
                    todo!()
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

    fn take_senders(&mut self) -> Vec<OutputSender<P>> {
        std::mem::replace(
            &mut self.sender_buffer,
            Vec::with_capacity(self.state.capacity()),
        )
    }

    fn run_inference(&self, input_buffer: &[P::Input]) -> InferenceResult<P> {
        self.predictor.predict_batch(input_buffer)
    }

    fn send_output(&mut self, output: InferenceResult<P>) {
        let batch = match output {
            Ok(b) => b,
            Err(e) => {
                self.send_errors(e);
                return;
            }
        };

        let senders = self.take_senders();
        debug_assert_eq!(batch.len(), senders.len());

        for (b, s) in batch.into_iter().zip(senders.into_iter()) {
            tokio::spawn(async move {
                let _ = s.send(Ok(b));
            });
        }
    }

    fn send_errors(&mut self, error: P::Error) {
        let senders = self.take_senders();
        for sender in senders.into_iter() {
            let e = Err(error.clone());
            tokio::spawn(async move {
                let _ = sender.send(e);
            });
        }
    }
}

async fn run_worker<P: Predictor + Send + Sync + 'static>(mut worker: InferenceWorker<P>) {
    worker.worker_loop().await;
}
