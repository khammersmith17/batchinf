use crate::config::BatcherConfig;
use crate::predictor::Predictor;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use tokio::select;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::oneshot::{Receiver as OneshotReceiver, Sender as OneshotSender};
use tokio::time::{Duration, Instant, sleep};

pub(crate) type OutputSender<P> =
    OneshotSender<Result<<P as Predictor>::Output, <P as Predictor>::Error>>;
pub(crate) type OutputReciever<P> =
    OneshotReceiver<Result<<P as Predictor>::Output, <P as Predictor>::Error>>;
pub(crate) type InputReceiver<P> = Receiver<(<P as Predictor>::Input, OutputSender<P>)>;
pub(crate) type InputSender<P> = Sender<(<P as Predictor>::Input, OutputSender<P>)>;
pub(crate) type InferenceResult<P> = Result<<P as Predictor>::Output, <P as Predictor>::Error>;

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

pub(crate) mod worker_states {
    pub(crate) const WAITING: u8 = 0_u8;
    pub(crate) const EXIT: u8 = 1_u8;
    pub(crate) const TIMED_OUT: u8 = 2_u8;
    pub(crate) const BUFFER_FULL: u8 = 3_u8;
}

pub(crate) struct WorkerState {
    queue_len: AtomicUsize,
    last_fire: AtomicUsize,
    state: AtomicU8,
    config: BatcherConfig,
}

impl WorkerState {
    fn capacity(&self) -> usize {
        self.config.size as usize
    }

    fn timeout(&self) -> u64 {
        self.config.timeout
    }

    fn increment_len(&self) {
        self.queue_len.fetch_add(1_usize, Ordering::Relaxed);
    }

    pub(crate) fn set_state(&self, state: WorkerStatus) {
        let state_key: u8 = state.into();
        self.state.store(state_key, Ordering::Release);
    }

    pub(crate) fn get_state(&self) -> WorkerStatus {
        let state_key = self.state.load(Ordering::Acquire);
        state_key.into()
    }

    fn reset_queue_len(&self) {
        self.queue_len.store(0_usize, Ordering::Release);
    }
}

pub(crate) struct InferenceWorker<P: Predictor + Send + Sync + 'static> {
    state: WorkerState,
    predicter: P,
    sender_buffer: Vec<OutputSender<P>>,
    input_receiver: InputReceiver<P>,
    input_buffer: Vec<P::Input>,
    next_inf: Instant,
}

impl<P: Predictor + Send + Sync + 'static> InferenceWorker<P> {
    fn run(self) {
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
                    let inf_results = self.run_inference(&self.input_buffer);
                    let senders = self.take_senders();
                    forward_inference_results(inf_results, senders);
                    self.state.reset_queue_len();
                    self.input_buffer.clear();
                    self.reset_next_inf();
                    self.state.set_state(WorkerStatus::Waiting);
                    todo!()
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

    fn run_inference(&self, input_buffer: &[P::Input]) -> Vec<InferenceResult<P>> {
        self.predicter.predict_batch(input_buffer)
    }

    async fn send_output(output: Vec<Result<P::Output, P::Error>>, senders: Vec<OutputSender<P>>) {
        for (out, sender) in output.into_iter().zip(senders.into_iter()) {
            tokio::spawn(async move {
                let _ = sender.send(out);
            });
        }
    }
}

fn forward_inference_results<Output, Error>(
    inf_results: Vec<Result<Output, Error>>,
    senders: Vec<OneshotSender<Result<Output, Error>>>,
) {
    for (output, sender) in inf_results.into_iter().zip(senders.into_iter()) {
        // TODO: handle errors here.
        let _ = sender.send(output);
    }
}

async fn run_worker<P: Predictor + Send + Sync + 'static>(mut worker: InferenceWorker<P>) {
    worker.worker_loop().await;
}
