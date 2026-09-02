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

/// When the user defined [Predictor::predict_batch] panics, that worker will be
/// removed from the pool by setting its state to [WorkerStatus::Exit].
///
/// This panic guard works by implementing this removal of the worker, the state gets set to Exit,
/// only when a thread is panicking.
struct InferencePanicGuard(WorkerState);

impl Drop for InferencePanicGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let state = &self.0;
            state.set_state(WorkerStatus::Exit)
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
    obs: Option<Arc<dyn BatcherMetrics>>,
}

impl<P: Predictor + Send + Sync + 'static> InferenceWorker<P> {
    pub(crate) fn new(
        state: WorkerState,
        predictor: P,
        input_receiver: InputReceiver<P>,
        obs: Option<Arc<dyn BatcherMetrics>>,
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
            obs,
        }
    }

    // Start the worker on a long running thread.
    pub(crate) fn start(self) {
        tokio::spawn(async move { run_worker(self).await });
    }

    // The worker loop.
    //
    // Drives accumulate_next_batch then runs inference when a batch is ready.
    async fn worker_loop(&mut self) {
        loop {
            self.accumulate_next_batch().await;
            match self.state.get_state() {
                // No inference data before timeout, restart accumulation phase.
                WorkerStatus::Waiting => unreachable!(
                    "accumulate_next_batch only returns with an empty buffer on timeout, which cannot occur"
                ),
                WorkerStatus::Running => {
                    self.run_inference();
                }
                WorkerStatus::Exit => {
                    self.run_inference();
                    // Inference worker exits.
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
                            self.emit_batch_start(BatchTrigger::Capacity);
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
                        //
                        // Only log a timeout when the worker has not exited.
                        if matches!(self.state.get_state(), WorkerStatus::Waiting) && !self.buffer.is_empty() {
                            self.state.set_state(WorkerStatus::Running);
                            self.emit_batch_start(BatchTrigger::Timeout);
                        }

                        return;
                }

            }
        }
    }

    /// Reset the clock for the next inference timeout.
    fn reset_next_inf(&mut self) {
        self.next_inf = Instant::now() + Duration::from_millis(self.state.timeout())
    }

    /// On each wait for a message from the channel, compute the remaining timeout.
    fn time_until_timeout(&self) -> Duration {
        let now = Instant::now();

        // Calculates the difference or returns Duration::ZERO if 'now' has passed 'deadline'
        let duration_remaining = self.next_inf.saturating_duration_since(now);

        Duration::from_millis(duration_remaining.as_millis() as u64)
    }

    /// Perform inference on a batch by calling the user defined batch inference method.
    fn run_inference(&mut self) {
        if self.buffer.is_empty() {
            return;
        };

        // Register the panic guard, to safely remove the worker on panic.
        let _guard = InferencePanicGuard(self.state.clone());
        let size = self.buffer.len();

        let start = Instant::now();
        let inf_results =
            tokio::task::block_in_place(|| self.predictor.predict_batch(&self.buffer.input()));

        let latency = Instant::now().duration_since(start);

        let senders = self.buffer.clear_and_take_senders();
        self.state.reset_queue_len();
        self.send_output(inf_results, senders, InfBatchMetrics { size, latency });
    }

    /// Send the output back out through the oneshot senders.
    fn send_output(
        &mut self,
        output: InferenceResult<P>,
        senders: Vec<OutputSender<P>>,
        metrics: InfBatchMetrics,
    ) {
        let batch = match output {
            Ok(b) => b,
            Err(e) => {
                self.send_errors(e, senders, metrics);
                return;
            }
        };

        self.emit_inference_ok(metrics);

        for (res, send) in batch.into_iter().zip(senders.into_iter()) {
            let _ = send.send(Ok(res));
        }
    }

    /// If the predict function result is Err, send all waiting the error.
    fn send_errors(
        &mut self,
        error: P::Error,
        senders: Vec<OutputSender<P>>,
        metrics: InfBatchMetrics,
    ) {
        self.emit_inference_err(metrics.size);
        for sender in senders.into_iter() {
            let e = Err(error.clone());
            let _ = sender.send(e);
        }
    }

    /*
     * The following 3 methods emit metrics when an obserability handler with the proper callbacks
     * is defined, otherwise it is a no-op.
     *
     * This could get compiled away, given if there is no observability metrics handler defined,
     * then it can be statically proven all these methods are no-ops.
     * */
    fn emit_batch_start(&self, trigger_type: BatchTrigger) {
        if let Some(ref obs) = self.obs {
            let size = self.buffer.len();
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

async fn run_worker<P: Predictor + Send + Sync + 'static>(mut worker: InferenceWorker<P>) {
    worker.worker_loop().await;
}
