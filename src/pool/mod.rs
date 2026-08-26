use crate::worker::{WorkerState, WorkerStatus};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::oneshot::Sender as OneshotSender;

pub(crate) enum FunnelMessage<Input, Output, Error> {
    Input((Input, OneshotSender<Result<Output, Error>>)),
    Exit,
}

pub(crate) struct WorkerRef<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: Send + Sync + 'static,
{
    state: WorkerState,
    worker_queue: Sender<(Input, OneshotSender<Result<Output, Error>>)>,
}

impl<Input, Output, Error> WorkerRef<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: Send + Sync + 'static,
{
    pub(crate) fn new(
        state: WorkerState,
        worker_queue: Sender<(Input, OneshotSender<Result<Output, Error>>)>,
    ) -> Self {
        Self {
            state,
            worker_queue,
        }
    }

    fn get_state(&self) -> WorkerStatus {
        self.state.get_state()
    }

    fn set_state_exit(&self) {
        self.state.set_state(WorkerStatus::Exit)
    }

    async fn push(&self, msg: (Input, OneshotSender<Result<Output, Error>>)) {
        let _ = self.worker_queue.send(msg).await;
    }
}

pub(crate) struct WorkerPool<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: Send + Sync + 'static,
{
    worker_pool: Vec<WorkerRef<Input, Output, Error>>,
    in_queue: Receiver<FunnelMessage<Input, Output, Error>>,
    sink: usize,
}

impl<Input, Output, Error> WorkerPool<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: Send + Sync + 'static,
{
    pub(crate) fn new(
        worker_pool: Vec<WorkerRef<Input, Output, Error>>,
        in_queue: Receiver<FunnelMessage<Input, Output, Error>>,
    ) -> Self {
        Self {
            worker_pool,
            in_queue,
            sink: 0_usize,
        }
    }

    pub(crate) fn start(self) {
        tokio::spawn(async { run_worker_pool(self) });
    }

    pub(crate) async fn run_pooler(&mut self) {
        // Fetch from the queue and resolve which sink to send to.
        while let Some(msg) = self.in_queue.recv().await {
            match msg {
                FunnelMessage::Input(input_msg) => {
                    let dest = self.determine_destination();

                    self.worker_pool[dest].push(input_msg).await;
                    self.increment_sink();
                }
                FunnelMessage::Exit => {
                    // On exit, set all worker states to exit.
                    for worker in &self.worker_pool {
                        worker.set_state_exit();
                    }
                }
            }
        }
    }

    pub(crate) fn determine_destination(&self) -> usize {
        // Based on current worker state, determine the next destination.
        // TODO: Improve this algorithm
        if self.worker_pool.len() == 1 {
            return 0_usize;
        }

        let mut tried = 0_usize;
        while tried < self.size() {
            if self.worker_pool[self.sink].get_state() == WorkerStatus::Waiting {
                return self.sink;
            }
            tried += 1
        }

        return self.sink;
    }

    #[inline]
    fn increment_sink(&mut self) {
        self.sink = (self.sink + 1) % self.worker_pool.len();
    }

    #[inline]
    fn size(&self) -> usize {
        self.worker_pool.len()
    }
}

async fn run_worker_pool<Input, Output, Error>(mut pool: WorkerPool<Input, Output, Error>)
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: Send + Sync + 'static,
{
    pool.run_pooler().await;
}
