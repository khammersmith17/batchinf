use crate::worker::{WorkerState, WorkerStatus};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::oneshot::Sender as OneshotSender;

pub(crate) enum FunnelMessage<Input, Output, Error> {
    Input((Input, OneshotSender<Result<Output, Error>>)),
    Exit,
}

struct WorkerRef<Input, Output, Error> {
    state: WorkerState,
    worker_queue: Sender<(Input, OneshotSender<Result<Output, Error>>)>,
}

impl<Input, Output, Error> WorkerRef<Input, Output, Error> {
    fn get_state(&self) -> WorkerStatus {
        self.state.get_state()
    }

    async fn push(&self, msg: (Input, OneshotSender<Result<Output, Error>>)) {
        let _ = self.worker_queue.send(msg).await;
    }
}

pub(crate) struct WorkerPool<Input, Output, Error> {
    worker_pool: Vec<WorkerRef<Input, Output, Error>>,
    in_queue: Receiver<FunnelMessage<Input, Output, Error>>,
    sink: usize,
}

impl<Input, Output, Error> WorkerPool<Input, Output, Error> {
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
                        worker.state.set_state(WorkerStatus::Exit)
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
