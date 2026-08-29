use crate::worker::{WorkerSnapshot, WorkerState, WorkerStatus};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::oneshot::Sender as OneshotSender;

pub(crate) type FunnelMessage<Input, Output, Error> = (Input, OneshotSender<Result<Output, Error>>);

pub(crate) struct WorkerRef<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: Send + Sync + 'static,
{
    state: WorkerState,
    worker_queue: Sender<FunnelMessage<Input, Output, Error>>,
}

impl<Input, Output, Error> WorkerRef<Input, Output, Error>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
    Error: Send + Sync + 'static,
{
    pub(crate) fn new(
        state: WorkerState,
        worker_queue: Sender<FunnelMessage<Input, Output, Error>>,
    ) -> Self {
        Self {
            state,
            worker_queue,
        }
    }

    fn snapshot(&self) -> WorkerSnapshot {
        self.state.snapshot()
    }

    pub(crate) fn capacity(&self) -> u64 {
        self.state.capacity()
    }


    async fn push(&self, msg: FunnelMessage<Input, Output, Error>) {
        // If the worker channel is closed (worker exited), the send error is dropped here.
        // The caller's oneshot receiver will return Err, which maps to BatchinfError::InternalError.
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
        tokio::spawn(async { run_worker_pool(self).await });
    }

    pub(crate) async fn run_pooler(&mut self) {
        // Fetch from the queue and resolve which sink to send to.
        while let Some(input_msg) = self.in_queue.recv().await {
            let dest = self.determine_destination();
            self.worker_pool[dest].push(input_msg).await;
            self.increment_sink();
        }
    }

    // A load aware round robin algorithm to determine the next worker that is dispatched to.
    pub(crate) fn determine_destination(&mut self) -> usize {
        if self.worker_pool.len() == 1 {
            return 0_usize;
        }

        let worker_count = self.size();
        for _ in 0..worker_count {
            let snapshot = self.worker_pool[self.sink].snapshot();
            if snapshot.status == WorkerStatus::Waiting
                && snapshot.queue_len < self.worker_pool[self.sink].capacity()
            {
                return self.sink;
            }
            self.increment_sink();
        }

        for _ in 0..worker_count {
            let snapshot = self.worker_pool[self.sink].snapshot();

            if snapshot.status != WorkerStatus::Exit {
                return self.sink;
            }
            self.increment_sink();
        }

        self.sink
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
