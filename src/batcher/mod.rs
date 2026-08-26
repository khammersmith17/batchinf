use crate::pool::FunnelMessage;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot::channel as oneshot_channel;

pub(crate) struct BatcherInner<Input, Output, Error> {
    funnel: Sender<FunnelMessage<Input, Output, Error>>,
}

impl<Input, Output, Error> BatcherInner<Input, Output, Error> {
    fn new(funnel: Sender<FunnelMessage<Input, Output, Error>>) -> Self {
        Self { funnel }
    }

    async fn predict(&self, input: Input) -> Result<Output, Error> {
        let (tx, rx) = oneshot_channel::<Result<Output, Error>>();
        let msg = (input, tx);
        let _ = self.funnel.send(FunnelMessage::Input(msg)).await;
        let Ok(result) = rx.blocking_recv() else {
            todo!()
        };
        result
    }
}

impl<Input, Output, Error> Drop for BatcherInner<Input, Output, Error> {
    fn drop(&mut self) {
        let _ = self.funnel.blocking_send(FunnelMessage::Exit);
    }
}

pub struct Batchinf<Input, Output, Error> {
    inner: Arc<BatcherInner<Input, Output, Error>>,
}

impl<Input, Output, Error> Batchinf<Input, Output, Error> {
    pub(crate) fn new(funnel: Sender<FunnelMessage<Input, Output, Error>>) -> Self {
        let inner = Arc::new(BatcherInner::new(funnel));

        Self { inner }
    }

    pub async fn predict(&self, input: Input) -> Result<Output, Error> {
        self.inner.predict(input).await
    }
}
