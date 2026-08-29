use crate::error::BatchinfError;
use crate::pool::FunnelMessage;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot::channel as oneshot_channel;

pub(crate) struct BatcherInner<Input, Output, Error> {
    funnel: Sender<FunnelMessage<Input, Output, Error>>,
}

impl<Input, Output, Error: std::error::Error + Clone + Send + Sync + 'static>
    BatcherInner<Input, Output, Error>
{
    fn new(funnel: Sender<FunnelMessage<Input, Output, Error>>) -> Self {
        Self { funnel }
    }

    async fn predict(&self, input: Input) -> Result<Output, BatchinfError<Error>> {
        let (tx, rx) = oneshot_channel::<Result<Output, Error>>();
        let _ = self.funnel.send((input, tx)).await;
        match rx.await {
            Ok(result) => match result {
                Ok(r) => Ok(r),
                Err(e) => Err(BatchinfError::InferenceError(e)),
            },
            Err(_) => Err(BatchinfError::InternalError),
        }
    }
}

pub struct Batchinf<Input, Output, Error> {
    inner: Arc<BatcherInner<Input, Output, Error>>,
}

impl<Input, Output, Error: std::error::Error + Clone + Send + Sync + 'static>
    Batchinf<Input, Output, Error>
{
    pub(crate) fn new(funnel: Sender<FunnelMessage<Input, Output, Error>>) -> Self {
        let inner = Arc::new(BatcherInner::new(funnel));

        Self { inner }
    }

    pub async fn predict(&self, input: Input) -> Result<Output, BatchinfError<Error>> {
        self.inner.predict(input).await
    }
}
