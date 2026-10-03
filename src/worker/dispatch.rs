use super::{InferenceResult, OutputSender};
use crate::{
    error::BatchinfError,
    observability::{BatcherMetrics, InfBatchMetrics, emitters},
    predictor::Predictor,
};
use std::sync::Arc;

/// Send the output back out through the oneshot senders.
pub(super) fn send_output<P: Predictor>(
    output: InferenceResult<P>,
    senders: Vec<OutputSender<P>>,
    metrics: InfBatchMetrics,
    obs: Option<Arc<dyn BatcherMetrics>>,
) {
    // When user defined predict errors, dispatch to error handler.
    let batch = match output {
        // Ensure that the predictors output buffer matches the number of senders we have.
        // This is a user error, so the user defined error is overriden.
        Ok(b) if b.len() != senders.len() => {
            send_errors::<P>(
                BatchinfError::InvalidPredictorOutput,
                senders,
                metrics,
                obs.clone(),
            );
            return;
        }
        Ok(b) => b,
        // User defined error, so P::Error is propogated forward.
        Err(e) => {
            send_errors::<P>(
                BatchinfError::InferenceError(e),
                senders,
                metrics,
                obs.clone(),
            );
            return;
        }
    };

    emitters::emit_inference_ok(obs, metrics);

    for (res, send) in batch.into_iter().zip(senders.into_iter()) {
        // Ignoring error here as receiver might have been closed due to timeout,
        // which is a valid state.
        let _ = send.send(Ok(res));
    }
}

/// Send `error` to all callers in the batch.
pub(super) fn send_errors<P: Predictor>(
    error: BatchinfError<P::Error>,
    senders: Vec<OutputSender<P>>,
    metrics: InfBatchMetrics,
    obs: Option<Arc<dyn BatcherMetrics>>,
) {
    emitters::emit_inference_err(obs, metrics.size);
    for sender in senders.into_iter() {
        let e = Err(error.clone());
        // Ignoring error here as receiver might have been closed due to timeout,
        // which is a valid state.
        let _ = sender.send(e);
    }
}
