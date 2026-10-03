use super::{WorkerBuffer, dispatch};
use crate::{
    observability::{BatcherMetrics, InfBatchMetrics, emitters},
    predictor::Predictor,
    setup::PredictorFactory,
};
use std::{sync::Arc, thread::Builder, time::Instant};
use tokio::sync::mpsc::Receiver;

type JobReceiver<P> = Receiver<WorkerBuffer<P>>;

// TODO: determine the number of retries here
fn get_predictor_from_factory<P: Predictor>(
    factory: PredictorFactory<P>,
    worker_id: usize,
) -> Result<P, ()> {
    for _ in 0..5 {
        let wid = worker_id;
        let f = Arc::clone(&factory);
        let panic_signal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || f(wid)));
        match panic_signal {
            Ok(p) => return Ok(p),
            Err(_) => {}
        }
    }

    Err(())
}

pub(super) fn spawn_inference_runner<P: Predictor + 'static>(
    factory: PredictorFactory<P>,
    recv: JobReceiver<P>,
    worker_id: usize,
    obs: Option<Arc<dyn BatcherMetrics>>,
) {
    let name = format!("batchinf-worker-{worker_id}");
    Builder::new()
        .name(name)
        .spawn(move || inference_runner(factory, recv, worker_id, obs))
        .expect("Unable to spawn thread");
}

fn inference_runner<P: Predictor>(
    factory: PredictorFactory<P>,
    mut recv: JobReceiver<P>,
    worker_id: usize,
    obs: Option<Arc<dyn BatcherMetrics>>,
) {
    // TODO: Add some signaling around handling a failure creating a predictor.
    let Ok(mut predictor) = get_predictor_from_factory(Arc::clone(&factory), worker_id) else {
        return;
    };

    // `blocking_recv` is fine here: this is a dedicated OS thread, not a runtime thread.
    while let Some(mut buffer) = recv.blocking_recv() {
        // Capture panic signal to handle crash and restart.
        let panic_signal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            predict(&mut predictor, buffer.input())
        }));

        // Inspect the result and handle panic in the case the user defined prediction method panicked.
        let (metrics, inf_result) = match panic_signal {
            Ok((m, ir)) => (m, ir),
            Err(_) => {
                emitters::emit_worker_panic(obs.clone());
                let Ok(new_predictor) = get_predictor_from_factory(Arc::clone(&factory), worker_id)
                else {
                    return;
                };
                predictor = new_predictor;
                continue;
            }
        };

        let senders = buffer.clear_and_take_senders();
        dispatch::send_output::<P>(inf_result, senders, metrics, obs.clone());
    }
}

/// Perform inference on a batch by calling the user defined batch inference method.
///
/// Captures inference latency and batch size for metric emission.
fn predict<P: Predictor>(
    predictor: &mut P,
    input: &[P::Input],
) -> (InfBatchMetrics, Result<Vec<P::Output>, P::Error>) {
    let start = Instant::now();

    // Runs on this worker's dedicated inference thread, so blocking here does not affect the
    // async runtime.
    let inf_results = predictor.predict_batch(input);

    let latency = Instant::now().duration_since(start);
    let metrics = InfBatchMetrics {
        size: input.len(),
        latency,
    };

    (metrics, inf_results)
}
