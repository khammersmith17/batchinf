pub mod batcher;
pub mod config;
mod pool;
pub mod predictor;
mod worker;

/*
* Implementation:
*   User implements the Predicter trait on a model type wrapper.
*
* The model runs in a dedicated tokio task.
* Inference inputs are buffered in the task, and inference is performed across all examples that
* are buffered. Inference occurs at either max buffer size, or timeout.
*
* Data is passed to this background thread through a channel and gets an associated oneshot::Sender.
* The calling thread enqueues the inference input and the oneshot::Sender, and waits on the
* oneshot::Receiver.
* */

pub fn get_batcher<P>(predictor: P, config: BatcherConfig) {
    todo!()
}

pub use config::BatcherConfig;
pub use predictor::Predictor;
