use std::sync::Arc;
use tokio::sync::watch;

/// Held by every inference thread for its whole lifetime. This serves as a sentinel that
/// represents when a worker has exited.
pub(crate) type ShutdownSender = Arc<watch::Sender<()>>;

/// Creates the shared sender handed to each inference thread, and the receiver kept by the
/// [`Batchinf`](`crate::Batchinf`) handle.
pub(crate) fn shutdown_channel() -> (ShutdownSender, watch::Receiver<()>) {
    let (tx, rx) = watch::channel(());
    (Arc::new(tx), rx)
}

/// Waits for a batcher to finish shutting down.
///
/// Created by [`Batchinf::shutdown_waiter`](`crate::Batchinf::shutdown_waiter`). Shutdown starts
/// when the last [`Batchinf`](`crate::Batchinf`) handle is dropped. It is complete once every
/// worker's inference thread has finished its current batch, dropped its predictor and exited.
/// Use this to make sure predictors, and any device memory they hold, have been released, for
/// example before the process exits or before building a new batcher on the same device.
///
/// A waiter does not keep the batcher alive, and can be cloned to wait from several places.
///
/// ```no_run
/// # async fn run(batcher: batchinf::Batchinf<u32, u32, std::io::Error>) {
/// let waiter = batcher.shutdown_waiter();
/// drop(batcher); // and every clone of it
/// waiter.wait().await;
/// // Every predictor has been dropped.
/// # }
/// ```
///
/// There is one case where a shutdown may be delayed: when a worker is building its predictor
/// during a shutdown. This cannot be interrupted. Backoff time for a retry _can_ be interrupted.
#[derive(Debug, Clone)]
#[must_use = "Shutdown should be awaited when used."]
pub struct ShutdownWaiter {
    rx: watch::Receiver<()>,
}

impl ShutdownWaiter {
    pub(crate) fn new(rx: watch::Receiver<()>) -> ShutdownWaiter {
        ShutdownWaiter { rx }
    }

    /// Resolves once every inference thread has exited and dropped its predictor.
    ///
    /// Never resolves while any [`Batchinf`](`crate::Batchinf`) handle is still alive.
    pub async fn wait(mut self) {
        // No value is ever sent, so `changed` only returns, with an error, once every thread has
        // dropped its sender. Loop anyway so a stray value could not end the wait early.
        while self.rx.changed().await.is_ok() {}
    }

    /// Returns `true` if shutdown has completed: every inference thread has exited and dropped
    /// its predictor.
    pub fn is_shutdown(&self) -> bool {
        self.rx.has_changed().is_err()
    }
}
