# Batchinf — Design

## Goal

A Rust crate providing a reusable abstraction for batching ML inference requests: individual requests are queued until either a size threshold or a timeout is hit, then dispatched together as a batch (dynamic batching). The crate is backend-agnostic via a trait boundary, with a worker pool for concurrency, load-aware routing, predictor rebuilds after a panic, and graceful, drop-driven shutdown.

## Overview

```
                      Batchinf ── Arc<BatchinfInner> ──┐  (pool, metrics hook, shutdown watch::Receiver)
                                                       ▼
  caller ──predict()──► WorkerPool (router) ──try_send──► request channel (mpsc, queue_size)
    ▲                        │  ▲                                  │
    │                        │  └── Notify: waiters retry          ▼
    │                        │                       accumulator task (tokio)
    │                        │                         builds a batch until full or timed out
    │                        │                                     │
    │                        │                       job channel (mpsc, depth 1)
    │                        │                                     ▼
    │                        │                       inference thread (OS thread, one per worker)
    │                        │                         owns the predictor, runs predict_batch,
    │                        │                         rebuilds it after a panic
    └──── oneshot ◄──────────┴──────────────── results sent from the inference thread
      Result<Output, BatchinfError>
```

Each worker is an accumulator task plus a dedicated inference thread. Inference never runs on the async runtime, so both the multi-thread and current-thread runtimes are supported. There is no supervisor task: every lifecycle transition is driven by a channel closing or by the inference thread itself.

## Core trait

```rust
pub trait Predictor {
    type Input: Send + 'static;
    type Output: Send + 'static;
    type Error: std::error::Error + Send + Sync + 'static;

    fn predict_batch(&mut self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>, Self::Error>;
}

pub type PredictorFactory<P, E> = Arc<dyn Fn(u8) -> Result<P, E> + Send + Sync>;
```

- Takes a slice of queued inputs, returns a `Vec<Output>` in the same order. Output order matching input order must be upheld by the implementer; a length mismatch is reported as `InvalidPredictorOutput`.
- The predictor is built by a factory, called with the worker index on that worker's inference thread. The predictor is created, used and dropped on that thread, so it needs neither `Clone`, `Send` nor `Sync`, and can bind to a device (for example `worker_id % n_gpus`).
- `&mut self` lets a predictor keep mutable state (scratch buffers, sessions) without interior mutability.
- `Error: Sync` because one error value is shared by every caller in a failed batch, as `InferenceError(Arc<E>)`. No `Clone` is needed.
- The factory returns `Result`; a panic in it is also caught and treated as a failure. Failed calls are retried with exponential backoff. Weights should be loaded once outside the factory and captured in the closure, so rebuilds are cheap.
- `get_batcher` rejects a zero-sized `Input` at compile time (an inline `const` assertion).

## Configuration

Built with `BatcherConfig::builder(batch_size, batch_timeout)`; every other option has a default. The struct is `#[non_exhaustive]` with public fields.

| Field | Default | Meaning |
|---|---|---|
| `batch_size: NonZeroU32` | required | Requests per batch. |
| `batch_timeout: Duration` | required | How long a partial batch waits, from its first request. `Duration::MAX` means "only fire when full". |
| `pool_size: NonZeroU8` | 1 | Number of workers. |
| `queue_size: NonZeroU32` | `batch_size` | Capacity of each worker's request channel. |
| `worker_start_retries: u8` | 3 | Factory retries after a failed call. |
| `worker_start_backoff_start: Duration` | 100 ms | First wait between factory calls; doubles each retry. |
| `max_worker_retry_wait: Duration` | 5 s | Cap on that wait. |
| `startup_timeout: Duration` | 60 s | How long `get_batcher` waits for every worker to build its predictor. |
| `metrics: Option<Arc<dyn BatcherMetrics>>` | `None` | Metrics hook. |

## Error model

```rust
pub enum BatchinfError<E> {
    InferenceError(Arc<E>),    // predict_batch returned Err; shared by every caller in the batch.
    InvalidPredictorOutput,    // predict_batch returned the wrong number of outputs.
    InternalError,             // predict_batch panicked, or the worker went away before replying.
    TimeoutError,              // predict_with_timeout deadline passed.
    NoAvailableWorkersError,   // Every worker has exited or its predictor is dead.
    QueueFullError,            // No room right now (try_predict / blocking_try_predict only).
}

pub enum StartupError<E> {     // Returned by get_batcher.
    Timeout,                                  // Not every worker was ready within startup_timeout.
    WorkerStartFailure { worker_id: u8, err: E }, // The factory returned Err on its last attempt.
    WorkerStartPanic { worker_id: u8 },       // The factory panicked on its last attempt.
    InternalError,                            // A worker stopped before reporting.
}
```

- `InferenceError` and `InvalidPredictorOutput` are broadcast to every caller in the affected batch.
- After a `predict_batch` panic, the inference thread sends `InternalError` explicitly to every caller in that batch.
- `StartupError<E>` implements `std::error::Error` when `E: Debug + Display`, so `anyhow::Error` works as `E`.

## Public handle

`Batchinf<Input, Output, Error>` wraps a single `Arc<BatchinfInner>` holding the pool, the metrics hook and the shutdown receiver, so a clone is one reference-count increment. `Clone` is implemented by hand so it puts no `Clone` bound on the type parameters.

- `predict(input)` — queues the request and awaits the result. If every queue is full it waits for room (backpressure) and never returns `QueueFullError`.
- `try_predict(input)` / `blocking_try_predict(input)` — fail fast with `QueueFullError` instead of waiting.
- `predict_with_timeout(input, Duration)` — `predict` with a deadline covering the wait for room, queueing and inference. A request that times out before its batch starts is skipped; one already running completes and its result is discarded.
- `shutdown_waiter()` — see Shutdown.
- `pool_status()` / `worker_status(idx)` — point-in-time `WorkerSnapshot`s. Worker order is fixed.

Dropping all `Batchinf` clones triggers graceful shutdown (see Shutdown).

## Worker state

Each worker shares a `WorkerState` (an `Arc`) between its accumulator, its inference thread and the router. It holds two atomics, each with a single writer.

**Accumulator status and batch length** — one `AtomicU64`, written only by the accumulator, so a router reads both fields with one load:

```
 63  62  61 .............. 0
[ status ][   batch_len    ]
```

| Bits | `WorkerStatus` | Meaning |
|------|---------|----------------------------------------------|
| `00` | Waiting | Accumulating requests into the next batch. |
| `01` | Exit    | Request channel closed; the accumulator has exited. |
| `10` | Running | A batch is ready and being handed to the inference thread (waits while the thread is busy and already has a batch queued). |
| `11` | Crashed | The inference thread is gone; the accumulator has exited. |

`batch_len` counts requests pulled into the current batch; it never reaches the status bits. With a single writer, `set_worker_state` is a load (`Acquire`), mask and store (`Release`) rather than a CAS loop. `increment_len` is `fetch_add(1, Release)`; `reset_batch_len` stores `0`, which also means `Waiting`. If another writer is ever added, `set_worker_state` must go back to a CAS loop.

**Predictor status** — one `AtomicU8`, written only by the inference thread: `Alive`, `Rebuilding` (after a panic, while the factory runs) or `Dead` (the factory kept failing; the thread has exited).

## Accumulator

Each worker runs `worker_loop` as a detached tokio task:

1. Wait on `recv()` for the first request. The batch timer starts here, not when the previous batch ended, so an idle worker does no work.
2. Compute the deadline with `checked_add`; a timeout that overflows (for example `Duration::MAX`) falls back to a deadline a year away, so the batch fires only when full. One `sleep_until` is pinned per batch.
3. `select!` between `recv()` and the timer:
   - New request: push to the buffer, `increment_len`, and `notify_one` (a slot opened in the channel). At `batch_size`, set `Running`.
   - Timer: set `Running`.
   - Channel closed: set `Exit` and return.
4. Swap in a fresh buffer and send the full one over the job channel (depth 1). Swapping lets the next batch accumulate while the thread runs the current one; with one batch running and one queued, the accumulator then waits on `send`, its request channel fills, and the router moves on.
5. If the send fails, the inference thread is gone: set `Crashed` and return, which drops the request channel's receiver. Otherwise `reset_batch_len` (back to `Waiting`) and loop.

`on_queue_depth` is emitted with the channel backlog just before and just after each hand-off.

## Inference thread

Spawned once per worker as `batchinf-worker-{id}`:

1. **Startup**: call the factory with retries and backoff, then report `Ok(())` or a `StartupError` to `get_batcher` over the startup channel. Retries stop early if that channel has closed (startup already failed, timed out, or the `get_batcher` future was dropped).
2. **Loop** over the job channel with `blocking_recv`:
   - `filter_canceled` removes requests whose caller has gone (`swap_remove` keeps inputs and result senders paired). An empty batch is skipped.
   - Run `predict_batch` inside `catch_unwind`, timing only the call.
   - On success or error, send results (or the shared error) and emit `on_batch_complete_ok` / `_err`.
3. **Panic**: set `Rebuilding`, emit `on_worker_panic`, send `InternalError` to the batch, **drop the old predictor** (so two never coexist in device memory), and rebuild from the factory. Retries stop early if the job channel has closed (shutdown). On success set `Alive` and `notify_waiters`; on failure set `Dead`, `notify_waiters`, and exit.
4. When the job channel closes, finish any queued batch, return (dropping the predictor and factory), then release the shutdown sender.

Panics outside `predict_batch` (in `BatcherMetrics` hooks or user `Drop` impls) are not caught; the documented contract is that those must not panic.

## WorkerBuffer

```rust
struct WorkerBuffer<P: Predictor> {
    sender_buffer: Vec<OutputSender<P>>,
    input_buffer:  Vec<P::Input>,
}
```

Invariant: `input_buffer.len() == sender_buffer.len()`. Both are allocated with capacity `batch_size`. The accumulator moves a full buffer to the inference thread and starts the next batch in a fresh one (`swap_replace`); the thread takes the senders with `mem::take` and drops the inputs.

## Worker pool and routing

```rust
pub(crate) struct WorkerPool<Input, Output, Error> {
    pool: Box<[WorkerRef<Input, Output, Error>]>,  // WorkerRef = WorkerState + request Sender
    signal: Arc<Notify>,
    start: AtomicUsize,
}
```

A fixed-size slice; slots are never added or removed, and a slot's sender never changes (rebuilds happen inside the inference thread). It is shared through `BatchinfInner`, so it needs no `Arc` of its own.

**Routing** (`try_push`), all non-blocking `try_send`:

- Single-worker pools take a fast path.
- Otherwise pick a start index round-robin (`fetch_add(1, Relaxed) % size`), then:
  1. Pass 1 tries workers that are `Waiting`, `Alive` and have room in the current batch (`batch_len < batch_size` covers the moment between filling the batch and setting `Running`).
  2. Pass 2 tries every worker that has not exited or crashed and whose predictor is `Alive`.
- If both passes fail: `QueueFullError` if any worker was alive or rebuilding, otherwise `NoAvailableWorkersError`.

**Waiting for room** (`predict`): on `QueueFullError`, register a `Notify` waiter with `enable()` *before* retrying `try_push`, then await it, so a slot freed in between is not missed. Wakeups:
- `notify_one` on every dequeue: one slot opened, so wake one waiter. Cheap when nobody waits (an atomic, no lock). A woken waiter that is dropped passes its notification on.
- `notify_waiters` when a rebuild finishes (a whole queue opens up) and when a worker dies (every waiter must see `NoAvailableWorkersError` if it was the last one).

## Startup

`get_batcher(factory, config)` is `async`. It creates the workers, then waits on a startup channel carrying `Result<(), StartupError<E>>` from each inference thread. It returns on the first failure, after `startup_timeout`, or once every worker has reported success. On failure the pool is dropped, which shuts down the workers that did start; a factory call already running finishes in the background and its predictor is dropped.

## Shutdown

Shutdown is driven entirely by ownership: there is no shutdown flag or command.

1. The last `Batchinf` is dropped, so `BatchinfInner` and the pool are dropped, dropping each worker's request `Sender` — the only one.
2. Each accumulator's `recv()` returns `None` after any buffered requests; it sets `Exit` and returns, dropping the job channel's `Sender` — again the only one. A partially accumulated batch is dropped: no caller can be waiting on it, since every waiting caller holds a handle.
3. Each inference thread finishes any queued batch, sees the job channel close, returns (dropping the predictor and factory), and only then drops its clone of the shutdown `watch::Sender`.
4. When the last thread exits, the watch channel closes and every `ShutdownWaiter` resolves.

`Batchinf::shutdown_waiter()` returns a cloneable `ShutdownWaiter` holding only the `watch::Receiver`, so it never keeps the batcher alive. No value is ever sent on that channel; `wait()` returns when `changed()` errors because every sender is gone. Shutting down the tokio runtime also triggers the same sequence from step 2, by cancelling the accumulator tasks. To handle `SIGTERM`, stop the server on the signal so it drops its handle.

## Observability

```rust
pub trait BatcherMetrics: Send + Sync + 'static {
    fn on_batch_trigger(&self, batch_size: usize, trigger: BatchTrigger) {}
    fn on_batch_complete_ok(&self, batch_size: usize, latency: Duration) {}
    fn on_batch_complete_err(&self, batch_size: usize, latency: Duration) {}
    fn on_request_timeout(&self) {}
    fn on_queue_depth(&self, worker_id: u8, queue_depth: usize) {}
    fn on_worker_panic(&self) {}
}
```

Every hook defaults to a no-op, so adding a hook is not a breaking change. `Debug` is not required: the crate implements `Debug` for `dyn BatcherMetrics` itself, which keeps `#[derive(Debug)]` working on `BatcherConfig` and `Batchinf`. Latencies cover `predict_batch` only.

Hooks run inline — `on_batch_trigger` and `on_queue_depth` on the accumulator task, `on_request_timeout` on the caller's task, the rest on the inference thread — so they must be cheap, non-blocking and must not panic.

Set with `BatcherConfigBuilder::metrics` and stored as `Option<Arc<dyn BatcherMetrics>>` in `BatcherConfig::metrics`. `Option<Arc<T>>` uses the null pointer optimisation — `None` is a null pointer, so the presence check in the emit methods is a single null pointer comparison. The only overhead when `Some` is virtual dispatch through the fat pointer.

## Out of scope

- Retry/requeue on failure — caller's responsibility.
- Continuous batching / autoregressive generation.
- Pluggable flush policies beyond count and timeout.
