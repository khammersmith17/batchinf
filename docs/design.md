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

    fn predict_batch(&mut self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>, Self::Error>;
}

pub type PredictorFactory<P, E> = Arc<dyn Fn(u8) -> Result<P, E> + Send + Sync>;
```

- Takes ownership of the queued inputs, returns a `Vec<Output>` in the same order. Output order matching input order must be upheld by the implementer. `BatchinfError` describes to the caller what might have failed.
- Inputs are passed by value because the inference thread owns each batch's buffer and has no further use for it. Moving the `Vec` is cheap and allows for pipeline in addition to letting a predictor hand inputs to APIs that take ownership (tokenizers, tensor constructors) without cloning them. The result senders are split off and kept by the thread, so a panic still reaches every caller in the batch.
- The predictor is built by a factory, called with the worker index on that worker's inference thread. The predictor is created, used and dropped on that thread, so it needs neither `Clone`, `Send` nor `Sync`, and can bind to a device (for example `worker_id % n_gpus`).
- `&mut self` lets a predictor keep mutable state (scratch buffers, sessions) without interior mutability.
- `Error: Sync` because one error value is shared by every caller in a failed batch, as `InferenceError(Arc<E>)`. 
- The factory returns `Result`; a panic in it is also caught and treated as a failure. Failed calls are retried with exponential backoff. Weights should be loaded once outside the factory and captured in the closure, so rebuilds are cheap.
- `get_batcher` rejects a zero-sized `Input` at compile time.

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
    InternalError,             // predict_batch panicked, the request was queued on a worker whose
                               // predictor panicked or died, or the worker went away before replying.
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
- After a `predict_batch` panic, the inference thread sends `InternalError` explicitly to every caller in that batch. The error here cannot be derived, but the panic trace is logged. Requests queued on the worker at that point also get `InternalError` (see [Failing queued requests](#failing-queued-requests)).
- `InferenceError` displays as `Unable to perform inference` and exposes the predictor's error through `source()` rather than its message, so error reporters that print the chain show it once.
- `StartupError<E>` hands the user defined error back to the user on panic during `Predictor` build.

## Public handle

`Batchinf<Input, Output, Error>` wraps a single `Arc<BatchinfInner>` holding the pool, the metrics hook and the shutdown receiver, so a clone is cheap. 

- `predict(input)` — queues the request and awaits the result. If every queue is full it waits for room and never returns `QueueFullError`.
- `try_predict(input)` / `blocking_try_predict(input)` — fail fast with `QueueFullError` instead of waiting.
- `predict_with_timeout(input, Duration)` — `predict` with a deadline covering the wait for room, queueing and inference. A request that times out before its batch starts is skipped; one already running completes and its result is discarded.
- `shutdown_waiter()` — see Shutdown.
- `pool_status()` / `worker_status(idx)` — point-in-time `WorkerSnapshot`s. Worker order is fixed in the worker pool.

Dropping all `Batchinf` clones triggers graceful shutdown (see Shutdown).

## Worker state

Each worker shares a `WorkerState` between its accumulator, its inference thread and the router. It holds two atomics, each with a single writer.

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
| `11` | Crashed | The inference thread is gone; the accumulator has exited. The predictor is always `Dead` by then. |

`batch_len` counts requests pulled into the current batch. The batch size is capped at `u32::MAX`, so state bits are never overwritten. `Acquire` and `Release` are used for ordering. Only the accumulator writes the worker status: `set_worker_state` is a load then a store, which is only safe with a single writer.

**Predictor status** — one `AtomicU8`, set by the inference thread: `Alive`, `Rebuilding` or `Dead`. `Rebuilding` describes the state where a `Predictor` panicked and is rebuilding. `Dead` describes the state where the `Predictor` could not be built after retries, or where a metrics hook panicked on either the accumulator or the inference thread (set by `PanicGuard`). Writes are plain stores, so both sides can write it safely.

## Accumulator

Each worker runs `worker_loop` as a detached tokio task:

1. Wait on `recv()` for the first request. The batch timer starts here, so an idle worker does no work.
2. Batch expiration is derived using `tokio::time::sleep_until`.
3. `select!` between `recv()`, the timer and `job_tx.closed()`:
   - On new request: push to the buffer, `increment_len`, and `notify_one` to notify when someone is waiting for capacity in the queue. At `batch_size`, end the batch.
   - Timer: end the batch.
   - Job channel closed: the inference thread has exited, so end the batch now rather than at its timeout.
   - Request channel closed: set `Exit` and return `BatchStatus::Exit`.
4. Check the predictor. If it is not `Alive`, fail the batch and drain the request channel (`BatchStatus::Flushed`), then loop. Otherwise swap in a fresh buffer and return the batch as `BatchStatus::Ready`. Swapping lets the next batch accumulate while the thread runs the current one.
5. Set `Running`, then hand off with `job_tx.reserve()`. With one batch running and one queued, the accumulator waits here, its request channel fills, and the router moves on. In effect, this allows for pipelining. Once a slot is free, check the predictor again:
   - `Alive`: `permit.send` the batch (`BatchHandoffResult::Sent`) and emit `on_batch_trigger`.
   - Not `Alive`: drop the permit, fail the batch and drain the request channel (`BatchHandoffResult::Flushed`).
   - `reserve` failed: the inference thread is gone (`BatchHandoffResult::Dead`). Set `Crashed` and return, which drops the request channel's receiver.
6. `reset_batch_len` and loop.

`on_queue_depth` is emitted with the channel backlog just before and just after each hand-off. After a sent batch, the second value is the backlog that built up during the wait. After a flush it is 0, so the gauge does not keep a stale backlog while the predictor rebuilds. When the accumulator returns because the inference thread is gone, it emits 0 as well.

A panic in the observability hooks is observed by `PanicGuard`, which sets the `PredictorStatus` to `Dead` and wakes waiting callers. It does not write the worker status, which only the accumulator writes.

## Inference thread

Spawned once per worker as `batchinf-inference-worker-{id}`:

1. **Startup**: call the factory with retries and backoff, then report `Ok(())` or a `StartupError` to `get_batcher` over the startup channel. Retries stop early if that channel has closed (startup already failed, timed out, or the `get_batcher` future was dropped).
2. **Loop** over the job channel with `blocking_recv`:
   - `filter_canceled` removes requests whose caller has gone when batch is non-empty.
   - Run `predict_batch` inside `catch_unwind`, timing only the call.
   - On success or error, send results or error and emit `on_batch_complete_ok` / `_err`.
3. **Panic**: set the `Predictor` state to `Rebuilding`, emit `on_worker_panic`, send `InternalError` to the batch, and fail the batch waiting in the job slot, if any.
    - The old crashed `Predictor` is then dropped and rebuilt from the factory. Retries stop early if the job channel has closed by polling every 20 ms. 
    - On a successful rebuild, fail any batch that landed in the job slot during the rebuild, then set the `Predictor` state to `Alive` and `notify_waiters`.
    - On a failure the `Predictor` state is set to `Dead`, `notify_waiters` so they can get a `NoAvailableWorkers` error, and exit the loop.
4. When the job channel closes, exit the loop and shutdown.

Panics outside `predict_batch` are not caught. This is limited to user defined observability hooks, thus the documented contract is that those must not panic. If one does, `PanicGuard` marks the predictor `Dead`, the thread exits, and the accumulator sees the closed job channel and fails what it holds.

## Failing queued requests

When a predictor panics, every request already queued on that worker fails with `InternalError` straight away instead of waiting for the rebuild, whose length depends on the factory and its retries. The same applies when the predictor goes `Dead`. New requests are not affected: the router stops sending requests to a worker whose predictor is not `Alive`.

A queued request can be in one of four places, and each is covered:

| Where the request is | Failed by |
|---|---|
| The batch running on the predictor | The inference thread, after `catch_unwind` (step 3 above). |
| The batch in the job slot | The inference thread, right after the panic. |
| The batch the accumulator is handing off | The accumulator, which checks the predictor after `reserve` and before `permit.send`. |
| The accumulating batch, and the request channel | The accumulator, when the batch ends (full, timeout, or `job_tx.closed()`). |

Checking after `reserve` rather than before matters: an accumulator can wait in `reserve` for a long time, and a check made before the wait can be stale by the time the batch is sent. One narrow race remains, between that check and `permit.send`, with no `.await` in between. A batch that slips through it lands in the job slot after the post-panic flush, and the inference thread fails it once the rebuild finishes.

## WorkerBuffer

```rust
struct WorkerBuffer<P: Predictor> {
    sender_buffer: Vec<OutputSender<P>>,
    input_buffer:  Vec<P::Input>,
}
```
The property `input_buffer.len() == sender_buffer.len()` must always hold.

Both buffers are allocated with capacity `batch_size` on construction. The accumulator moves a full buffer to the inference thread and starts the next batch in a fresh one. The thread splits it into inputs and senders. The inputs are moved to the user defined `predict_batch`, and result senders continue to be owned by the loop until the results are dispatched back to the callers over the `oneshot` channels.

## Worker pool and routing

```rust
pub(crate) struct WorkerPool<Input, Output, Error> {
    pool: Box<[WorkerRef<Input, Output, Error>]>,  // WorkerRef = WorkerState + request Sender
    signal: Arc<Notify>,
    start: AtomicUsize,
}
```

A fixed-size slice and a slot's sender is always valid as `Predictors` are rebuilt inplace. It is shared through `BatchinfInner`.

**Routing** (`try_push`), all non-blocking `try_send`:

- Single-worker pools attempt to forward to the only worker.
- Otherwise pick a start index round-robin, then:
  1. Pass 1 tries workers that are `WorkerStatus::Waiting`, `PredictorStatus::Alive` and have room in the current batch as a batch might have just become full. The first worker, if any, that satifies these conditions is selected.
  2. Pass 2 tries every worker that has not exited or crashed and whose predictor is `Alive`, as a fallback. The first worker that can accept work is selected.
- If both passes fail, `QueueFullError` is returned if any worker was alive or rebuilding, otherwise `NoAvailableWorkersError` indicating there are currently no workers how can accept work at any point. This will be case when all workers are rebuilding or dead.

**Waiting for room** (`predict`): on `QueueFullError`, a `Notify` waiter is registered before retrying `try_push`, then await it, so a slot freed in between is not missed. 
On wakeups:
- `notify_one` is called on every dequeue in all accumulators to notify a caller waiting for capacity.
- `notify_waiters` when a rebuild finishes to indicate another worker can accept work, and when a worker dies to bubble up a `NoAvailableWorkers` if there are no workers currently available after a failed rebuild or a panic in an observability hook.

## Startup

`get_batcher(factory, config)` creates the workers, then waits on a startup channel carrying `Result<(), StartupError<E>>` from each inference thread. It returns on the first failure, after `startup_timeout`, or once every worker has reported success. 

On failure the pool is dropped, which shuts down the workers that did start and an `Err` is bubbled up. Any factories that are still running are failed.

## Shutdown

Shutdown is driven entirely by ownership by dropping the channels that forward requests to workers.

1. When the last `Batchinf` is dropped, `BatchinfInner` is the dropped as the reference count becomes 0. Dropping `BatchinfInner` then drops the pool, which in turn closes all channels after the are drained.
2. Each accumulator's `recv()` then returns `None` after any buffered requests
    - The accumulators state is then set to `Exit` and returns
    - The same mechanism is used in the `Predictor` thread to signal exit.
3. Each inference thread finishes any queued batch
    - Then the `watch::Sender` is dropped.
4. When the last thread exits, the watch channel closes and every `ShutdownWaiter` resolves.

`Batchinf::shutdown_waiter()` returns a cloneable `ShutdownWaiter` holding the `watch::Receiver`. Its lifetime is not coupled to the lifetime of `BatchinfInner`. A user can then await the `watch::Receiver` to gracefully exit.

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

Hooks run inline — `on_batch_trigger` and `on_queue_depth` on the accumulator task, `on_request_timeout` on the caller's task, the rest on the inference thread — so they must be cheap, non-blocking and must not panic.

Set with `BatcherConfigBuilder::metrics`. Given it is `Option<Arc<dyn BatcherMetrics>`, it is a no-op when not provided. Also, this leverages the null pointer optimization, given that the `Option<T>` only holds a pointer when `Some`.

## Logging
Logging is configurable by a user. The modes can be found in the README.

## Out of scope (For now)

- Retry/requeue on failure is delegated to the user.
- Continuous batching / autoregressive generation.
- Inference trigger policies beyond count and timeout.
