# Batchinf — Design

## Goal

A Rust crate providing a reusable abstraction for batching ML inference requests: individual requests are queued until either a size threshold or a timeout is hit, then dispatched together as a batch. The crate is backend-agnostic via a trait boundary, with a worker pool for concurrency, load-aware routing, worker crash recovery, and graceful shutdown.

## Overview

```
                                         ┌─────────────────────────────────┐
                                         │           WorkerPool            │
  caller ──predict()──► Batchinf ───────►│                                 │
    ▲                      │             │  WorkerRef[0] ──mpsc──► Worker  │
    │                      │ oneshot rx  │  WorkerRef[1] ──mpsc──► Worker  │
    └──────────────────────┘             │  WorkerRef[N] ──mpsc──► Worker  │
      Result<Output, Err>                │                                 │
                                         └─────────────────────────────────┘
                                                  ▲ Weak ref
                                                  │
  SIGTERM ──► signal task ──► AtomicBool ────► ControlPlane (250ms poll)
                                                  │
                                         crashed? ├──► restart worker
                                         exiting? └──► set_exit on workers
```

## Core trait

```rust
pub trait Predictor: Clone {
    type Input: Send + Sync + 'static;
    type Output: Send + Sync + 'static;
    type Error: std::error::Error + Clone + Send + Sync + 'static;

    fn predict_batch(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>, Self::Error>;
}
```

- Takes a slice of queued inputs, returns a `Vec<Output>` in the same order. Output order matching input order must be upheld by trait implementer.
- `Clone` is required so the predictor can be cloned once per pool worker at startup, plus once more into the control plane for crash recovery.
- `Error: Clone` is required so a single inference error can be broadcast to every caller in the failed batch without heap allocation per caller.
    - Errors in this case should be cheap.
- `predict_batch` is called synchronously inside `tokio::task::block_in_place`, so it may block without stalling the async runtime. This is preferred over `spawn_blocking` because it avoids the `'static` bound that would prevent
  borrowing local state.
- Input buffer is borrowed so a single allocation needs to performed per worker, and the math for stateless inference tasks is typically pure.

## Configuration

```rust
pub struct BatcherConfig {
    pub batch_timeout: NonZeroU32,  // milliseconds
    pub batch_size: NonZeroU32,
    pub pool_size: NonZeroU32,
}
```

`u32` sized values imply some constraints.

## Error model

```rust
pub enum BatchinfError<E> {
    InferenceError(E),         // Predict_batch returned Err, E is user implemented Predictor::Error.
    InvalidPredictorOutput,    // Predict_batch returned wrong number of outputs.
    InternalError,             // Worker exited before sending a result (e.g. after panic).
    TimeoutError,              // Predict_with_timeout deadline exceeded.
    NoAvailableWorkersError,   // All workers are in Exit or Crashed state.
    QueueFullError,            // All worker channels are full.
}
```

- `InferenceError(E)` and `InvalidPredictorOutput` are broadcast to every caller in the affected batch.
- `InternalError` is produced when the worker's oneshot sender is dropped during unwind.

## Public handle

`Batchinf<Input, Output, Error>` is the user-facing type. It is cheap to clone, effectively the size of two fat pointer. Each instance holds a pointer to the `WorkerPool` 
Key methods:
- `predict(input) -> Result<Output, BatchinfError<Error>>` — queues a request and awaits the result.
- `predict_with_timeout(input, Duration) -> Result<Output, BatchinfError<Error>>`
  - `predict` that provides allows timeouts on queued inference requests. When timed out, the inference result is ignored.
- `pool_status() -> Vec<WorkerSnapshot>` — point-in-time snapshot of all workers.
- `worker_status(idx) -> Option<WorkerSnapshot>` — snapshot of a single worker. Ordering of workers is static, so queries are consistent.

Dropping all `Batchinf` clones triggers graceful shutdown (see Shutdown).

## Packed worker state

Each worker exposes a single `AtomicU64` that encodes both its operational status and its current queue length in one word, so a router can read a consistent snapshot of both fields with a single atomic load:

```
 63  62  61 .............. 0
[ state  ][   queue_len    ]
```

The top 2 bits hold the status:

| Bits | Status  | Meaning                                      |
|------|---------|----------------------------------------------|
| `00` | Waiting | Accumulating requests into the next batch    |
| `01` | Exit    | Exited; no longer accepting requests         |
| `10` | Running | Executing `predict_batch`                    |
| `11` | Crashed | `predict_batch` panicked; awaiting restart   |


In practice, the queue len will never overwrite the state bits.

Operations:
- `increment_len`: `fetch_add(1, Release)` — increments queue_len; state bits are unaffected since batch sizes fit in 62 bits.
- `set_state(status)`: CAS loop — preserves queue_len bits, replaces state bits. Success ordering: `Release`. Failure ordering: `Relaxed`.
- `reset_queue_len`: `store(0, Release)` — clears both queue_len and state bits, implicitly returning the worker to `Waiting` (state `0b00`).
- `get_state` / `snapshot`: `load(Acquire)`.

## Worker loop

Each worker runs `worker_loop` in a detached `tokio::task::spawn`. The loop alternates between accumulation and inference:

**Accumulation (`accumulate_next_batch`):**

1. Block on `receiver.recv()` for the first item. Start the batch timer once the first item arrives rather than when inference of the previous batch ends. Eliminates unnecessary work when the worker is not busy. 
2. Loop with `tokio::select!`:
   - On new item: push to `WorkerBuffer`, increment queue_len. If
     `buffer.len() == capacity`, set state to `Running` and break.
   - On timeout: set state to `Running` and return.
   - On channel close: set state to `Exit` and return.

**Inference (`run_inference`):**

1. Register `InferencePanicGuard` — a Drop guard that sets state to `Crashed` if the thread is unwinding. This is the mechanism by which the control plane detects a crash.
2. Call `block_in_place(|| predictor.predict_batch(&buffer.input()))`.
3. Take senders from `WorkerBuffer` via `clear_and_take_senders` (uses `mem::replace` to preserve allocation).
4. Validate output length. Broadcast `InferenceError` or `InvalidPredictorOutput` on mismatch or error; otherwise zip outputs to senders.

After inference, if state is `Running`, call `reset_queue_len` (returns to `Waiting`) and loop. If state is `Exit`, break.

## WorkerBuffer

```rust
struct WorkerBuffer<P: Predictor> {
    sender_buffer: Vec<OutputSender<P>>,
    input_buffer:  Vec<P::Input>,
}
```

Invariant: `input_buffer.len() == sender_buffer.len()` at all times. `clear_and_take_senders` clears `input_buffer` in-place (retaining capacity) and `mem::replace`s `sender_buffer` with a fresh `Vec` of the same capacity, returning the old senders for dispatch. Both buffers are pre-allocated to `batch_size` and reused across batches.

## Worker pool and routing

```rust
pub(crate) struct WorkerPool<Input, Output, Error> {
    pool: Arc<[RwLock<WorkerRef<Input, Output, Error>>]>,
}
```

A fixed-size slice of `RwLock<WorkerRef>`. Slots are never added or removed; crashed workers are restarted in-place by swapping the channel sender within the existing slot. The `Arc` is shared across all `Batchinf` clones; the control plane holds a `Weak` reference.

**Routing algorithm** (`WorkerPool::push`):

Single-worker pools take a fast path — no routing logic.

For multi-worker pools:

1. Pick a random start index.
2. Scan all workers in ring order. For each worker in `Waiting` state with `queue_len < capacity`, attempt a `try_send` (non-blocking). Return on success. Record the first non-Exit/non-Crashed worker as `fallback`.
3. If no suitable Waiting worker was found, push to `fallback`. If that also fails, return `QueueFullError` or `NoAvailableWorkersError`.

All pushes use `try_send`; the pool never blocks awaiting channel capacity. Push acquires a read lock on the target slot; crash restart acquires a write lock.

## Control plane

A background task (`supervisor_loop`) holds a `Weak` reference to the pool and runs a poll loop every 250ms:

- **Crash detection**: reads each worker's snapshot under a read lock. If `Crashed`, upgrades to a write lock, re-checks state (TOCTOU guard), then calls `restart_worker`: creates a new channel, resets queue_len, spawns a new `InferenceWorker` with the same `WorkerState`, and swaps the sender via `replace_queue`.
- **All-exited detection**: if every worker is in `Exit` state, the control plane exits.
- **Queue depth metric**: emits `on_queue_depth` with the total across all workers each poll cycle.
- **Pool liveness**: if `pool_weak.upgrade()` returns `None` (all `Batchinf` clones dropped), the control plane exits immediately.

## Shutdown

**Drop-driven shutdown** (all `Batchinf` clones dropped):

1. `Arc` refcount reaches zero — pool is dropped.
2. Control plane's `pool_weak.upgrade()` returns `None` — control plane exits.
3. Each `WorkerRef`'s `Sender` is dropped — workers' `recv()` returns `None`
   — each worker sets `Exit`, flushes its in-progress batch, and exits.

**Signal-driven shutdown** (SIGTERM / Windows ctrl-shutdown):

1. A spawned signal handler sets an `AtomicBool` flag (`Ordering::Release`).
2. The supervisor loop checks the flag each iteration (`Ordering::Acquire`), breaks on true, and calls `shutdown_workers`.
3. `shutdown_workers` polls every 500ms: for each non-Exit worker, acquires a write lock and calls `set_exit` — sets state to `Exit` and replaces the sender with a dummy channel whose receiver is immediately dropped. This closes the worker's channel, causing `recv()` to return `None` and triggering flush + exit. Loops until all workers are in `Exit` state.

## Observability

```rust
pub trait BatcherMetrics: Debug + Send + Sync + 'static {
    fn on_batch_trigger(&self, batch_size: usize, trigger: BatchTrigger);
    fn on_batch_complete_ok(&self, batch_size: usize, latency: Duration);
    fn on_batch_complete_err(&self, batch_size: usize);
    fn on_request_timeout(&self);
    fn on_queue_depth(&self, queue_depth: usize);
}
```

Passed as `Option<Arc<dyn BatcherMetrics>>`. `Option<Arc<T>>` uses the null pointer optimisation — `None` is a null pointer, so the presence check in the emit methods is a single null pointer comparison. The only overhead when `Some is virtual dispatch through the fat pointer.

## Out of scope

- Retry/requeue on failure — caller's responsibility.
- Continuous batching / autoregressive generation.
- Pluggable flush policies beyond count and timeout.
