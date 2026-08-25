# Rust Inference Batching Crate — Design Summary

## Goal

A Rust crate providing a reusable abstraction for batching ML inference
requests: individual requests are queued until either a size threshold or a
timeout is hit, then dispatched together. The crate is backend-agnostic —
`ort`, `burn`, `candle`, or anything else — via a trait boundary, with a
worker pool for concurrency and simple least-loaded routing.

## Prior art

- **`batched-fn`** — macro-based, channel-backed (`flume`) queue with
  `max_batch_size` + `max_delay`. Closest prior art; unmaintained, single-model.
- **`ort_batcher`** — same pattern, hard-coded to `ort` + `f32` tensors.
- **`text-embeddings-inference` (TEI)** — production reference architecture.
  Router → tokenizer → Queue → batch manager → backend (Candle/ORT/Python).
  Notably batches by **token budget**, not just count/timeout, since request
  cost is heterogeneous. Worth generalizing as a pluggable flush policy.
- **`tritonserver-rs`** — wraps Triton's batching scheduler via FFI; different
  tradeoff (Triton's process model, not in-process pure-Rust).
- Nothing on crates.io currently unifies `ort`/`burn`/`candle` under one
  batching abstraction — this is a real gap.

## Why unifying ort/burn/candle is nontrivial

The three don't share a tensor type or execution model:

| Crate | Tensor type | Batching mechanism |
|---|---|---|
| `ort` | `ort::Value` (often built from `ndarray`) | concat raw arrays along dim 0 before building `Value` |
| `candle` | `candle::Tensor` (own `Device`/`DType`) | `Tensor::cat(&tensors, 0)` |
| `burn` | generic `Tensor<B>`, `B: Backend` | backend-generic, viral generics or trait-object erasure |

Resolution: don't try to share a tensor type. Put the abstraction boundary at
"batchable input/output," and let each backend adapter own its own
concat/split logic internally.

## Core trait

```rust
trait Predictor: Send + Sync {
    type Input;
    type Output;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Called once per flushed batch. Implementor receives all queued items
    /// for this flush and is responsible for fusing them into one backend
    /// call (e.g. stacking into a single ort::Value / candle::Tensor /
    /// burn::Tensor) to get the actual GPU-utilization benefit of batching.
    /// Reads and is implemented like a normal single-request handler —
    /// the queueing/timeout mechanics are invisible to the implementor.
    fn predict_batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>, Self::Error>;
}
```

Implementors get a `Vec<Input>`, do one fused forward pass, return
`Vec<Output>` in the same order. Optional helper types (`TensorBuffer`,
`Batch`, described below) make the concat/split mechanical, but an
implementor can also hand-roll it directly against `ort`/`candle`/`burn`.

**Rejected alternative:** `predict` operating on one item at a time in a
loop inside the flush. Keeps the device/thread warm and smooths bursty
traffic, but loses the fused-matmul throughput win that's the actual point
of batching for GPU-bound inference — sequential per-item calls underutilize
the GPU the same way unbatched serving does. Rejected in favor of the
`Vec<Input> -> Vec<Output>` fused signature above.

## Optional helper types for building `Input`/`Output`

For backends that want a ready-made contiguous layout:

```rust
struct TensorBuffer<T: bytemuck::Pod> {
    data: Vec<T>,           // flat, len = batch_size * item_len
    item_len: usize,
    shape_tail: SmallVec<[usize; 4]>, // per-item dims after batch dim
}

struct Batch {
    tensors: HashMap<&'static str, ErasedTensorBuffer>, // dtype-erased, one entry per named model input
    batch_size: usize,
}
```

Design notes:

- **Element bound is `bytemuck::Pod + Zeroable`, not `num_traits::Float`.**
  Token-based models need `i64`/`u32` (`input_ids`, `attention_mask`), vision
  models may want `u8`. `Pod` allows any plain-old-data element to back the
  buffer without per-element conversion in the core crate; dtype-specific
  meaning stays in the backend adapter.
- **A `Batch` is a named set of typed buffers sharing a batch size**, not a
  single buffer — real models commonly have multiple inputs with different
  dtypes and shapes (e.g. `input_ids: i64` + `attention_mask: i64` +
  `pixel_values: f32`). A single-tensor model just uses a `Batch` with one
  entry.
- **Per-tensor contiguity is already the optimal layout** — it matches what
  every backend's call signature wants (one `Value`/`Tensor` per named
  input). A single heterogeneous byte buffer spanning all inputs buys
  nothing; you'd split it back into per-tensor views before calling any
  backend anyway.
- **Flatten at flush time, not live during collection.** Each pending
  request holds its own owned data; the flushing task does one
  copy/`extend` pass per named tensor when building the batch. Avoids
  synchronization on the hot path.

## Queue + worker

```rust
struct QueuedRequest<In, Out, Err> {
    input: In,
    respond_to: oneshot::Sender<Result<Out, Arc<Err>>>,
}
```

Each `predict()` (public API) call: build a request, send `(input, tx)` on
an `mpsc`, `.await` the paired `oneshot::Receiver`.

Worker loop (one per pool member), simplified:

```rust
let mut pending: Vec<QueuedRequest<..>> = Vec::new();
loop {
    if pending.is_empty() {
        match rx.recv().await {
            Some(req) => pending.push(req),
            None => break, // channel closed, shut down
        }
    }
    let deadline = sleep_until(first_item_time + max_wait);
    tokio::select! {
        _ = &mut deadline => flush(&mut pending).await,
        Some(req) = rx.recv() => {
            pending.push(req);
            if pending.len() >= max_batch_size {
                flush(&mut pending).await;
            }
        }
    }
}
```

Key points:

- Timeout is measured from the **first** item in the current pending batch,
  not reset per arrival — otherwise a steady trickle never flushes.
- `flush` calls `predict_batch` on `spawn_blocking` (or a dedicated thread),
  since inference is CPU/GPU-bound and must not stall the async runtime.
- The `select!` structure makes double-flush races structurally impossible —
  one task, sequential control flow, no flush-guard atomic needed unless an
  out-of-band trigger (e.g. an admin "flush now" endpoint) is added later.

## Fan-out (success and failure)

- **Success:** split the returned `Vec<Output>` back to the `pending`
  oneshots in order. Output order matching input order is a documented
  invariant of `Predictor`; an optional debug-only length check
  (`output.len() == pending.len()`) is cheap insurance against a buggy impl.
- **Failure:** `predict_batch` returns one `Err(e)` for the whole flush, but
  each queued item still gets exactly one shot — **no retry inside the
  crate**. The batcher wraps `e` once in `Arc<Self::Error>` and clones the
  `Arc` into every pending oneshot in that flush:

  ```rust
  Err(e) => {
      let shared = Arc::new(e);
      for req in pending.drain(..) {
          let _ = req.respond_to.send(Err(Arc::clone(&shared)));
      }
  }
  ```

  Caller-visible type is `Result<Output, Arc<Predictor::Error>>`. `Arc` is
  used (rather than a `Clone` bound on `Predictor::Error`) so implementors
  aren't constrained to a cloneable error type, and the wrap only happens on
  the already-exceptional failure path. Retry/requeue policy is explicitly
  out of scope — that's the caller's business logic (same worker vs. a
  different one, backoff, give up, etc.), not something the batcher should
  have an opinion on.

## Worker pool and routing

- Pool is sharded: each worker owns its own `mpsc` receiver, its own
  `pending` buffer, its own timeout — no shared-queue mutex.
- Pool constructed from `Vec<Predictor>` (or a factory `Fn() -> P` called N
  times); size naturally tracks number of independent backend instances
  (e.g. devices/sessions) rather than being an arbitrary integer the core
  invents meaning for. `Vec` of length 1 degenerates to the single-worker
  case with no special-casing.
- `max_batch_size` / `max_wait` apply **per worker**, not globally across
  the pool — falls out of the sharded design for free; total in-flight
  capacity is `N × max_batch_size`.

### Worker state for routing

Single `AtomicU64` per worker, MSB = busy flag, low 63 bits = pending queue
count — packed into one word specifically so a router can read both fields
as one consistent snapshot (two separate atomics can't be read as a pair
atomically regardless of memory ordering used):

```rust
struct WorkerState(AtomicU64);
const BUSY_BIT: u64 = 1 << 63;

impl WorkerState {
    fn set_busy(&self, busy: bool) {
        if busy { self.0.fetch_or(BUSY_BIT, Ordering::Relaxed); }
        else    { self.0.fetch_and(!BUSY_BIT, Ordering::Relaxed); }
    }
    fn add_load(&self, n: u64) { self.0.fetch_add(n, Ordering::Relaxed); }
    fn sub_load(&self, n: u64) { self.0.fetch_sub(n, Ordering::Relaxed); }
    fn snapshot(&self) -> (bool, u64) {
        let v = self.0.load(Ordering::Relaxed);
        (v & BUSY_BIT != 0, v & !BUSY_BIT)
    }
}
```

- `busy`: set `true` immediately before `predict_batch`, `false`
  immediately after — brackets the blocking call exactly.
- `load`: `fetch_add` on enqueue (public `predict()` call site),
  `fetch_sub(pending.len())` once after a flush completes.
- `Relaxed` ordering throughout — this is pure scheduling heuristic with no
  dependent memory to publish/acquire, so `Acquire`/`Release`/`SeqCst` would
  add synchronization cost without fixing anything (in particular, no
  ordering can make two *separate* atomics read as a transaction — hence
  packing into one `AtomicU64` instead of using stricter ordering on two).
- 63 bits for the count is not a practical overflow risk (would require
  ~10^18 queued-but-unflushed items).

Router:

```rust
fn pick_worker(workers: &[WorkerHandle]) -> usize {
    workers.iter().enumerate()
        .min_by_key(|(_, w)| w.state.snapshot()) // (busy, load), false < true
        .map(|(i, _)| i)
        .unwrap()
}
```

Idle workers are preferred over busy ones outright; queue depth breaks ties.
Degrades gracefully to pure load-based routing when all workers are idle or
all are busy. The read-then-route sequence is still a benign heuristic race
under concurrency (two callers can pick the same "least loaded" worker
before either's counter update lands) — accepted as self-correcting rather
than fixed with a lock, to avoid reintroducing the contention the sharded
design exists to avoid.

The per-worker `AtomicU64` can also be exposed read-only via the batcher
handle (`batcher.worker_loads() -> Vec<(bool, u64)>`) as a free
Prometheus/metrics hook.

## Explicitly out of scope (v1)

- Retry/requeue on failure — caller's responsibility.
- Continuous batching / autoregressive generation (output batch size ≠
  input batch size) — a materially different scheduling problem.
- Cross-worker shared-queue load balancing (would require a lock around a
  shared `pending` buffer; sharded-per-worker is the default, revisit only
  if profiling shows uneven batch fill).
- Pluggable flush policies beyond count/timeout (e.g. TEI-style token
  budget) — noted as a natural extension point (`should_flush(&[Request]) ->
  bool` trait) but not required for v1.
