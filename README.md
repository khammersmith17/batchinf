# Batchinf
This crate provides a framework agnostic approach to embedding ML inference batching into a service. It is designed to be drop into common Rust server frameworks when you want to own the server where model inference is performed.

As a user, you implement the core inference function (`Predictor::predict_batch`), akin to `predict` commonly found in other frameworks, that takes a batch of records and performs inference on all of them, and this crate provides the plumbing to plug that into your service.

Inference batching is a technique that improves the utilization of a device, say GPU or TPU, by dispatching the inference call across a large number of inference examples. This improves both device utilization and throughput by amortizing the inference cost. This crate currently only supports dynamic inference batching. Continuous batching is a different implementation. Dynamic inference batching lends well to stateless inference, such as expensive embedding generation or expensive image classification.

Dynamic inference batching adds some overhead, so amortizing the cost of device usage provides benefit when inference is expensive. If inference is pretty cheap, the overhead here may actually lower throughput. There is some plumbing overhead for robust dynamic batching, which also adds overhead. When inference takes at least tens of milliseconds, and device dispatch is more expensive, then the overhead relative to inference is minimized and the amortization of dispatching to the device significantly improves throughput.


## High Level Design
All inference is dispatched through `Batchinf`, which is cheap to clone and holds all state required to dispatch to the inference workers. This fits nicely into `with_state` in a framework like `axum` or other methods of dependency injection to generate a handle to the batching pool.

Each batch is determined by whichever of the following conditions satisfy first: batch size or batch timeout. The clock for the batch timeout starts when the first inference request of the batch is collected. Tune these paremeters accordingly based on your expected throughput/toleration for latency.

Each worker has an accumulator worker that runs within the tokio runtime, which collects inference examples routed from endpoint handlers. Once the batch constraints defined are satisfied, then the accumulated buffers are dispatched to an OS thread where the each predictor instance in the pool runs on its own OS thread. At this point, the accumulator begins accepting requests and buffering inference requests again. This allows for each worker in the pool to pipeline work.

Inference requests are dispatched to works using a load aware round robin algorithm. The round robin approach determines which work is queried first. From the start of the worker search space, each worker is queried in ring order to determine if it is in a state that can accept work. If all workers are busy, then the worker space is traversed again to find the first worker that is alive, and will be able to accept work at some point in the future.

When the `predict_batch` implementation panics, it will be taken out of the pool and attempted to be rebuilt. If it cannot be rebuilt for whatever reason, then it will be taken out of the pool. The state of all accumulators and inference worker threads can be polled using the `Batchinf::pool_status` to get the status of all workers in the pool, and the `Batchinf::worker_status` to get the status of a particular worker in the pool. The number of workers in the pool can be configured using the `pool_size` configuration parameter. This determines the number of inference workers, in the case the box the server is running on has multiple devices. Each inference worker adds an accumulator worker that runs as a long running task in the `tokio` runtime and a worker that runs in its own dedicated OS thread.

Shutdown is graceful and driven by ownership: dropping the last `Batchinf` handle lets every worker finish its in-flight batches, drop its predictor and exit. To wait for that, for example before the process exits or before reusing a GPU, get a `ShutdownWaiter` from `Batchinf::shutdown_waiter` before dropping the handle and await it. This will block until all workers have exited, when awaited.

To shut down on `SIGTERM` (on Kubernetes, for example), stop your server on the signal so it drops its handle.

## Observability
Observability is a first class citizen through the `BatcherMetrics` trait, where any observability backend can be used to emit system metrics, see the trait methods for more detail. Pass your implementation to `BatcherConfigBuilder::metrics`. Hooks run inline, so they should be cheap and must not panic. If an observability method panics, it will remove the associated accumulator from the pool. Other workers in the pool, if `pool_size` > 1, will continue.

## Logging
The crate emits [`tracing`](https://docs.rs/tracing) events under the `batchinf` target. It does not install a subscriber or read any environment variables itself. It is configurable based on your needs.

With `tracing-subscriber`'s `EnvFilter`, for example, run with `RUST_LOG=batchinf=info`, or add `batchinf=info` to your existing filter.

| Level | What is logged |
|---|---|
| `error` | A worker stopped serving: its predictor could not be rebuilt, or its batching task failed. |
| `warn` | [`Predictor::predict_batch`] panicked (with the panic message), a predictor factory call failed (with the error and the delay before the next attempt), or a worker could not start. |
| `info` | Worker lifecycle: batching tasks and inference threads starting and exiting, predictors built and rebuilt. |
| `debug` | Per batch: when a batch starts filling and why it fired (full, or timed out). Shutdown cutting a startup or rebuild short. |
| `trace` | Per batch: inference starting and results being returned. |

Every event from a worker carries a `worker_id` field. Events from the batching task are inside a `batchinf_accumulator` span, and events from the inference thread inside a `batchinf_inference` span. Your own `tracing` calls inside `predict_batch` run in the inference thread's span too, so they are tagged with the worker as well.

