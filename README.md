# Batchinf
This crate provides a framework agnostic approach to embedding ML inference batching into a service.

As a user, you implement the core inference function (`Predictor::predict_batch`), akin to `predict` in other frameworks, that takes a batch of records and performs inference on all of them, and this crate provides the plumbing to plug that into your service.

Inference batching is a technique that improves the utilization of a device, say GPU or TPU, by dispatching the inference call across a large number of inference examples. This improves both device utilization and throughput by amortizing the inference cost. This crate currently only supports dynamic inference batching. Continuous batching is a different implementation. Dynamic inference batching lends well to stateless inference, such as expensive embedding generation or expensive image classification.

Dynamic inference batching adds some overhead, so amortizing the cost of device usage provides benefit when inference is expensive. If inference is pretty cheap, the overhead here may actually lower throughput. There is some plumbing overhead for robust dynamic batching, which also adds overhead. When inference takes at least tens of milliseconds, and device dispatch is more expensive, then the overhead relative to inference is minimized and the amortization of dispatching to the device significantly improves throughput.

All inference is dispatched through `Batchinf`, which is cheap to clone and holds all state required to dispatch to the inference workers. This fits nicely into `with_state` in a framework like `axum` or other methods of dependency injection to generate a handle to the batching pool.

Each worker runs inference on its own dedicated OS thread, so inference never blocks the async runtime. If `predict_batch` panics, the callers in that batch get an error and the worker rebuilds its predictor from your factory.

Shutdown is graceful and driven by ownership: dropping the last `Batchinf` handle lets every worker finish its in-flight batches, drop its predictor and exit. To wait for that, for example before the process exits or before reusing a GPU, get a `ShutdownWaiter` from `Batchinf::shutdown_waiter` before dropping the handle and await it. To shut down on `SIGTERM` (on Kubernetes, for example), stop your server on the signal so it drops its handle.

Observability is a first class citizen through the `BatcherMetrics` trait, where any observability backend can be used to emit system metrics. Pass your implementation to `BatcherConfigBuilder::metrics`; every hook defaults to a no-op. Hooks run inline, so they should be cheap and must not panic.
