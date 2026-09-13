# Batchinf
This crate provides a framework agnostic approach to embedding ML inference batching into a service.

As a user, you implement the core inference function (`Predictor::predict_batch`), akin to `predict` in other frameworks, that takes a batch of records and performs inference on all of them, and this crate provides the plumbing to plug that into your service.

Inference batching is a technique that improves the utilization of a device, say GPU or TPU, by dispatching the inference call across a large number of inference examples. This improves both device utilization and throughput by amortizing the inference cost. This crate currently only supports static inference batching. Continuous batching is a different implementation. Static inference batching lends well to stateless inference, such as expensive embedding generation or expensive image classification.

Static inference batching adds some overhead, so amortizing the cost of device usage provides benefit when inference is expensive. If inference is pretty cheap, the overhead here actually lowers throughput. When inference takes at least a few milliseconds, and device dispatch is more expensive, then the overhead is minimized and the amortization of dispatching to the device significantly improves throughput.

All inference is dispatched through `Batchinf`, which is cheap to clone and holds all state required to dispatch to the inference workers. This fits nicely into state in a framework like `axum`.

This implementation also restarts workers that have panicked, and gracefully handles termination on `SIGTERM`, when running on Kubernetes for example.

Observability is a first class citizen through the `BatcherMetrics` trait, where any observability backend can be used to emit system metrics.
