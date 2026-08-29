# Batchinf
This crate provides a framework agnostic approach to embedding ML inference batching into a service. I did not see a crate available that provided this functionality in a framework agnostic way. Some exists for ONNX, but not one where you could use either ONNX, candle, burn, etc.

The idea here is that as a user, you implement the core inference function, akin to predict in other frameworks, that takes a batch of records and performs inference on all of them, and this crate provides the plumming to plug that into you service.

Inference batching is a techninque that improves the utilization of a device, say GPU or TPU, by dispatching the inference call across a large number of inference examples. This improves both device utilization and throughput by amortizing the inference cost.

The core design is in `docs/design.md`.
