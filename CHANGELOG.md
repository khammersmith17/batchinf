# 0.1.2
- Notify callers of failures earlier when a predictor crashes. Queued requests are returned immediately to the caller when the predictor crashes with `BatchinfError::InternalError`.
    - Previously, a batch of inference requests would wait for the `Predictor` to be rebuilt.
- A panic in a `BatcherMetrics` callback now marks the predictor `PredictorStatus::Dead`.
    - Previously, the predictor would continue to have state `Alive`.
- Fix potential hang on a dead predictor.
    - The accumulator now wakes up when the channel closes.
    - Previously, the crash would not have been observed by the accumulator until the batch was triggered.
- Additional tests to cover the failure modes mentioned above.
- `BatchinfError::InferenceError` no longer includes the predictor's error in its message; it is returned from `Error::source()` instead. Fixes double display when printed.
- Add harness benchmarks on simulated predictor.
- Update CI to use `cargo nextest` given timing based tests.
- Observability fixes:
    - `on_batch_trigger` only fires when the inference batch is handed off to the predictor. Batches failed due to a `Predictor` crash do not emit.
    - `on_queue_depth` now reports 0 when all inference requests that are queued are failed due to a `Predictor` crash.
- Documentation cleanup.
# 0.1.1
- CI changes to not run CI when non source code changes.
- Handle panic in user defined code in the predictor thread.
