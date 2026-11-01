# 0.1.2
- Notify callers of failures earlier when a predictor crashes, queued requests are returned immediately to the caller when the predictor crashes.
- Fix some behavior around crashing predictor and reflecting that state.
- Fix bug where the timeout is unbounded and the accumulator does not wake until the batch is set.
- Additional tests to cover the failure modes mentioned above.
- Add harness benchmarks on simulated predictor.
- Update CI to use `cargo nextest` given time based tests.
- Documentation cleanup.
# 0.1.1
- CI changes to not run CI when non source code changes.
- Handle panic in user defined code in the predictor thread.
