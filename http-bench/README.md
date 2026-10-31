# HTTP throughput benchmark

A simulated model served over HTTP with [axum], to show what batchinf does for throughput once a
real network stack and real clients are involved. `benches/throughput.rs` in the crate root
measures the same thing in-process.

The server exposes the same model twice, each with its own simulated device:

| endpoint                  | what it does                                               |
| ------------------------- | ---------------------------------------------------------- |
| `POST /predict`           | goes through batchinf, which batches concurrent requests   |
| `POST /predict/unbatched` | runs every request as its own one-item call on the device  |

One device call costs a fixed 5ms plus 50µs per item, the shape that makes batching pay off on a
GPU. Calls on a device are serialized by a mutex, so neither endpoint can win by running calls in
parallel. Requests and responses are JSON: `{"features": [0.1, 0.2]}` → `{"score": 0.3}`.

## Running

```sh
# terminal 1
cargo run --release

# terminal 2 (Python 3.10+, standard library only)
python3 harness.py --concurrency 256 --duration 10 --min-speedup 5
```

The harness benchmarks each endpoint in turn and prints a table:

```
256 concurrent clients, 5.0s per endpoint

endpoint              requests  errors     req/s  speedup    p50 ms    p99 ms
----------------------------------------------------------------------------
/predict/unbatched        1042       0       159     1.0x    1565.4    1586.7
/predict                 19456       0      3843    24.2x      63.5     159.8
```

`--min-speedup` makes it exit non-zero if `/predict` falls short of that multiple of the
unbatched throughput. Run `python3 harness.py --help` for the other options.

## Server configuration

Environment variables, with defaults:

| variable           | default | meaning                                      |
| ------------------ | ------- | -------------------------------------------- |
| `PORT`             | 3000    | listen port (binds 127.0.0.1)                |
| `BATCH_SIZE`       | 32      | batchinf batch size                          |
| `BATCH_TIMEOUT_MS` | 2       | how long a partial batch waits for more work |
| `POOL_SIZE`        | 1       | batchinf workers (all share the one device)  |
| `CALL_OVERHEAD_MS` | 5       | fixed cost of one device call                |
| `PER_ITEM_US`      | 50      | extra cost per item in a call                |

[axum]: https://github.com/tokio-rs/axum
