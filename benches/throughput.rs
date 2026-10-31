//! Throughput benchmark: does dynamic batching actually help when inference is expensive?
//!
//! Every scenario runs against the same simulated device. A call on the device costs a large
//! fixed overhead (kernel launch, host-to-device copy, ...) plus a small per-item cost, which is
//! the shape that makes batching pay off on a GPU/TPU. The device is a single shared resource
//! guarded by a mutex, so no scenario can gain throughput by running calls in parallel; the only
//! lever is how many items each call carries.
//!
//! The load is closed-loop: `CLIENTS` concurrent callers each send one request, wait for the
//! answer, and send the next, until the run's duration has passed.
//!
//! Scenarios:
//! - `unbatched`: no harness. Each request takes the device for a call of one item. This is the
//!   naive server that shares one model across requests.
//! - `batchinf b=1`: the harness with `batch_size = 1`, which shows the harness's own overhead.
//! - `batchinf b=N`: the harness with larger batches.
//!
//! Run with `cargo bench --bench throughput`. `BENCH_SECS` sets each scenario's duration
//! (default 2). The run fails if batching does not beat the unbatched baseline by `MIN_SPEEDUP`.

use batchinf::{BatcherConfig, Predictor, PredictorFactory, get_batcher};
use std::{
    convert::Infallible,
    num::{NonZeroU8, NonZeroU32},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

/// Fixed cost of one call on the device, paid regardless of batch size.
const CALL_OVERHEAD: Duration = Duration::from_millis(5);
/// Extra cost per item in the call.
const PER_ITEM_COST: Duration = Duration::from_micros(50);
/// Concurrent closed-loop callers.
const CLIENTS: usize = 256;
/// How long a not-yet-full batch waits for more requests.
const BATCH_TIMEOUT: Duration = Duration::from_millis(2);
/// Batch sizes to run through the harness.
const BATCH_SIZES: [u32; 4] = [1, 8, 32, 64];
/// The smallest speedup over `unbatched` that the `b=32` scenario must reach. The cost model
/// above allows roughly 25x, so this leaves wide margin for scheduler noise.
const MIN_SPEEDUP: f64 = 5.0;
const CHECKED_BATCH_SIZE: u32 = 32;

/// A single accelerator shared by every caller. Holding the lock means owning the device.
#[derive(Clone, Default)]
struct Device(Arc<Mutex<()>>);

impl Device {
    fn run(&self, items: usize) {
        let _guard = self.0.lock().unwrap();
        // Sleep rather than spin: the host thread waits on the device, it does not compute.
        std::thread::sleep(CALL_OVERHEAD + PER_ITEM_COST * items as u32);
    }
}

/// A model whose only behavior is to occupy the device for as long as a real call would.
struct SimulatedModel {
    device: Device,
}

impl Predictor for SimulatedModel {
    type Input = u64;
    type Output = u64;
    type Error = Infallible;

    fn predict_batch(&mut self, inp: Vec<u64>) -> Result<Vec<u64>, Infallible> {
        self.device.run(inp.len());
        Ok(inp)
    }
}

struct Report {
    name: String,
    completed: usize,
    elapsed: Duration,
    latencies: Vec<Duration>,
}

impl Report {
    fn new(name: String, elapsed: Duration, per_client: Vec<Vec<Duration>>) -> Self {
        let mut latencies: Vec<Duration> = per_client.into_iter().flatten().collect();
        latencies.sort_unstable();
        Self {
            name,
            completed: latencies.len(),
            elapsed,
            latencies,
        }
    }

    fn throughput(&self) -> f64 {
        self.completed as f64 / self.elapsed.as_secs_f64()
    }

    fn percentile(&self, p: f64) -> Duration {
        let idx = ((self.latencies.len() - 1) as f64 * p).round() as usize;
        self.latencies[idx]
    }
}

/// Runs `CLIENTS` closed-loop callers until `duration` passes. `call` performs one request.
async fn drive<F, Fut>(name: String, duration: Duration, call: F) -> Report
where
    F: Fn(u64) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = ()> + Send,
{
    let start = Instant::now();
    let deadline = start + duration;
    let clients: Vec<_> = (0..CLIENTS)
        .map(|c| {
            let call = call.clone();
            tokio::spawn(async move {
                let mut latencies = Vec::new();
                let mut i = 0_u64;
                while Instant::now() < deadline {
                    let sent = Instant::now();
                    call(c as u64 * 1_000_000 + i).await;
                    latencies.push(sent.elapsed());
                    i += 1;
                }
                latencies
            })
        })
        .collect();

    let mut per_client = Vec::with_capacity(CLIENTS);
    for client in clients {
        per_client.push(client.await.unwrap());
    }
    // Measured to the last completion, so requests still in flight at the deadline count.
    Report::new(name, start.elapsed(), per_client)
}

async fn run_unbatched(duration: Duration) -> Report {
    let device = Device::default();
    drive("unbatched".into(), duration, move |x| {
        let device = device.clone();
        async move {
            let out = tokio::task::spawn_blocking(move || {
                device.run(1);
                x
            })
            .await
            .unwrap();
            assert_eq!(out, x);
        }
    })
    .await
}

async fn run_batched(batch_size: u32, duration: Duration) -> Report {
    let device = Device::default();
    let factory: PredictorFactory<SimulatedModel, Infallible> = Arc::new(move |_| {
        Ok(SimulatedModel {
            device: device.clone(),
        })
    });
    let batch_size = NonZeroU32::new(batch_size).unwrap();
    let config = BatcherConfig::builder(batch_size, BATCH_TIMEOUT)
        .pool_size(NonZeroU8::MIN)
        // Room for the next batch to accumulate while the current one is on the device.
        .queue_size(batch_size.saturating_mul(NonZeroU32::new(2).unwrap()))
        .build();
    // Startup is outside the measured window.
    let batcher = get_batcher(factory, config).await.unwrap();

    let name = format!("batchinf b={batch_size}");
    drive(name, duration, move |x| {
        let batcher = batcher.clone();
        async move {
            assert_eq!(batcher.predict(x).await.unwrap(), x);
        }
    })
    .await
}

fn print_table(reports: &[Report]) {
    let baseline = reports[0].throughput();
    println!();
    println!(
        "device: {CALL_OVERHEAD:?} per call + {PER_ITEM_COST:?} per item, \
         {CLIENTS} concurrent clients, batch timeout {BATCH_TIMEOUT:?}"
    );
    println!();
    println!(
        "{:<16} {:>10} {:>12} {:>9} {:>11} {:>11}",
        "scenario", "requests", "req/s", "speedup", "p50", "p99"
    );
    println!("{}", "-".repeat(74));
    for r in reports {
        println!(
            "{:<16} {:>10} {:>12.0} {:>8.1}x {:>11.2?} {:>11.2?}",
            r.name,
            r.completed,
            r.throughput(),
            r.throughput() / baseline,
            r.percentile(0.50),
            r.percentile(0.99),
        );
    }
    println!();
}

fn main() {
    let secs: u64 = std::env::var("BENCH_SECS")
        .ok()
        .map(|s| {
            s.parse()
                .expect("BENCH_SECS must be a whole number of seconds")
        })
        .unwrap_or(2);
    let duration = Duration::from_secs(secs);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        // The unbatched baseline parks one blocking thread per in-flight request.
        .max_blocking_threads(CLIENTS)
        .build()
        .unwrap();

    let reports = rt.block_on(async {
        let mut reports = vec![run_unbatched(duration).await];
        for b in BATCH_SIZES {
            reports.push(run_batched(b, duration).await);
        }
        reports
    });

    print_table(&reports);

    let baseline = reports[0].throughput();
    let checked = reports
        .iter()
        .find(|r| r.name == format!("batchinf b={CHECKED_BATCH_SIZE}"))
        .unwrap();
    let speedup = checked.throughput() / baseline;
    assert!(
        speedup >= MIN_SPEEDUP,
        "{} reached {speedup:.1}x over unbatched, expected at least {MIN_SPEEDUP}x",
        checked.name
    );
    println!(
        "ok: {} is {speedup:.1}x the unbatched throughput",
        checked.name
    );
}
