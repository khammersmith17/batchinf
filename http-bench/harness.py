#!/usr/bin/env python3
"""Closed-loop HTTP load generator for the batchinf benchmark server.

Each of --concurrency clients holds one keep-alive connection, sends a request, waits for the
response, and sends the next, until --duration seconds have passed. Endpoints are benchmarked one
after another, and the table reports throughput relative to the unbatched endpoint.

Standard library only. HTTP/1.1 is spoken directly over asyncio streams, which keeps the client
cheap enough that it is not the bottleneck at a few thousand requests per second.

    python3 harness.py --concurrency 256 --duration 10
"""

import argparse
import asyncio
import json
import random
import resource
import sys
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

BASELINE = "/predict/unbatched"
# File descriptors needed beyond one socket per client: stdio, the event loop's selector, etc.
FD_HEADROOM = 64


@dataclass
class Result:
    path: str
    elapsed: float = 0.0
    latencies: list[float] = field(default_factory=list)
    errors: int = 0

    @property
    def throughput(self) -> float:
        return len(self.latencies) / self.elapsed

    def percentile(self, p: float) -> float:
        lat = sorted(self.latencies)
        return lat[round((len(lat) - 1) * p)]


def raise_fd_limit(needed: int) -> None:
    """
    Raises the soft open-file limit to `needed`, which is often 256 by default on macOS.
    Change/ignore this if this constraint is not relevant on your setup.

    Exits with a message if the hard limit is lower, since that takes a privileged change.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < needed:
        if hard != resource.RLIM_INFINITY and hard < needed:
            sys.exit(
                f"error: {needed} open files needed but the hard limit is {hard}; "
                "lower --concurrency or raise the limit"
            )
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))


def build_request(host: str, path: str, features: int) -> bytes:
    body = json.dumps({"features": [random.random() for _ in range(features)]}).encode()
    head = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "\r\n"
    )
    return head.encode() + body


async def read_response(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split(" ", 2)[1])
    length = 0
    for line in lines[1:]:
        name, _, value = line.partition(":")
        if name.lower() == "content-length":
            length = int(value)
    return status, await reader.readexactly(length)


async def client(
    host: str, port: int, request: bytes, deadline: float, result: Result
) -> None:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        while time.perf_counter() < deadline:
            sent = time.perf_counter()
            writer.write(request)
            await writer.drain()
            status, body = await read_response(reader)
            if status == 200 and "score" in json.loads(body):
                result.latencies.append(time.perf_counter() - sent)
            else:
                result.errors += 1
    finally:
        writer.close()


async def run_clients(
    host: str,
    port: int,
    request: bytes,
    deadline: float,
    result: Result,
    concurrency: int,
) -> None:
    # A TaskGroup cancels the remaining clients as soon as one fails, so an error ends the run
    # straight away instead of after the deadline.
    async with asyncio.TaskGroup() as tg:
        for _ in range(concurrency):
            tg.create_task(client(host, port, request, deadline, result))


async def run_benchmark(
    host: str, port: int, path: str, args: argparse.Namespace
) -> Result:
    request = build_request(f"{host}:{port}", path, args.features)
    if args.warmup > 0:
        await run_clients(
            host,
            port,
            request,
            time.perf_counter() + args.warmup,
            Result(path),
            args.concurrency,
        )
    result = Result(path)
    start = time.perf_counter()
    await run_clients(
        host, port, request, start + args.duration, result, args.concurrency
    )
    # Measured to the last response, so requests still in flight at the deadline count.
    result.elapsed = time.perf_counter() - start
    return result


def print_table(results: list[Result], args: argparse.Namespace) -> None:
    baseline = next((r.throughput for r in results if r.path == BASELINE), None)
    print(f"\n{args.concurrency} concurrent clients, {args.duration}s per endpoint\n")
    print(
        f"{'endpoint':<20} {'requests':>9} {'errors':>7} {'req/s':>9} "
        f"{'speedup':>8} {'p50 ms':>9} {'p99 ms':>9}"
    )
    print("-" * 76)
    for r in results:
        speedup = f"{r.throughput / baseline:.1f}x" if baseline else "-"
        print(
            f"{r.path:<20} {len(r.latencies):>9} {r.errors:>7} {r.throughput:>9.0f} "
            f"{speedup:>8} {r.percentile(0.5) * 1e3:>9.1f} {r.percentile(0.99) * 1e3:>9.1f}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="http://127.0.0.1:3000")
    parser.add_argument("--endpoints", nargs="+", default=[BASELINE, "/predict"])
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument(
        "--duration", type=float, default=10.0, help="seconds per endpoint"
    )
    parser.add_argument(
        "--warmup", type=float, default=1.0, help="seconds per endpoint"
    )
    parser.add_argument("--features", type=int, default=16, help="floats per request")
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=0.0,
        help="fail unless /predict reaches this multiple of the baseline",
    )
    args = parser.parse_args()

    url = urlsplit(args.url)
    host, port = url.hostname or "127.0.0.1", url.port or 80
    raise_fd_limit(args.concurrency + FD_HEADROOM)

    results = []
    for path in args.endpoints:
        print(f"benchmarking {path} ...", flush=True)
        results.append(asyncio.run(run_benchmark(host, port, path, args)))
    print_table(results, args)

    if any(r.errors for r in results):
        print("error: some requests failed", file=sys.stderr)
        return 1
    if args.min_speedup:
        by_path = {r.path: r.throughput for r in results}
        speedup = by_path["/predict"] / by_path[BASELINE]
        if speedup < args.min_speedup:
            print(
                f"error: /predict reached {speedup:.1f}x, expected {args.min_speedup}x",
                file=sys.stderr,
            )
            return 1
        print(f"ok: /predict is {speedup:.1f}x the unbatched throughput")
    return 0


if __name__ == "__main__":
    sys.exit(main())
