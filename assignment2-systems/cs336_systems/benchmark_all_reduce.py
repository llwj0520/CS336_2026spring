from __future__ import annotations

import argparse
import csv
import socket
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


DEFAULT_WORLD_SIZES = [2, 4, 6]
DEFAULT_SIZES_MB = [1, 10, 100, 1024]
CSV_FIELDNAMES = [
    "backend",
    "world_size",
    "size_mb",
    "num_elements",
    "mean_ms",
    "std_ms",
    "min_ms",
    "max_ms",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark single-node distributed all-reduce."
    )
    parser.add_argument(
        "--world-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_WORLD_SIZES,
    )
    parser.add_argument(
        "--sizes-mb",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES_MB,
        help="Tensor sizes in MiB; use 1024 for 1 GiB.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--backend",
        choices=["auto", "gloo", "nccl"],
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("all_reduce_benchmark.csv"),
    )
    return parser.parse_args()


def select_backend(requested_backend):
    if requested_backend == "auto":
        return "nccl" if torch.cuda.is_available() else "gloo"
    return requested_backend


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return server_socket.getsockname()[1]


def synchronize(backend):
    if backend == "nccl":
        torch.cuda.synchronize()


def benchmark_size(
    rank,
    world_size,
    backend,
    size_mb,
    warmup,
    repetitions,
    device,
):
    size_bytes = size_mb * 1024**2
    num_elements = size_bytes // torch.tensor([], dtype=torch.float32).element_size()
    data = torch.ones(num_elements, dtype=torch.float32, device=device)

    for _ in range(warmup):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        synchronize(backend)

    local_timings = []

    for _ in range(repetitions):
        dist.barrier()
        synchronize(backend)

        start = time.perf_counter()
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        synchronize(backend)
        end = time.perf_counter()

        local_timings.append((end - start) * 1000)

    timings_by_rank = [None for _ in range(world_size)]
    dist.all_gather_object(timings_by_rank, local_timings)

    if rank != 0:
        return None

    all_timings = [
        timing
        for rank_timings in timings_by_rank
        for timing in rank_timings
    ]

    return {
        "backend": backend,
        "world_size": world_size,
        "size_mb": size_mb,
        "num_elements": num_elements,
        "mean_ms": statistics.mean(all_timings),
        "std_ms": statistics.stdev(all_timings) if len(all_timings) > 1 else 0.0,
        "min_ms": min(all_timings),
        "max_ms": max(all_timings),
    }


def append_result(output_path, result):
    with output_path.open("a", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDNAMES)
        writer.writerow(result)


def distributed_worker(
    rank,
    world_size,
    backend,
    sizes_mb,
    warmup,
    repetitions,
    master_port,
    output_path,
):
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        for size_mb in sizes_mb:
            result = benchmark_size(
                rank=rank,
                world_size=world_size,
                backend=backend,
                size_mb=size_mb,
                warmup=warmup,
                repetitions=repetitions,
                device=device,
            )

            if rank == 0:
                append_result(Path(output_path), result)
                print(
                    f"backend={backend}, world_size={world_size}, "
                    f"size={size_mb} MiB, mean={result['mean_ms']:.3f} ms"
                )

            del result
            if backend == "nccl":
                torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()


def initialize_output(output_path):
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()


def read_results(output_path):
    with output_path.open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    for row in rows:
        for key in ("mean_ms", "std_ms", "min_ms", "max_ms"):
            row[key] = float(row[key])
    return rows


def print_markdown_table(results):
    print("| backend | processes | size | mean ms | std ms | min ms | max ms |")
    print("|---|---:|---:|---:|---:|---:|---:|")

    for result in results:
        print(
            f"| {result['backend']} "
            f"| {result['world_size']} "
            f"| {result['size_mb']} MiB "
            f"| {result['mean_ms']:.3f} "
            f"| {result['std_ms']:.3f} "
            f"| {result['min_ms']:.3f} "
            f"| {result['max_ms']:.3f} |"
        )


def main():
    args = parse_args()
    backend = select_backend(args.backend)

    if backend == "nccl":
        available_gpus = torch.cuda.device_count()
        required_gpus = max(args.world_sizes)
        if available_gpus < required_gpus:
            raise RuntimeError(
                f"Requested up to {required_gpus} processes, "
                f"but only {available_gpus} CUDA GPUs are available."
            )

    output_path = args.output.resolve()
    initialize_output(output_path)

    for world_size in args.world_sizes:
        master_port = find_free_port()
        mp.spawn(
            distributed_worker,
            args=(
                world_size,
                backend,
                args.sizes_mb,
                args.warmup,
                args.repetitions,
                master_port,
                str(output_path),
            ),
            nprocs=world_size,
            join=True,
        )

    results = read_results(output_path)
    print_markdown_table(results)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
