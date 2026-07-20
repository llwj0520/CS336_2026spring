from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path

import torch
import triton  # pyright: ignore[reportMissingImports]

from cs336_systems.triton_attention import FlashAttentionTriton


DEFAULT_SEQUENCE_LENGTHS = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
]
DEFAULT_DIMENSIONS = [16, 32, 64, 128]
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch attention against Triton FlashAttention-2."
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQUENCE_LENGTHS,
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=DEFAULT_DIMENSIONS,
    )
    parser.add_argument(
        "--dtypes",
        choices=DTYPES,
        nargs="+",
        default=list(DTYPES),
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("flash_attention_benchmark.csv"),
    )
    return parser.parse_args()


def regular_pytorch_attention(q, k, v, future_mask):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores.masked_fill_(future_mask, -1e6)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def clear_gradients(q, k, v):
    q.grad = None
    k.grad = None
    v.grad = None


def make_inputs(sequence_length, d, dtype):
    shape = (1, sequence_length, d)
    q = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad_output = torch.randn(shape, device="cuda", dtype=dtype)
    return q, k, v, grad_output


def make_future_mask(sequence_length):
    query_positions = torch.arange(sequence_length, device="cuda")[:, None]
    key_positions = torch.arange(sequence_length, device="cuda")[None, :]
    return query_positions < key_positions


def benchmark_implementation(
    implementation_name,
    sequence_length,
    d,
    dtype,
    warmup_ms,
    rep_ms,
):
    q, k, v, grad_output = make_inputs(sequence_length, d, dtype)
    future_mask = None

    if implementation_name == "pytorch":
        future_mask = make_future_mask(sequence_length)

        def attention():
            return regular_pytorch_attention(q, k, v, future_mask)

    elif implementation_name == "triton":

        def attention():
            return FlashAttentionTriton.apply(q, k, v, True)

    else:
        raise ValueError(f"Unknown implementation: {implementation_name}")

    # Trigger Triton and torch.compile before collecting timings.
    warmup_output = attention()
    warmup_output.backward(grad_output)
    clear_gradients(q, k, v)
    del warmup_output
    torch.cuda.synchronize()

    forward_ms = triton.testing.do_bench(
        attention,
        warmup=warmup_ms,
        rep=rep_ms,
    )

    backward_output = attention()

    def backward():
        clear_gradients(q, k, v)
        backward_output.backward(grad_output, retain_graph=True)

    backward_ms = triton.testing.do_bench(
        backward,
        warmup=warmup_ms,
        rep=rep_ms,
    )

    clear_gradients(q, k, v)

    def forward_backward():
        clear_gradients(q, k, v)
        output = attention()
        output.backward(grad_output)

    forward_backward_ms = triton.testing.do_bench(
        forward_backward,
        warmup=warmup_ms,
        rep=rep_ms,
    )

    return {
        "forward_ms": float(forward_ms),
        "backward_ms": float(backward_ms),
        "forward_backward_ms": float(forward_backward_ms),
    }


def benchmark_configuration(
    implementation_name,
    sequence_length,
    d,
    dtype_name,
    warmup_ms,
    rep_ms,
):
    dtype = DTYPES[dtype_name]

    try:
        timings = benchmark_implementation(
            implementation_name=implementation_name,
            sequence_length=sequence_length,
            d=d,
            dtype=dtype,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )
    except RuntimeError as error:
        if not isinstance(error, torch.cuda.OutOfMemoryError) and (
            "out of memory" not in str(error).lower()
        ):
            raise

        timings = {
            "forward_ms": None,
            "backward_ms": None,
            "forward_backward_ms": None,
        }
        status = "OOM"
    else:
        status = "ok"
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "sequence_length": sequence_length,
        "d": d,
        "dtype": dtype_name,
        "implementation": implementation_name,
        **timings,
        "status": status,
    }


def write_csv(results, output_path):
    fieldnames = [
        "sequence_length",
        "d",
        "dtype",
        "implementation",
        "forward_ms",
        "backward_ms",
        "forward_backward_ms",
        "status",
    ]

    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def format_timing(value):
    return "-" if value is None else f"{value:.3f}"


def print_markdown_table(results):
    print(
        "| seq_len | D | dtype | implementation | "
        "forward ms | backward ms | forward+backward ms | status |"
    )
    print("|---:|---:|---|---|---:|---:|---:|---|")

    for result in results:
        print(
            f"| {result['sequence_length']} "
            f"| {result['d']} "
            f"| {result['dtype']} "
            f"| {result['implementation']} "
            f"| {format_timing(result['forward_ms'])} "
            f"| {format_timing(result['backward_ms'])} "
            f"| {format_timing(result['forward_backward_ms'])} "
            f"| {result['status']} |"
        )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU.")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    results = []

    for dtype_name in args.dtypes:
        for d in args.dimensions:
            for sequence_length in args.sequence_lengths:
                for implementation_name in ("pytorch", "triton"):
                    print(
                        f"Benchmarking {implementation_name}: "
                        f"N={sequence_length}, D={d}, dtype={dtype_name}"
                    )

                    result = benchmark_configuration(
                        implementation_name=implementation_name,
                        sequence_length=sequence_length,
                        d=d,
                        dtype_name=dtype_name,
                        warmup_ms=args.warmup_ms,
                        rep_ms=args.rep_ms,
                    )
                    results.append(result)
                    write_csv(results, args.output)

    print_markdown_table(results)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
