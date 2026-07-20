import argparse
import statistics
import time
from contextlib import nullcontext

import torch

from cs336_basics.model import BasicsTransformerLM


MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")

    parser.add_argument(
        "--memory-profile",
        choices=["forward", "training"],
        default=None,
        help="Profile either inference-only forward or a full training step.",
    )
    parser.add_argument(
        "--memory-snapshot",
        default="memory_snapshot.pickle",
        help="Path for the CUDA memory snapshot.",
    )
    return parser.parse_args()


def create_model(args, device):
    config = MODEL_CONFIGS[args.model_size]

    model = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=args.context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    )
    return model.to(device).train()


def autocast_context(use_mixed_precision):
    if use_mixed_precision:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def run_step(model, input_ids, run_backward, use_mixed_precision, optimizer=None):
    with autocast_context(use_mixed_precision):
        logits = model(input_ids)

        if run_backward:
            loss = logits.float().mean()

    if run_backward:
        loss.backward()

        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def make_input_ids(args, device):
    return torch.randint(
        low=0,
        high=10_000,
        size=(args.batch_size, args.context_length),
        device=device,
    )


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU.")

    device = torch.device("cuda")
    model = create_model(args, device)
    input_ids = make_input_ids(args, device)

    for _ in range(args.warmup):
        run_step(model, input_ids, args.backward, args.mixed_precision)
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)

    timings = []

    for _ in range(args.steps):
        torch.cuda.synchronize()
        start = time.perf_counter()

        run_step(model, input_ids, args.backward, args.mixed_precision)

        torch.cuda.synchronize()
        end = time.perf_counter()

        timings.append(end - start)
        model.zero_grad(set_to_none=True)

    mean_ms = statistics.mean(timings) * 1000
    std_ms = statistics.stdev(timings) * 1000 if len(timings) > 1 else 0.0

    mode = "forward + backward" if args.backward else "forward"
    precision = "BF16 mixed precision" if args.mixed_precision else "FP32"

    print(f"model: {args.model_size}")
    print(f"context length: {args.context_length}")
    print(f"mode: {mode}")
    print(f"precision: {precision}")
    print(f"mean: {mean_ms:.2f} ms")
    print(f"std: {std_ms:.2f} ms")


def profile_memory(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA memory profiling requires an NVIDIA CUDA GPU.")

    device = torch.device("cuda")
    model = create_model(args, device)
    input_ids = make_input_ids(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=100_000)

    try:
        if args.memory_profile == "forward":
            with torch.no_grad():
                with autocast_context(args.mixed_precision):
                    model(input_ids)
        else:
            run_step(
                model=model,
                input_ids=input_ids,
                run_backward=True,
                use_mixed_precision=args.mixed_precision,
                optimizer=optimizer,
            )

        torch.cuda.synchronize()
        torch.cuda.memory._dump_snapshot(args.memory_snapshot)

        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"memory profile mode: {args.memory_profile}")
        print(f"peak allocated memory: {peak_mb:.2f} MB")
        print(f"snapshot saved to: {args.memory_snapshot}")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    args = parse_args()

    if args.memory_profile is None:
        benchmark(args)
    else:
        profile_memory(args)