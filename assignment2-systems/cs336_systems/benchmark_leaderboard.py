from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

import cs336_basics.model as model_module
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.fsdp import FSDP
from cs336_systems.triton_attention import FlashAttentionTriton


WORLD_SIZE = 2
GLOBAL_BATCH_SIZE = 2
WARMUP_STEPS = 1
BENCHMARK_STEPS = 3

VOCAB_SIZE = 151_936
CONTEXT_LENGTH = 32_768
D_MODEL = 4_096
D_FF = 11_008
NUM_LAYERS = 34
NUM_HEADS = 32
COMPUTE_DTYPE = torch.bfloat16

#让多个 Python 进程组成一个分布式通信组，并让每个进程绑定对应的 GPU
def setup_process_group(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29504"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def install_triton_flash_attention() -> None:
    def flash_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        *leading_dimensions, n_queries, d_head = q.shape
        n_keys = k.shape[-2]

        q_flat = q.reshape(-1, n_queries, d_head).contiguous()
        k_flat = k.reshape(-1, n_keys, d_head).contiguous()
        v_flat = v.reshape(-1, n_keys, d_head).contiguous()

        output_flat = FlashAttentionTriton.apply(
            q_flat,
            k_flat,
            v_flat,
            mask is not None,
        )

        return output_flat.view(
            *leading_dimensions,
            n_queries,
            d_head,
        )

    # BasicsTransformerLM resolves this module-level function at runtime.
    model_module.scaled_dot_product_attention = flash_attention


def create_model(device: torch.device) -> FSDP:
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
    ).to(device)

    return FSDP(
        model,
        compute_dtype=COMPUTE_DTYPE,
    )


def run_training_step(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)
    loss = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        targets.reshape(-1),
        reduction="sum",
    )

    loss.backward()
    model.finish_gradient_synchronization()
    optimizer.step()

    return loss.detach()


def benchmark_worker(rank: int, world_size: int) -> None:
    setup_process_group(rank, world_size)

    try:
        device = torch.device("cuda", rank)
        install_triton_flash_attention()

        torch.manual_seed(rank)
        torch.cuda.manual_seed(rank)

        model = create_model(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
        )

        local_batch_size = GLOBAL_BATCH_SIZE // world_size
        input_ids = torch.randint(
            0,
            VOCAB_SIZE,
            (local_batch_size, CONTEXT_LENGTH),
            device=device,
        )
        targets = torch.randint(
            0,
            VOCAB_SIZE,
            (local_batch_size, CONTEXT_LENGTH),
            device=device,
        )

        for _ in range(WARMUP_STEPS):
            run_training_step(
                model,
                optimizer,
                input_ids,
                targets,
            )

        dist.barrier()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        global_step_times_ms: list[float] = []
        loss = None

        for _ in range(BENCHMARK_STEPS):
            dist.barrier()
            torch.cuda.synchronize(device)
            start = time.perf_counter()

            loss = run_training_step(
                model,
                optimizer,
                input_ids,
                targets,
            )

            torch.cuda.synchronize(device)
            local_time_ms = (time.perf_counter() - start) * 1_000
            step_time = torch.tensor(
                local_time_ms,
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(step_time, op=dist.ReduceOp.MAX)
            global_step_times_ms.append(step_time.item())

        peak_memory = torch.tensor(
            torch.cuda.max_memory_allocated(device) / 1024**3,
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)

        if rank == 0:
            mean_ms = sum(global_step_times_ms) / len(global_step_times_ms)
            best_ms = min(global_step_times_ms)
            print(f"mean training step: {mean_ms:.3f} ms")
            print(f"best training step: {best_ms:.3f} ms")
            print(f"peak allocated memory: {peak_memory.item():.3f} GiB")
            print(f"final loss: {loss.item():.4f}")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if GLOBAL_BATCH_SIZE % WORLD_SIZE != 0:
        raise ValueError("GLOBAL_BATCH_SIZE must be divisible by WORLD_SIZE")

    mp.spawn(
        benchmark_worker,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )
