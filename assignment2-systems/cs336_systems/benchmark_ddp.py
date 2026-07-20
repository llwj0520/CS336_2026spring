from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.ddp import NaiveDDP
from cs336_systems.ddp import (
    FlatDDP,
    NaiveDDP,
    OverlapDDP,
)

WORLD_SIZE = 2
VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 128
LOCAL_BATCH_SIZE = 4
WARMUP_STEPS = 5
BENCHMARK_STEPS = 10
DDP_IMPLEMENTATION = "flat"

XL_CONFIG = {
    "d_model": 1600,
    "d_ff": 6400,
    "num_layers": 48,
    "num_heads": 25,
}


def setup_process_group(
    rank,
    world_size,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

    # rank 0 使用 cuda:0，rank 1 使用 cuda:1。
    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def create_model(device):
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=XL_CONFIG["d_model"],
        num_layers=XL_CONFIG["num_layers"],
        num_heads=XL_CONFIG["num_heads"],
        d_ff=XL_CONFIG["d_ff"],
    )

    return model.to(device)

def run_training_step(
    ddp_model,
    optimizer,
    input_ids,
    targets,
):
    # 让所有 rank 在相同位置开始训练步骤。
    # barrier 不计入训练步骤时间。
    dist.barrier()
    torch.cuda.synchronize()
    step_start = time.perf_counter()

    with torch.cuda.nvtx.range("zero_grad"):
        optimizer.zero_grad(
            set_to_none=True,
        )

    with torch.cuda.nvtx.range("forward_and_loss"):
        logits = ddp_model(input_ids)

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                VOCAB_SIZE,
            ),
            targets.reshape(-1),
        )

    # OverlapDDP 会在这个范围内通过 hook
    # 启动异步梯度通信。
    with torch.cuda.nvtx.range("backward"):
        loss.backward()

    # NaiveDDP 和 FlatDDP 的通信还没有开始。
    # 对 OverlapDDP 来说，大部分通信已经发生在 backward 中。
    torch.cuda.synchronize()
    communication_start = time.perf_counter()

    with torch.cuda.nvtx.range(
        "finish_gradient_synchronization"
    ):
        ddp_model.finish_gradient_synchronization()

    # 等待 GPU 通信操作真正完成。
    torch.cuda.synchronize()
    communication_end = time.perf_counter()

    with torch.cuda.nvtx.range("optimizer_step"):
        optimizer.step()

    torch.cuda.synchronize()
    step_end = time.perf_counter()

    # 完整训练步骤耗时。
    step_time_ms = (
        step_end - step_start
    ) * 1000

    # NaiveDDP/FlatDDP：梯度通信时间。
    # OverlapDDP：backward 结束后的剩余等待时间。
    communication_time_ms = (
        communication_end
        - communication_start
    ) * 1000

    return (
        loss,
        step_time_ms,
        communication_time_ms,
    )

def benchmark_worker(
    rank,
    world_size,
):
    setup_process_group(
        rank,
        world_size,
    )

    try:
        device = torch.device(
            "cuda",
            rank,
        )

        # 让不同 rank 生成不同的本地数据。
        torch.manual_seed(rank)
        torch.cuda.manual_seed(rank)

        model = create_model(device)

       # 选择要进行性能测试的 DDP 实现。
        if DDP_IMPLEMENTATION == "naive":
            ddp_model = NaiveDDP(model)
        elif DDP_IMPLEMENTATION == "flat":
            ddp_model = FlatDDP(model)
        elif DDP_IMPLEMENTATION == "overlap":
            ddp_model = OverlapDDP(model)
        else:
            raise ValueError(
                "DDP_IMPLEMENTATION must be "
                "'naive', 'flat', or 'overlap'"
            )

        optimizer = torch.optim.AdamW(
            ddp_model.parameters(),
            lr=1e-4,
        )

        input_ids = torch.randint(
            low=0,
            high=VOCAB_SIZE,
            size=(
                LOCAL_BATCH_SIZE,
                CONTEXT_LENGTH,
            ),
            device=device,
        )

        targets = torch.randint(
            low=0,
            high=VOCAB_SIZE,
            size=(
                LOCAL_BATCH_SIZE,
                CONTEXT_LENGTH,
            ),
            device=device,
        )

        # 预热阶段：执行训练，但不记录时间。
        for _ in range(WARMUP_STEPS):
            run_training_step(
                ddp_model,
                optimizer,
                input_ids,
                targets,
            )

        # 等待所有 rank 完成预热。
        dist.barrier()

        step_times_ms = []
        communication_times_ms = []

        # 进行多次正式训练
        for _ in range(BENCHMARK_STEPS):
            (loss,step_time_ms,communication_time_ms,) = run_training_step(
                ddp_model,
                optimizer,
                input_ids,
                targets,
            )

            step_times_ms.append(step_time_ms)
            communication_times_ms.append(
                communication_time_ms
            )

        # 计算多次测量的平均时间。
        mean_step_time_ms = (
            sum(step_times_ms)
            / len(step_times_ms)
        )

        mean_communication_time_ms = (
            sum(communication_times_ms)
            / len(communication_times_ms)
        )

        # 将当前 rank 的两个平均时间放到 GPU tensor 中。
        timings = torch.tensor(
            [
                mean_step_time_ms,
                mean_communication_time_ms,
            ],
            dtype=torch.float64,
            device=device,
        )

        # 取所有 rank 中最大的时间。
        # DDP 的整体速度由最慢的 rank 决定。
        dist.all_reduce(
            timings,
            op=dist.ReduceOp.MAX,
        )

        max_step_time_ms = timings[0].item()
        max_communication_time_ms = timings[1].item()

        if rank == 0:
            communication_ratio = (
                max_communication_time_ms
                / max_step_time_ms
                * 100
            )

            print(f"loss: {loss.item():.4f}")
            print(
                "mean step time: "
                f"{max_step_time_ms:.3f} ms"
            )
            print(
                "mean communication time: "
                f"{max_communication_time_ms:.3f} ms"
            )
            print(
                "communication ratio: "
                f"{communication_ratio:.2f}%"
            )
            print(
                f"DDP implementation: "
                f"{DDP_IMPLEMENTATION}"
            )

    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(
        benchmark_worker,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )