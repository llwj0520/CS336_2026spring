# 比较普通 AdamW 和分片 AdamW 的 GPU 内存。
# 比较两者每步训练时间。
# 分析它与 ZeRO Stage 1 的区别。

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.ddp import NaiveDDP
from cs336_systems.sharded_optimizer import (
    ShardedOptimizer,
)


WORLD_SIZE = 2
VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 128
LOCAL_BATCH_SIZE = 4
WARMUP_STEPS = 5
BENCHMARK_STEPS = 10

# 可选 "regular" 或 "sharded"。
OPTIMIZER_MODE = "sharded"

XL_CONFIG = {
    "d_model": 1600,
    "d_ff": 6400,
    "num_layers": 48,
    "num_heads": 25,
}

#让创建出来的两个进程互相认识并能够通信
def setup_process_group(rank,world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"

    # rank 0 使用 cuda:0，rank 1 使用 cuda:1
    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

def get_memory_stats(
    device,
):
    # CUDA 异步执行，读取前先等待 GPU 完成。
    torch.cuda.synchronize(device)

    bytes_per_gib = 1024**3

    return {
        "allocated_gib": (
            torch.cuda.memory_allocated(device)
            / bytes_per_gib
        ),
        "reserved_gib": (
            torch.cuda.memory_reserved(device)
            / bytes_per_gib
        ),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device)
            / bytes_per_gib
        ),
    }

def create_model(
    device,
):
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=XL_CONFIG["d_model"],
        num_layers=XL_CONFIG["num_layers"],
        num_heads=XL_CONFIG["num_heads"],
        d_ff=XL_CONFIG["d_ff"],
    )

    # 将模型参数移动到当前 rank 的 GPU。
    return model.to(device)

def run_training_step(
    ddp_model,
    optimizer,
    input_ids,
    targets,
):
    # 清空上一步的梯度。
    optimizer.zero_grad(
        set_to_none=True
    )

    # 前向传播
    logits = ddp_model(input_ids)

    # 计算损失。
    loss = F.cross_entropy(
        logits.reshape(
            -1,
            VOCAB_SIZE,
        ),
        targets.reshape(-1),
    )

    # 计算当前 rank 的本地梯度。
    loss.backward()

    # 在 rank 之间同步并平均梯度。
    ddp_model.finish_gradient_synchronization()

    # 普通优化器更新全部参数；
    # 分片优化器更新本地参数并广播。
    optimizer.step()

    return loss

def benchmark_worker(rank, world_size):
    setup_process_group(rank, world_size)

    try:
        device = torch.device("cuda", rank)
        # 不同 rank 使用不同的本地数据。
        torch.manual_seed(rank)
        torch.cuda.manual_seed(rank)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # 第一处：模型初始化后的显存。
        model = create_model(device)
        ddp_model = NaiveDDP(model)
        dist.barrier()

        memory_after_model = get_memory_stats(device)
        print(
            f"rank {rank}, after model initialization: "
            f"{memory_after_model}"
        )

        torch.cuda.reset_peak_memory_stats(device)

        # 选择普通优化器/优化器分片。
        if OPTIMIZER_MODE == "regular":
            optimizer = torch.optim.AdamW(
                ddp_model.parameters(),
                lr=1e-4,
            )
        elif OPTIMIZER_MODE == "sharded":
            optimizer = ShardedOptimizer(
                ddp_model.parameters(),
                torch.optim.AdamW,
                lr=1e-4,
            )
        else:
            raise ValueError(
                "OPTIMIZER_MODE must be "
                "'regular' or 'sharded'"
            )

        input_ids = torch.randint(
            0,
            VOCAB_SIZE,
            (LOCAL_BATCH_SIZE, CONTEXT_LENGTH),
            device=device,
        )
        targets = torch.randint(
            0,
            VOCAB_SIZE,
            (LOCAL_BATCH_SIZE, CONTEXT_LENGTH),
            device=device,
        )
        
        #梯度归零
        optimizer.zero_grad(set_to_none=True)
        #前向传播
        logits = ddp_model(input_ids)
        #损失函数
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
        )

        # 反向传播计算本地梯度，然后对各rank 之间求和取平均梯度值。
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        dist.barrier()

        # 第二处：optimizer.step() 前的显存。
        memory_before_step = get_memory_stats(device)
        print(
            f"rank {rank}, before optimizer step: "
            f"{memory_before_step}"
        )

        torch.cuda.reset_peak_memory_stats(device)

        # 普通 AdamW 更新全部参数；
        # 分片优化器更新本地参数并广播。
        optimizer.step()
        dist.barrier()

        # 第三处：optimizer.step() 后的显存。
        memory_after_step = get_memory_stats(device)
        print(
            f"rank {rank}, after optimizer step: "
            f"{memory_after_step}"
        )

        # 预热：不记录这些训练步骤的时间。
        for _ in range(WARMUP_STEPS):
            run_training_step(
                ddp_model,
                optimizer,
                input_ids,
                targets,
            )

        # 所有 rank 完成预热后再正式计时。
        dist.barrier()

        step_times_ms = []

        for _ in range(BENCHMARK_STEPS):
            # 不把等待其他 rank 的时间放进训练步骤计时。
            dist.barrier()
            torch.cuda.synchronize(device)

            step_start = time.perf_counter()

            loss = run_training_step(
                ddp_model,
                optimizer,
                input_ids,
                targets,
            )

            # 等待 GPU 计算和通信真正完成。
            torch.cuda.synchronize(device)

            step_time_ms = (
                time.perf_counter()
                - step_start
            ) * 1000

            step_times_ms.append(
                step_time_ms
            )

        mean_step_time_ms = (
            sum(step_times_ms)
            / len(step_times_ms)
        )

        # DDP 整体速度由最慢的 rank 决定。
        timing = torch.tensor(
            mean_step_time_ms,
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(
            timing,
            op=dist.ReduceOp.MAX,
        )

        max_mean_step_time_ms = (
            timing.item()
        )

        if rank == 0:
            print(
                f"optimizer mode: "
                f"{OPTIMIZER_MODE}"
            )
            print(
                "mean training step time: "
                f"{max_mean_step_time_ms:.3f} ms"
            )
            print(
                f"final loss: "
                f"{loss.item():.4f}"
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