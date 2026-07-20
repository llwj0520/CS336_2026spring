'''
shared_optimizer只分片 AdamW 状态
模型参数：完整
梯度：完整
优化器状态：分片

FSDP 进一步变成：
模型参数：分片
梯度：分片
优化器状态：自然分片
'''

from __future__ import annotations

import torch
import torch.distributed as dist

from cs336_basics.model import Embedding, Linear


class FSDP(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self._sharded_parameter_info = {}
        self._ordered_sharded_infos = []

        shardable_modules = [
            submodule
            for submodule in self.module.modules()
            if isinstance(submodule, (Linear, Embedding))
        ]

        for index, submodule in enumerate(shardable_modules):
            info = self._prepare_sharded_module(
                submodule,
                index,
            )
            self._ordered_sharded_infos.append(info)

        sharded_parameters = set(
            self._sharded_parameter_info
        )
        self._replicated_parameters = [
            parameter
            for parameter in self.module.parameters()
            if parameter not in sharded_parameters
        ]

    def _prepare_sharded_module(
        self,
        submodule,
        index,
    ):
        parameter = submodule.weight

        if parameter not in self._sharded_parameter_info:
            full_shape = tuple(parameter.shape)
            full_numel = parameter.numel()
            shard_numel = (
                full_numel + self.world_size - 1
            ) // self.world_size
            padded_numel = shard_numel * self.world_size

            with torch.no_grad():
                dist.broadcast(parameter, src=0)

                padded = torch.zeros(
                    padded_numel,
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                padded[:full_numel].copy_(
                    parameter.detach().reshape(-1)
                )

                start = self.rank * shard_numel
                parameter.data = padded[
                    start : start + shard_numel
                ].clone()

            info = {
                "parameter": parameter,
                "full_shape": full_shape,
                "full_numel": full_numel,
                "shard_numel": shard_numel,
            }
            self._sharded_parameter_info[parameter] = info

            parameter.register_post_accumulate_grad_hook(
                self._make_gradient_hook(info)
            )
        else:
            info = self._sharded_parameter_info[parameter]

        submodule.register_forward_pre_hook(
            self._make_forward_pre_hook(info)
        )
        submodule.register_forward_hook(
            self._make_forward_post_hook(
                info,
                index,
            )
        )
        submodule.register_full_backward_pre_hook(
            self._make_backward_pre_hook(info)
        )
        submodule.register_full_backward_hook(
            self._make_backward_post_hook(index)
        )

        return info

    def _prefetch_weight(self, info):
        parameter = info["parameter"]

        # CPU 测试继续使用同步路径。
        if not parameter.is_cuda:
            return

        if (
            "prefetch_handle" in info
            or "saved_local_shard" in info
        ):
            return

        communication_dtype = (
            self.compute_dtype
            if self.compute_dtype is not None
            else parameter.dtype
        )

        local_shard = parameter.data.to(
            communication_dtype
        ).contiguous()

        gathered_flat = torch.empty(
            info["shard_numel"] * self.world_size,
            dtype=communication_dtype,
            device=parameter.device,
        )

        handle = dist.all_gather_into_tensor(
            gathered_flat,
            local_shard,
            async_op=True,
        )

        # 通信完成前保留相关对象。
        info["prefetch_input"] = local_shard
        info["prefetch_buffer"] = gathered_flat
        info["prefetch_handle"] = handle

    def _prefetch_index(self, index):
        if 0 <= index < len(
            self._ordered_sharded_infos
        ):
            self._prefetch_weight(
                self._ordered_sharded_infos[index]
            )

    def _all_gather_weight(self, info):
        parameter = info["parameter"]

        if "prefetch_handle" in info:
            info["prefetch_handle"].wait()

            full_flat = info.pop(
                "prefetch_buffer"
            )
            info.pop("prefetch_input")
            info.pop("prefetch_handle")
        else:
            communication_dtype = (
                self.compute_dtype
                if self.compute_dtype is not None
                else parameter.dtype
            )

            local_shard = parameter.data.to(
                communication_dtype
            )

            gathered_shards = [
                torch.empty_like(local_shard)
                for _ in range(self.world_size)
            ]

            dist.all_gather(
                gathered_shards,
                local_shard,
            )
            full_flat = torch.cat(gathered_shards)

        return full_flat[
            : info["full_numel"]
        ].view(info["full_shape"])

    def _materialize_full_weight(self, info):
        parameter = info["parameter"]
        info["saved_local_shard"] = parameter.data
        parameter.data = self._all_gather_weight(info)

    def _reshard_weight(self, info):
        parameter = info["parameter"]
        parameter.data = info.pop(
            "saved_local_shard"
        )

    def _make_forward_pre_hook(self, info):
        def hook(module, inputs):
            self._materialize_full_weight(info)

        return hook

    def _make_forward_post_hook(
        self,
        info,
        index,
    ):
        def hook(module, inputs, output):
            self._reshard_weight(info)

            # 第 i 层完成后预取第 i+2 层。
            self._prefetch_index(index + 2)

        return hook

    def _make_backward_pre_hook(self, info):
        def hook(module, grad_output):
            self._materialize_full_weight(info)

        return hook

    def _make_backward_post_hook(self, index):
        def hook(module, grad_input, grad_output):
            # backward 顺序相反，因此预取 i-2。
            self._prefetch_index(index - 2)

        return hook

    def _reduce_scatter_gradient(
        self,
        full_gradient,
        info,
    ):
        parameter = info["parameter"]
        padded_gradient = torch.zeros(
            info["shard_numel"] * self.world_size,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        padded_gradient[: info["full_numel"]].copy_(
            full_gradient.detach().reshape(-1).to(
                parameter.dtype
            )
        )

        local_gradient = torch.empty(
            info["shard_numel"],
            dtype=parameter.dtype,
            device=parameter.device,
        )

        if dist.get_backend() == "gloo":
            # CPU 测试使用等价的 all-reduce + slice。
            dist.all_reduce(
                padded_gradient,
                op=dist.ReduceOp.SUM,
            )
            start = self.rank * info["shard_numel"]
            local_gradient.copy_(
                padded_gradient[
                    start : start + info["shard_numel"]
                ]
            )
        else:
            handle = dist.reduce_scatter_tensor(
                local_gradient,
                padded_gradient,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            handle.wait()

        local_gradient.div_(self.world_size)
        return local_gradient

    def _make_gradient_hook(self, info):
        def hook(parameter):
            full_gradient = parameter.grad

            # 恢复 FP32 主权重分片。
            self._reshard_weight(info)

            if full_gradient is not None:
                parameter.grad = (
                    self._reduce_scatter_gradient(
                        full_gradient,
                        info,
                    )
                )

        return hook

    def _start_backward_prefetch(self, gradient):
        count = len(self._ordered_sharded_infos)

        # backward 从最后一层开始。
        self._prefetch_index(count - 1)
        self._prefetch_index(count - 2)

        return gradient

    def forward(self, *inputs, **kwargs):
        # 前两层没有更早的层帮助预取。
        self._prefetch_index(0)
        self._prefetch_index(1)

        output = self.module(*inputs, **kwargs)

        # backward 开始时预取最后两层。
        if (
            isinstance(output, torch.Tensor)
            and output.requires_grad
        ):
            output.register_hook(
                self._start_backward_prefetch
            )

        return output

    def finish_gradient_synchronization(self):
        # 同步 RMSNorm 等未分片的小参数。
        handles = []

        for parameter in self._replicated_parameters:
            if parameter.grad is None:
                continue

            parameter.grad.div_(self.world_size)

            handles.append(
                dist.all_reduce(
                    parameter.grad,
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                )
            )

        for handle in handles:
            handle.wait()

    def gather_full_params(self):
        full_parameters = {}

        for name, parameter in self.module.named_parameters():
            info = self._sharded_parameter_info.get(
                parameter
            )

            if info is None:
                full_parameters[name] = (
                    parameter.detach().clone()
                )
                continue

            local_shard = parameter.detach()
            gathered_shards = [
                torch.empty_like(local_shard)
                for _ in range(self.world_size)
            ]

            dist.all_gather(
                gathered_shards,
                local_shard,
            )

            full_flat = torch.cat(gathered_shards)
            full_parameters[name] = (
                full_flat[: info["full_numel"]]
                .view(info["full_shape"])
                .clone()
            )

        return full_parameters