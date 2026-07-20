#优化器状态分片

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs: Any,
    ):
        # 记录真正要使用的优化器类型，如 AdamW
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        # 当前进程信息。
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # 记录每个参数由哪个 rank 负责。
        self._parameter_owners = {}

        # 记录每个 rank 当前负责的参数元素数量，
        # 后面用它尽量均衡地分配参数。
        self._shard_sizes = [
            0 for _ in range(self.world_size)
        ]

        # 稍后创建只包含本 rank 参数的本地优化器。
        self.local_optimizer = None

        # 初始化 torch.optim.Optimizer 基类。
        super().__init__(
            params,
            kwargs,
        )
        # 构造只属于当前 rank 的参数组。
        local_param_groups = []

        for param_group in self.param_groups:
            local_parameters = [
                parameter
                for parameter in param_group["params"]
                if self._parameter_owners[parameter] == self.rank
            ]

            # 当前参数组可能没有分配给这个 rank 的参数。
            if not local_parameters:
                continue

            local_param_group = {
                key: value
                for key, value in param_group.items()
                if key != "params"
            }
            local_param_group["params"] = (
                local_parameters
            )

            local_param_groups.append(
                local_param_group
            )

        # 只有本 rank 负责的参数才会进入本地优化器。
        if local_param_groups:
            self.local_optimizer = (
                self.optimizer_cls(
                    local_param_groups,
                    **self.optimizer_kwargs,
                )
            )

            # 对外暴露本地优化器真正持有的状态。
            self.state = self.local_optimizer.state
    
    def add_param_group(
        self,
        param_group: dict[str, Any],
    ):
        # 先让 PyTorch 基类检查并保存参数组。
        super().add_param_group(
            param_group
        )

        # super() 会把整理后的参数组放在列表最后。
        normalized_group = self.param_groups[-1]

        for parameter in normalized_group["params"]:
            # 找到当前负责参数量最少的 rank。
            owner_rank = min(
                range(self.world_size),
                key=lambda rank: self._shard_sizes[rank],
            )

            # 记录该参数由哪个 rank 更新。
            self._parameter_owners[parameter] = (
                owner_rank
            )

            # 使用参数元素数量更新该 rank 的负载。
            self._shard_sizes[owner_rank] += (
                parameter.numel()
            )

        # 如果本地优化器已经创建，说明这是训练期间
        # 新增加的参数组，需要同步加入本地优化器。
        if self.local_optimizer is not None:
            local_parameters = [
                parameter
                for parameter in normalized_group["params"]
                if self._parameter_owners[parameter]
                == self.rank
            ]

            if local_parameters:
                local_param_group = {
                    key: value
                    for key, value in normalized_group.items()
                    if key != "params"
                }
                local_param_group["params"] = (
                    local_parameters
                )

                self.local_optimizer.add_param_group(
                    local_param_group
                )

    def step(
        self,
        closure=None,
        **kwargs: Any,
    ):
        loss = None

        # 本地优化器只更新当前 rank 负责的参数。
        if self.local_optimizer is not None:
            loss = self.local_optimizer.step(
                closure=closure,
                **kwargs,
            )

        # 将每个参数从负责它的 rank 广播给所有其他 rank，即同步所有参数
        with torch.no_grad():
            for param_group in self.param_groups:
                for parameter in param_group["params"]:
                    owner_rank = (
                        self._parameter_owners[
                            parameter
                        ]
                    )

                    dist.broadcast(
                        parameter,
                        src=owner_rank,
                    )

        return loss