from __future__ import annotations

import torch
import torch.distributed as dist
from torch._utils import (
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
)

class NaiveDDP(torch.nn.Module):
    def __init__(self, module:torch.nn.Module,):
        super(). __init__()
        self.module=module
        
        #注意：broadcast rank0的时候不用计算梯度，同步初始化权重
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(
                    parameter,
                    src=0,
                )
    
    def forward( self,*inputs,**kwargs,):
        return self.module(
            *inputs,
            **kwargs,
        )
    
    def finish_gradient_synchronization(self):
        
        #拿到进程数量
        world_size=dist.get_world_size()
        #逐个处理模型参数
        for parameter in self.module.parameters():
            #如果参数没有梯度，就跳过当前参数。
            if parameter.grad is None:
                continue
            
            #将所有 rank 的梯度求和，并把求和结果写回每个 rank 的 parameter.grad
            dist.all_reduce(
                parameter.grad,
                op=dist.ReduceOp.SUM,
            )
            
            #计算平均梯度:/world_size
            parameter.grad.div_(world_size)


class FlatDDP(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
    ):
        super().__init__()
        self.module = module

        # 所有 rank 使用 rank 0 的初始参数。
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(
                    parameter,
                    src=0,
                )

    def forward(
        self,
        *inputs,
        **kwargs,
    ):
        return self.module(
            *inputs,
            **kwargs,
        )
    
    def finish_gradient_synchronization(self):
        # 只收集本次 backward 中实际产生了梯度的参数。
        parameters_with_grad = [
            parameter
            for parameter in self.module.parameters()
            if parameter.grad is not None
        ]

        # 模型可能没有可训练参数，避免 flatten 空列表。
        if not parameters_with_grad:
            return

        gradients = [
            parameter.grad
            for parameter in parameters_with_grad
        ]

        # 将不同形状的梯度拼成一个一维 tensor
        flat_gradient = _flatten_dense_tensors(
            gradients
        )

        # 将所有 rank 的扁平梯度相加。
        dist.all_reduce(
            flat_gradient,
            op=dist.ReduceOp.SUM,
        )

        # 得到所有 rank 的平均梯度。
        flat_gradient.div_(
            dist.get_world_size()
        )
        
        # 按照原始梯度的形状拆分扁平梯度。
        synchronized_gradients = (
            _unflatten_dense_tensors(
                flat_gradient,
                gradients,
            )
        )

        # 将同步后的平均梯度复制回对应参数。
        for parameter, synchronized_gradient in zip(
            parameters_with_grad,
            synchronized_gradients,
        ):
            parameter.grad.copy_(
                synchronized_gradient
            )

class OverlapDDP(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
    ):
        super().__init__()
        self.module = module

        # 保存异步 all-reduce 返回的通信句柄。
        self._communication_handles = []

        # 所有 rank 使用相同的初始参数。
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(
                    parameter,
                    src=0,
                )

        # 为每个需要梯度的参数注册 backward hook。
        for parameter in self.module.parameters():
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(
                    self._on_gradient_ready
                )

    def forward(
        self,
        *inputs,
        **kwargs,
    ):
        return self.module(
            *inputs,
            **kwargs,
        )

    def _on_gradient_ready(
        self,
        parameter,
    ):
        # 正常情况下 hook 被调用时，parameter.grad 已经存在。
        if parameter.grad is None:
            return

        # 提前除以 world_size。
        # all_reduce 求和后得到的就是平均梯度。
        parameter.grad.div_(
            dist.get_world_size()
        )

        # async_op=True：启动通信后立即返回，
        # 让 backward 继续计算其他参数的梯度。
        handle = dist.all_reduce(
            parameter.grad,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )

        # 保存通信句柄，之后必须等待通信完成。
        self._communication_handles.append(
            handle
        )

    def finish_gradient_synchronization(self):
            # 等待本次 backward 启动的所有异步通信。
            for handle in self._communication_handles:
                handle.wait()

            # 当前训练步骤结束，清空句柄，
            # 避免下一步再次等待旧的通信。
            self._communication_handles.clear()