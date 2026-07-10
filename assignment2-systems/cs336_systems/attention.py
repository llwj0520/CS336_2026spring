from __future__ import annotations

import math

import torch


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention and save values for backward."""
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if is_causal:
            n_queries, n_keys = scores.shape[-2:]
            query_positions = torch.arange(n_queries, device=scores.device)[:, None]
            key_positions = torch.arange(n_keys, device=scores.device)[None, :]
            scores = scores.masked_fill(query_positions < key_positions, -1e6)

        lse = torch.logsumexp(scores, dim=-1)
        probabilities = torch.exp(scores - lse.unsqueeze(-1))
        output = torch.matmul(probabilities, v)

        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = is_causal
        ctx.scale = scale
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, lse = ctx.saved_tensors

        scores = torch.matmul(q, k.transpose(-2, -1)) * ctx.scale
        if ctx.is_causal:
            n_queries, n_keys = scores.shape[-2:]
            query_positions = torch.arange(n_queries, device=scores.device)[:, None]
            key_positions = torch.arange(n_keys, device=scores.device)[None, :]
            scores = scores.masked_fill(query_positions < key_positions, -1e6)

        probabilities = torch.exp(scores - lse.unsqueeze(-1))
        grad_v = torch.matmul(probabilities.transpose(-2, -1), grad_output)
        grad_probabilities = torch.matmul(grad_output, v.transpose(-2, -1))
        row_sum = (grad_output * output).sum(dim=-1, keepdim=True)
        grad_scores = probabilities * (grad_probabilities - row_sum)
        grad_q = torch.matmul(grad_scores, k) * ctx.scale
        grad_k = torch.matmul(grad_scores.transpose(-2, -1), q) * ctx.scale

        return grad_q, grad_k, grad_v, None
