from __future__ import annotations

import math

import torch
from cs336_systems.flash_backward import (
    compiled_flash_attention_backward,
)


Q_TILE_SIZE = 16
K_TILE_SIZE = 16


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        is_causal=False,
    ):
        """
        使用纯 PyTorch 按照 FlashAttention-2 Algorithm 1
        分块计算前向传播。

        q: [..., N_QUERIES, D]
        k: [..., N_KEYS, D]
        v: [..., N_KEYS, D]
        """
        if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
            raise ValueError("q, k and v must have at least 2 dimensions")

        if q.shape[:-2] != k.shape[:-2]:
            raise ValueError("q and k must have the same batch dimensions")

        if k.shape != v.shape:
            raise ValueError("k and v must have the same shape")

        if q.shape[-1] != k.shape[-1]:
            raise ValueError("q and k must have the same hidden dimension")

        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        d = q.shape[-1]

        scale = 1.0 / math.sqrt(d)

        # 将所有 batch 维度合并
        q_flat = q.reshape(-1, n_queries, d)
        k_flat = k.reshape(-1, n_keys, d)
        v_flat = v.reshape(-1, n_keys, d)

        batch_size = q_flat.shape[0]

        output_flat = torch.empty(
            (batch_size, n_queries, d),
            device=q.device,
            dtype=q.dtype,
        )

        # 使用 float32 保存 logsumexp，提高数值稳定性。
        lse_flat = torch.empty(
            (batch_size, n_queries),
            device=q.device,
            dtype=torch.float32,
        )

        # Algorithm 1 外层循环：遍历 Q tiles。
        for query_start in range(
            0,
            n_queries,
            Q_TILE_SIZE,
        ):
            query_end = min(
                query_start + Q_TILE_SIZE,
                n_queries,
            )

            q_tile = q_flat[
                :,
                query_start:query_end,
                :,
            ].float()

            query_tile_length = query_end - query_start

            # Algorithm 1 第 6 行。
            output_accumulator = torch.zeros(
                (
                    batch_size,
                    query_tile_length,
                    d,
                ),
                device=q.device,
                dtype=torch.float32,
            )

            denominator = torch.zeros(
                (
                    batch_size,
                    query_tile_length,
                ),
                device=q.device,
                dtype=torch.float32,
            )

            row_max = torch.full(
                (
                    batch_size,
                    query_tile_length,
                ),
                -torch.inf,
                device=q.device,
                dtype=torch.float32,
            )

            # Algorithm 1 内层循环：遍历 K/V tiles。
            for key_start in range(
                0,
                n_keys,
                K_TILE_SIZE,
            ):
                key_end = min(
                    key_start + K_TILE_SIZE,
                    n_keys,
                )

                k_tile = k_flat[
                    :,
                    key_start:key_end,
                    :,
                ].float()

                v_tile = v_flat[
                    :,
                    key_start:key_end,
                    :,
                ].float()

                # Algorithm 1 第 9 行：
                # S = QK^T / sqrt(D)
                scores = torch.matmul(
                    q_tile,
                    k_tile.transpose(-2, -1),
                ) * scale

                if is_causal:
                    query_positions = torch.arange(
                        query_start,
                        query_end,
                        device=q.device,
                    )

                    key_positions = torch.arange(
                        key_start,
                        key_end,
                        device=q.device,
                    )

                    causal_mask = (
                        query_positions[:, None]
                        >= key_positions[None, :]
                    )

                    scores = scores.masked_fill(
                        ~causal_mask.unsqueeze(0),
                        -1e6,
                    )

                # Algorithm 1 第 10 行：
                # m_new = max(m_old, rowmax(S))
                tile_row_max = scores.max(
                    dim=-1
                ).values

                new_row_max = torch.maximum(
                    row_max,
                    tile_row_max,
                )

                # 当最大值改变时，修正之前累积的结果。
                correction = torch.exp(
                    row_max - new_row_max
                )

                # Algorithm 1 第 11 行：计算当前 tile 的未归一化概率
                # P_tilde = exp(S - m_new)
                probabilities = torch.exp(
                    scores
                    - new_row_max.unsqueeze(-1)
                )

                # Algorithm 1 第 12 行：在线更新 softmax 分母
                # l_new = exp(m_old-m_new) * l_old + rowsum(P_tilde)
                denominator = (
                    correction * denominator
                    + probabilities.sum(dim=-1)
                )

                # Algorithm 1 第 13 行：
                # O_new = exp(m_old-m_new) * O_old
                #         + P_tilde @ V
                output_accumulator = (
                    correction.unsqueeze(-1)
                    * output_accumulator
                    + torch.matmul(
                        probabilities,
                        v_tile,
                    )
                )

                row_max = new_row_max

            # Algorithm 1 第 15 行。
            output_tile = (
                output_accumulator
                / denominator.unsqueeze(-1)
            )

            # Algorithm 1 第 16 行。
            lse_tile = (
                row_max
                + torch.log(denominator)
            )

            output_flat[
                :,
                query_start:query_end,
                :,
            ] = output_tile.to(q.dtype)

            lse_flat[
                :,
                query_start:query_end,
            ] = lse_tile

        output_shape = (
            *q.shape[:-2],
            n_queries,
            d,
        )

        lse_shape = (
            *q.shape[:-2],
            n_queries,
        )

        output = output_flat.view(output_shape)
        lse = lse_flat.view(lse_shape)

        # PDF 要求为 backward 保存 Q、K、V、O 和 L。
        ctx.save_for_backward(
            q,
            k,
            v,
            output,
            lse,
        )

        ctx.is_causal = is_causal
        ctx.scale = scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, lse = ctx.saved_tensors

        grad_q, grad_k, grad_v = (
            compiled_flash_attention_backward(
                q,
                k,
                v,
                output,
                grad_output,
                lse,
                ctx.is_causal,
                ctx.scale,
            )
        )

        # is_causal 是布尔值，不需要梯度。
        return grad_q, grad_k, grad_v, None