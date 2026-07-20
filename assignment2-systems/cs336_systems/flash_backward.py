from __future__ import annotations

import torch


def flash_attention_backward(
    q,
    k,
    v,
    output,
    grad_output,
    lse,
    is_causal,
    scale,
):
    """
    根据 PDF 公式（13）到（19）计算 FlashAttention backward。

    输入：
        q:           [batch_size, n_queries, D]
        k:           [batch_size, n_keys, D]
        v:           [batch_size, n_keys, D]
        output:      [batch_size, n_queries, D]
        grad_output: [batch_size, n_queries, D]
        lse:         [batch_size, n_queries]

    返回：
        grad_q、grad_k、grad_v
    """
    q_dtype = q.dtype
    k_dtype = k.dtype
    v_dtype = v.dtype

    # 使用 float32 执行反向计算，减少精度误差。
    q_float = q.float()
    k_float = k.float()
    v_float = v.float()
    output_float = output.float()
    grad_output_float = grad_output.float()
    lse_float = lse.float()

    # PDF 公式（13）：
    # S = QK^T / sqrt(D)
    scores = torch.matmul(
        q_float,
        k_float.transpose(-2, -1),
    ) * scale

    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]

        query_positions = torch.arange(
            n_queries,
            device=q.device,
        )

        key_positions = torch.arange(
            n_keys,
            device=q.device,
        )

        causal_mask = (
            query_positions[:, None]
            >= key_positions[None, :]
        )

        scores = scores.masked_fill(
            ~causal_mask,
            -1e6,
        )

    # PDF 公式（14）：
    # P = exp(S - L)
    probabilities = torch.exp(
        scores - lse_float.unsqueeze(-1)
    )

    # PDF 要求首先计算：
    # D = rowsum(O * dO)
    d_vector = (
        output_float * grad_output_float
    ).sum(
        dim=-1,
        keepdim=True,
    )

    # PDF 公式（15）：
    # dV = P^T dO
    grad_v = torch.matmul(
        probabilities.transpose(-2, -1),
        grad_output_float,
    )

    # PDF 公式（16）：
    # dP = dO V^T
    grad_probabilities = torch.matmul(
        grad_output_float,
        v_float.transpose(-2, -1),
    )

    # PDF 公式（17）：
    # dS = P * (dP - D)
    grad_scores = probabilities * (
        grad_probabilities - d_vector
    )

    # PDF 公式（18）：
    # dQ = dS K / sqrt(D)
    grad_q = torch.matmul(
        grad_scores,
        k_float,
    ) * scale

    # PDF 公式（19）：
    # dK = dS^T Q / sqrt(D)
    grad_k = torch.matmul(
        grad_scores.transpose(-2, -1),
        q_float,
    ) * scale

    return (
        grad_q.to(q_dtype),
        grad_k.to(k_dtype),
        grad_v.to(v_dtype),
    )


# PDF 要求使用 torch.compile。
compiled_flash_attention_backward = torch.compile(
    flash_attention_backward,
    dynamic=True,
)
