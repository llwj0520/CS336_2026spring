from __future__ import annotations

import math

import torch
import triton  # pyright: ignore[reportMissingImports]
import triton.language as tl  # pyright: ignore[reportMissingImports]


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q = tl.load(
        Q_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    output_accumulator = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    denominator = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    row_max = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for key_start in range(0, N_KEYS, K_TILE_SIZE):
        k = tl.load(
            K_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        v = tl.load(
            V_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        scores = tl.dot(q, tl.trans(k)) * scale
        key_offsets = key_start + tl.arange(0, K_TILE_SIZE)
        valid_mask = key_offsets[None, :] < N_KEYS

        if is_causal:
            causal_mask = query_offsets[:, None] >= key_offsets[None, :]
            valid_mask = valid_mask & causal_mask

        scores = tl.where(valid_mask, scores, -1e6)

        tile_row_max = tl.max(scores, axis=1)
        new_row_max = tl.maximum(row_max, tile_row_max)
        correction = tl.exp(row_max - new_row_max)
        probabilities = tl.exp(scores - new_row_max[:, None])

        denominator = correction * denominator + tl.sum(probabilities, axis=1)
        output_accumulator *= correction[:, None]
        output_accumulator = tl.dot(
            probabilities.to(v.dtype),
            v,
            acc=output_accumulator,
        )
        row_max = new_row_max

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    output = output_accumulator / denominator[:, None]
    logsumexp = row_max + tl.log(denominator)

    tl.store(
        O_block_ptr,
        output.to(q.dtype),
        boundary_check=(0, 1),
    )
    tl.store(L_block_ptr, logsumexp, boundary_check=(0,))


@triton.jit
def flash_bwd_preprocess_kernel(
    O_ptr,
    DO_ptr,
    Delta_ptr,
    stride_ob,
    stride_oq,
    stride_od,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_db,
    stride_dq,
    N_QUERIES,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    d_offsets = tl.arange(0, D)
    query_mask = query_offsets < N_QUERIES

    o_ptrs = (
        O_ptr
        + batch_index * stride_ob
        + query_offsets[:, None] * stride_oq
        + d_offsets[None, :] * stride_od
    )
    do_ptrs = (
        DO_ptr
        + batch_index * stride_dob
        + query_offsets[:, None] * stride_doq
        + d_offsets[None, :] * stride_dod
    )

    output = tl.load(o_ptrs, mask=query_mask[:, None], other=0.0)
    grad_output = tl.load(do_ptrs, mask=query_mask[:, None], other=0.0)
    delta = tl.sum(output.to(tl.float32) * grad_output.to(tl.float32), axis=1)

    delta_ptrs = Delta_ptr + batch_index * stride_db + query_offsets * stride_dq
    tl.store(delta_ptrs, delta, mask=query_mask)


@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    DO_ptr,
    L_ptr,
    Delta_ptr,
    DK_ptr,
    DV_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_lb,
    stride_lq,
    stride_db,
    stride_dq,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    stride_dvb,
    stride_dvk,
    stride_dvd,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    key_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
    d_offsets = tl.arange(0, D)
    key_mask = key_offsets < N_KEYS

    k_ptrs = (
        K_ptr
        + batch_index * stride_kb
        + key_offsets[:, None] * stride_kk
        + d_offsets[None, :] * stride_kd
    )
    v_ptrs = (
        V_ptr
        + batch_index * stride_vb
        + key_offsets[:, None] * stride_vk
        + d_offsets[None, :] * stride_vd
    )
    k = tl.load(k_ptrs, mask=key_mask[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)

    grad_k = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    grad_v = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    for query_start in range(0, N_QUERIES, Q_TILE_SIZE):
        query_offsets = query_start + tl.arange(0, Q_TILE_SIZE)
        query_mask = query_offsets < N_QUERIES

        q_ptrs = (
            Q_ptr
            + batch_index * stride_qb
            + query_offsets[:, None] * stride_qq
            + d_offsets[None, :] * stride_qd
        )
        do_ptrs = (
            DO_ptr
            + batch_index * stride_dob
            + query_offsets[:, None] * stride_doq
            + d_offsets[None, :] * stride_dod
        )
        l_ptrs = L_ptr + batch_index * stride_lb + query_offsets * stride_lq
        delta_ptrs = Delta_ptr + batch_index * stride_db + query_offsets * stride_dq

        q = tl.load(q_ptrs, mask=query_mask[:, None], other=0.0)
        grad_output = tl.load(do_ptrs, mask=query_mask[:, None], other=0.0)
        lse = tl.load(l_ptrs, mask=query_mask, other=0.0)
        delta = tl.load(delta_ptrs, mask=query_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * scale
        valid_mask = query_mask[:, None] & key_mask[None, :]
        if is_causal:
            valid_mask = valid_mask & (query_offsets[:, None] >= key_offsets[None, :])

        probabilities = tl.where(
            valid_mask,
            tl.exp(scores - lse[:, None]),
            0.0,
        )

        grad_v = tl.dot(
            tl.trans(probabilities.to(grad_output.dtype)),
            grad_output,
            acc=grad_v,
        )
        grad_probabilities = tl.dot(grad_output, tl.trans(v))
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        grad_k = tl.dot(
            tl.trans(grad_scores.to(q.dtype)),
            q,
            acc=grad_k,
        )

    grad_k *= scale

    grad_k_ptrs = (
        DK_ptr
        + batch_index * stride_dkb
        + key_offsets[:, None] * stride_dkk
        + d_offsets[None, :] * stride_dkd
    )
    grad_v_ptrs = (
        DV_ptr
        + batch_index * stride_dvb
        + key_offsets[:, None] * stride_dvk
        + d_offsets[None, :] * stride_dvd
    )
    tl.store(grad_k_ptrs, grad_k, mask=key_mask[:, None])
    tl.store(grad_v_ptrs, grad_v, mask=key_mask[:, None])


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    DO_ptr,
    L_ptr,
    Delta_ptr,
    DQ_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_lb,
    stride_lq,
    stride_db,
    stride_dq,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    d_offsets = tl.arange(0, D)
    query_mask = query_offsets < N_QUERIES

    q_ptrs = (
        Q_ptr
        + batch_index * stride_qb
        + query_offsets[:, None] * stride_qq
        + d_offsets[None, :] * stride_qd
    )
    do_ptrs = (
        DO_ptr
        + batch_index * stride_dob
        + query_offsets[:, None] * stride_doq
        + d_offsets[None, :] * stride_dod
    )
    l_ptrs = L_ptr + batch_index * stride_lb + query_offsets * stride_lq
    delta_ptrs = Delta_ptr + batch_index * stride_db + query_offsets * stride_dq

    q = tl.load(q_ptrs, mask=query_mask[:, None], other=0.0)
    grad_output = tl.load(do_ptrs, mask=query_mask[:, None], other=0.0)
    lse = tl.load(l_ptrs, mask=query_mask, other=0.0)
    delta = tl.load(delta_ptrs, mask=query_mask, other=0.0)
    grad_q = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for key_start in range(0, N_KEYS, K_TILE_SIZE):
        key_offsets = key_start + tl.arange(0, K_TILE_SIZE)
        key_mask = key_offsets < N_KEYS

        k_ptrs = (
            K_ptr
            + batch_index * stride_kb
            + key_offsets[:, None] * stride_kk
            + d_offsets[None, :] * stride_kd
        )
        v_ptrs = (
            V_ptr
            + batch_index * stride_vb
            + key_offsets[:, None] * stride_vk
            + d_offsets[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=key_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k)) * scale
        valid_mask = query_mask[:, None] & key_mask[None, :]
        if is_causal:
            valid_mask = valid_mask & (query_offsets[:, None] >= key_offsets[None, :])

        probabilities = tl.where(
            valid_mask,
            tl.exp(scores - lse[:, None]),
            0.0,
        )
        grad_probabilities = tl.dot(grad_output, tl.trans(v))
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        grad_q = tl.dot(
            grad_scores.to(k.dtype),
            k,
            acc=grad_q,
        )

    grad_q *= scale
    grad_q_ptrs = (
        DQ_ptr
        + batch_index * stride_dqb
        + query_offsets[:, None] * stride_dqq
        + d_offsets[None, :] * stride_dqd
    )
    tl.store(grad_q_ptrs, grad_q, mask=query_mask[:, None])


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("q, k, and v must be CUDA tensors")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("q, k, and v must be three-dimensional")

        batch_size, n_queries, d = q.shape
        k_batch_size, n_keys, k_d = k.shape

        if batch_size != k_batch_size:
            raise ValueError("q and k must have the same batch size")
        if k.shape != v.shape:
            raise ValueError("k and v must have the same shape")
        if d != k_d:
            raise ValueError("q and k must have the same hidden dimension")
        if d < 16 or (d & (d - 1)) != 0:
            raise ValueError("D must be a power of two and at least 16")

        q_tile_size = 16
        k_tile_size = 16
        output = torch.empty_like(q)
        logsumexp = torch.empty(
            (batch_size, n_queries),
            device=q.device,
            dtype=torch.float32,
        )
        scale = 1.0 / math.sqrt(d)
        grid = (triton.cdiv(n_queries, q_tile_size), batch_size)

        flash_fwd_kernel[grid](
            q,
            k,
            v,
            output,
            logsumexp,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            logsumexp.stride(0),
            logsumexp.stride(1),
            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            scale=scale,
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            is_causal=is_causal,
            num_warps=4,
        )

        ctx.save_for_backward(q, k, v, output, logsumexp)
        ctx.is_causal = is_causal
        ctx.scale = scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, lse = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        delta = torch.empty(
            (q.shape[0], q.shape[1]),
            device=q.device,
            dtype=torch.float32,
        )

        batch_size, n_queries, d = q.shape
        n_keys = k.shape[1]
        q_tile_size = 16
        k_tile_size = 16

        preprocess_grid = (
            triton.cdiv(n_queries, q_tile_size),
            batch_size,
        )
        flash_bwd_preprocess_kernel[preprocess_grid](
            output,
            grad_output,
            delta,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            delta.stride(0),
            delta.stride(1),
            N_QUERIES=n_queries,
            D=d,
            Q_TILE_SIZE=q_tile_size,
            num_warps=4,
        )

        dkdv_grid = (
            triton.cdiv(n_keys, k_tile_size),
            batch_size,
        )
        flash_bwd_dkdv_kernel[dkdv_grid](
            q,
            k,
            v,
            grad_output,
            lse,
            delta,
            grad_k,
            grad_v,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            lse.stride(0),
            lse.stride(1),
            delta.stride(0),
            delta.stride(1),
            grad_k.stride(0),
            grad_k.stride(1),
            grad_k.stride(2),
            grad_v.stride(0),
            grad_v.stride(1),
            grad_v.stride(2),
            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            scale=ctx.scale,
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            is_causal=ctx.is_causal,
            num_warps=4,
        )

        dq_grid = (
            triton.cdiv(n_queries, q_tile_size),
            batch_size,
        )
        flash_bwd_dq_kernel[dq_grid](
            q,
            k,
            v,
            grad_output,
            lse,
            delta,
            grad_q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            lse.stride(0),
            lse.stride(1),
            delta.stride(0),
            delta.stride(1),
            grad_q.stride(0),
            grad_q.stride(1),
            grad_q.stride(2),
            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            scale=ctx.scale,
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            is_causal=ctx.is_causal,
            num_warps=4,
        )

        return grad_q, grad_k, grad_v, None
