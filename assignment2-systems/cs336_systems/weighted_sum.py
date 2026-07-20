import torch
import triton  # pyright: ignore[reportMissingImports]
import triton.language as tl  # pyright: ignore[reportMissingImports]


@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,
    output_ptr,
    x_stride_row,
    x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros(
        (ROWS_TILE_SIZE,),
        dtype=tl.float32,
    )

    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        x_block = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        weight_block = tl.load(
            weight_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )

        # y_i = sum_j(x_ij * weight_j)
        output += tl.sum(
            x_block * weight_block[None, :],
            axis=1,
        )

        x_block_ptr = x_block_ptr.advance(
            (0, D_TILE_SIZE)
        )

        weight_block_ptr = weight_block_ptr.advance(
            (D_TILE_SIZE,)
        )

    tl.store(
        output_block_ptr,
        output,
        boundary_check=(0,),
    )


@triton.jit
def weighted_sum_bwd(
    x_ptr,
    weight_ptr,
    grad_output_ptr,
    grad_x_ptr,
    partial_grad_weight_ptr,
    stride_xr,
    stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr,
    stride_gxd,
    stride_gwb,
    stride_gwd,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    grad_output_block_ptr = tl.make_block_ptr(
        base=grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        base=grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        base=partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    grad_output = tl.load(
        grad_output_block_ptr,
        boundary_check=(0,),
        padding_option="zero",
    )

    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        weight_block = tl.load(
            weight_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )

        # PDF 公式（2）：
        # grad_x[i, j] = grad_output[i] * weight[j]
        grad_x_block = (
            grad_output[:, None] * weight_block[None, :]
        )

        tl.store(
            grad_x_block_ptr,
            grad_x_block,
            boundary_check=(0, 1),
        )

        x_block = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        # PDF 公式（3）：
        # grad_weight[j] = sum_i(x[i, j] * grad_output[i])
        partial_grad_weight = tl.sum(
            x_block * grad_output[:, None],
            axis=0,
            keep_dims=True,
        )

        tl.store(
            partial_grad_weight_block_ptr,
            partial_grad_weight,
            boundary_check=(1,),
        )

        x_block_ptr = x_block_ptr.advance(
            (0, D_TILE_SIZE)
        )

        weight_block_ptr = weight_block_ptr.advance(
            (D_TILE_SIZE,)
        )

        grad_x_block_ptr = grad_x_block_ptr.advance(
            (0, D_TILE_SIZE)
        )

        partial_grad_weight_block_ptr = (
            partial_grad_weight_block_ptr.advance(
                (0, D_TILE_SIZE)
            )
        )


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        input_shape = x.shape
        D = input_shape[-1]

        assert weight.ndim == 1, (
            "weight must be a one-dimensional tensor"
        )
        assert weight.shape[0] == D, (
            "weight length must match x's last dimension"
        )
        assert x.is_cuda and weight.is_cuda, (
            "inputs must be CUDA tensors"
        )
        assert x.is_contiguous(), "x must be contiguous"
        assert weight.is_contiguous(), "weight must be contiguous"

        # [..., D] -> [NUM_ROWS, D]
        x_2d = x.reshape(-1, D)
        n_rows = x_2d.shape[0]

        rows_tile_size = 16
        d_tile_size = max(
            1,
            triton.next_power_of_2(D) // 16,
        )

        ctx.save_for_backward(x_2d, weight)
        ctx.input_shape = input_shape
        ctx.ROWS_TILE_SIZE = rows_tile_size
        ctx.D_TILE_SIZE = d_tile_size

        output = torch.empty(
            (n_rows,),
            device=x.device,
            dtype=x.dtype,
        )

        grid = (
            triton.cdiv(n_rows, rows_tile_size),
        )

        weighted_sum_fwd[grid](
            x_2d,
            weight,
            output,
            x_2d.stride(0),
            x_2d.stride(1),
            weight.stride(0),
            output.stride(0),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=rows_tile_size,
            D_TILE_SIZE=d_tile_size,
        )

        return output.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors

        rows_tile_size = ctx.ROWS_TILE_SIZE
        d_tile_size = ctx.D_TILE_SIZE

        n_rows, D = x.shape
        n_row_tiles = triton.cdiv(
            n_rows,
            rows_tile_size,
        )

        grad_output = grad_output.contiguous().reshape(-1)

        grad_x = torch.empty_like(x)

        partial_grad_weight = torch.empty(
            (n_row_tiles, D),
            device=x.device,
            dtype=x.dtype,
        )

        grid = (n_row_tiles,)

        weighted_sum_bwd[grid](
            x,
            weight,
            grad_output,
            grad_x,
            partial_grad_weight,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            grad_output.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            partial_grad_weight.stride(0),
            partial_grad_weight.stride(1),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=rows_tile_size,
            D_TILE_SIZE=d_tile_size,
        )

        grad_weight = partial_grad_weight.sum(dim=0)

        grad_x = grad_x.view(ctx.input_shape)

        return grad_x, grad_weight


weighted_sum = WeightedSumFunc.apply


def test_weighted_sum():
    torch.manual_seed(0)

    x = torch.randn(
        4,
        8,
        64,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    weight = torch.randn(
        64,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    reference_x = (
        x.detach().clone().requires_grad_(True)
    )
    reference_weight = (
        weight.detach().clone().requires_grad_(True)
    )

    triton_output = weighted_sum(x, weight)

    pytorch_output = (
        reference_x * reference_weight
    ).sum(dim=-1)

    torch.testing.assert_close(
        triton_output,
        pytorch_output,
        rtol=1e-4,
        atol=1e-4,
    )

    grad_output = torch.randn_like(triton_output)

    triton_output.backward(grad_output)
    pytorch_output.backward(grad_output)

    torch.testing.assert_close(
        x.grad,
        reference_x.grad,
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        weight.grad,
        reference_weight.grad,
        rtol=1e-4,
        atol=1e-4,
    )

    print("Forward pass correct")
    print("Backward pass correct")


if __name__ == "__main__":
    test_weighted_sum()