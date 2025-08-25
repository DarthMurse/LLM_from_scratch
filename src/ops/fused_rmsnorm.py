import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_forward(x_ptr, w_ptr, y_ptr, rvar_ptr, stride, D, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * stride
    y_ptr += row_idx * stride
    rvar_ptr += row_idx

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D
    w = tl.load(w_ptr + cols, mask=mask, other=0)
    v = tl.load(x_ptr + cols, mask=mask, other=0).to(tl.float32)
    var = tl.sum(v * v) / D + 1e-6
    rvar = tl.rsqrt(var)
    tl.store(rvar_ptr, rvar)
    x = tl.load(x_ptr + cols, mask=mask)
    y = x * rvar * w
    tl.store(y_ptr + cols, y, mask=mask)

@triton.jit
def _rmsnorm_backward_dx(x_ptr, w_ptr, dy_ptr, rvar_ptr, dx_ptr, partial_dw_ptr, stride, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Assume BLOCK_SIZE >= D
    row_idx = tl.program_id(0)
    row_offset = row_idx * BLOCK_SIZE_M
    x_block_ptr = tl.make_block_ptr(x_ptr, (M, N), (stride, 1), (row_offset, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    dx_block_ptr = tl.make_block_ptr(dx_ptr, (M, N), (stride, 1), (row_offset, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    dy_block_ptr = tl.make_block_ptr(dy_ptr, (M, N), (stride, 1), (row_offset, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    w_block_ptr = tl.make_block_ptr(w_ptr, (N,), (1,), (0,), (BLOCK_SIZE_N,), order=(0,))
    partial_dw_ptr = tl.make_block_ptr(partial_dw_ptr, (tl.cdiv(M, BLOCK_SIZE_M), N), (stride, 1), (row_idx, 0), (1, BLOCK_SIZE_N), order=(1, 0))
    rvar_block_ptr = tl.make_block_ptr(rvar_ptr, (M,), (1,), (row_offset,), (BLOCK_SIZE_M,), order=(0,))

    rvar = tl.load(rvar_block_ptr, boundary_check=(0,), padding_option="zero")
    scale = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    for i in range(tl.cdiv(N, BLOCK_SIZE_N)):
        cur_x_ptr = tl.advance(x_block_ptr, (0, i * BLOCK_SIZE_N))
        cur_dy_ptr = tl.advance(dy_block_ptr, (0, i * BLOCK_SIZE_N))
        cur_w_ptr = tl.advance(w_block_ptr, (i * BLOCK_SIZE_N))
        x = tl.load(cur_x_ptr, boundary_check=(0, 1), padding_option="zero")        
        dy = tl.load(cur_dy_ptr, boundary_check=(0, 1), padding_option="zero")
        w = tl.load(cur_w_ptr, boundary_check=(0,), padding_option="zero")
        scale += tl.sum(x * dy * w[None, :], axis=1)
    scale /= N
    for i in range(tl.cdiv(N, BLOCK_SIZE_N)):
        cur_x_ptr = tl.advance(x_block_ptr, (0, i * BLOCK_SIZE_N))
        cur_dy_ptr = tl.advance(dy_block_ptr, (0, i * BLOCK_SIZE_N))
        cur_dx_ptr = tl.advance(dx_block_ptr, (0, i * BLOCK_SIZE_N))
        cur_w_ptr = tl.advance(w_block_ptr, (i * BLOCK_SIZE_N))
        cur_partial_dw_ptr = tl.advance(partial_dw_ptr, (0, i * BLOCK_SIZE_N))
        x = tl.load(cur_x_ptr, boundary_check=(0, 1), padding_option="zero")        
        dy = tl.load(cur_dy_ptr, boundary_check=(0, 1), padding_option="zero")
        w = tl.load(cur_w_ptr, boundary_check=(0,), padding_option="zero")
        dx = rvar[:, None] * (dy * w[None, :] - x * scale[:, None] * rvar[:, None] * rvar[:, None])
        partial_dw = tl.sum(rvar[:, None] * x * dy, axis=0, keep_dims=True)
        tl.store(cur_dx_ptr, dx, boundary_check=(0, 1))
        tl.store(cur_partial_dw_ptr, partial_dw, boundary_check=(0, 1))

class rmsnorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        assert N == w.shape[0], f"The dimension of input and weight doesn't match, {D} and {w.shape[0]}"
        rvar = torch.empty([M], dtype=torch.float32, device=x.device)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 32 #min(BLOCK_SIZE, MAX_FUSED_SIZE // BLOCK_SIZE_M)
        _rmsnorm_forward[(M, )](x_arg, w, y, rvar, x_arg.stride(0), N, BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        ctx.save_for_backward(x, w, rvar)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.BLOCK_SIZE_M = BLOCK_SIZE_M
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.num_warps = num_warps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, rvar = ctx.saved_tensors
        dx = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        BLOCK_SIZE_M = ctx.BLOCK_SIZE_M
        BLOCK_SIZE_N = ctx.BLOCK_SIZE_N
        BLOCK_SIZE = ctx.BLOCK_SIZE
        num_warps = ctx.num_warps
        M_num = int((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M) 
        partial_dw = torch.empty([M_num, N]).to(x.device)
        _rmsnorm_backward_dx[(M_num, )](x_arg, w, dy, rvar, dx, partial_dw, x_arg.stride(0), M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps=num_warps, num_ctas=1)
        dw = partial_dw.sum(dim=0)
        return dx, dw

def triton_rmsnorm(x, weight):
    return rmsnorm.apply(x, weight)

def torch_rmsnorm(x, weight):
    var = torch.mean(x * x, dim=-1, keepdim=True) + 1e-6
    result = torch.rsqrt(var) * x * weight
    return result

def test():
    data1 = torch.rand([2, 3, 4], requires_grad=True).to("cuda")
    weight1 = torch.rand([4], requires_grad=True).to("cuda")
    data2 = data1.clone()
    weight2 = weight1.clone()
    triton_out = triton_rmsnorm(data1, weight1)
    torch_out = torch_rmsnorm(data2, weight2)
    print("triton rmsnorm forward:", triton_out)
    print("torch rmsnorm forward:", torch_out)
    triton_loss = (triton_out.flatten() - 0.5).pow(2).mean()
    torch_loss = (torch_out.flatten() - 0.5).pow(2).mean()
    data1.retain_grad()
    weight1.retain_grad()
    triton_loss.backward()
    print("triton rmsnorm backward:", data1.grad, weight1.grad)
    data2.retain_grad()
    weight2.retain_grad()
    torch_loss.backward()
    print("torch rmsnorm backward:", data2.grad, weight2.grad)

def benchmark_rmsnorm():
    from benchmark import benchmark
    data1 = torch.rand([8, 1024, 8192], requires_grad=True).to("cuda")
    data2 = data1.clone()
    weight = torch.rand([8192]).to("cuda")
    data1.retain_grad()
    data2.retain_grad()
    dy = torch.rand_like(data1)
    def triton_func():
        x = triton_rmsnorm(data1, weight)
        x.backward(dy)

    def torch_func():
        x = torch_rmsnorm(data2, weight)
        x.backward(dy)

    print("benchmarking triton rmsnorm ...")
    benchmark(triton_func, warmup=20, rep=10)
    print("benchmarking torch rmsnorm ...")
    benchmark(torch_func, warmup=20, rep=10)

if __name__ == "__main__":
    test()
    benchmark_rmsnorm()
