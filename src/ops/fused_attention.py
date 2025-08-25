import torch
import torch.nn.functional as F
import os
import triton
import triton.language as tl
import math

def get_config():
    return [
            triton.Config({'TILE_SIZE': 16}, num_warps=4, num_stages=3),
            triton.Config({'TILE_SIZE': 16}, num_warps=8, num_stages=3),
            triton.Config({'TILE_SIZE': 16}, num_warps=4, num_stages=4),
            triton.Config({'TILE_SIZE': 16}, num_warps=8, num_stages=4),
            triton.Config({'TILE_SIZE': 32}, num_warps=4, num_stages=3),
            triton.Config({'TILE_SIZE': 32}, num_warps=8, num_stages=3),
            triton.Config({'TILE_SIZE': 32}, num_warps=4, num_stages=4),
            triton.Config({'TILE_SIZE': 32}, num_warps=8, num_stages=4),
            triton.Config({'TILE_SIZE': 64}, num_warps=4, num_stages=4),
            triton.Config({'TILE_SIZE': 64}, num_warps=8, num_stages=4),
            triton.Config({'TILE_SIZE': 64}, num_warps=4, num_stages=5),
            triton.Config({'TILE_SIZE': 64}, num_warps=8, num_stages=5),
            ]

@triton.autotune(
        get_config(),
        key=['S', 'D']
        )
@triton.jit 
def _attention_forward(q_ptr, k_ptr, v_ptr, o_ptr, log_ptr, S, D, KERNEL_D: tl.constexpr, TILE_SIZE: tl.constexpr):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    batch_offset = batch_idx * S
    q_ptr += batch_offset * D
    q_block_ptr = tl.make_block_ptr(q_ptr, 
                                    shape=(S, D), 
                                    strides=(D, 1), 
                                    offsets=(0, 0), 
                                    block_shape=(TILE_SIZE, KERNEL_D), 
                                    order=(1, 0))
    k_ptr += batch_offset * D
    k_block_ptr = tl.make_block_ptr(k_ptr, 
                                    shape=(S, D), 
                                    strides=(D, 1), 
                                    offsets=(0, 0), 
                                    block_shape=(TILE_SIZE, KERNEL_D), 
                                    order=(1, 0))
    v_ptr += batch_offset * D
    v_block_ptr = tl.make_block_ptr(v_ptr, 
                                    shape=(S, D), 
                                    strides=(D, 1), 
                                    offsets=(0, 0), 
                                    block_shape=(TILE_SIZE, KERNEL_D), 
                                    order=(1, 0))
    o_ptr += batch_offset * D
    o_block_ptr = tl.make_block_ptr(o_ptr, 
                                    shape=(S, D), 
                                    strides=(D, 1), 
                                    offsets=(0, 0), 
                                    block_shape=(TILE_SIZE, KERNEL_D), 
                                    order=(1, 0))
    log_ptr += batch_offset
    log_block_ptr = tl.make_block_ptr(log_ptr,
                                          shape=(S,),
                                          strides=(1,),
                                          offsets=(0,),
                                          block_shape=(TILE_SIZE,),
                                          order=(0,))

    cur_q_ptr = tl.advance(q_block_ptr, (seq_idx * TILE_SIZE, 0))
    q_tile = tl.load(cur_q_ptr, boundary_check=(0, 1), padding_option="zero")
    o_tile = tl.zeros([TILE_SIZE, KERNEL_D], dtype=tl.float32)
    row_max = tl.zeros([TILE_SIZE], dtype=tl.float32) + float("-inf")
    row_exp_sum = tl.zeros([TILE_SIZE], dtype=tl.float32)
    scale = 1.44269504089 / tl.sqrt(D * 1.0)
    rows = tl.arange(0, TILE_SIZE)
    mask = rows[:, None] >= rows[None, :]
    for i in range(seq_idx + 1):
        cur_k_ptr = tl.advance(k_block_ptr, (TILE_SIZE * i, 0))
        cur_v_ptr = tl.advance(v_block_ptr, (TILE_SIZE * i, 0))
        k_tile = tl.load(cur_k_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.load(cur_v_ptr, boundary_check=(0, 1), padding_option="zero")
        p_tile = tl.dot(q_tile, k_tile.trans(1, 0)) * scale
        if i == seq_idx:
            p_tile = tl.where(mask, p_tile, float("-inf"))
        row_max_tile = tl.max(p_tile, axis=1)
        new_row_max = tl.maximum(row_max_tile, row_max)
        diff_exp = tl.math.exp2(row_max - new_row_max)
        p_scale_exp = tl.math.exp2(p_tile - new_row_max[:, None])
        row_exp_sum = row_exp_sum * diff_exp + tl.sum(p_scale_exp, axis=1)
        o_tile = diff_exp[:, None] * o_tile + tl.dot(p_scale_exp.to(v_tile.dtype), v_tile)
        row_max = new_row_max
    o_tile = o_tile / row_exp_sum[:, None]
    cur_o_ptr = tl.advance(o_block_ptr, (seq_idx * TILE_SIZE, 0))
    tl.store(cur_o_ptr, o_tile.to(q_tile.dtype), boundary_check=(0, 1))
    cur_log_ptr = tl.advance(log_block_ptr, (seq_idx * TILE_SIZE,))
    tl.store(cur_log_ptr, (row_max + tl.log2(row_exp_sum)).to(q_tile.dtype), boundary_check=(0,))

@triton.autotune(
        get_config(),
        key=['S', 'D']
        )
@triton.jit
def _attention_backward(do_ptr, o_ptr, q_ptr, k_ptr, v_ptr, log_ptr, dq_ptr, dk_ptr, dv_ptr, S, D, KERNEL_D: tl.constexpr, TILE_SIZE: tl.constexpr):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    batch_offset = batch_idx * S
    do_ptr += batch_offset * D
    do_block_ptr = tl.make_block_ptr(do_ptr, 
                                    shape=(S, D), 
                                    strides=(D, 1), 
                                    offsets=(0, 0), 
                                    block_shape=(TILE_SIZE, KERNEL_D), 
                                    order=(1, 0))
    o_ptr += batch_offset * D
    o_block_ptr = tl.make_block_ptr(o_ptr, 
                                    shape=(S, D), 
                                    strides=(D, 1), 
                                    offsets=(0, 0), 
                                    block_shape=(TILE_SIZE, KERNEL_D), 
                                    order=(1, 0))
    q_ptr += batch_offset * D
    q_block_ptr = tl.make_block_ptr(q_ptr, 
                                    shape=(S, D), 
                                    strides=(D, 1), 
                                    offsets=(0, 0), 
                                    block_shape=(TILE_SIZE, KERNEL_D), 
                                    order=(1, 0))
    k_ptr += batch_offset * D
    k_block_ptr = tl.make_block_ptr(k_ptr, 
                                    shape=(S, D), 
                                    strides=(D, 1), 
                                    offsets=(0, 0), 
                                    block_shape=(TILE_SIZE, KERNEL_D), 
                                    order=(1, 0))
    v_ptr += batch_offset * D
    v_block_ptr = tl.make_block_ptr(v_ptr, 
                                    shape=(S, D), 
                                    strides=(D, 1), 
                                    offsets=(0, 0), 
                                    block_shape=(TILE_SIZE, KERNEL_D), 
                                    order=(1, 0))
    dq_ptr += batch_offset * D
    dq_block_ptr = tl.make_block_ptr(dq_ptr, 
                                    shape=(S, D), 
                                    strides=(D, 1), 
                                    offsets=(0, 0), 
                                    block_shape=(TILE_SIZE, KERNEL_D), 
                                    order=(1, 0))
    dk_ptr += batch_offset * D
    dk_block_ptr = tl.make_block_ptr(dk_ptr, 
                                    shape=(S, D), 
                                    strides=(D, 1), 
                                    offsets=(0, 0), 
                                    block_shape=(TILE_SIZE, KERNEL_D), 
                                    order=(1, 0))
    dv_ptr += batch_offset * D
    dv_block_ptr = tl.make_block_ptr(dv_ptr, 
                                    shape=(S, D), 
                                    strides=(D, 1), 
                                    offsets=(0, 0), 
                                    block_shape=(TILE_SIZE, KERNEL_D), 
                                    order=(1, 0))
    log_ptr += batch_offset
    log_block_ptr = tl.make_block_ptr(log_ptr,
                                          shape=(S,),
                                          strides=(1,),
                                          offsets=(0,),
                                          block_shape=(TILE_SIZE,),
                                          order=(0,))
    
    rows = tl.arange(0, TILE_SIZE)
    scale = 1.44269504089 / tl.sqrt(D * 1.0)
    mask = rows[:, None] >= rows[None, :]
    cur_k_ptr = tl.advance(k_block_ptr, (seq_idx * TILE_SIZE, 0))
    k_tile = tl.load(cur_k_ptr, boundary_check=(0, 1), padding_option="zero")
    dv_tile = tl.zeros([TILE_SIZE, KERNEL_D], dtype=tl.float32)
    dk_tile = tl.zeros([TILE_SIZE, KERNEL_D], dtype=tl.float32)

    for i in range(seq_idx + 1):
        offset_row = tl.arange(0, TILE_SIZE) + seq_idx * TILE_SIZE
        offset_col = tl.arange(0, KERNEL_D)
        cur_dq_ptr = dq_ptr + offset_row[:, None] * D + offset_col[None, :]
        dq_mask = (offset_row[:, None] < S) & (offset_col[None, :] < D)
        cur_q_ptr = tl.advance(q_block_ptr, (i * TILE_SIZE, 0))
        cur_v_ptr = tl.advance(v_block_ptr, (i * TILE_SIZE, 0))
        cur_o_ptr = tl.advance(o_block_ptr, (i * TILE_SIZE, 0))
        cur_do_ptr = tl.advance(do_block_ptr, (i * TILE_SIZE, 0))
        cur_log_ptr = tl.advance(log_block_ptr, (i * TILE_SIZE,))
        q_tile = tl.load(cur_q_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.load(cur_v_ptr, boundary_check=(0, 1), padding_option="zero")
        o_tile = tl.load(cur_o_ptr, boundary_check=(0, 1), padding_option="zero")
        do_tile = tl.load(cur_do_ptr, boundary_check=(0, 1), padding_option="zero")
        log_tile = tl.load(cur_log_ptr, boundary_check=(0,), padding_option="zero")
        D_tile = tl.sum(o_tile * do_tile, axis=1)
        p_tile = tl.dot(q_tile, k_tile.trans(1, 0)) * scale 
        if i == seq_idx:
            p_tile = tl.where(mask, p_tile, float("-inf"))
        p_tile = tl.math.exp2(p_tile - log_tile[:, None])
        dv_tile += tl.dot(p_tile.trans(1, 0).to(do_tile.dtype), do_tile)
        dp_tile = p_tile * (tl.dot(do_tile.to(v_tile.dtype), v_tile.trans(1, 0)) - D_tile[:, None]) * scale
        dk_tile += tl.dot(dp_tile.trans(1, 0).to(q_tile.dtype), q_tile)
        dq_add = tl.dot(dp_tile.to(k_tile.dtype), k_tile)
        # Atomic add seems not to support block pointer, so using array of pointers instead.
        tl.atomic_add(cur_dq_ptr, dq_add, mask=dq_mask)
    cur_dv_ptr = tl.advance(dv_block_ptr, (seq_idx * TILE_SIZE, 0))
    cur_dk_ptr = tl.advance(dk_block_ptr, (seq_idx * TILE_SIZE, 0))
    tl.store(cur_dv_ptr, dv_tile, boundary_check=(0, 1))
    tl.store(cur_dk_ptr, dk_tile, boundary_check=(0, 1))

class attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        # Assume q, k, v have same shape [B, H, S, D]
        B, H, S, D = q.shape
        q_arg = q.reshape(-1, q.shape[-2], q.shape[-1])
        k_arg = k.reshape(-1, k.shape[-2], k.shape[-1])
        v_arg = v.reshape(-1, v.shape[-2], v.shape[-1])
        o_arg = torch.empty_like(q_arg)
        log = torch.empty([B*H, S], dtype=q_arg.dtype, device=q_arg.device)
        KERNEL_D = max(16, triton.next_power_of_2(D))
        grid = lambda META: (B*H, triton.cdiv(S, META['TILE_SIZE']))
        _attention_forward[grid](q_arg, k_arg, v_arg, o_arg, log, S, D, KERNEL_D=KERNEL_D)
        o = o_arg.reshape(B, H, S, D)
        ctx.save_for_backward(q_arg, k_arg, v_arg, o_arg, log)
        return o

    @staticmethod
    def backward(ctx, do):
        q_arg, k_arg, v_arg, o_arg, log = ctx.saved_tensors
        B, H, S, D = do.shape
        do_arg = do.reshape(B*H, S, D)
        dq_arg = torch.zeros_like(do_arg)
        dk_arg = torch.empty_like(do_arg)
        dv_arg = torch.empty_like(do_arg)
        KERNEL_D = max(16, triton.next_power_of_2(D))
        grid = lambda META: (B*H, triton.cdiv(S, META['TILE_SIZE']))
        _attention_backward[grid](do_arg, o_arg, q_arg, k_arg, v_arg, log, dq_arg, dk_arg, dv_arg, S, D, KERNEL_D)
        dq = dq_arg.reshape(B, H, S, D)
        dk = dk_arg.reshape(B, H, S, D)
        dv = dv_arg.reshape(B, H, S, D)
        return dq, dk, dv

def torch_attention(q, k, v):
    # q, k, v: [B, H, S, D_H]
    B, H, S, D = q.shape

    mask = torch.ones([S, S], device=q.device, dtype=torch.bool).tril()
    attn_bias = torch.where(mask, 0, float("-inf"))
    attn_weight = q @ k.transpose(-1, -2) / math.sqrt(D) + attn_bias
    attn_weight = F.softmax(attn_weight, dim=-1)
    out = attn_weight.to(q.dtype) @ v
    return out 

def triton_attention(q, k, v):
    return attention.apply(q, k, v)

def relative_error(x1, x2, rel=True):
    abs_error = (x1 - x2).abs()
    re_error = abs_error / (x1.abs() + 1e-6)
    if rel:
        return re_error.max()
    else:
        return abs_error.max()

def test():
    dtype = torch.bfloat16
    q1 = torch.rand([10, 20, 100, 50], requires_grad=True, dtype=dtype).to("cuda")
    k1 = torch.rand([10, 20, 100, 50], requires_grad=True, dtype=dtype).to("cuda")
    v1 = torch.rand([10, 20, 100, 50], requires_grad=True, dtype=dtype).to("cuda")
    q2 = q1.clone()
    k2 = k1.clone()
    v2 = v1.clone()
    dy = torch.rand_like(q1).to(torch.float32)

    rel = False
    out1 = torch_attention(q1, k1, v1).to(torch.float32)
    out2 = triton_attention(q2, k2, v2).to(torch.float32)
    #print("torch attention forward: ", out1)
    #print("triton attention forward: ", out2)
    print("forward diff:", relative_error(out1, out2, rel))
    q1.retain_grad()
    k1.retain_grad()
    v1.retain_grad()
    out1.backward(dy)
    #print("torch attention backward: ", q1.grad, k1.grad, v1.grad)
    q2.retain_grad()
    k2.retain_grad()
    v2.retain_grad()
    out2.backward(dy)
    #print("triton attention backward: ", q2.grad, k2.grad, v2.grad)
    print("backward diff:", relative_error(q1.grad, q2.grad, rel), relative_error(k1.grad, k2.grad, rel), relative_error(v1.grad, v2.grad, rel))

def benchmark_attention():
    from benchmark import benchmark
    shape = [1, 12, 16384, 64]
    q = torch.rand(shape, requires_grad=True).to("cuda")
    k = torch.rand(shape, requires_grad=True).to("cuda")
    v = torch.rand(shape, requires_grad=True).to("cuda")
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    dy = torch.rand_like(q).to(torch.float32)
    def triton_func():
        x = triton_attention(q, k, v)
        x.backward(dy)
    def torch_func():
        x = torch_attention(q, k, v)
        x.backward(dy)
    print("benchmarking triton attention ...")
    benchmark(triton_func, warmup=30, rep=100)
    print("benchmarking torch attention ...")
    benchmark(torch_func, warmup=30, rep=100)

if __name__ == "__main__":
    #test()
    benchmark_attention()
