import torch
import triton
import triton.language as tl

@triton.jit
def kernel(a, b, SIZE: tl.constexpr):
    offset = tl.arange(0, SIZE)
    a_row = a + offset
    b_row = b + offset
    a_tile = tl.load(a_row)
    b_tile = tl.math.exp2(a_tile)
    tl.store(b_row, b_tile)

def test():
    a = torch.rand([4]).to("cuda") * 10
    b = torch.empty_like(a)
    kernel[(1, )](a, b, 4)
    c = torch.exp2(a)
    #b = b.to(torch.bfloat16)
    print((b-c).max())

if __name__ == "__main__":
    torch.manual_seed(42)
    test()
