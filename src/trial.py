import torch
import triton
import triton.language as tl

@triton.jit
def kernel(a, b):
    pid = tl.program_id(0)
    val = tl.load(a + pid)
    tl.atomic_add(b, val)

def test():
    a = torch.rand([4]).to("cuda") * 10
    b = torch.zeros([1]).to("cuda")
    kernel[(4, )](a, b)
    #b = b.to(torch.bfloat16)
    print(a.sum())
    print(b)

if __name__ == "__main__":
    torch.manual_seed(42)
    test()
