import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import os
from typing import Optional, Callable

class DDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.handles = []
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self.hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs)

    def hook(self, p):
        handle = dist.all_reduce(p.grad, async_op=True)
        self.handles.append(handle)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def get_ddp_dataloader(dataset, rank, world_size, batch_size):
    size = int(len(dataset) / batch_size)
    assert batch_size % world_size == 0, f"batch_size {batch_size} not divisible by world_size {world_size}"
    micro_batch = batch_size // world_size
    sampler = [list(range(i * batch_size + rank * micro_batch, i * batch_size + (rank + 1) * micro_batch)) for i in range(size)]
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    return dataloader

