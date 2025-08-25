import torch
import torch.nn as nn
from torch.optim import Optimizer
from collections.abc import Callable, Iterable
from typing import Optional
import math

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad 
                state["t"] = t + 1

        return loss

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0 or betas[0] > 1:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if betas[1] < 0 or betas[1] > 1:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self.eps = eps

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                m = state.get("m", torch.zeros_like(p.grad))
                v = state.get("v", torch.zeros_like(p.grad))
                t = state.get("t", 0)
                betas_pow = state.get("betas_pow", betas)
                m.data = betas[0] * m.data + (1 - betas[0]) * p.grad.data
                v.data = betas[1] * v.data + (1 - betas[1]) * p.grad.pow(2).data
                lr_t = lr * math.sqrt(1 - betas_pow[1]) / (1 - betas_pow[0])
                p.data -= lr_t * m / (torch.sqrt(v) + self.eps)
                p.data *= (1 - lr * weight_decay)

                state["t"] = t + 1
                state["betas_pow"] = (betas_pow[0] * betas[0], betas_pow[1] * betas[1])
                state["m"] = m
                state["v"] = v

        return loss

def cosine_lr_schedule(optimizer, t, max_lr, min_lr, tw, tc):
    lr = min_lr
    if t < tw:
         lr = t / tw * max_lr
    elif t <= tc:
         lr = min_lr + 0.5 * (1 + math.cos(math.pi * (t - tw) / (tc - tw))) * (max_lr - min_lr)
    else:
         lr = min_lr
    for group in optimizer.param_groups:
        group["lr"] = lr

def gradient_clipping(params, theta):
    p_list = []
    for p in params:
        if p.grad is None:
            continue
        p_list.append(p.grad.flatten())
    big_tensor = torch.concat(p_list, dim=0)
    m = big_tensor.abs().max()
    norm = m * (big_tensor / m).norm(2)
    if norm.isnan():
        for p in params:
            print(p.grad)

    if norm > theta:
        for p in params:
            if p.grad is None:
                continue
            p.grad.data *= theta / (norm + 1e-6)

if __name__ == "__main__":
    class MLP(nn.Module):
        def __init__(self, in_features, classes):
            super().__init__()
            self.linear1 = nn.Linear(in_features, int(in_features // 2))
            self.linear2 = nn.Linear(int(in_features // 2), classes)
            self.act1 = nn.ReLU()
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.act1(x)
            y = self.linear2(x)
            return y

    model = MLP(100, 3).to("cuda:1")
    optimizer = AdamW(model.parameters(), lr=1, weight_decay=0)
    #optimizer = SGD(model.parameters(), lr=1)
    loss_func = nn.CrossEntropyLoss()
    data = torch.randn([64, 100]).to("cuda:1")
    classes = torch.randint(low=0, high=3, size=[64]).to("cuda:1")

    for i in range(100):
        y = model(data)
        loss = loss_func(y, classes)
        cosine_lr_schedule(optimizer, i, 1, 0.1, 10, 100)
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)
        optimizer.step()

        print(f"iter {i}: train_loss: {loss.item()}")
