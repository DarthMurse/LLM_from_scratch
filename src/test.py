import torch
from model import *
from optimizer import *

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        for n, p in self.named_parameters():
            if p.grad is not None:
                print(f"{n} param dtype: {p.dtype}, grad dtype: {p.grad.dtype}")
            else:
                print(f"{n} param dtype: {p.dtype}, grad dtype: None")
        print(f"x dtype: {x.dtype}")
        x = self.fc1(x)
        print(f"fc1 dtype: {x.dtype}")
        x = self.relu(x)
        print(f"relu dtype: {x.dtype}")
        x = self.ln(x)
        print(f"layernorm dtype: {x.dtype}")
        x = self.fc2(x)
        print(f"fc2 dtype: {x.dtype}")
        return x

model = ToyModel(256, 2).cuda()
x = torch.rand([2, 256]).cuda()
y = torch.randint(2, [2]).cuda()
loss_func = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    t = model(x)
    loss = loss_func(t, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    t = model(x)
    print(f"loss dtype: {loss.dtype}")