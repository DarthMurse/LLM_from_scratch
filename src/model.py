import torch
import torch.nn as nn
from dataclasses import dataclass
import math

@dataclass
class ModelConfig:
    hidden_dim: 768
    intermediate_dim: 2048
    num_layer: 12
    vocab_size: 32000
    num_head: 12
    num_kv_head: 12
    block_size: 1024
    theta: 10000
    bias: False
    q_per_kv: int
    
    def __post_init__(self):
        self.q_per_kv = num_head / num_kv_head
        self.head_dim = self.hidden_dim / self.num_head

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn([in_features, out_features]) / math.sqrt(in_features))
        if bias:
            self.bias = nn.Parameter(torch.rand([out_features]) * 2 - 1)
        else:
            self.bias = None

    def forward(self, x):
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

def softmax(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True)
    exp_x = torch.exp(x)
    y = exp_x / exp_x.sum(dim=dim, keepdim=True)
    return y

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit, target):
        # logits: [..., vocab_size], logits before softmax
        # target: [...], value \in [0, vocab_size-1], ground truth
        logit = logit.flatten(-2)
        target = target.flatten()
        assert logit.shape[0] == target.shape[0], "logits and target must have the same dimension"
        selected_logit = torch.gather(logit, -1, target.unsqueeze(-1))
        processed_logit = torch.exp(logit - selected_logit)
        loss = torch.log(processed_logit.sum(-1)).mean()
        return loss 

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.eps = 1e-6

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean() + self.eps)
        return scale * x

class Embedding(nn.Module):
    def __init__(self, index_num, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn([index_num, dim]))

    def forward(self, x: torch.LongTensor):
        return self.weight[x]

class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.up_proj = Linear(config.hidden_dim, config.intermediate_dim, config.bias)
        self.gate_proj = Linear(config.hidden_dim, config.intermediate_dim, config.bias)
        self.down_proj = Linear(config.intermediate_dim, config.hidden_dim, config.bias)

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        gate = gate / (1 + torch.exp(gate))
        up = up * gate 
        y = self.down_proj(up)
        return y

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.Sequential([Block(config) for i in range(config.num_layer)])
        self.norm = RMSNorm(config.hidden_dim)
        self.head = Linear(config.hidden_dim, config.vocab_size, config.bias)
        self.rope = RotaryEmbedding(config)

    def forward(self, x):
        x = self.embed(x)
        for i in range(self.config.num_layer):
            x = self.layers[i](x, self.rope.cos, self.rope.sin)
        x = self.norm(x)
        y = self.head(x)
        return y

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.hidden_dim)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        y = x + self.mlp(self.mlp_norm(x))
        return y

class RotaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        angle = torch.arange(config.head_dim / 2)
        angle = 1 / (config.theta ** (angle * 2 / config.head_dim))
        seq_len = torch.arange(config.block_size)
        angle = torch.outer(seq_len, angle)
        self.register_buffer("cos", torch.cos(angle))
        self.register_buffer("sin", torch.sin(angle))

    def forward(self, x):
        # x: [B, H, S, D_H]
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1 = x1 * self.cos + x2 * self.sin 
        y2 = x2 * self.cos - x1 * self.sin
        y = torch.concat((y1, y2), dim=-1)
        return y
    
def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    y1 = x1 * cos + x2 * sin 
    y2 = x2 * cos - x1 * sin
    y = torch.concat((y1, y2), dim=-1)
    return y

class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.qkv_proj = Linear(config.hidden_dim, config.hidden_dim + config.hidden_dim / config.q_per_kv * 2, config.bias)
        self.out_proj = Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, x, cos, sin):
        z = self.qkv_proj(x)
        q, k, v = torch.split(z, [self.config.hidden_dim, self.config.hidden_dim / self.config.q_per_kv, self.config.hidden_dim / self.config.q_per_kv], dim=-1)
        
        B, S, D = q.shape
        q = q.reshape(-1, -1, self.config.num_head, self.config.head_dim) # [B, S, H, D_h]
        k = k.reshape(-1, -1, self.config.num_kv_head, self.config.head_dim) # [B, S, H_kv, D_h]
        v = v.reshape(-1, -1, self.config.num_kv_head, self.config.head_dim) # [B, S, H_kv, D_h]

        q = q.transpose(1, 2) # [B, H, S, D_h]
        k = k.transpose(1, 2) # [B, H_kv, S, D_h]
        v = v.transpose(1, 2) # [B, H_kv, S, D_h]
        
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        k = k.expand([1, self.config.q_per_kv, 1, 1])
        v = v.expand([1, self.config.q_per_kv, 1, 1])

        scale = 1 / math.sqrt(self.config.head_dim)
        mask = torch.ones([S, S], device=q.device, dtype=torch.bool).triu()
        attn_bias = torch.where(mask, 0, float("-inf"))
        attn_weight = q @ k.transpose(-1, -2) * scale + attn_bias
        attn_weight = softmax(attn_weight)
        out = attn_weight @ v # [B, H, S, D_h]
        out = out.transpose(1, 2).reshape(B, S, D)
        y = self.out_proj(out)
        return y

if __name__ == "__main__":
    A = torch.rand([2,2])
