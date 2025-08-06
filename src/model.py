import torch
import torch.nn as nn
from dataclasses import dataclass
import math

@dataclass
class ModelConfig:
    hidden_dim: int = 512
    intermediate_dim: int = 1344
    num_layer: int = 4
    vocab_size: int = 10000
    num_head: int = 16
    num_kv_head: int = 16
    block_size: int = 256
    theta: int = 10000
    bias: bool = False
    
    def __post_init__(self):
        self.q_per_kv = int(self.num_head / self.num_kv_head)
        self.head_dim = int(self.hidden_dim / self.num_head)

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
    x = x - x.max(dim=dim, keepdim=True).values.detach()
    exp_x = torch.exp(x)
    y = exp_x / exp_x.sum(dim=dim, keepdim=True)
    return y

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit, target):
        # logits: [..., vocab_size], logits before softmax
        # target: [...], value \in [0, vocab_size-1], ground truth
        assert logit.shape[0] == target.shape[0], f"logits and target must have the same dimension, current shape {logit.shape}, {target.shape}"
        max_logit = logit.max(dim=-1).values.detach()
        selected_logit = torch.gather(logit, -1, target.unsqueeze(-1))
        processed_logit = torch.exp(logit - max_logit.unsqueeze(-1))
        loss = (torch.log(processed_logit.sum(-1)) + max_logit - selected_logit).mean()
        return loss 

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.eps = 1e-6

    def forward(self, x):
        scale = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
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
        gate = gate / (1 + torch.exp(-gate))
        up = up * gate 
        y = self.down_proj(up)
        return y

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([Block(config) for i in range(config.num_layer)])
        self.norm = RMSNorm(config.hidden_dim)
        self.head = Linear(config.hidden_dim, config.vocab_size, config.bias)
        self.rope = RotaryEmbedding(config)

    def forward(self, x, inference=False):
        x = self.embed(x)
        for i in range(self.config.num_layer):
            x = self.layers[i](x, self.rope.cos, self.rope.sin, inference)
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

    def forward(self, x, cos, sin, inference=False):
        x = x + self.attn(self.attn_norm(x), cos, sin, inference)
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
        self.qkv_proj = Linear(config.hidden_dim, config.hidden_dim + int(config.hidden_dim / config.q_per_kv * 2), config.bias)
        self.out_proj = Linear(config.hidden_dim, config.hidden_dim)

        # Setting for inference
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, cos, sin, inference=False):
        z = self.qkv_proj(x)
        q, k, v = torch.split(z, [self.config.hidden_dim, int(self.config.hidden_dim / self.config.q_per_kv), int(self.config.hidden_dim / self.config.q_per_kv)], dim=-1)
        
        B, S, D = q.shape
        q = q.reshape(B, S, self.config.num_head, self.config.head_dim) # [B, S, H, D_h]
        k = k.reshape(B, S, self.config.num_kv_head, self.config.head_dim) # [B, S, H_kv, D_h]
        v = v.reshape(B, S, self.config.num_kv_head, self.config.head_dim) # [B, S, H_kv, D_h]

        q = q.transpose(1, 2) # [B, H, S, D_h]
        k = k.transpose(1, 2) # [B, H_kv, S, D_h]
        v = v.transpose(1, 2) # [B, H_kv, S, D_h]

        if inference:
            current_idx = self.k_cache.shape[2] if self.k_cache is not None else 0
            current_len = q.shape[2]
            cos = cos[current_idx: current_idx + current_len]
            sin = sin[current_idx: current_idx + current_len]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if inference:
            if self.k_cache is None or self.v_cache is None:
                self.k_cache, self.v_cache = k, v
            else:
                self.k_cache = torch.concat([self.k_cache, k], dim=2)
                self.v_cache = torch.concat([self.v_cache, v], dim=2)
                k = self.k_cache
                v = self.v_cache

        k = k.repeat(1, self.config.q_per_kv, 1, 1)
        v = v.repeat(1, self.config.q_per_kv, 1, 1)

        scale = 1 / math.sqrt(self.config.head_dim)
        if not inference:
            mask = torch.ones([S, S], device=q.device, dtype=torch.bool).tril()
            attn_bias = torch.where(mask, 0, float("-inf"))
        else:
            attn_bias = 0
        attn_weight = q @ k.transpose(-1, -2) * scale + attn_bias
        attn_weight = softmax(attn_weight, dim=-1)
        out = attn_weight @ v # [B, H, S, D_h]
        out = out.transpose(1, 2).reshape(B, S, D)
        y = self.out_proj(out)
        return y

if __name__ == "__main__":
    config = ModelConfig()
    model = Transformer(config)
    params = model.parameters()
    for p in params:
        print(p.dtype)
