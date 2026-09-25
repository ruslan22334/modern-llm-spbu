"""small decoder-only language model with grouped-query attention (gqa)"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_dim: int = 768
    max_seq_len: int = 128
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if min(
            self.vocab_size,
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.ffn_dim,
            self.max_seq_len,
        ) <= 0:
            raise ValueError("configuration dimensions must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head dimension must be even for RoPE")
        if self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError("rope_theta and norm_eps must be positive")


class RMSNorm(nn.Module):
    """normalize by root mean square, with float32 reduction for stability"""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight).to(x.dtype)


class SwiGLU(nn.Module):
    """llama feed-forward network: down(SiLU(gate(x)) * up(x))"""

    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.up = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def apply_rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    """rotate adjacent feature pairs according to each token's position
    x shape [batch, heads, tokens, head_dim], only Q and K use RoPE
    """
    tokens, head_dim = x.shape[-2:]
    if head_dim % 2:
        raise ValueError("RoPE needs an even head dimension")

    positions = torch.arange(tokens, device=x.device, dtype=torch.float32)
    frequencies = theta ** (
        -torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
        / head_dim
    )
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos()[None, None].to(x.dtype)
    sin = angles.sin()[None, None].to(x.dtype)

    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class GQAAttention(nn.Module):
    """causal gqa with RoPE on Q and K"""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("attention input must have shape [batch, tokens, d_model]")
        batch, tokens, _ = x.shape
        if tokens > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")

        def heads(projection: nn.Linear, count: int) -> torch.Tensor:
            return projection(x).view(batch, tokens, count, self.head_dim).transpose(1, 2)

        q = apply_rope(heads(self.q_proj, self.n_heads), self.rope_theta)
        k = apply_rope(heads(self.k_proj, self.n_kv_heads), self.rope_theta)
        v = heads(self.v_proj, self.n_kv_heads)

        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True
        )
        joined = attended.transpose(1, 2).contiguous().view(batch, tokens, self.d_model)
        return self.out_proj(joined)


class DecoderBlock(nn.Module):
    """one pre-normalized Llama block with attention and feed-forward paths"""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = GQAAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.feed_forward = SwiGLU(config.d_model, config.ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class LlamaLM(nn.Module):
    """small autoregressive language model with the main Llama 3 components"""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        if not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("token count must be within max_seq_len")

        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))
