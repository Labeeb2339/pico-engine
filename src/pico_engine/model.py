"""LLaMA/Qwen-style transformer forward pass, from scratch.

Implements RMSNorm, rotary embeddings (rotate-half), grouped-query attention
with a KV cache, and SwiGLU MLP. Weights come from :mod:`engine` as fp32 torch
tensors already dequantized; the GGUF linear convention is ``out = x @ W``
(i.e. weights are stored (in_features, out_features), no transpose).

Reference architecture: Qwen2.5 (the target model), which is LLaMA-style.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    n_layers: int
    hidden: int          # n_embd
    n_head: int          # query heads
    n_kv_head: int       # kv heads (GQA)
    head_dim: int
    ffn_dim: int         # intermediate size
    rope_base: float
    eps: float           # RMSNorm epsilon
    context_len: int


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class RMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = weight  # (hidden,)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., hidden)
        rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rstd * self.weight


class Transformer:
    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.w = weights  # name -> tensor on device (already in numpy/C order)
        # token embedding: GGUF (hidden, vocab) -> loaded as (vocab, hidden); direct lookup
        self.embed = weights["token_embd.weight"]

        # precompute rotary cos/sin tables: (context_len, head_dim//2)
        inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, cfg.head_dim, 2, device=device) / cfg.head_dim))
        pos = torch.arange(cfg.context_len, device=device, dtype=torch.float32)
        angles = pos[:, None] * inv_freq[None, :]
        self.cos = angles.cos()  # (ctx, hd/2)
        self.sin = angles.sin()

    # ---- weight accessors (GGUF layout: x @ W) ----
    def _w(self, name: str) -> torch.Tensor:
        return self.w[name]

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor, positions: torch.Tensor,
                cache: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Run ``token_ids`` (L,) at ``positions`` (L,), updating ``cache``.

        Returns logits (vocab_size,) for the last position. Each cache entry is
        (k, v) with shape (n_kv_head, cache_len, head_dim).
        """
        cfg = self.cfg
        L = token_ids.shape[0]
        x = self.embed[token_ids]  # (L, hidden)

        for i in range(cfg.n_layers):
            x, cache[i] = self._layer(i, x, positions, cache[i])
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.eps) * self._w("output_norm.weight")
        logits = x[-1] @ self._w("output.weight").T  # (vocab,)
        return logits

    def _layer(self, i: int, x: torch.Tensor, positions: torch.Tensor,
               cache: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        cfg = self.cfg
        L = x.shape[0]
        # attention norm
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.eps) * self._w(f"blk.{i}.attn_norm.weight")

        q = h @ self._w(f"blk.{i}.attn_q.weight").T + self._w(f"blk.{i}.attn_q.bias")  # (L, hidden)
        k = h @ self._w(f"blk.{i}.attn_k.weight").T + self._w(f"blk.{i}.attn_k.bias")  # (L, n_kv*head_dim)
        v = h @ self._w(f"blk.{i}.attn_v.weight").T + self._w(f"blk.{i}.attn_v.bias")
        q = q.view(L, cfg.n_head, cfg.head_dim)
        k = k.view(L, cfg.n_kv_head, cfg.head_dim)
        v = v.view(L, cfg.n_kv_head, cfg.head_dim)

        # rotary embeddings (rotate half)
        q, k = self._rope(q, positions), self._rope(k, positions)

        # append to cache and read back the full sequence (cache stores (n_kv, S, hd))
        k_full = torch.cat([cache[0], k.transpose(0, 1)], dim=1)
        v_full = torch.cat([cache[1], v.transpose(0, 1)], dim=1)
        new_cache = (k_full, v_full)

        # GQA: broadcast kv heads to query heads
        n_rep = cfg.n_head // cfg.n_kv_head
        k_rep = k_full.repeat_interleave(n_rep, dim=0)   # (n_head, S, hd)
        v_rep = v_full.repeat_interleave(n_rep, dim=0)

        # scaled dot-product attention with causal mask
        scores = torch.einsum("qhd,hkd->hqk", q, k_rep) / (cfg.head_dim ** 0.5)
        S = k_full.shape[1]
        causal = positions[:, None] >= torch.arange(S, device=self.device)[None, :]
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = F.softmax(scores.float(), dim=-1)          # (n_head, L, S)
        out = torch.einsum("hqk,hkd->qhd", attn, v_rep)   # (L, n_head, hd)
        out = out.reshape(L, cfg.hidden) @ self._w(f"blk.{i}.attn_output.weight").T

        x = x + out

        # SwiGLU MLP
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.eps) * self._w(f"blk.{i}.ffn_norm.weight")
        gate = _silu(h @ self._w(f"blk.{i}.ffn_gate.weight").T)
        up = h @ self._w(f"blk.{i}.ffn_up.weight").T
        x = x + (gate * up) @ self._w(f"blk.{i}.ffn_down.weight").T

        return x, new_cache

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Qwen2 uses "rotate-half" RoPE: rotate the pair (x[i], x[i+d/2]) by theta_i.
        # (HF: q_embed = q*cos + rotate_half(q)*sin, rotate_half(q) = [-x2, x1])
        c = self.cos[positions]  # (L, hd/2)
        s = self.sin[positions]
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([x1 * c.unsqueeze(1) - x2 * s.unsqueeze(1),
                          x1 * s.unsqueeze(1) + x2 * c.unsqueeze(1)], dim=-1)
