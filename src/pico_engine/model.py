"""LLaMA/Qwen-style transformer forward pass, from scratch.

Implements RMSNorm, rotary embeddings (rotate-half), grouped-query attention
with a KV cache, and SwiGLU MLP. Weights come from :mod:`engine` already dequantized; the GGUF linear
convention is ``out = x @ W.T`` (weights stored (in, out), no transpose).

The hot elementwise/reduction ops (RMSNorm, RoPE) and single-token decode
attention run as fused Triton kernels (:mod:`kernels`) because the M=1 decode
path is CPU-dispatch-bound, not compute/bandwidth-bound. The matmuls stay in
torch (cuBLAS) — a custom GEMV measured neutral at M=1.

Reference architecture: Qwen2.5 (the target model), which is LLaMA-style.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .kernels import rmsnorm, rope, silu_mul, decode_attn


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


class Transformer:
    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.w = weights  # name -> tensor on device
        # token embedding: GGUF (hidden, vocab) -> loaded as (vocab, hidden); direct lookup
        self.embed = weights["token_embd.weight"]

        # rotary cos/sin tables — (context_len, head_dim//2)
        inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, cfg.head_dim, 2, device=device) / cfg.head_dim))
        pos = torch.arange(cfg.context_len, device=device, dtype=torch.float32)
        angles = pos[:, None] * inv_freq[None, :]
        self.cos = angles.cos()
        self.sin = angles.sin()

        # Fuse the per-layer QKV and gate/up projections into single matmuls
        # (fewer kernel launches for the launch-bound M=1 decode path).
        self.qkv_w, self.qkv_b, self.gu_w = [], [], []
        for i in range(cfg.n_layers):
            self.qkv_w.append(torch.cat([
                self.w[f"blk.{i}.attn_q.weight"],
                self.w[f"blk.{i}.attn_k.weight"],
                self.w[f"blk.{i}.attn_v.weight"],
            ], dim=0))
            self.qkv_b.append(torch.cat([
                self.w[f"blk.{i}.attn_q.bias"],
                self.w[f"blk.{i}.attn_k.bias"],
                self.w[f"blk.{i}.attn_v.bias"],
            ]))
            self.gu_w.append(torch.cat([
                self.w[f"blk.{i}.ffn_gate.weight"],
                self.w[f"blk.{i}.ffn_up.weight"],
            ], dim=0))

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
        x = self.embed[token_ids]  # (L, hidden), fp16

        for i in range(cfg.n_layers):
            x, cache[i] = self._layer(i, x, positions, cache[i])
        x = rmsnorm(x, self._w("output_norm.weight"), cfg.eps)
        logits = x[-1] @ self._w("output.weight").T  # (vocab,)
        return logits

    def _layer(self, i: int, x: torch.Tensor, positions: torch.Tensor,
               cache: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        cfg = self.cfg
        L = x.shape[0]
        # attention norm
        h = rmsnorm(x, self._w(f"blk.{i}.attn_norm.weight"), cfg.eps)

        qkv = h @ self.qkv_w[i].T + self.qkv_b[i]  # (L, n_head*hd + 2*n_kv*hd)
        q, k, v = qkv.split([cfg.hidden, cfg.n_kv_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim], dim=-1)
        q = q.view(L, cfg.n_head, cfg.head_dim)
        k = k.view(L, cfg.n_kv_head, cfg.head_dim)
        v = v.view(L, cfg.n_kv_head, cfg.head_dim)

        # rotary embeddings (rotate-half, fused)
        c = self.cos[positions]
        s = self.sin[positions]
        q = rope(q, c, s)
        k = rope(k, c, s)

        # append to cache (stores (n_kv, S, hd))
        k_full = torch.cat([cache[0], k.transpose(0, 1)], dim=1)
        v_full = torch.cat([cache[1], v.transpose(0, 1)], dim=1)
        new_cache = (k_full, v_full)

        # GQA attention. Prefill (L>1) uses fused SDPA with a native causal mask.
        # Single-token decode uses a dedicated Triton kernel (one launch) instead
        # of SDPA's dispatch, which was the dominant CPU cost on the launch-bound
        # decode path.
        if L > 1:
            out = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),   # (1, n_head, L, hd)
                k_full.unsqueeze(0),              # (1, n_kv, S, hd)
                v_full.unsqueeze(0),              # (1, n_kv, S, hd)
                is_causal=True,
                enable_gqa=True,
            )
            out = out.squeeze(0).transpose(0, 1)  # (L, n_head, hd)
        else:
            out = decode_attn(q[0], k_full, v_full).unsqueeze(0)  # (1, n_head, hd)
        out = out.reshape(L, cfg.hidden) @ self._w(f"blk.{i}.attn_output.weight").T

        x = x + out

        # SwiGLU MLP
        h = rmsnorm(x, self._w(f"blk.{i}.ffn_norm.weight"), cfg.eps)
        gate_up = h @ self.gu_w[i].T                      # (L, 2*ffn_dim)
        act = silu_mul(gate_up, cfg.ffn_dim)              # (L, ffn_dim) = silu(gate)*up
        x = torch.addmm(x, act, self._w(f"blk.{i}.ffn_down.weight").T)

        return x, new_cache
