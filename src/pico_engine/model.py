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

from .kernels import rmsnorm, rope, silu_mul, decode_attn, q8_0_gemv, q5_0_gemv, q5_0_gemv_norm, q6_k_gemv, q4_k_gemv


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
    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor], device: torch.device,
                 quant: dict | None = None):
        self.cfg = cfg
        self.device = device
        self.w = weights  # name -> tensor on device
        self.quant = quant or {}
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
        a pre-allocated (k, v) buffer of shape (n_kv_head, capacity, head_dim),
        written in place (no per-step ``torch.cat``).
        """
        cfg = self.cfg
        x = self.embed[token_ids]  # (L, hidden), fp32
        # RoPE tables + scalar position hoisted out of the per-layer loop (the
        # gather used to run 48x per token).
        cos = self.cos[positions]
        sin = self.sin[positions]
        pos = int(positions[-1].item())  # last absolute position (one sync)
        for i in range(cfg.n_layers):
            x = self._layer(i, x, pos, cos, sin, cache[i])
        x = rmsnorm(x, self._w("output_norm.weight"), cfg.eps)
        if "output" in self.quant:
            qs, d = self.quant["output"]
            logits = q8_0_gemv(x[-1], qs, d)          # Q8_0 quantized projection
        else:
            logits = x[-1] @ self._w("output.weight").T
        return logits

    def _layer(self, i: int, x: torch.Tensor, pos: int, cos: torch.Tensor, sin: torch.Tensor,
               cache: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        L = x.shape[0]
        k_buf, v_buf = cache            # (n_kv, capacity, hd)
        capacity = k_buf.shape[1]
        # attention norm
        h = rmsnorm(x, self._w(f"blk.{i}.attn_norm.weight"), cfg.eps)

        qkv = F.linear(h, self.qkv_w[i], self.qkv_b[i])  # fused matmul + bias (1 launch)
        # rotary embeddings: rope q (14 heads) + k (2 heads) in one fused launch
        qk = qkv[:, :cfg.hidden + cfg.n_kv_head * cfg.head_dim].view(L, cfg.n_head + cfg.n_kv_head, cfg.head_dim)
        qk = rope(qk, cos, sin)
        q = qk[:, :cfg.n_head]                                    # (L, n_head, hd)
        k = qk[:, cfg.n_head:]                                    # (L, n_kv, hd)
        v = qkv[:, cfg.hidden + cfg.n_kv_head * cfg.head_dim:].view(L, cfg.n_kv_head, cfg.head_dim)

        # write k/v into the pre-allocated cache at their positions (in place,
        # no torch.cat / reallocation per step)
        k_t = k.transpose(0, 1)         # (n_kv, L, hd)
        v_t = v.transpose(0, 1)
        if L == 1:
            k_buf[:, pos, :] = k_t[:, 0, :]
            v_buf[:, pos, :] = v_t[:, 0, :]
            S = pos + 1
        else:
            k_buf[:, :L, :] = k_t
            v_buf[:, :L, :] = v_t
            S = L

        # GQA attention. Prefill (L>1) uses fused SDPA with a native causal mask.
        # Single-token decode uses the fused Triton kernel (one launch).
        if L > 1:
            out = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),        # (1, n_head, L, hd)
                k_buf[:, :S, :].contiguous().unsqueeze(0),
                v_buf[:, :S, :].contiguous().unsqueeze(0),
                is_causal=True,
                enable_gqa=True,
            )
            out = out.squeeze(0).transpose(0, 1)       # (L, n_head, hd)
        else:
            out = decode_attn(q[0], k_buf, v_buf, S=S, stride=capacity).unsqueeze(0)
        # o-projection + residual fused into one addmm (1 launch instead of matmul+add)
        x = torch.addmm(x, out.reshape(L, cfg.hidden), self._w(f"blk.{i}.attn_output.weight").T)

        # SwiGLU MLP
        if L == 1 and "gu" in self.quant:
            gu_q5, gu_d = self.quant["gu"]
            # fused RMSNorm + Q5_0 gate/up GEMV (one launch instead of norm + matmul)
            gate_up = q5_0_gemv_norm(x[0], self._w(f"blk.{i}.ffn_norm.weight"),
                                     gu_q5[i], gu_d[i], cfg.eps).unsqueeze(0)
        else:
            h = rmsnorm(x, self._w(f"blk.{i}.ffn_norm.weight"), cfg.eps)
            gate_up = h @ self.gu_w[i].T                      # (L, 2*ffn_dim)
        act = silu_mul(gate_up, cfg.ffn_dim)              # (L, ffn_dim) = silu(gate)*up
        if L == 1 and "down" in self.quant:
            entry = self.quant["down"][i]
            if entry is not None:
                if entry[0] == "q6":
                    _, qs, dsc = entry
                    x = q6_k_gemv(act[0], qs, dsc, residual=x[0]).unsqueeze(0)
                else:
                    _, qs, sc, mn = entry
                    x = q4_k_gemv(act[0], qs, sc, mn, residual=x[0]).unsqueeze(0)
                return x
        x = torch.addmm(x, act, self._w(f"blk.{i}.ffn_down.weight").T)

        return x
