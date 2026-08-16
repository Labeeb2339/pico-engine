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

from contextlib import nullcontext
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
    ffn_dim: int         # intermediate size (dense FFN, or shared-expert FFN for MoE)
    rope_base: float
    eps: float           # RMSNorm epsilon
    context_len: int
    # MoE (defaults 0 => dense)
    num_experts: int = 0
    top_k: int = 0
    expert_ffn_dim: int = 0
    shared_ffn_dim: int = 0

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0


class Transformer:
    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor], device: torch.device,
                 quant: dict | None = None):
        self.cfg = cfg
        self.device = device
        self.w = weights  # name -> tensor on device
        self.quant = quant or {}
        # token embedding: GGUF (hidden, vocab) -> loaded as (vocab, hidden); direct lookup.
        # MoE keeps it as an fp32 table in RAM (quant["token_embd"]); dense uses the weight dict.
        self.embed = quant["token_embd"] if cfg.is_moe else weights["token_embd.weight"]

        # rotary cos/sin tables — (context_len, head_dim//2), on the embedding device
        emb_dev = self.embed.device
        inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, cfg.head_dim, 2, device=emb_dev) / cfg.head_dim))
        pos = torch.arange(cfg.context_len, device=emb_dev, dtype=torch.float32)
        angles = pos[:, None] * inv_freq[None, :]
        self.cos = angles.cos()
        self.sin = angles.sin()

        # Dense-only: fuse the per-layer QKV and gate/up projections into single matmuls.
        self.qkv_w, self.qkv_b, self.gu_w = [], [], []
        if not cfg.is_moe:
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
        written in place (no per-step ``torch.cat``). The decode path uses only
        device-tensor positions (no ``.item()`` sync), so it can be captured
        and replayed as a CUDA graph.
        """
        cfg = self.cfg
        if cfg.is_moe:
            return self._forward_moe(token_ids, positions, cache)
        L = token_ids.shape[0]
        # Prefill (L>1) runs the batched matmuls + SDPA in fp16 (tensor cores +
        # flash attention); decode (L=1) stays fp32 so the CUDA-graph capture is
        # unaffected.
        use_fp16 = L > 1 and self.device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            x = self.embed[token_ids]  # (L, hidden)
            # RoPE tables hoisted out of the per-layer loop (the gather used to run
            # 48x per token).
            cos = self.cos[positions]
            sin = self.sin[positions]
            for i in range(cfg.n_layers):
                x = self._layer(i, x, positions, cos, sin, cache[i])
            x = rmsnorm(x, self._w("output_norm.weight"), cfg.eps)
            if "output" in self.quant:
                qs, d = self.quant["output"]
                logits = q8_0_gemv(x[-1], qs, d)          # Q8_0 quantized projection
            else:
                logits = x[-1] @ self._w("output.weight").T
            return logits

    def _layer(self, i: int, x: torch.Tensor, positions: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor,
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
        # no torch.cat / reallocation per step; index_copy_ with a device
        # position tensor so the write is CUDA-graph capturable)
        k_t = k.transpose(0, 1)         # (n_kv, L, hd)
        v_t = v.transpose(0, 1)
        if L == 1:
            k_buf.index_copy_(1, positions, k_t.contiguous())
            v_buf.index_copy_(1, positions, v_t.contiguous())
            S = positions + 1           # (1,) device tensor = valid cache length
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

    # ---- MoE (qwen2moe) correctness path: CPU blocked GEMV, no fp32 materialization ----

    @staticmethod
    def _gemv(entry, x, bias=None, residual=None):
        """Blocked CPU GEMV against a quantized 2D weight.

        entry = (tag, q, s1, s2): q int8 (N, K), s1/s2 fp32 scales. Dequantizes
        one block at a time so the full fp32 weight is never materialized.
        """
        tag, q, s1, s2 = entry
        x = x.float()
        N, K = q.shape
        block = 16 if tag == "q6" else 32
        nb = K // block
        acc = torch.zeros(N, dtype=torch.float32)
        for b in range(nb):
            qb = q[:, b * block:(b + 1) * block].float()
            xb = x[b * block:(b + 1) * block]
            if tag == "q4":
                acc += s1[:, b] * (qb @ xb) - s2[:, b] * xb.sum()
            elif tag == "q5":
                acc += s1[:, b] * (qb @ xb - 16.0 * xb.sum())
            else:  # q6 / q8
                acc += s1[:, b] * (qb @ xb)
        if bias is not None:
            acc = acc + bias
        if residual is not None:
            acc = acc + residual
        return acc

    def _gemv_expert(self, entry, e, x):
        """Slice expert ``e``'s 2D weight out of the 3D (num_experts, N, K) entry."""
        tag, q, s1, s2 = entry
        return self._gemv((tag, q[e], s1[e], None if s2 is None else s2[e]), x)

    @staticmethod
    def _rmsnorm_cpu(x, w, eps):
        var = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + eps) * w

    @staticmethod
    def _rope_cpu(x, cos, sin):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def _moe_ffn(self, i, x):
        cfg = self.cfg
        h = self._rmsnorm_cpu(x, self._w(f"blk.{i}.ffn_norm.weight"), cfg.eps)
        # router: softmax over all experts -> top-k -> normalize
        gate_inp = self._w(f"blk.{i}.ffn_gate_inp.weight")  # (num_experts, hidden)
        router_logits = h @ gate_inp.T
        routing = torch.softmax(router_logits, dim=-1)
        topk_w, topk_idx = torch.topk(routing, cfg.top_k)
        topk_w = topk_w / topk_w.sum()
        # sparse experts
        ge, ue, de = (self.quant["experts"][i][k] for k in ("gate", "up", "down"))
        y = torch.zeros(cfg.hidden, dtype=torch.float32)
        for j in range(cfg.top_k):
            e = int(topk_idx[j])
            act = torch.nn.functional.silu(self._gemv_expert(ge, e, h)) * self._gemv_expert(ue, e, h)
            y = y + topk_w[j] * self._gemv_expert(de, e, act)
        # shared expert (always-on, gated by sigmoid)
        sh = self.quant["shared"][i]
        sh_act = torch.nn.functional.silu(self._gemv(sh["ffn_gate_shexp"], h)) * self._gemv(sh["ffn_up_shexp"], h)
        sh_down = self._gemv(sh["ffn_down_shexp"], sh_act)
        sh_gate_w = self._w(f"blk.{i}.ffn_gate_inp_shexp.weight")
        y = y + torch.sigmoid(sh_gate_w @ h) * sh_down
        return x + y

    def _moe_layer(self, i, x, pos, k_buf, v_buf):
        cfg = self.cfg
        hd = cfg.head_dim
        h = self._rmsnorm_cpu(x, self._w(f"blk.{i}.attn_norm.weight"), cfg.eps)
        a = self.quant["attn"][i]
        q = self._gemv(a["attn_q"], h, bias=self._w(f"blk.{i}.attn_q.bias")).view(cfg.n_head, hd)
        k = self._gemv(a["attn_k"], h, bias=self._w(f"blk.{i}.attn_k.bias")).view(cfg.n_kv_head, hd)
        v = self._gemv(a["attn_v"], h, bias=self._w(f"blk.{i}.attn_v.bias")).view(cfg.n_kv_head, hd)
        q = self._rope_cpu(q, self.cos[pos], self.sin[pos])
        k = self._rope_cpu(k, self.cos[pos], self.sin[pos])
        k_buf[:, pos] = k
        v_buf[:, pos] = v
        S = pos + 1
        # GQA attention (this model: n_head == n_kv_head, group=1)
        scores = torch.einsum('hd,hsd->hs', q, k_buf[:, :S]) * (hd ** -0.5)
        scores = torch.softmax(scores, dim=-1)
        out = torch.einsum('hs,hsd->hd', scores, v_buf[:, :S]).reshape(cfg.hidden)
        x = self._gemv(a["attn_output"], out, residual=x)
        return self._moe_ffn(i, x)

    def _forward_moe(self, token_ids, positions, cache):
        """Single-token-at-a-time MoE forward (CPU, correctness-first)."""
        cfg = self.cfg
        x = self.embed[token_ids]  # (L, hidden)
        L = x.shape[0]
        for t in range(L):
            x_t = x[t]
            for i in range(cfg.n_layers):
                x_t = self._moe_layer(i, x_t, int(positions[t]), cache[i][0], cache[i][1])
            x[t] = x_t
        h = self._rmsnorm_cpu(x[-1], self._w("output_norm.weight"), cfg.eps)
        return self._gemv(self.quant["output"], h)
