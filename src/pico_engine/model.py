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
from .scan import selective_scan, chunked_selective_scan


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
    # Mamba / SSM (defaults 0 => not mamba)
    arch: str = ""
    d_inner: int = 0
    d_state: int = 0
    d_conv: int = 0
    dt_rank: int = 0

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def is_mamba(self) -> bool:
        return self.arch == "mamba"


class Transformer:
    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor], device: torch.device,
                 quant: dict | None = None, n_gpu_layers: int = 0):
        self.cfg = cfg
        self.device = device
        self.w = weights  # name -> tensor on device
        self.quant = quant or {}
        # token embedding: GGUF (hidden, vocab) -> loaded as (vocab, hidden); direct lookup.
        # MoE keeps it as an fp32 table in RAM (quant["token_embd"]); dense uses the weight dict.
        self.embed = quant["token_embd"] if cfg.is_moe else weights["token_embd.weight"]

        # rotary cos/sin tables — (context_len, head_dim//2), on the embedding device
        # (Mamba has no attention/RoPE, so this is skipped.)
        if not cfg.is_mamba:
            emb_dev = self.embed.device
            inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, cfg.head_dim, 2, device=emb_dev) / cfg.head_dim))
            pos = torch.arange(cfg.context_len, device=emb_dev, dtype=torch.float32)
            angles = pos[:, None] * inv_freq[None, :]
            self.cos = angles.cos()
            self.sin = angles.sin()

        # Dense-only: fuse the per-layer QKV and gate/up projections into single matmuls.
        self.qkv_w, self.qkv_b, self.gu_w = [], [], []
        if not cfg.is_moe and not cfg.is_mamba:
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

        # Mamba: the GGUF stores `ssm_a` as A = -exp(A_log) already (precomputed),
        # so the forward uses it directly — no log/exp round-trip.
        if cfg.is_mamba:
            self._A = [self._w(f"blk.{i}.ssm_a").float() for i in range(cfg.n_layers)]

        # MoE partial offload: move the first n_gpu_layers' weights (int8 + F32) to GPU.
        if cfg.is_moe and n_gpu_layers > 0 and device.type == "cuda":
            def _mv(entry, dev):
                tag, q, s1, s2 = entry
                return (tag, q.to(dev), s1.to(dev), None if s2 is None else s2.to(dev))

            for i in range(min(n_gpu_layers, cfg.n_layers)):
                for name in list(self.w):
                    if name.startswith(f"blk.{i}."):
                        self.w[name] = self.w[name].to(device)
                for nm in ("attn_q", "attn_k", "attn_v", "attn_output"):
                    self.quant["attn"][i][nm] = _mv(self.quant["attn"][i][nm], device)
                for k in ("gate", "up", "down"):
                    self.quant["experts"][i][k] = _mv(self.quant["experts"][i][k], device)
                for nm in ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"):
                    self.quant["shared"][i][nm] = _mv(self.quant["shared"][i][nm], device)

    def _layer_dev(self, i: int) -> torch.device:
        return self.quant["attn"][i]["attn_q"][1].device

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
        if cfg.is_mamba:
            if token_ids.shape[0] > 1:
                return self._forward_mamba_prefill(token_ids, cache)
            return self._forward_mamba(token_ids, positions, cache)
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
        """Device-aware quantized GEMV.

        entry = (tag, q, s1, s2): q int8 (N, K), s1/s2 fp32 scales. On CUDA it
        dispatches to the fused Triton GEMV kernels; on CPU it uses a vectorized
        reduce (no per-block Python loop, no full-fp32 materialization).
        """
        tag, q, s1, s2 = entry
        if q.device.type == "cuda":
            if tag == "q4":
                out = q4_k_gemv(x, q, s1, s2, residual)
            elif tag == "q6":
                out = q6_k_gemv(x, q, s1, residual)
            elif tag == "q8":
                out = q8_0_gemv(x, q, s1)
                if residual is not None:
                    out = out + residual
            else:  # q5
                out = q5_0_gemv(x, q, s1)
                if residual is not None:
                    out = out + residual
            if bias is not None:
                out = out + bias
            return out
        x = x.float()
        N, K = q.shape
        block = 16 if tag == "q6" else 32
        nb = K // block
        q3 = q.float().reshape(N, nb, block)
        x3 = x.reshape(nb, block)
        dot = (q3 * x3).sum(-1)          # (N, nb)
        xs = x3.sum(-1)                  # (nb,)
        if tag == "q4":
            acc = (s1 * dot - s2 * xs).sum(-1)
        elif tag == "q5":
            acc = (s1 * (dot - 16.0 * xs)).sum(-1)
        else:  # q6 / q8
            acc = (s1 * dot).sum(-1)
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
        # router: softmax over all experts -> top-k. Qwen1.5-MoE has norm_topk_prob=False,
        # so the raw top-k softmax weights are used (no re-normalization).
        gate_inp = self._w(f"blk.{i}.ffn_gate_inp.weight")  # (num_experts, hidden)
        router_logits = h @ gate_inp.T
        routing = torch.softmax(router_logits, dim=-1)
        topk_w, topk_idx = torch.topk(routing, cfg.top_k)
        # sparse experts
        ge, ue, de = (self.quant["experts"][i][k] for k in ("gate", "up", "down"))
        y = torch.zeros(cfg.hidden, dtype=torch.float32, device=x.device)
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
        cos = self.cos[pos].to(x.device)
        sin = self.sin[pos].to(x.device)
        q = self._rope_cpu(q, cos, sin)
        k = self._rope_cpu(k, cos, sin)
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
        """Single-token-at-a-time MoE forward, dispatching each layer to its device."""
        cfg = self.cfg
        x = self.embed[token_ids]  # (L, hidden), on the embedding device (CPU)
        L = x.shape[0]
        x_last = None
        for t in range(L):
            x_t = x[t]
            for i in range(cfg.n_layers):
                dev = self._layer_dev(i)
                if x_t.device != dev:
                    x_t = x_t.to(dev)
                x_t = self._moe_layer(i, x_t, int(positions[t]), cache[i][0], cache[i][1])
            x_last = x_t
        out_dev = self.quant["output"][1].device
        h = self._rmsnorm_cpu(x_last.to(out_dev), self._w("output_norm.weight"), cfg.eps)
        return self._gemv(self.quant["output"], h)

    # ---- Mamba (SSM) forward: causal conv1d + selective scan ----

    def _mamba_ssm(self, i, x, z, ssm_state):
        """Single-token selective-scan SSM (sequential recurrence, correctness-first)."""
        cfg = self.cfg
        p = f"blk.{i}"
        x_proj_out = x @ self._w(f"{p}.ssm_x.weight").T   # (dt_rank + 2*d_state,)
        dt_raw = x_proj_out[:cfg.dt_rank]
        B = x_proj_out[cfg.dt_rank:cfg.dt_rank + cfg.d_state]
        C = x_proj_out[cfg.dt_rank + cfg.d_state:]
        dt = self._w(f"{p}.ssm_dt.weight") @ dt_raw + self._w(f"{p}.ssm_dt.bias")
        dt = F.softplus(dt)
        A = self._A[i]                                     # (d_inner, d_state) = -exp(A_log)
        D = self._w(f"{p}.ssm_d")                          # (d_inner,)
        dA = torch.exp(dt[..., None] * A)                  # (d_inner, d_state)
        dB = dt[..., None] * B[None, :]                    # (d_inner, d_state)
        dBx = dB * x[..., None]                            # (d_inner, d_state)
        new_state = ssm_state * dA + dBx                   # (d_inner, d_state)
        ssm_state.copy_(new_state)
        y = (new_state * C[None, :]).sum(-1)               # (d_inner,)
        y = y + x * D                                       # D skip connection
        return y * F.silu(z)                                # gated output

    def _mamba_layer(self, i, x, cache_i):
        cfg = self.cfg
        conv_state, ssm_state = cache_i
        p = f"blk.{i}"
        h = self._rmsnorm_cpu(x, self._w(f"{p}.attn_norm.weight"), cfg.eps)
        xz = h @ self._w(f"{p}.ssm_in.weight").T           # (2*d_inner,) = [x, z]
        x_ssm = xz[:cfg.d_inner]
        z = xz[cfg.d_inner:]
        # causal depthwise conv1d (single token)
        conv_w = self._w(f"{p}.ssm_conv1d.weight")         # (d_inner, d_conv)
        conv_b = self._w(f"{p}.ssm_conv1d.bias")
        x_new = torch.cat([conv_state, x_ssm.unsqueeze(-1)], dim=-1)  # (d_inner, d_conv)
        conv_state.copy_(x_new[:, 1:])
        x_conv = (conv_w * x_new).sum(-1) + conv_b
        x_act = F.silu(x_conv)
        y = self._mamba_ssm(i, x_act, z, ssm_state)
        out = y @ self._w(f"{p}.ssm_out.weight").T         # (d_model,)
        return x + out

    def _forward_mamba(self, token_ids, positions, cache):
        cfg = self.cfg
        x = self.embed[token_ids]  # (L, d_model)
        L = x.shape[0]
        x_last = None
        for t in range(L):
            x_t = x[t]
            for i in range(cfg.n_layers):
                x_t = self._mamba_layer(i, x_t, cache[i])
            x_last = x_t
        h = self._rmsnorm_cpu(x_last, self._w("output_norm.weight"), cfg.eps)
        return h @ self.embed.T  # tied embedding -> logits (vocab,)

    # ---- parallel Mamba prefill: batched matmuls + associative-scan SSM ----

    def _causal_conv1d(self, x, conv_w, conv_b):
        """Causal depthwise conv1d over the full sequence (parallel, F.conv1d)."""
        # x: (L, d_inner), conv_w: (d_inner, d_conv), conv_b: (d_inner,)
        d_inner, d_conv = conv_w.shape
        x = x.transpose(0, 1).unsqueeze(0)          # (1, d_inner, L)
        x = F.conv1d(x, conv_w.unsqueeze(1), conv_b, padding=d_conv - 1, groups=d_inner)
        x = x[..., :-(d_conv - 1)]                   # drop the future-leaking tail
        return x.squeeze(0).transpose(0, 1)          # (L, d_inner)

    def _ssm_prefill(self, i, x_act, z, ssm_state):
        """Parallel selective scan for the whole sequence, updating ssm_state."""
        cfg = self.cfg
        p = f"blk.{i}"
        x_proj = self._w(f"{p}.ssm_x.weight")
        dt_proj = self._w(f"{p}.ssm_dt.weight")
        dt_bias = self._w(f"{p}.ssm_dt.bias")
        A = self._A[i]                                # (d_inner, d_state) = -exp(A_log)
        D = self._w(f"{p}.ssm_d")                     # (d_inner,)
        L = x_act.shape[0]
        x_dbl = x_act @ x_proj.T                      # (L, dt_rank + 2*d_state)
        dt_raw = x_dbl[:, :cfg.dt_rank]
        B = x_dbl[:, cfg.dt_rank:cfg.dt_rank + cfg.d_state]
        C = x_dbl[:, cfg.dt_rank + cfg.d_state:]
        dt = F.softplus(dt_raw @ dt_proj.T + dt_bias)  # (L, d_inner)
        a = torch.exp(dt[:, :, None] * A[None])       # (L, d_inner, d_state)
        b = dt[:, :, None] * B[:, None, :] * x_act[:, :, None]
        c = C[:, None, :].expand(L, cfg.d_inner, cfg.d_state)
        y, h_last = self._scan_fn(L)(
            a.permute(1, 0, 2).contiguous(),
            b.permute(1, 0, 2).contiguous(),
            c.permute(1, 0, 2).contiguous(),
            x_act.t().contiguous(),
            D,
        )
        ssm_state.copy_(h_last)
        return y.t()                                  # (L, d_inner)

    @staticmethod
    def _scan_fn(seq_len):
        # single-block scan caps ~1024 tokens (shared-memory wall); chunked beyond
        return selective_scan if seq_len <= 1024 else chunked_selective_scan

    def _mamba_layer_prefill(self, i, x, cache_i):
        cfg = self.cfg
        conv_state, ssm_state = cache_i
        p = f"blk.{i}"
        h = self._rmsnorm_cpu(x, self._w(f"{p}.attn_norm.weight"), cfg.eps)
        xz = h @ self._w(f"{p}.ssm_in.weight").T      # (L, 2*d_inner)
        x_ssm = xz[:, :cfg.d_inner]
        z = xz[:, cfg.d_inner:]
        x_conv = self._causal_conv1d(
            x_ssm, self._w(f"{p}.ssm_conv1d.weight"), self._w(f"{p}.ssm_conv1d.bias"))
        conv_state.copy_(x_ssm[-(cfg.d_conv - 1):].t().contiguous())
        x_act = F.silu(x_conv)
        y = self._ssm_prefill(i, x_act, z, ssm_state)
        y = y * F.silu(z)
        out = y @ self._w(f"{p}.ssm_out.weight").T    # (L, d_model)
        return x + out

    def _forward_mamba_prefill(self, token_ids, cache):
        cfg = self.cfg
        x = self.embed[token_ids]                     # (L, d_model)
        for i in range(cfg.n_layers):
            x = self._mamba_layer_prefill(i, x, cache[i])
        h = self._rmsnorm_cpu(x[-1], self._w("output_norm.weight"), cfg.eps)
        return h @ self.embed.T                       # tied embedding -> logits (vocab,)
