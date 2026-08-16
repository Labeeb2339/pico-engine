"""Fused Triton kernels for the inference engine.

The torch-native forward is launch-bound: each decode step runs hundreds of
tiny elementwise/reduction kernels (RMSNorm alone is ~5 ops, RoPE ~4, called
dozens of times per token). These fused kernels collapse the hot ones into a
single launch each.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd(x_ptr, w_ptr, out_ptr, n, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n
    x = tl.load(x_ptr + row * n + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    var = tl.sum(x * x, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(out_ptr + row * n + cols, x * rstd * w, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight."""
    x = x.contiguous()
    weight = weight.contiguous()
    L, n = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n)
    _rmsnorm_fwd[(L,)](x, weight, out, n, eps, BLOCK=BLOCK, num_warps=4)
    return out


@triton.jit
def _rope_fwd(x_ptr, cos_ptr, sin_ptr, out_ptr, n_head, half, H: tl.constexpr):
    pid = tl.program_id(0)
    l = pid // n_head
    j = tl.arange(0, H)
    cos = tl.load(cos_ptr + l * half + j)
    sin = tl.load(sin_ptr + l * half + j)
    x1 = tl.load(x_ptr + pid * 2 * half + j)
    x2 = tl.load(x_ptr + pid * 2 * half + j + half)
    tl.store(out_ptr + pid * 2 * half + j, x1 * cos - x2 * sin)
    tl.store(out_ptr + pid * 2 * half + j + half, x1 * sin + x2 * cos)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Fused rotate-half RoPE. x: (L, n_head, hd); cos/sin: (L, hd//2)."""
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    L, n_head, hd = x.shape
    half = hd // 2
    out = torch.empty_like(x)
    _rope_fwd[(L * n_head,)](x, cos, sin, out, n_head, half, H=half, num_warps=2)
    return out


@triton.jit
def _silu_mul_fwd(gate_up_ptr, out_ptr, ffn, n, BLOCK: tl.constexpr):
    # gate_up is (L, 2*ffn) with gate in [:, :ffn] and up in [:, ffn:].
    # out is (L, ffn) = silu(gate) * up.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    row = offs // ffn
    col = offs % ffn
    g = tl.load(gate_up_ptr + row * 2 * ffn + col, mask=mask)
    u = tl.load(gate_up_ptr + row * 2 * ffn + ffn + col, mask=mask)
    act = g * tl.sigmoid(g) * u
    tl.store(out_ptr + offs, act, mask=mask)


def silu_mul(gate_up: torch.Tensor, ffn: int) -> torch.Tensor:
    """Fused silu(gate) * up, reading gate/up directly from the interleaved
    (L, 2*ffn) gate/up tensor (no split, no copies)."""
    gate_up = gate_up.contiguous()
    L = gate_up.shape[0]
    out = torch.empty(L, ffn, device=gate_up.device, dtype=gate_up.dtype)
    n = L * ffn
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _silu_mul_fwd[grid](gate_up, out, ffn, n, BLOCK=BLOCK, num_warps=4)
    return out


@triton.jit
def _decode_attn_fwd(q_ptr, k_ptr, v_ptr, o_ptr, S, scale,
                     HD: tl.constexpr, GROUP: tl.constexpr, BLOCK_S: tl.constexpr):
    # One program per query head. Single-query (decode) GQA attention with
    # online softmax over the cached keys/values.
    h = tl.program_id(0)
    kv = h // GROUP                    # kv head this query head maps to (GQA)
    offs_d = tl.arange(0, HD)
    q = tl.load(q_ptr + h * HD + offs_d).to(tl.float32)   # (HD,)

    m_i = tl.full((), float("-inf"), tl.float32)
    l_i = tl.zeros((), tl.float32)
    acc = tl.zeros((HD,), tl.float32)

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        k = tl.load(k_ptr + kv * S * HD + offs_s[:, None] * HD + offs_d[None, :],
                    mask=mask_s[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptr + kv * S * HD + offs_s[:, None] * HD + offs_d[None, :],
                    mask=mask_s[:, None], other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1) * scale          # (BLOCK_S,)
        s = tl.where(mask_s, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)                                # (BLOCK_S,)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    acc = acc / l_i
    tl.store(o_ptr + h * HD + offs_d, acc.to(tl.float16))


def decode_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                scale: float | None = None) -> torch.Tensor:
    """Single-query GQA attention for decode.

    q: (n_head, hd); k/v: (n_kv, S, hd) (the running cache). Returns
    (n_head, hd). One Triton launch replaces the SDPA dispatch (which was the
    dominant CPU cost on the launch-bound decode path).
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    n_head, hd = q.shape
    n_kv, S, _ = k.shape
    if scale is None:
        scale = hd ** -0.5
    group = n_head // n_kv
    out = torch.empty(n_head, hd, device=q.device, dtype=q.dtype)
    _decode_attn_fwd[(n_head,)](q, k, v, out, S, scale,
                                HD=hd, GROUP=group, BLOCK_S=64, num_warps=4)
    return out
