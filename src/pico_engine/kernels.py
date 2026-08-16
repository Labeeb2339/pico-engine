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
