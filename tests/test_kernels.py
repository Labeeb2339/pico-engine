"""Fused Triton kernels: correctness against torch references.

The non-contiguous RoPE case is load-bearing: during prefill, q/k come from
``qkv.split(...)`` as strided views, and the Triton kernel assumes row-major
layout. The wrapper forces ``.contiguous()``; this test pins that regression.
"""

import torch
import pytest

from pico_engine.kernels import rmsnorm, rope


def test_rmsnorm_matches_torch():
    torch.manual_seed(0)
    for L in (1, 5):
        x = torch.randn(L, 896, device="cuda")
        w = torch.randn(896, device="cuda")
        eps = 1e-6
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
        out = rmsnorm(x, w, eps)
        assert (out - ref).abs().max().item() < 1e-5


def test_rope_matches_torch_contiguous():
    torch.manual_seed(0)
    L, nh, hd = 2, 14, 64
    x = torch.randn(L, nh, hd, device="cuda")
    c = torch.randn(L, hd // 2, device="cuda")
    s = torch.randn(L, hd // 2, device="cuda")
    half = hd // 2
    x1, x2 = x[..., :half], x[..., half:]
    ref = torch.cat([x1 * c.unsqueeze(1) - x2 * s.unsqueeze(1),
                     x1 * s.unsqueeze(1) + x2 * c.unsqueeze(1)], dim=-1)
    assert (rope(x, c, s) - ref).abs().max().item() < 1e-5


def test_rope_handles_non_contiguous_prefill():
    # Reproduces the prefill path: q comes from a strided qkv.split(...) view.
    torch.manual_seed(0)
    L, n_head, n_kv, hd = 5, 14, 2, 64
    qkv = torch.randn(L, n_head * hd + 2 * n_kv * hd, device="cuda")
    q, k, _ = qkv.split([n_head * hd, n_kv * hd, n_kv * hd], dim=-1)
    q = q.view(L, n_head, hd)
    k = k.view(L, n_kv, hd)
    assert not q.is_contiguous()  # the exact bug condition
    c = torch.randn(L, hd // 2, device="cuda")
    s = torch.randn(L, hd // 2, device="cuda")
    half = hd // 2

    def ref(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * c.unsqueeze(1) - x2 * s.unsqueeze(1),
                          x1 * s.unsqueeze(1) + x2 * c.unsqueeze(1)], dim=-1)

    assert (rope(q, c, s) - ref(q)).abs().max().item() < 1e-5
    assert (rope(k, c, s) - ref(k)).abs().max().item() < 1e-5
