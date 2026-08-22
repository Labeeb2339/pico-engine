"""Fused Triton kernels: correctness against torch references.

The non-contiguous RoPE case is load-bearing: during prefill, q/k come from
``qkv.split(...)`` as strided views, and the Triton kernel assumes row-major
layout. The wrapper forces ``.contiguous()``; this test pins that regression.
"""

import pytest
import torch

from pico_engine.kernels import decode_attn, rmsnorm, rope, silu_mul

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")


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


def test_silu_mul_matches_torch():
    torch.manual_seed(0)
    for L in (1, 5):
        ffn = 4864
        gate_up = torch.randn(L, 2 * ffn, device="cuda")
        gate, up = gate_up[:, :ffn], gate_up[:, ffn:]
        ref = (gate * torch.sigmoid(gate)) * up
        out = silu_mul(gate_up, ffn)
        assert (out - ref).abs().max().item() < 1e-4


def test_decode_attn_matches_torch():
    # Single-query GQA attention (the decode path). q is fp16 as in the model.
    torch.manual_seed(0)
    n_head, n_kv, hd = 14, 2, 64
    group = n_head // n_kv
    for S in (1, 17, 130):
        q = torch.randn(n_head, hd, device="cuda", dtype=torch.float16)
        k = torch.randn(n_kv, S, hd, device="cuda", dtype=torch.float16)
        v = torch.randn(n_kv, S, hd, device="cuda", dtype=torch.float16)
        out = decode_attn(q, k, v)
        k_rep = k.repeat_interleave(group, dim=0)
        v_rep = v.repeat_interleave(group, dim=0)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(1), k_rep, v_rep).squeeze(1)
        assert (out.float() - ref.float()).abs().max().item() < 0.05
