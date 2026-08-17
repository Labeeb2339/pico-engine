"""Test the parallel selective-scan kernel against a sequential reference."""
import pytest
import torch

from pico_engine.scan import selective_scan, chunked_selective_scan


def _ssm_seq(a, b, c, x, d):
    """Reference full SSM: h_t = a_t*h_{t-1} + b_t; y_t = C_t^T h_t + D*x_t."""
    N, seq, DSTATE = a.shape
    state = torch.zeros(N, DSTATE, device=a.device)
    ys = torch.zeros(N, seq, device=a.device)
    for t in range(seq):
        state = a[:, t] * state + b[:, t]
        ys[:, t] = (state * c[:, t]).sum(-1) + d * x[:, t]
    return ys, state


@pytest.mark.parametrize("seq", [16, 64, 200, 512])
@pytest.mark.parametrize("n", [1, 8])
def test_scan_matches_sequential(seq, n):
    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    DSTATE = 16
    torch.manual_seed(0)
    a = torch.rand(n, seq, DSTATE, device="cuda") * 0.9  # |decay| < 1 for stability
    b = torch.randn(n, seq, DSTATE, device="cuda")
    c = torch.randn(n, seq, DSTATE, device="cuda")
    x = torch.randn(n, seq, device="cuda")
    d = torch.randn(n, device="cuda")
    y_ref, h_ref = _ssm_seq(a, b, c, x, d)
    y_tri, h_tri = selective_scan(a, b, c, x, d)
    assert (y_ref - y_tri).abs().max().item() < 1e-3
    assert (h_ref - h_tri).abs().max().item() < 1e-3


@pytest.mark.parametrize("seq", [1, 257, 511, 1000, 2048])
@pytest.mark.parametrize("chunk_size", [64, 256])
def test_chunked_scan_matches_sequential(seq, chunk_size):
    """Chunked scan must match the sequential reference for any length."""
    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    DSTATE = 16
    n = 4
    torch.manual_seed(0)
    a = torch.rand(n, seq, DSTATE, device="cuda") * 0.9
    b = torch.randn(n, seq, DSTATE, device="cuda")
    c = torch.randn(n, seq, DSTATE, device="cuda")
    x = torch.randn(n, seq, device="cuda")
    d = torch.randn(n, device="cuda")
    y_ref, h_ref = _ssm_seq(a, b, c, x, d)
    y_chk, h_chk = chunked_selective_scan(a, b, c, x, d, chunk_size=chunk_size)
    assert (y_ref - y_chk).abs().max().item() < 1e-3
    assert (h_ref - h_chk).abs().max().item() < 1e-3


def test_chunked_matches_single_block_where_both_fit():
    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    DSTATE, seq, n = 16, 512, 4
    torch.manual_seed(1)
    a = torch.rand(n, seq, DSTATE, device="cuda") * 0.9
    b = torch.randn(n, seq, DSTATE, device="cuda")
    c = torch.randn(n, seq, DSTATE, device="cuda")
    x = torch.randn(n, seq, device="cuda")
    d = torch.randn(n, device="cuda")
    y_sb, h_sb = selective_scan(a, b, c, x, d)
    y_ch, h_ch = chunked_selective_scan(a, b, c, x, d, chunk_size=256)
    assert (y_sb - y_ch).abs().max().item() < 1e-3
    assert (h_sb - h_ch).abs().max().item() < 1e-3
