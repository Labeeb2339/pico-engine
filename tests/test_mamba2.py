"""Test Mamba-2 (SSD) state-passing scan against a sequential reference."""
import os

import pytest
import torch

from pico_engine.mamba2 import ssd_state_passing


def _ssd_seq(x, dt, A, B, C, D):
    L, H, P = x.shape
    N = B.shape[-1]
    h = torch.zeros(H, N, P, device=x.device, dtype=x.dtype)
    ys = []
    for t in range(L):
        a = torch.exp(dt[t] * A)
        h = a[:, None, None] * h + B[t][None, :, None] * x[t][:, None, :]
        ys.append((h * C[t][None, :, None]).sum(dim=1) + D[:, None] * x[t])
    return torch.stack(ys)


@pytest.mark.parametrize("chunk_size", [32, 64, 256])
@pytest.mark.parametrize("L", [16, 100, 333])  # 333 crosses chunk boundaries unevenly
def test_state_passing_matches_sequential(L, chunk_size):
    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    H, P, N = 8, 16, 32
    torch.manual_seed(0)
    x = torch.randn(L, H, P, device="cuda")
    dt = torch.rand(L, H, device="cuda") * 0.1 + 0.001
    A = -torch.rand(H, device="cuda") * 0.9 - 0.01      # mild decay, in (-0.91, -0.01)
    B = torch.randn(L, N, device="cuda")
    C = torch.randn(L, N, device="cuda")
    D = torch.rand(H, device="cuda")
    y_seq = _ssd_seq(x, dt, A, B, C, D)
    y_sp = ssd_state_passing(x, dt, A, B, C, D, chunk_size=chunk_size)
    assert (y_seq - y_sp).abs().max().item() < 1e-2


def test_state_passing_handles_fast_decay_heads():
    """Extreme A values (trained fast-decay heads) must not produce NaN."""
    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    L, H, P, N = 64, 8, 16, 32
    torch.manual_seed(1)
    x = torch.randn(L, H, P, device="cuda")
    dt = torch.rand(L, H, device="cuda") * 20 + 0.01   # large dt
    A = -torch.exp(torch.randn(H, device="cuda") * 4)  # A up to ~-e^4 (very fast decay)
    B = torch.randn(L, N, device="cuda")
    C = torch.randn(L, N, device="cuda")
    D = torch.rand(H, device="cuda")
    y_sp = ssd_state_passing(x, dt, A, B, C, D, chunk_size=32)
    assert torch.isfinite(y_sp).all()
    y_seq = _ssd_seq(x, dt, A, B, C, D)
    assert (y_seq - y_sp).abs().max().item() < 1e-2


def test_mamba2_model_prefill_smoke():
    """Skip if the checkpoint isn't present; otherwise prefill must be finite + sane."""
    path = "models/mamba2-130m.bin"
    if not torch.cuda.is_available() or not os.path.exists(path):
        pytest.skip("cuda or checkpoint not available")
    from pico_engine.mamba2 import Mamba2Model
    from pico_engine.engine import Engine
    m = Mamba2Model(path, device="cuda")
    e = Engine("models/mamba-130m-f16.gguf", device="cpu")  # just the tokenizer
    ids = torch.tensor(e.tok.encode("The capital of France is"), device="cuda")
    with torch.no_grad():
        logits = m.prefill(ids)
    assert torch.isfinite(logits).all()
    # sane distribution: the top token should not collapse to probability 1
    probs = torch.softmax(logits[-1], dim=-1)
    assert probs.max().item() < 0.99
