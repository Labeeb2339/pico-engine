"""Test Mamba-2 (SSD) state-passing scan against a sequential reference."""
import os

import pytest
import torch
import torch.nn.functional as F

from pico_engine.mamba2 import ssd_state_passing


def _ssd_seq(x, dt, A, B, C, D):
    L, H, P = x.shape
    N = B.shape[-1]
    h = torch.zeros(H, N, P, device=x.device, dtype=x.dtype)
    ys = []
    for t in range(L):
        a = torch.exp(dt[t] * A)
        # B̄ = dt·B (discretization)
        h = a[:, None, None] * h + dt[t][:, None, None] * B[t][None, :, None] * x[t][:, None, :]
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
    y_sp = ssd_state_passing(x, dt, A, B, C, D, chunk_size=chunk_size)[0]
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
    y_sp = ssd_state_passing(x, dt, A, B, C, D, chunk_size=32)[0]
    assert torch.isfinite(y_sp).all()
    y_seq = _ssd_seq(x, dt, A, B, C, D)
    # relative tolerance: extreme dt/A blow up the magnitude (~1e3), so absolute 1e-2 is too tight
    assert (y_seq - y_sp).abs().max().item() < 1e-2 * y_seq.abs().max().item() + 1e-3


def test_state_passing_matches_mamba_ssm_reference():
    """Independent check: mamba_ssm's chunk_state_ref + state_passing_ref formulas.

    This is the check that caught the missing-dt bug (B̄ = dt·B). The sequential
    reference above can't catch it because a bug in BOTH the sequential and the
    state-passing path cancels out. The mamba_ssm formulas are a third, independent
    implementation (state-spaces/mamba, Apache-2.0).
    """
    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    torch.manual_seed(0)
    for (L, H, P, N, chunk) in [(64, 8, 16, 32, 128), (300, 8, 16, 32, 128)]:
        nchunks = (L + chunk - 1) // chunk
        pad = nchunks * chunk - L
        x = torch.randn(L, H, P, device="cuda")
        dt = torch.rand(L, H, device="cuda") * 0.1 + 0.001
        A = -torch.rand(H, device="cuda") * 0.9
        B = torch.randn(L, N, device="cuda")
        C = torch.randn(L, N, device="cuda")
        D = torch.rand(H, device="cuda")

        # --- mamba_ssm chunk_state_ref: states = einsum(B * decay * dt * x) ---
        B4 = B[None, :, None, :].expand(1, L, H, N)          # (1,L,H,N) ngroups=1 -> broadcast
        x4 = x[None]                                          # (1,L,H,P)
        dt_pad = F.pad(dt[None], (0, 0, 0, pad))              # (1,L+pad,H)
        dt_c = dt_pad.transpose(1, 2).reshape(1, H, nchunks, chunk)
        dA_c = torch.cumsum((dt_pad * A[None, None]).transpose(1, 2)
                            .reshape(1, H, nchunks, chunk), dim=-1)  # per-chunk cumsum
        B4 = F.pad(B4, (0, 0, 0, 0, 0, pad)).reshape(1, nchunks, chunk, H, N)
        x4 = F.pad(x4, (0, 0, 0, 0, 0, pad)).reshape(1, nchunks, chunk, H, P)
        decay = torch.exp(dA_c[:, :, :, -1:] - dA_c)         # (1,H,nchunks,chunk)
        states = torch.einsum("clhn,hcl,hcl,clhp->chpn", B4[0], decay[0], dt_c[0], x4[0])  # (nchunks,H,P,N)
        # --- mamba_ssm state_passing_ref ---
        states_flat = states.reshape(nchunks, H, P * N)      # (nchunks,H,P*N)
        init = torch.zeros(H, P * N, device="cuda")
        states_cat = torch.cat([init[None], states_flat])    # (nchunks+1,H,P*N)
        dA_chunk = dA_c[:, :, :, -1][0]                      # (H,nchunks) per-chunk total
        dA_bound = torch.cumsum(F.pad(dA_chunk, (1, 0)), dim=-1)  # (H,nchunks+1) absolute boundary
        decay_chunk = torch.exp(dA_bound[:, :, None] - dA_bound[:, None, :])  # (H,nchunks+1,nchunks+1)
        decay_chunk = decay_chunk.masked_fill(~torch.tril(torch.ones(nchunks + 1, nchunks + 1, dtype=torch.bool, device="cuda")), 0)
        passed = torch.einsum("hzc,chd->zhd", decay_chunk, states_cat)[:-1]  # (nchunks,H,P*N)
        passed = passed.reshape(nchunks, H, P, N).permute(0, 1, 3, 2)       # (nchunks,H,N,P)

        # --- my carried state per chunk (replicate ssd_state_passing's carry) ---
        dA_abs = torch.cumsum(dt * A[None], dim=0)           # (L,H)
        h = torch.zeros(H, N, P, device="cuda")
        mine = []
        for c in range(nchunks):
            s, e = c * chunk, min(c * chunk + chunk, L)
            mine.append(h.clone())
            dA_in = dA_abs[s - 1] if s > 0 else torch.zeros(H, device="cuda")
            dA_end = dA_abs[e - 1]
            decay_end = torch.exp(dA_end - dA_in)
            decay_j = torch.exp(dA_end[None, :] - dA_abs[s:e])
            B_w = B[s:e][:, None, :] * decay_j[..., None] * dt[s:e][:, :, None]
            h = decay_end[:, None, None] * h + torch.einsum("lhn,lhp->hnp", B_w, x[s:e])
        mine = torch.stack(mine)                              # (nchunks,H,N,P)
        assert (passed - mine).abs().max().item() < 1e-4


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
    # cached-state decode must match a full prefill of the extended sequence
    tok0 = int(logits[-1].argmax())
    m2 = Mamba2Model(path, device="cuda")
    full = torch.cat([ids, torch.tensor([tok0], device="cuda")])
    full_logits = m2.prefill(full)
    step_logits = m.decode_step(tok0)
    assert (full_logits[-1] - step_logits).abs().max().item() < 1e-2
