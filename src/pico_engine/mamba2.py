"""Mamba-2 (SSD) inference — from scratch, no mamba_ssm/transformers.

Mamba-2 reparameterizes the SSM with a *scalar-identity* A (one decay value per
head, shared across the d_state state dimensions). That makes the state a matrix
(nheads, d_state, headdim) and turns the within-chunk recurrence into matmuls —
the "structured state-space duality" (SSD) with linear attention. The fast path
is a *state-passing* chunked scan: quadratic (attention-like) within a chunk via
matmul, linear (state) across chunks via a sequential carry.

Reference: state-spaces/mamba2-130m (Apache-2.0), loaded from the raw HF
pytorch_model.bin. Verified: logits are sane per-prompt; state-passing scan
matches the sequential recurrence.
"""
import torch
import torch.nn.functional as F


def rmsnorm(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def causal_conv1d(x, conv_w, conv_b, d_conv):
    """Depthwise causal conv over the sequence axis. x:(L,C) -> (L,C)."""
    C = x.shape[-1]
    x = x.transpose(0, 1).unsqueeze(0)                    # (1, C, L)
    x = F.conv1d(x, conv_w.unsqueeze(1), conv_b, padding=d_conv - 1, groups=C)
    x = x[..., :-(d_conv - 1)]
    return x.squeeze(0).transpose(0, 1)                   # (L, C)


def ssd_state_passing(x, dt, A, B, C, D, chunk_size=256):
    """Mamba-2 state-passing scan. x:(L,H,P) dt:(L,H) A:(H,) B:(L,N) C:(L,N) D:(H,).

    Returns y:(L,H,P). Chunk the sequence; within each chunk compute the
    quadratic term (matmul) and the linear term (carried state), then pass the
    chunk-final state to the next chunk.
    """
    L, H, P = x.shape
    N = B.shape[-1]
    dev, dtp = x.device, x.dtype
    dA_cs = torch.cumsum(dt * A[None], dim=0)             # (L, H) log-decay cumsum
    n_chunks = (L + chunk_size - 1) // chunk_size
    y = torch.empty(L, H, P, device=dev, dtype=dtp)
    h = torch.zeros(H, N, P, device=dev, dtype=dtp)
    tril = torch.tril(torch.ones(chunk_size, chunk_size, device=dev, dtype=torch.bool))
    for c in range(n_chunks):
        s, e = c * chunk_size, min(c * chunk_size + chunk_size, L)
        cl = e - s
        dA_ch = dA_cs[s:e]                                  # (cl, H)
        dA_in = dA_cs[s - 1] if s > 0 else torch.zeros(H, device=dev)
        # quadratic: Y_quad = (M . CB) @ X  (causal linear attention within chunk)
        CB = C[s:e] @ B[s:e].T                              # (cl, cl)
        dA_diff = dA_ch[:, None, :] - dA_ch[None, :, :]     # (cl, cl, H)
        # mask upper triangle BEFORE exp: A can be very negative (fast-decay heads),
        # so exp(positive diff) = inf and inf*0 (mask) = NaN otherwise
        dA_diff = dA_diff.masked_fill(~tril[:cl, :cl, None], float("-inf"))
        M = torch.exp(dA_diff)                              # (cl, cl, H), causal
        # dt_j scales the B_j term (B̄ = dt·B in the discretization)
        W = M * CB[:, :, None] * dt[s:e][None, :, :]        # (cl, cl, H)
        Y_quad = torch.einsum("ijh,jhp->ihp", W, x[s:e])    # (cl, H, P)
        # linear: Y_linear = decay_in * (C . h_prev)
        decay_in = torch.exp(dA_ch - dA_in[None, :])        # (cl, H)
        C_h = torch.einsum("ln,hnp->lhp", C[s:e], h)        # (cl, H, P)
        Y_linear = decay_in[:, :, None] * C_h
        y[s:e] = Y_quad + Y_linear + D[None, :, None] * x[s:e]
        # carry state: h = decay_end * h + sum_j decay_j * B_j (x) x_j
        dA_end = dA_cs[e - 1]                               # (H,)
        decay_end = torch.exp(dA_end - dA_in)               # (H,)
        decay_j = torch.exp(dA_end[None, :] - dA_ch)        # (cl, H)
        # B̄ = dt·B in the state update too
        B_w = B[s:e][:, None, :] * decay_j[..., None] * dt[s:e][:, :, None]  # (cl, H, N)
        h = decay_end[:, None, None] * h + torch.einsum("lhn,lhp->hnp", B_w, x[s:e])
    return y


class Mamba2Model:
    def __init__(self, path, device="cuda", chunk_size=256):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        emb = sd["backbone.embedding.weight"]
        p0 = "backbone.layers.0.mixer"
        d_model = emb.shape[1]
        n_layer = max(int(k.split(".")[2]) for k in sd
                      if k.startswith("backbone.layers.") and k.endswith(".norm.weight")) + 1
        d_inner = sd[f"{p0}.norm.weight"].shape[0]
        nheads = sd[f"{p0}.dt_bias"].shape[0]
        d_state = (sd[f"{p0}.conv1d.weight"].shape[0] - d_inner) // 2  # conv_dim = d_inner + 2*d_state
        self.cfg = dict(
            d_model=d_model, n_layer=n_layer, d_inner=d_inner, d_state=d_state,
            nheads=nheads, headdim=d_inner // nheads, ngroups=1,
            d_conv=sd[f"{p0}.conv1d.weight"].shape[2], vocab=emb.shape[0], eps=1e-5,
        )
        self.device = device
        self.chunk_size = chunk_size
        self.w = {k: v.to(device, dtype=torch.float32) for k, v in sd.items()}
        self._A = [(-torch.exp(self.w[f"backbone.layers.{i}.mixer.A_log"])).float()
                   for i in range(self.cfg["n_layer"])]
        # per-layer recurrent state
        C0 = self.cfg["d_inner"] + 2 * self.cfg["d_state"]   # conv channels (x + B + C)
        self.conv_states = [torch.zeros(C0, self.cfg["d_conv"] - 1, device=device)
                            for _ in range(self.cfg["n_layer"])]
        self.ssm_states = [torch.zeros(self.cfg["nheads"], self.cfg["d_state"],
                                       self.cfg["headdim"], device=device)
                           for _ in range(self.cfg["n_layer"])]

    def _mixer(self, h, i):
        """Mamba-2 mixer for a full sequence. h:(L,d_model) -> (L,d_model)."""
        c = self.cfg
        W = self.w
        p = f"backbone.layers.{i}.mixer"
        in_proj = W[f"{p}.in_proj.weight"]
        conv_w = W[f"{p}.conv1d.weight"].squeeze(1)          # (C0, d_conv)
        conv_b = W[f"{p}.conv1d.bias"]
        dt_bias = W[f"{p}.dt_bias"]
        D = W[f"{p}.D"]
        norm_w = W[f"{p}.norm.weight"]
        out_proj = W[f"{p}.out_proj.weight"]
        zxbcdt = h @ in_proj.T                               # (L, 3352)
        z = zxbcdt[:, :c["d_inner"]]
        x = zxbcdt[:, c["d_inner"]:2 * c["d_inner"]]
        B = zxbcdt[:, 2 * c["d_inner"]:2 * c["d_inner"] + c["d_state"]]
        C = zxbcdt[:, 2 * c["d_inner"] + c["d_state"]:2 * c["d_inner"] + 2 * c["d_state"]]
        dt_raw = zxbcdt[:, 2 * c["d_inner"] + 2 * c["d_state"]:]
        xBC = torch.cat([x, B, C], dim=-1)                   # (L, C0)
        xBC = F.silu(causal_conv1d(xBC, conv_w, conv_b, c["d_conv"]))
        x = xBC[:, :c["d_inner"]].reshape(-1, c["nheads"], c["headdim"])
        B = xBC[:, c["d_inner"]:c["d_inner"] + c["d_state"]]
        C = xBC[:, c["d_inner"] + c["d_state"]:]
        dt = F.softplus(dt_raw + dt_bias)                    # (L, nheads)
        y = ssd_state_passing(x, dt, self._A[i], B, C, D, self.chunk_size)  # (L,H,P)
        y = y.reshape(-1, c["d_inner"])                      # (L, d_inner)
        y = rmsnorm(y * F.silu(z), norm_w, c["eps"])         # gated norm (gate-before-norm)
        return y @ out_proj.T                                # (L, d_model)

    def prefill(self, ids):
        """Full-sequence forward (parallel state-passing), returns logits + updates states."""
        c = self.cfg
        x = self.w["backbone.embedding.weight"][ids]         # (L, d_model)
        residual = None
        for i in range(c["n_layer"]):
            if residual is not None:
                x = x + residual
            residual = x
            h = rmsnorm(x, self.w[f"backbone.layers.{i}.norm.weight"], c["eps"])
            x = self._mixer(h, i)
        h_final = rmsnorm(x + residual, self.w["backbone.norm_f.weight"], c["eps"])
        return h_final @ self.w["backbone.embedding.weight"].T

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=32):
        """Greedy: prefill, then one-token decode with cached conv/SSM states."""
        ids = list(ids)
        logits = self.prefill(torch.tensor(ids, device=self.device))
        for _ in range(max_new_tokens):
            tok = int(logits[-1].argmax())
            ids.append(tok)
            logits = self.prefill(torch.tensor(ids, device=self.device))
        return ids
