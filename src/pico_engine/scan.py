"""Parallel selective-scan (S6) via Triton associative_scan.

The Mamba SSM recurrence  h_t = a_t * h_{t-1} + b_t  (h in R^d_state) is a
first-order linear recurrence. It admits an associative combine

    (a1, b1) . (a2, b2) = (a1*a2, a2*b1 + b2)

so the whole sequence can be scanned in parallel (Blelloch-style) instead of
looping token-by-token. This is the SSM analogue of FlashAttention: the same
math, restructured from O(L) sequential to O(log L) parallel depth.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _scan_combine(a1, b1, a2, b2):
    # apply (a1, b1) then (a2, b2): (a1*a2, a2*b1 + b2)
    return a1 * a2, a2 * b1 + b2


@triton.jit
def _selective_scan_kernel(a_ptr, b_ptr, c_ptr, x_ptr, d_ptr, y_ptr, h_ptr, seq,
                           stride_n, stride_s, stride_xn,
                           DSTATE: tl.constexpr, BLOCK_SEQ: tl.constexpr):
    pid = tl.program_id(0)  # one program per (batch, d_inner) channel
    offs_s = tl.arange(0, BLOCK_SEQ)
    offs_d = tl.arange(0, DSTATE)
    mask2d = offs_s[:, None] < seq
    base = pid * stride_n + offs_s[:, None] * stride_s + offs_d[None, :]
    a = tl.load(a_ptr + base, mask=mask2d, other=1.0)
    b = tl.load(b_ptr + base, mask=mask2d, other=0.0)
    c = tl.load(c_ptr + base, mask=mask2d, other=0.0)
    # parallel associative scan over the sequence axis
    a, b = tl.associative_scan((a, b), 0, combine_fn=_scan_combine)
    # y_t = sum_s h_t[s] * C_t[s] + D * x_t
    x = tl.load(x_ptr + pid * stride_xn + offs_s, mask=offs_s < seq, other=0.0)
    d = tl.load(d_ptr + pid)
    y = tl.sum(b * c, axis=1) + d * x
    tl.store(y_ptr + pid * stride_xn + offs_s, y, mask=offs_s < seq)
    # final state = h at the last valid position (seq-1)
    h_last = tl.sum(tl.where(offs_s[:, None] == seq - 1, b, 0.0), axis=0)
    tl.store(h_ptr + pid * DSTATE + offs_d, h_last)


def selective_scan(a, b, c, x, d):
    """Parallel selective scan.

    a, b, c: (N, seq, d_state) fp32 contiguous   (A_t = exp(dt*A), b = dt*B*x, C_t)
    x:       (N, seq) fp32 contiguous            (the SSM input)
    d:       (N,) fp32 contiguous                (the D skip vector)
    returns (y, h_last):
      y:      (N, seq) = sum_s h_t[s]*C_t[s] + D*x_t
      h_last: (N, d_state) = the state at the final position (for decode carry-over)
    """
    N, seq, DSTATE = a.shape
    BLOCK_SEQ = triton.next_power_of_2(seq)
    y = torch.empty(N, seq, device=a.device, dtype=a.dtype)
    h_last = torch.empty(N, DSTATE, device=a.device, dtype=a.dtype)
    _selective_scan_kernel[(N,)](
        a, b, c, x, d, y, h_last, seq,
        a.stride(0), a.stride(1), x.stride(0),
        DSTATE=DSTATE, BLOCK_SEQ=BLOCK_SEQ,
    )
    return y, h_last
