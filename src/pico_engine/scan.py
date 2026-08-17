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


# ---------------------------------------------------------------------------
# Chunked (multi-block) scan — removes the single-block sequence-length cap.
#
# A single associative_scan over the whole sequence needs the whole sequence
# resident in one block (registers), which caps the length. The standard fix is
# a two-level Blelloch scan:
#   1. scan within each chunk          -> local scan + per-chunk aggregate
#   2. scan the chunk aggregates       -> the "carry" into each chunk
#   3. propagate the carry             -> h_t = carry_a * h_local + carry_b
# ---------------------------------------------------------------------------

@triton.jit
def _chunk_scan_kernel(a_ptr, b_ptr, a_rel_ptr, h_rel_ptr, agg_a_ptr, agg_b_ptr, seq, num_chunks,
                       DSTATE: tl.constexpr, CHUNK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = tl.arange(0, CHUNK)
    offs_d = tl.arange(0, DSTATE)
    pos = pid_c * CHUNK + offs_c
    mask = pos[:, None] < seq
    base = pid_n * (seq * DSTATE) + pos[:, None] * DSTATE + offs_d[None, :]
    a = tl.load(a_ptr + base, mask=mask, other=1.0)
    b = tl.load(b_ptr + base, mask=mask, other=0.0)
    a, b = tl.associative_scan((a, b), 0, combine_fn=_scan_combine)
    # within-chunk prefix product A_t and state B_t (both relative to chunk start)
    seq_pad = num_chunks * CHUNK
    rel_base = pid_n * (seq_pad * DSTATE) + pos[:, None] * DSTATE + offs_d[None, :]
    tl.store(a_rel_ptr + rel_base, a, mask=mask)
    tl.store(h_rel_ptr + rel_base, b, mask=mask)
    # chunk aggregate = scan output at the last slot (padded slots are identity)
    agg_a = tl.sum(tl.where(offs_c[:, None] == CHUNK - 1, a, 0.0), axis=0)
    agg_b = tl.sum(tl.where(offs_c[:, None] == CHUNK - 1, b, 0.0), axis=0)
    agg_base = pid_n * (num_chunks * DSTATE) + pid_c * DSTATE + offs_d
    tl.store(agg_a_ptr + agg_base, agg_a)
    tl.store(agg_b_ptr + agg_base, agg_b)


@triton.jit
def _carry_scan_kernel(agg_a_ptr, agg_b_ptr, carry_b_ptr, hlast_ptr,
                       num_chunks,
                       DSTATE: tl.constexpr, BLOCK_CHUNKS: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_CHUNKS)
    offs_d = tl.arange(0, DSTATE)
    mask = offs_k[:, None] < num_chunks
    base = pid_n * (num_chunks * DSTATE) + offs_k[:, None] * DSTATE + offs_d[None, :]
    agg_a = tl.load(agg_a_ptr + base, mask=mask, other=1.0)
    agg_b = tl.load(agg_b_ptr + base, mask=mask, other=0.0)
    agg_a, agg_b = tl.associative_scan((agg_a, agg_b), 0, combine_fn=_scan_combine)
    # inclusive scan of the chunk aggregates; propagate kernel reads carry[k-1]
    # (only the b component is needed as the incoming state, since h_0 = 0)
    tl.store(carry_b_ptr + base, agg_b, mask=mask)
    # final state = state after all chunks = inclusive scan at num_chunks-1
    hlast = tl.sum(tl.where(offs_k[:, None] == num_chunks - 1, agg_b, 0.0), axis=0)
    tl.store(hlast_ptr + pid_n * DSTATE + offs_d, hlast)


@triton.jit
def _propagate_kernel(a_rel_ptr, h_rel_ptr, carry_b_ptr, c_ptr, x_ptr, d_ptr, y_ptr,
                      seq, num_chunks,
                      DSTATE: tl.constexpr, CHUNK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = tl.arange(0, CHUNK)
    offs_d = tl.arange(0, DSTATE)
    pos = pid_c * CHUNK + offs_c
    mask = pos[:, None] < seq
    # incoming state = inclusive aggregate scan at chunk pid_c-1 (0 if pid_c==0)
    carry_base = pid_n * (num_chunks * DSTATE) + (pid_c - 1) * DSTATE + offs_d
    carry_b = tl.load(carry_b_ptr + carry_base, mask=pid_c > 0, other=0.0)
    # full state h_t = A_t * h_in + B_t  (A_t = within-chunk prefix product)
    seq_pad = num_chunks * CHUNK
    rel_base = pid_n * (seq_pad * DSTATE) + pos[:, None] * DSTATE + offs_d[None, :]
    a_rel = tl.load(a_rel_ptr + rel_base, mask=mask, other=1.0)
    b_rel = tl.load(h_rel_ptr + rel_base, mask=mask, other=0.0)
    h = a_rel * carry_b[None, :] + b_rel
    # y = sum_s h_t[s]*C_t[s] + D*x_t
    c_base = pid_n * (seq * DSTATE) + pos[:, None] * DSTATE + offs_d[None, :]
    c = tl.load(c_ptr + c_base, mask=mask, other=0.0)
    x = tl.load(x_ptr + pid_n * seq + pos, mask=pos < seq, other=0.0)
    d = tl.load(d_ptr + pid_n)
    y = tl.sum(h * c, axis=1) + d * x
    tl.store(y_ptr + pid_n * seq + pos, y, mask=pos < seq)


def chunked_selective_scan(a, b, c, x, d, chunk_size=256):
    """Chunked parallel selective scan (handles arbitrary sequence length).

    Same contract as selective_scan but splits the sequence into chunks and runs
    a two-level Blelloch scan (chunk-local -> carry -> propagate), so no single
    block needs the whole sequence resident.
    """
    N, seq, DSTATE = a.shape
    num_chunks = (seq + chunk_size - 1) // chunk_size
    seq_pad = num_chunks * chunk_size
    dev, dt = a.device, a.dtype
    y = torch.empty(N, seq, device=dev, dtype=dt)
    h_last = torch.empty(N, DSTATE, device=dev, dtype=dt)
    a_rel = torch.empty(N, seq_pad, DSTATE, device=dev, dtype=dt)
    h_rel = torch.empty(N, seq_pad, DSTATE, device=dev, dtype=dt)
    agg_a = torch.empty(N, num_chunks, DSTATE, device=dev, dtype=dt)
    agg_b = torch.empty(N, num_chunks, DSTATE, device=dev, dtype=dt)
    carry_b = torch.empty(N, num_chunks, DSTATE, device=dev, dtype=dt)
    _chunk_scan_kernel[(N, num_chunks)](
        a, b, a_rel, h_rel, agg_a, agg_b, seq, num_chunks, DSTATE=DSTATE, CHUNK=chunk_size)
    _carry_scan_kernel[(N,)](
        agg_a, agg_b, carry_b, h_last, num_chunks,
        DSTATE=DSTATE, BLOCK_CHUNKS=triton.next_power_of_2(num_chunks))
    _propagate_kernel[(N, num_chunks)](
        a_rel, h_rel, carry_b, c, x, d, y, seq, num_chunks,
        DSTATE=DSTATE, CHUNK=chunk_size)
    return y, h_last
