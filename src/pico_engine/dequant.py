"""GGML block-quantization dequantization, implemented from the ggml reference.

Reference: ggml/src/ggml-common.h (block structs) and ggml/src/ggml-quants.c
(dequantize_row_* functions). All formats are little-endian, as GGUF requires.

Dequantization always produces float32; the model layer casts to its compute
dtype afterwards.
"""

from __future__ import annotations

import numpy as np

# ggml type enum (ggml.h)
GGML_F32 = 0
GGML_F16 = 1
GGML_Q5_0 = 6
GGML_Q8_0 = 8
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
GGML_Q8_K = 15

_QK = 256  # K-quant super-block size


def nbytes_for(ggml_type: int, n: int) -> int:
    """Bytes a quantized tensor of ``n`` elements occupies in the data section."""
    if ggml_type == GGML_F32:
        return n * 4
    if ggml_type == GGML_F16:
        return n * 2
    if ggml_type == GGML_Q8_0:
        assert n % 32 == 0, f"Q8_0 needs n%32==0, got {n}"
        return (n // 32) * 34
    if ggml_type == GGML_Q5_0:
        assert n % 32 == 0, f"Q5_0 needs n%32==0, got {n}"
        return (n // 32) * 22  # d(2) + qh(4) + qs(16)
    if ggml_type == GGML_Q4_K:
        assert n % _QK == 0, f"Q4_K needs n%256==0, got {n}"
        return (n // _QK) * 144  # d(2) + dmin(2) + scales(12) + qs(128)
    if ggml_type == GGML_Q6_K:
        assert n % _QK == 0, f"Q6_K needs n%256==0, got {n}"
        return (n // _QK) * 210  # d(2) + ql(128) + qh(64) + scales(16)
    if ggml_type == GGML_Q8_K:
        assert n % _QK == 0, f"Q8_K needs n%256==0, got {n}"
        return (n // _QK) * 292  # d(4, f32) + qs(256) + bsums(32)
    raise NotImplementedError(f"ggml type {ggml_type} not supported")


def _dequant_q5_0(blocks: np.ndarray, n: int) -> np.ndarray:
    nb = n // 32
    blocks = blocks[: nb * 22].reshape(nb, 22)
    d = blocks[:, :2].view(np.float16).astype(np.float32).reshape(nb)  # (nb,)
    qh = blocks[:, 2:6].view(np.uint32).reshape(nb, 1)         # 32 high bits
    qs = blocks[:, 6:22]                                        # (nb, 16)

    j = np.arange(16, dtype=np.uint32)
    # element j   uses qs[j] low nibble  + qh bit j    as the 5th bit
    # element j+16 uses qs[j] high nibble + qh bit j+16 as the 5th bit
    #   (C: xh_1 = (qh >> (j+12)) & 0x10  ->  bit (j+12+4) = bit j+16)
    x0 = ((qs & 0xF) | (((qh >> j) & 1) << 4).astype(np.uint8)).astype(np.int32) - 16
    x1 = ((qs >> 4) | (((qh >> (j + 16)) & 1) << 4).astype(np.uint8)).astype(np.int32) - 16

    out = np.empty((nb, 32), dtype=np.float32)
    out[:, :16] = x0 * d[:, None]
    out[:, 16:] = x1 * d[:, None]
    return out.reshape(-1)


def _dequant_q8_0(blocks: np.ndarray, n: int) -> np.ndarray:
    nb = n // 32
    blocks = blocks[: nb * 34].reshape(nb, 34)
    d = blocks[:, :2].view(np.float16).astype(np.float32).reshape(nb)
    qs = blocks[:, 2:].view(np.int8).astype(np.float32)
    return (qs * d[:, None]).reshape(-1)


def unpack_q5_0(blocks: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack Q5_0 into (q5 int8 0..31, d fp32) *without* the -16 offset.

    The GEMV kernel applies ``d * (sum(q5*x) - 16*sum(x))`` itself, so this
    splits the quantized tensor into its two kernel-friendly pieces (int8 values
    + per-block fp32 scale) rather than dequantizing to fp32.
    """
    nb = n // 32
    blocks = blocks[: nb * 22].reshape(nb, 22)
    d = blocks[:, :2].view(np.float16).astype(np.float32).reshape(nb)
    qh = blocks[:, 2:6].view(np.uint32).reshape(nb, 1)
    qs = blocks[:, 6:22]
    j = np.arange(16, dtype=np.uint32)
    lo = ((qs & 0xF) | (((qh >> j) & 1) << 4).astype(np.uint8))
    hi = ((qs >> 4) | (((qh >> (j + 16)) & 1) << 4).astype(np.uint8))
    q5 = np.concatenate([lo, hi], axis=-1).reshape(-1).astype(np.int8)
    return q5, d


def unpack_q8_0(blocks: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack Q8_0 into (qs int8, d fp32) for the quantized GEMV kernel."""
    nb = n // 32
    blocks = blocks[: nb * 34].reshape(nb, 34)
    d = blocks[:, :2].view(np.float16).astype(np.float32).reshape(nb)
    qs = blocks[:, 2:].view(np.int8)
    return qs.reshape(-1), d


def unpack_q6_k(blocks: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack Q6_K into (q int8 -32..31, dsc fp32 per-16) for the GEMV kernel.

    Each 256-element super-block stores ql(128)+qh(64)+scales(16 int8)+d(fp16).
    Per 16-element sub-block the value is ``(6bit - 32) * d * scale``; the
    kernel reads the pre-applied combined scale ``dsc = d * scale`` so it only
    does ``sum(dsc * q * x)``.
    """
    nb = n // 256
    blocks = blocks[: nb * 210].reshape(nb, 210)
    ql = blocks[:, 0:128]                                   # (nb, 128)
    qh = blocks[:, 128:192]                                 # (nb, 64)
    sc = blocks[:, 192:208].view(np.int8).astype(np.float32)  # (nb, 16)
    d = blocks[:, 208:210].view(np.float16).astype(np.float32).reshape(nb, 1)

    q = np.empty((nb, 256), dtype=np.int8)
    for h in range(2):
        ql_h = ql[:, h * 64:(h + 1) * 64]                   # (nb, 64)
        qh_h = qh[:, h * 32:(h + 1) * 32]                   # (nb, 32)
        lo = (ql_h & 0xF).astype(np.int16)
        hi = (ql_h >> 4).astype(np.int16)
        b0 = (qh_h & 3).astype(np.int16)
        b1 = ((qh_h >> 2) & 3).astype(np.int16)
        b2 = ((qh_h >> 4) & 3).astype(np.int16)
        b3 = ((qh_h >> 6) & 3).astype(np.int16)
        q[:, h * 128 + 0:h * 128 + 32] = (lo[:, :32] | (b0 << 4)) - 32
        q[:, h * 128 + 32:h * 128 + 64] = (lo[:, 32:] | (b1 << 4)) - 32
        q[:, h * 128 + 64:h * 128 + 96] = (hi[:, :32] | (b2 << 4)) - 32
        q[:, h * 128 + 96:h * 128 + 128] = (hi[:, 32:] | (b3 << 4)) - 32
    dsc = (sc * d).astype(np.float32)                       # (nb, 16)
    return q.reshape(-1), dsc.reshape(-1)


def unpack_q4_k(blocks: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack Q4_K into (q int8 0..15, sc fp32 per-32, mn fp32 per-32).

    Each 256-element super-block has 8 sub-blocks of 32 elements; sub-block j
    stores a 4-bit nibble plus a 6-bit scale ``sc[j]`` and 6-bit min ``mn[j]``:
    ``value = nibble * sc[j] * d - mn[j] * dmin``. The kernel applies the
    combined ``sc*d`` and ``mn*dmin`` so it only does ``sum(sc*q*x - mn*x)``.
    """
    nb = n // 256
    blocks = blocks[: nb * 144].reshape(nb, 144)
    d = blocks[:, :2].view(np.float16).astype(np.float32).reshape(nb)
    dmin = blocks[:, 2:4].view(np.float16).astype(np.float32).reshape(nb)
    scales = blocks[:, 4:16].astype(np.int32)               # (nb, 12)
    qs = blocks[:, 16:144]                                  # (nb, 128)

    q = np.empty((nb, 256), dtype=np.int8)
    sc_out = np.empty((nb, 8), dtype=np.float32)
    mn_out = np.empty((nb, 8), dtype=np.float32)
    for k in range(4):
        qseg = qs[:, k * 32:(k + 1) * 32]
        lo = (qseg & 0xF).astype(np.int8)
        hi = (qseg >> 4).astype(np.int8)
        for which, nib in ((0, lo), (1, hi)):
            j = 2 * k + which
            sc6, mn6 = _scale_min_k4(j, scales)
            q[:, j * 32:(j + 1) * 32] = nib
            sc_out[:, j] = sc6 * d
            mn_out[:, j] = mn6 * dmin
    return q.reshape(-1), sc_out.reshape(-1), mn_out.reshape(-1)


def _scale_min_k4(j: int, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack the j-th 6-bit scale and 6-bit min from a 12-byte scales array."""
    if j < 4:
        s = scales[:, j] & 63
        m = scales[:, j + 4] & 63
    else:
        s = (scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4)
        m = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    return s.astype(np.float32), m.astype(np.float32)


def _dequant_q4_K(blocks: np.ndarray, n: int) -> np.ndarray:
    nb = n // _QK
    blocks = blocks[: nb * 144].reshape(nb, 144)
    d = blocks[:, :2].view(np.float16).astype(np.float32).reshape(nb)
    dmin = blocks[:, 2:4].view(np.float16).astype(np.float32).reshape(nb)
    scales = blocks[:, 4:16].astype(np.int32)
    qs = blocks[:, 16:144]

    out = np.empty((nb, _QK), dtype=np.float32)
    # 8 sub-blocks of 32; each 32-byte qs segment feeds two sub-blocks
    # (low nibble -> even sub-block, high nibble -> odd sub-block).
    for k in range(4):
        qseg = qs[:, k * 32:(k + 1) * 32]
        lo = (qseg & 0xF).astype(np.float32)
        hi = (qseg >> 4).astype(np.float32)
        for which, nib in ((0, lo), (1, hi)):
            j = 2 * k + which
            sc, mn = _scale_min_k4(j, scales)
            out[:, j * 32:(j + 1) * 32] = (sc * d)[:, None] * nib - (mn * dmin)[:, None]
    return out.reshape(-1)


def _dequant_q6_K(blocks: np.ndarray, n: int) -> np.ndarray:
    nb = n // _QK
    blocks = blocks[: nb * 210].reshape(nb, 210)
    # block_q6_K layout: ql(128) + qh(64) + scales(16) + d(2)  [d is LAST]
    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    sc = blocks[:, 192:208].view(np.int8).astype(np.float32)  # 16 scales
    d = blocks[:, 208:210].view(np.float16).astype(np.float32).reshape(nb, 1)

    out = np.empty((nb, _QK), dtype=np.float32)
    for n_half in range(2):
        ql_h = ql[:, n_half * 64:(n_half + 1) * 64]   # (nb, 64)
        qh_h = qh[:, n_half * 32:(n_half + 1) * 32]   # (nb, 32)
        sc_h = sc[:, n_half * 8:(n_half + 1) * 8]     # (nb, 8)
        lo = (ql_h & 0xF).astype(np.float32)
        hi = (ql_h >> 4).astype(np.float32)
        bits = np.stack([qh_h & 3, (qh_h >> 2) & 3, (qh_h >> 4) & 3, (qh_h >> 6) & 3], axis=-1).astype(np.int32)  # (nb,32,4)
        # q1 (lo[l]) uses scale[is], q2 (lo[l+32]) scale[is+2], q3 (hi[l]) scale[is+4], q4 (hi[l+32]) scale[is+6]
        # is = l//16
        is_idx = (np.arange(32) // 16).astype(np.int64)
        q1 = (lo[:, :32] + (bits[:, :, 0] << 4) - 32) * d * sc_h[:, is_idx]
        q2 = (lo[:, 32:] + (bits[:, :, 1] << 4) - 32) * d * sc_h[:, is_idx + 2]
        q3 = (hi[:, :32] + (bits[:, :, 2] << 4) - 32) * d * sc_h[:, is_idx + 4]
        q4 = (hi[:, 32:] + (bits[:, :, 3] << 4) - 32) * d * sc_h[:, is_idx + 6]
        out[:, n_half * 128 + 0: n_half * 128 + 32] = q1
        out[:, n_half * 128 + 32: n_half * 128 + 64] = q2
        out[:, n_half * 128 + 64: n_half * 128 + 96] = q3
        out[:, n_half * 128 + 96: n_half * 128 + 128] = q4
    return out.reshape(-1)


def _dequant_q8_K(blocks: np.ndarray, n: int) -> np.ndarray:
    nb = n // _QK
    blocks = blocks[: nb * 292].reshape(nb, 292)
    d = blocks[:, :4].view(np.float32).reshape(nb, 1)
    qs = blocks[:, 4:260].view(np.int8).astype(np.float32)
    return (qs * d).reshape(-1)


def dequantize(ggml_type: int, raw: bytes, n: int) -> np.ndarray:
    """Dequantize ``n`` elements of ``raw`` bytes into a float32 array."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    if ggml_type == GGML_F32:
        return np.frombuffer(raw, dtype=np.float32).astype(np.float32).reshape(-1)
    if ggml_type == GGML_F16:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(-1)
    if ggml_type == GGML_Q8_0:
        return _dequant_q8_0(arr, n)
    if ggml_type == GGML_Q5_0:
        return _dequant_q5_0(arr, n)
    if ggml_type == GGML_Q4_K:
        return _dequant_q4_K(arr, n)
    if ggml_type == GGML_Q6_K:
        return _dequant_q6_K(arr, n)
    if ggml_type == GGML_Q8_K:
        return _dequant_q8_K(arr, n)
    raise NotImplementedError(f"dequantize for ggml type {ggml_type} not implemented")
