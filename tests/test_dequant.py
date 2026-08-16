"""Dequantization correctness — hand-crafted blocks with known values.

These tests pin the exact bit-level decoding for each GGML block format,
implemented from ggml's dequantize_row_* reference.
"""

import struct

import numpy as np
import pytest

from pico_engine.dequant import (
    GGML_Q5_0, GGML_Q6_K, GGML_Q8_0, GGML_Q4_K, dequantize, nbytes_for,
)


def _fp16(x: float) -> bytes:
    return struct.pack("<H", np.float16(x).view(np.uint16))


def test_q8_0_known_values():
    scale = 0.5
    qs = (np.arange(32, dtype=np.int8) - 16).astype(np.int8)  # -16..15
    block = _fp16(scale) + qs.tobytes()
    out = dequantize(GGML_Q8_0, block, 32)
    assert out.shape == (32,)
    assert np.allclose(out, qs.astype(np.float32) * scale)


def test_q5_0_nibble_and_highbit():
    scale = 1.0
    qh = 0b10000000000000000000000000000000  # set bit 31 (high bit of element 31)
    qs = np.zeros(16, dtype=np.uint8)
    qs[0] = 0x53  # low nibble 3, high nibble 5
    block = _fp16(scale) + struct.pack("<I", qh) + qs.tobytes()
    out = dequantize(GGML_Q5_0, block, 32)
    # element 0 = low nibble(3) | high-bit 0  - 16 = -13
    # element 16 = high nibble(5) | high-bit 0 - 16 = -11
    # element 31 = nibble 0 | high-bit 1 - 16 = 0 (bit 31 -> element 31's 5th bit)
    assert out[0] == -13 and out[16] == -11 and out[31] == 0.0
    assert out[1] == -16  # untouched nibble 0, no high bit


def test_q4_k_zero_nibbles_with_scale():
    # 144-byte block: d, dmin, 12 scale bytes, 128 qs bytes.
    # d = 1.0, dmin = 0.5, all scales/min = 1, all nibbles = 0
    block = _fp16(1.0) + _fp16(0.5)
    block += bytes([0x01] * 12)  # scale[0..3]=1, min[0..3]=1, rest packed
    # note: 6-bit packing for j>=4 is non-trivial; this tests j<4 path.
    block += bytes(128)  # all nibbles 0
    out = dequantize(GGML_Q4_K, block, 256)
    # first 4 sub-blocks: scale=1, min=1 -> weight = d*1*0 - dmin*1 = -0.5
    assert np.allclose(out[:128], -0.5)


def test_q6_k_zero_with_scale():
    # 210-byte block: ql(128) + qh(64) + scales(16) + d(2)
    ql = bytes(128)  # all nibbles 0
    qh = bytes(64)   # all high 2-bits 0
    scales = bytes([1] * 16)  # all scales 1
    d = _fp16(0.5)
    block = ql + qh + scales + d
    out = dequantize(GGML_Q6_K, block, 256)
    # weight = d * scale * (q - 32); q = 0 -> -32; so weight = 0.5 * 1 * -32 = -16
    assert np.allclose(out, -16.0)


def test_nbytes_for_roundtrip():
    assert nbytes_for(GGML_Q8_0, 32) == 34
    assert nbytes_for(GGML_Q5_0, 32) == 22
    assert nbytes_for(GGML_Q4_K, 256) == 144
    assert nbytes_for(GGML_Q6_K, 256) == 210
