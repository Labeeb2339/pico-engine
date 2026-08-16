"""Minimal GGUF parser (spec v2/v3).

Reads the header, metadata key/value pairs, and tensor infos. Tensor *data* is
not loaded here — see :mod:`dequant` for the type-aware reader.

Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

GGUF_MAGIC = 0x46554747  # b"GGUF"

# GGUF metadata value types (gguf.md "Value Types")
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int        # byte offset relative to the data-section start
    n_elements: int
    n_bytes: int       # size of the (quantized) tensor data


@dataclass
class GGUF:
    path: Path
    version: int
    metadata: dict[str, object]
    tensors: list[TensorInfo]
    data_start: int    # absolute file offset where tensor data begins
    alignment: int


def _read_string(f: BinaryIO) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, vtype: int) -> object:
    if vtype == _U8:
        return struct.unpack("<B", f.read(1))[0]
    if vtype == _I8:
        return struct.unpack("<b", f.read(1))[0]
    if vtype == _U16:
        return struct.unpack("<H", f.read(2))[0]
    if vtype == _I16:
        return struct.unpack("<h", f.read(2))[0]
    if vtype == _U32:
        return struct.unpack("<I", f.read(4))[0]
    if vtype == _I32:
        return struct.unpack("<i", f.read(4))[0]
    if vtype == _F32:
        return struct.unpack("<f", f.read(4))[0]
    if vtype == _BOOL:
        return struct.unpack("<?", f.read(1))[0]
    if vtype == _U64:
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == _I64:
        return struct.unpack("<q", f.read(8))[0]
    if vtype == _F64:
        return struct.unpack("<d", f.read(8))[0]
    if vtype == _STR:
        return _read_string(f)
    if vtype == _ARR:
        (etype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [_read_value(f, etype) for _ in range(n)]
    raise ValueError(f"unknown GGUF value type {vtype}")


def load(path: str | Path) -> GGUF:
    path = Path(path)
    with open(path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file (magic {magic:#x})")

        metadata: dict[str, object] = {}
        for _ in range(n_kv):
            key = _read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            metadata[key] = _read_value(f, vtype)

        alignment = int(metadata.get("general.alignment", 32))

        from .dequant import nbytes_for

        tensors: list[TensorInfo] = []
        for _ in range(n_tensors):
            name = _read_string(f)
            (ndim,) = struct.unpack("<I", f.read(4))
            shape = struct.unpack(f"<{ndim}Q", f.read(8 * ndim))
            (ggml_type,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            n_elements = 1
            for d in shape:
                n_elements *= d
            tensors.append(
                TensorInfo(name, tuple(shape), ggml_type, offset, n_elements,
                           nbytes_for(ggml_type, n_elements))
            )

        data_start = f.tell()
        # spec: padding aligns the start of tensor_data to a multiple of ALIGNMENT
        data_start += (alignment - (data_start % alignment)) % alignment

    return GGUF(path, version, metadata, tensors, data_start, alignment)
