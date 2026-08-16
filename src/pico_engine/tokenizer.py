"""Byte-level BPE tokenizer, reconstructed from the GGUF tokenizer metadata.

Qwen2.5 uses a GPT-2-style byte-level BPE. The GGUF file embeds the vocab
(``tokenizer.ggml.tokens``) and merge ranks (``tokenizer.ggml.merges``); this
module rebuilds the encoder/decoder from those, plus the standard GPT-2
byte-to-unicode table and split pattern.
"""

from __future__ import annotations

import regex as re
from pathlib import Path


def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}

# GPT-2 split pattern (also used by the Qwen2.5 BPE family).
_SPLIT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    def __init__(self, vocab: list[str], merges: list[str], bos_id: int, eos_id: int):
        self.vocab = vocab
        self.vocab_id = {t: i for i, t in enumerate(vocab)}
        # merge rank: "a b" -> rank (lower = earlier = higher priority)
        self.merges = {}
        for i, m in enumerate(merges):
            self.merges[m] = i
        self.bos_id = bos_id
        self.eos_id = eos_id

    def _bpe(self, token: str) -> list[str]:
        word = list(token)
        while len(word) > 1:
            best_rank = float("inf")
            best_i = -1
            for i in range(len(word) - 1):
                rank = self.merges.get(word[i] + " " + word[i + 1])
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_i = i
            if best_i < 0:
                break
            # merge best_i and best_i+1
            merged = word[best_i] + word[best_i + 1]
            word = word[:best_i] + [merged] + word[best_i + 2:]
        return word

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in _SPLIT.finditer(text):
            piece = match.group(0)
            chars = "".join(_BYTE_ENCODER[b] for b in piece.encode("utf-8"))
            for sub in self._bpe(chars):
                if sub in self.vocab_id:
                    ids.append(self.vocab_id[sub])
                else:
                    # fall back to raw bytes (should not happen for a valid vocab)
                    for c in sub:
                        b = _BYTE_DECODER[c]
                        ids.append(self.vocab_id[_BYTE_ENCODER[b]])
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.vocab[i] for i in ids)
        # undo byte-encoder mapping
        return bytearray(_BYTE_DECODER[c] for c in text).decode("utf-8", errors="replace")

    @property
    def n_vocab(self) -> int:
        return len(self.vocab)


def from_gguf(metadata: dict) -> BPETokenizer:
    """Build a BPE tokenizer from a parsed GGUF metadata dict."""
    model = metadata.get("tokenizer.ggml.model", "")
    if model != "gpt2":
        raise ValueError(f"unsupported tokenizer model {model!r} (only gpt2/BPE)")
    vocab = metadata["tokenizer.ggml.tokens"]
    merges = metadata.get("tokenizer.ggml.merges", [])
    bos = int(metadata.get("tokenizer.ggml.bos_token_id", -1))
    eos = int(metadata.get("tokenizer.ggml.eos_token_id", -1))
    return BPETokenizer(vocab, merges, bos, eos)
