"""GGUF inference engine: load, dequantize, run generation."""

from __future__ import annotations

from pathlib import Path

import torch

from .dequant import dequantize, unpack_q5_0, unpack_q8_0
from .gguf import load as load_gguf
from .model import ModelConfig, Transformer
from .sampler import sample as sample_token
from .tokenizer import from_gguf as tokenizer_from_gguf


class Engine:
    def __init__(self, gguf_path: str | Path, device: str = "cuda"):
        self.device = torch.device(device)
        self.gguf = load_gguf(gguf_path)
        self.cfg = self._build_config()
        self.weights = self._load_weights()
        self.quant = self._prep_quant()
        self.model = Transformer(self.cfg, self.weights, self.device, self.quant)
        self.tok = tokenizer_from_gguf(self.gguf.metadata)

    def _build_config(self) -> ModelConfig:
        m = self.gguf.metadata
        arch = m["general.architecture"]
        n_layers = int(m[f"{arch}.block_count"])
        hidden = int(m[f"{arch}.embedding_length"])
        n_head = int(m[f"{arch}.attention.head_count"])
        n_kv_head = int(m[f"{arch}.attention.head_count_kv"])
        eps = float(m[f"{arch}.attention.layer_norm_rms_epsilon"])
        rope_base = float(m[f"{arch}.rope.freq_base"])
        ffn_dim = int(m[f"{arch}.feed_forward_length"])
        context_len = int(m[f"{arch}.context_length"])

        vocab_size = len(m["tokenizer.ggml.tokens"])

        head_dim = None
        for t in self.gguf.tensors:
            if t.name == "blk.0.attn_k.weight":
                head_dim = t.shape[1] // n_kv_head
                break
        assert head_dim is not None, "could not derive head_dim"

        return ModelConfig(vocab_size, n_layers, hidden, n_head, n_kv_head,
                           head_dim, ffn_dim, rope_base, eps, context_len)

    def _load_weights(self) -> dict[str, torch.Tensor]:
        weights: dict[str, torch.Tensor] = {}
        with open(self.gguf.path, "rb") as f:
            for t in self.gguf.tensors:
                f.seek(self.gguf.data_start + t.offset)
                raw = f.read(t.n_bytes)
                arr = dequantize(t.ggml_type, raw, t.n_elements)
                # GGUF lists dims fastest-first (ggml ne[0] = contiguous); numpy is
                # C-order (last dim contiguous), so reverse the shape.
                w = torch.from_numpy(arr).reshape(t.shape[::-1]).to(self.device)
                weights[t.name] = w
        return weights

    def _empty_cache(self, capacity: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        empty = lambda: torch.zeros(self.cfg.n_kv_head, capacity, self.cfg.head_dim, device=self.device)
        return [(empty(), empty()) for _ in range(self.cfg.n_layers)]

    def _prep_quant(self) -> dict:
        """Preprocess quantized tensors into a kernel-friendly split form.

        The model multiplies directly against the quantized values instead of
        fp32-dequantized copies (the fp32 matmul reads ~4x the bytes). Q8_0 /
        Q5_0 become (qs int8, d fp32); the output projection and gate/up use them.
        """
        import numpy as np

        quant: dict = {}
        by_name = {t.name: t for t in self.gguf.tensors}
        with open(self.gguf.path, "rb") as f:
            def raw_of(t):
                f.seek(self.gguf.data_start + t.offset)
                return f.read(t.n_bytes)

            # output projection (Q8_0)
            t = by_name.get("output.weight")
            if t is not None and t.ggml_type == 8:
                N, K = t.shape[::-1]
                qs, d = unpack_q8_0(np.frombuffer(raw_of(t), dtype=np.uint8), N * K)
                quant["output"] = (
                    torch.from_numpy(qs.reshape(N, K)).to(self.device),
                    torch.from_numpy(d.reshape(N, K // 32)).to(self.device),
                )

            # gate + up (Q5_0) fused per layer -> (gu_q5, gu_d) lists
            gu_q5, gu_d = [], []
            for i in range(self.cfg.n_layers):
                g = by_name.get(f"blk.{i}.ffn_gate.weight")
                u = by_name.get(f"blk.{i}.ffn_up.weight")
                if g is None or u is None or g.ggml_type != 6 or u.ggml_type != 6:
                    return quant  # not Q5_0; keep fp32 fallback
                Ng, K = g.shape[::-1]
                Nu, _ = u.shape[::-1]
                qg, dg = unpack_q5_0(np.frombuffer(raw_of(g), dtype=np.uint8), Ng * K)
                qu, du = unpack_q5_0(np.frombuffer(raw_of(u), dtype=np.uint8), Nu * K)
                q5 = torch.from_numpy(np.concatenate([qg, qu])).to(self.device).view(Ng + Nu, K)
                d = torch.from_numpy(np.concatenate([dg, du])).to(self.device).view(Ng + Nu, K // 32)
                gu_q5.append(q5)
                gu_d.append(d)
            quant["gu"] = (gu_q5, gu_d)
        return quant

    @torch.inference_mode()
    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.7,
                 top_k: int = 0, top_p: float = 0.9) -> tuple[str, list[int]]:
        ids = self.tok.encode(prompt)
        cache = self._empty_cache(len(ids) + max_tokens)

        if not ids:
            return "", []

        # prefill
        tokens = torch.tensor(ids, device=self.device)
        positions = torch.arange(len(ids), device=self.device)
        logits = self.model.forward(tokens, positions, cache)

        generated: list[int] = []
        for _ in range(max_tokens):
            nxt = sample_token(logits, temperature, top_k, top_p)
            if nxt == self.tok.eos_id:
                break
            pos = len(ids) + len(generated)  # position of this new token (0-indexed)
            generated.append(nxt)
            logits = self.model.forward(
                torch.tensor([nxt], device=self.device),
                torch.tensor([pos], device=self.device),
                cache,
            )

        return self.tok.decode(generated), generated
