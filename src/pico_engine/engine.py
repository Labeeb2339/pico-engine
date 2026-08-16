"""GGUF inference engine: load, dequantize, run generation."""

from __future__ import annotations

from pathlib import Path

import torch

from .dequant import dequantize, unpack_q5_0, unpack_q8_0, unpack_q6_k, unpack_q4_k
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
            gu_ok = True
            for i in range(self.cfg.n_layers):
                g = by_name.get(f"blk.{i}.ffn_gate.weight")
                u = by_name.get(f"blk.{i}.ffn_up.weight")
                if g is None or u is None or g.ggml_type != 6 or u.ggml_type != 6:
                    gu_ok = False
                    break
                Ng, K = g.shape[::-1]
                Nu, _ = u.shape[::-1]
                qg, dg = unpack_q5_0(np.frombuffer(raw_of(g), dtype=np.uint8), Ng * K)
                qu, du = unpack_q5_0(np.frombuffer(raw_of(u), dtype=np.uint8), Nu * K)
                q5 = torch.from_numpy(np.concatenate([qg, qu])).to(self.device).view(Ng + Nu, K)
                d = torch.from_numpy(np.concatenate([dg, du])).to(self.device).view(Ng + Nu, K // 32)
                gu_q5.append(q5)
                gu_d.append(d)
            if gu_ok:
                quant["gu"] = (gu_q5, gu_d)

            # ffn_down (Q6_K or Q4_K per layer) -> per-layer entries ("q6"/"q4"/None)
            down = []
            for i in range(self.cfg.n_layers):
                t = by_name.get(f"blk.{i}.ffn_down.weight")
                if t is None:
                    down.append(None)
                    continue
                N, K = t.shape[::-1]
                raw = np.frombuffer(raw_of(t), dtype=np.uint8)
                if t.ggml_type == 14:  # Q6_K
                    qs, dsc = unpack_q6_k(raw, N * K)
                    down.append(("q6",
                                 torch.from_numpy(qs.reshape(N, K)).to(self.device),
                                 torch.from_numpy(dsc.reshape(N, K // 16)).to(self.device)))
                elif t.ggml_type == 12:  # Q4_K
                    qs, sc, mn = unpack_q4_k(raw, N * K)
                    down.append(("q4",
                                 torch.from_numpy(qs.reshape(N, K)).to(self.device),
                                 torch.from_numpy(sc.reshape(N, K // 32)).to(self.device),
                                 torch.from_numpy(mn.reshape(N, K // 32)).to(self.device)))
                else:
                    down.append(None)
            quant["down"] = down
            return quant

    def _build_decode_graph(self, cache):
        """Capture the single-token decode forward into a CUDA graph.

        The decode step is CPU-dispatch-bound (~200+ kernel launches/token), so
        a graph replay collapses that dispatch to one call. Returns the graph,
        or None if capture fails (falls back to eager).
        """
        self._static_ids = torch.zeros(1, dtype=torch.long, device=self.device)
        self._static_pos = torch.zeros(1, dtype=torch.long, device=self.device)
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.model.forward(self._static_ids, self._static_pos, cache)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._static_logits = self.model.forward(self._static_ids, self._static_pos, cache)
            self._graph = g
            return g
        except Exception:
            self._graph = None
            return None

    def _decode_step(self, token: int, pos: int, cache, graph):
        if graph is not None:
            self._static_ids.copy_(torch.tensor([token], device=self.device))
            self._static_pos.copy_(torch.tensor([pos], device=self.device))
            graph.replay()
            return self._static_logits
        return self.model.forward(
            torch.tensor([token], device=self.device),
            torch.tensor([pos], device=self.device),
            cache,
        )

    @torch.inference_mode()
    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.7,
                 top_k: int = 0, top_p: float = 0.9) -> tuple[str, list[int]]:
        ids = self.tok.encode(prompt)
        if not ids:
            return "", []

        cache = self._empty_cache(len(ids) + max_tokens)

        # Capture the decode graph *before* prefill (the capture warmup writes
        # a scratch position into the cache; the prefill below then overwrites
        # the real positions).
        graph = self._build_decode_graph(cache) if self.device.type == "cuda" else None

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
            logits = self._decode_step(nxt, pos, cache, graph)

        return self.tok.decode(generated), generated
