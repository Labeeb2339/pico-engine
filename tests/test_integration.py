"""End-to-end integration: load a real GGUF and generate coherent text.

GPU-gated, and skipped unless a model is provided via ``--model``.
"""

import os
import sys

import pytest

MODEL = os.environ.get("PICO_ENGINE_MODEL", "")


@pytest.mark.skipif(not MODEL, reason="set PICO_ENGINE_MODEL to run integration test")
def test_generate_coherent():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")

    from pico_engine.engine import Engine

    eng = Engine(MODEL)
    out, ids = eng.generate("The capital of France is", max_tokens=16, temperature=0.0)
    assert "Paris" in out, f"expected 'Paris', got {out!r}"
    assert len(ids) > 0


@pytest.mark.skipif(not MODEL, reason="set PICO_ENGINE_MODEL to run integration test")
def test_decode_graph_matches_eager():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")

    from pico_engine.engine import Engine

    eng = Engine(MODEL)
    ids = eng.tok.encode("The capital of France is")
    cache = eng._empty_cache(len(ids) + 16)
    graph = eng._build_decode_graph(cache)
    assert graph is not None, "graph capture failed unexpectedly"

    tokens = torch.tensor(ids, device=eng.device)
    positions = torch.arange(len(ids), device=eng.device)
    logits = eng.model.forward(tokens, positions, cache)

    nxt = int(logits.argmax())
    for step in range(8):
        pos = len(ids) + step
        eager = eng.model.forward(
            torch.tensor([nxt], device=eng.device),
            torch.tensor([pos], device=eng.device),
            cache,
        ).clone()
        graph_logits = eng._decode_step(nxt, pos, cache, graph).clone()
        assert torch.allclose(eager, graph_logits, atol=1e-4), \
            f"graph diverged from eager at step {step}"
        nxt = int(eager.argmax())
