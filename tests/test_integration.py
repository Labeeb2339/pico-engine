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
