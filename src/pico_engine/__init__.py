"""pico-engine: a from-scratch GGUF inference engine for LLaMA-style models."""

from .gguf import GGUF, load as load_gguf
from . import dequant

__all__ = ["Engine", "ModelConfig", "Transformer", "GGUF", "load_gguf", "dequant"]


def __getattr__(name: str):
    """Lazy-load the CUDA/torch-heavy names.

    ``Engine`` pulls in ``torch`` (via ``engine`` -> ``sampler``) and
    ``ModelConfig``/``Transformer`` pull in ``triton`` (via ``model`` ->
    ``kernels``). Deferring them keeps the lightweight submodules (``dequant``,
    ``tokenizer``, ``gguf``) importable on CPU-only machines, which is what the
    CI exercises.
    """
    if name == "Engine":
        from .engine import Engine
        return Engine
    if name in ("ModelConfig", "Transformer"):
        from .model import ModelConfig, Transformer
        return {"ModelConfig": ModelConfig, "Transformer": Transformer}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
