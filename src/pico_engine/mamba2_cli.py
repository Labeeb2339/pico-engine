"""Command-line entry point for raw Mamba-2 checkpoints."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .gguf import load as load_gguf
from .mamba2 import Mamba2Model
from .tokenizer import from_gguf as tokenizer_from_gguf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a raw Mamba-2 checkpoint with pico-engine",
        epilog=(
            "example: pico-engine-mamba2 models/mamba2-130m.bin "
            "--tokenizer-gguf models/mamba-130m-f16.gguf --prompt Hello"
        ),
    )
    parser.add_argument("checkpoint", type=Path, help="path to pytorch_model.bin")
    parser.add_argument(
        "--tokenizer-gguf",
        type=Path,
        required=True,
        help="GGUF whose tokenizer metadata matches the checkpoint",
    )
    parser.add_argument("--prompt", default="Hello", help="input prompt")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint file does not exist: {args.checkpoint}")
    if not args.tokenizer_gguf.is_file():
        parser.error(f"tokenizer GGUF does not exist: {args.tokenizer_gguf}")
    if args.max_tokens < 0:
        parser.error("--max-tokens must be non-negative")
    if not torch.cuda.is_available():
        parser.error(
            "Mamba-2 inference requires CUDA-enabled PyTorch and an NVIDIA GPU"
        )

    tokenizer = tokenizer_from_gguf(load_gguf(args.tokenizer_gguf).metadata)
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        parser.error("the prompt produced no tokens")

    started = time.perf_counter()
    model = Mamba2Model(args.checkpoint, device="cuda")
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    all_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_tokens,
        eos_id=tokenizer.eos_id,
    )
    generated_ids = all_ids[len(prompt_ids) :]
    generated = tokenizer.decode(generated_ids)
    generation_seconds = time.perf_counter() - started

    rate = len(generated_ids) / generation_seconds if generation_seconds else 0.0
    print(
        f"\n[mamba2] loaded in {load_seconds:.1f}s; generated {len(generated_ids)} tokens "
        f"in {generation_seconds:.2f}s ({rate:.1f} tok/s)"
    )
    print("=" * 60)
    print(args.prompt, end="")
    print(generated, flush=True)


if __name__ == "__main__":
    main()
