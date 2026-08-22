"""pico-engine CLI: load a GGUF and generate text."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .engine import Engine


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run a GGUF model with the from-scratch pico-engine",
        epilog=(
            "example: pico-engine models/qwen2.5-0.5b-instruct-q4_k_m.gguf "
            '--prompt "Question: What is the capital of Sarawak? Answer:" '
            "--max-tokens 11 --temperature 0"
        ),
    )
    p.add_argument("model", type=Path, help="path to the .gguf file")
    p.add_argument("--prompt", default="Hello", help="input prompt")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="MoE layers to offload to CUDA (dense models already run on CUDA)",
    )
    p.add_argument(
        "--no-graph",
        action="store_true",
        help="disable CUDA-graph decode and use the slower eager fallback",
    )
    args = p.parse_args()

    if not args.model.is_file():
        p.error(f"model file does not exist: {args.model}")
    if args.max_tokens < 0:
        p.error("--max-tokens must be non-negative")
    if args.n_gpu_layers < 0:
        p.error("--n-gpu-layers must be non-negative")
    if not torch.cuda.is_available():
        p.error(
            "CUDA-enabled PyTorch and an NVIDIA GPU are required by this CLI; "
            "on Windows, install the CUDA wheel shown in README.md"
        )

    t0 = time.perf_counter()
    eng = Engine(args.model, n_gpu_layers=args.n_gpu_layers)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out, ids = eng.generate(
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        use_graph=not args.no_graph,
    )
    gen_s = time.perf_counter() - t0

    print(
        f"\n[engine] loaded in {load_s:.1f}s; generated {len(ids)} tokens in {gen_s:.2f}s "
        f"({len(ids) / gen_s:.1f} tok/s)"
    )
    print("=" * 60)
    print(args.prompt, end="")
    print(out, flush=True)


if __name__ == "__main__":
    main()
