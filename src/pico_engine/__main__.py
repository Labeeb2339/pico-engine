"""pico-engine CLI: load a GGUF and generate text."""

from __future__ import annotations

import argparse
import time

from .engine import Engine


def main() -> None:
    p = argparse.ArgumentParser(description="Run a GGUF model with a from-scratch engine")
    p.add_argument("model", help="path to the .gguf file")
    p.add_argument("--prompt", default="Hello", help="input prompt")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    args = p.parse_args()

    t0 = time.perf_counter()
    eng = Engine(args.model)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out, ids = eng.generate(args.prompt, args.max_tokens, args.temperature, args.top_k, args.top_p)
    gen_s = time.perf_counter() - t0

    print(f"\n[engine] loaded in {load_s:.1f}s; generated {len(ids)} tokens in {gen_s:.2f}s "
          f"({len(ids)/gen_s:.1f} tok/s)")
    print("=" * 60)
    print(args.prompt, end="")
    print(out, flush=True)


if __name__ == "__main__":
    main()
