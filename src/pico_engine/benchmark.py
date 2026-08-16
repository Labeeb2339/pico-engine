"""Benchmark pico-engine against llama.cpp on the same GGUF.

Reports prefill time and decode tokens/sec for greedy generation. The two
engines share the exact same quantized weights, so this is an apples-to-apples
throughput comparison. llama.cpp is the reference via ``llama-cpp-python``.
"""

from __future__ import annotations

import argparse
import time

import torch

from .engine import Engine


def bench_my_engine(model: str, prompt: str, n_tokens: int):
    eng = Engine(model)
    ids = eng.tok.encode(prompt)
    cache = eng._empty_cache(len(ids) + n_tokens)
    # capture the decode graph before prefill (warmup writes a scratch position)
    graph = eng._build_decode_graph(cache)

    tokens = torch.tensor(ids, device=eng.device)
    positions = torch.arange(len(ids), device=eng.device)

    t0 = time.perf_counter()
    logits = eng.model.forward(tokens, positions, cache)
    prefill_s = time.perf_counter() - t0

    generated: list[int] = []
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        nxt = int(logits.argmax())
        if nxt == eng.tok.eos_id:
            break
        pos = len(ids) + len(generated)
        generated.append(nxt)
        logits = eng._decode_step(nxt, pos, cache, graph)
    decode_s = time.perf_counter() - t0
    return prefill_s, decode_s, len(generated), eng.tok.decode(generated)


def bench_llama_cpp(model: str, prompt: str, n_tokens: int):
    from llama_cpp import Llama

    llm = Llama(model_path=model, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    t0 = time.perf_counter()
    out = llm.create_completion(prompt, max_tokens=n_tokens, temperature=0.0, echo=False)
    total_s = time.perf_counter() - t0
    n_gen = int(out["usage"]["completion_tokens"])
    return total_s, n_gen, out["choices"][0]["text"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", help="path to .gguf")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--n-tokens", type=int, default=128)
    args = p.parse_args()

    print(f"prompt: {args.prompt!r} | {args.n_tokens} tokens greedy\n")

    mine = bench_my_engine(args.model, args.prompt, args.n_tokens)
    ref = bench_llama_cpp(args.model, args.prompt, args.n_tokens)

    print(f"{'':24s} {'pico-engine':>14s} {'llama.cpp':>14s}")
    print(f"{'prefill (s)':24s} {mine[0]:14.3f} {'—':>14s}")
    print(f"{'decode (s)':24s} {mine[1]:14.3f} {ref[0]:14.3f}")
    print(f"{'tokens':24s} {mine[2]:14d} {ref[1]:14d}")
    print(f"{'decode tok/s':24s} {mine[2]/mine[1]:14.1f} {ref[1]/ref[0]:14.1f}")
    print(f"{'speed vs llama.cpp':24s} {((mine[2]/mine[1])/(ref[1]/ref[0])):14.2f}x")
    print(f"\n[pico-engine] {args.prompt}{mine[3]}")
    print(f"[llama.cpp]   {args.prompt}{ref[2]}")


if __name__ == "__main__":
    main()
