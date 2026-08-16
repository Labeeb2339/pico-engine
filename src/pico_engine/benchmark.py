"""Benchmark pico-engine against llama.cpp on the same GGUF.

Reports prefill and decode tokens/sec for greedy generation. The two engines
share the exact same quantized weights, so this is an apples-to-apples
throughput comparison. llama.cpp is the reference via ``llama-cpp-python``.
"""

from __future__ import annotations

import argparse
import time

import torch

from .engine import Engine


def bench_my_engine(model: str, prompt: str, n_tokens: int, use_graph: bool):
    eng = Engine(model)
    ids = eng.tok.encode(prompt)
    cache = eng._empty_cache(len(ids) + n_tokens)
    # capture the decode graph before prefill (warmup writes a scratch position)
    graph = eng._build_decode_graph(cache) if use_graph else None

    tokens = torch.tensor(ids, device=eng.device)
    positions = torch.arange(len(ids), device=eng.device)

    t0 = time.perf_counter()
    logits = eng.model.forward(tokens, positions, cache)
    torch.cuda.synchronize()
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
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0
    return len(ids), prefill_s, len(generated), decode_s, eng.tok.decode(generated)


def bench_llama_cpp(model: str, prompt: str, n_tokens: int):
    from llama_cpp import Llama

    llm = Llama(model_path=model, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    toks = llm.tokenize(prompt.encode("utf-8"))

    t0 = time.perf_counter()
    llm.eval(toks)
    prefill_s = time.perf_counter() - t0

    generated: list[int] = []
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        tok = llm.sample(temp=0.0)  # greedy
        generated.append(tok)
        if tok == llm.token_eos():
            break
        llm.eval([tok])
    decode_s = time.perf_counter() - t0

    text = llm.detokenize(generated).decode("utf-8", errors="replace")
    return len(toks), prefill_s, len(generated), decode_s, text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", help="path to .gguf")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--n-tokens", type=int, default=128)
    p.add_argument("--no-graph", action="store_true",
                   help="disable the CUDA-graph decode path (eager fallback)")
    args = p.parse_args()

    print(f"prompt: {args.prompt!r} | {args.n_tokens} tokens greedy"
          f" | graph={'off' if args.no_graph else 'on'}\n")

    mine = bench_my_engine(args.model, args.prompt, args.n_tokens, use_graph=not args.no_graph)
    ref = bench_llama_cpp(args.model, args.prompt, args.n_tokens)

    n_pre, pre_s, n_dec, dec_s, _ = mine
    r_pre, r_pre_s, r_dec, r_dec_s, _ = ref

    print(f"{'':24s} {'pico-engine':>14s} {'llama.cpp':>14s}")
    print(f"{'prefill tok/s':24s} {n_pre/pre_s:14.1f} {r_pre/r_pre_s:14.1f}")
    print(f"{'decode tok/s':24s} {n_dec/dec_s:14.1f} {r_dec/r_dec_s:14.1f}")
    print(f"{'decode vs llama.cpp':24s} {(n_dec/dec_s)/(r_dec/r_dec_s):14.2f}x")
    print(f"\n[pico-engine] {args.prompt}{mine[4]}")
    print(f"[llama.cpp]   {args.prompt}{ref[4]}")


if __name__ == "__main__":
    main()
