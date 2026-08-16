# pico-engine

A **from-scratch LLM inference engine**: a GGUF loader + LLaMA-style transformer
forward pass that runs a real quantized model end to end — no `llama.cpp`, no
HuggingFace, no vendor runtime. The whole pipeline is written here:

- **GGUF parser** (`gguf.py`) — header, metadata KV pairs, tensor infos, alignment.
- **Dequantization** (`dequant.py`) — Q5_0, Q8_0, Q4_K, Q6_K, F16, F32 block
  formats, decoded bit-for-bit from ggml's `dequantize_row_*` reference.
- **Transformer forward** (`model.py`) — RMSNorm, rotary embeddings (rotate-half),
  grouped-query attention with a KV cache, SwiGLU MLP.
- **BPE tokenizer** (`tokenizer.py`) — byte-level BPE rebuilt from the GGUF
  metadata (`tokenizer.ggml.tokens` / `tokenizer.ggml.merges`).
- **Sampling** (`sampler.py`) — temperature / top-k / top-p.
- **Benchmark** (`benchmark.py`) — tokens/sec against `llama.cpp`.

It targets **Qwen2.5-0.5B-Instruct** (GGUF) and produces coherent output:

```
$ python -m pico_engine models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --prompt "The capital of France is" --max-tokens 24 --temperature 0

The capital of France is Paris. It is the largest city in Europe and the
second largest in the world. It is also the capital of the department of Paris.
```

## Why this exists

The from-scratch ML series (`pico-kernels`, `picolm`, `pico-diffusion`) builds
every layer of the stack — kernels, training, diffusion. This is the last
missing piece: **serving** a real model. It is, in essence, a small
`llama.cpp`: it reads the same file format, reverses the same quantization,
and runs the same forward pass, all written by hand.

## Verified, not vibed

- **Dequantization is bit-exact.** Every block format matches the reference
  `gguf` package's dequantizer to `0.0` max absolute error (Q5_0, Q8_0, Q4_K,
  Q6_K). `tests/test_dequant.py` pins the bit-level decoding with hand-crafted
  blocks.
- **Tokenizer matches llama.cpp exactly.** Token IDs for arbitrary prompts are
  identical to `llama-cpp-python`'s tokenizer.
- **Output is coherent and correct** — factual prompts, instruction prompts,
  and creative prompts all produce sensible text (see `tests/test_integration.py`).

## Benchmark vs llama.cpp

Same GGUF, same prompt, greedy decoding, 128 tokens, RTX 5070 Laptop:

| engine | prefill | decode tok/s |
|--------|---------|--------------|
| pico-engine | 0.21 s | **23.2** |
| llama.cpp | — | **102.6** |

pico-engine is **~4.4× slower** than llama.cpp — and that gap is honest. The
decode path is *launch-bound*, not bandwidth- or compute-bound: each generated
token runs ~250 small torch ops (24 layers × ~10 kernels), and the M=1 matmuls
never amortize their launch cost. Fusing the per-layer QKV and gate/up
projections into single matmuls (3→1 and 2→1) moved it from 21.4 to 23.2 tok/s;
**fp16 and `scaled_dot_product_attention` did not help** — fp16's memory savings
don't matter when you're not bandwidth-bound, and flash attention pays off at
long sequences, not M=1 decode. llama.cpp's advantage is hand-fused CUDA kernels
(RMSNorm+QKV, fused SwiGLU), which is the real next step. (Greedy outputs diverge
slightly from llama.cpp — fp32-vs-fp16 numerics, not a bug.)

```
$ python -m pico_engine.benchmark models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

## Supported quantization

| GGML type | used for | bytes/block |
|-----------|----------|-------------|
| F32       | norms, QKV biases | 4 / elem |
| F16       | — | 2 / elem |
| Q5_0      | Q/K/output/gate/up weights | 22 / 32 |
| Q8_0      | token embeddings, V, output | 34 / 32 |
| Q4_K      | ffn_down (some layers) | 144 / 256 |
| Q6_K      | ffn_down (some layers) | 210 / 256 |

(The official Qwen "Q4_K_M" file is actually a mix — Q5_0 for most weights,
Q8_0 for embeddings, Q6_K/Q4_K for `ffn_down`. All handled.)

## Install & run

```bash
pip install -r requirements.txt
# download a model, e.g. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
python -m pico_engine <model.gguf> --prompt "Hello" --max-tokens 64
```

## What I would do next

- Write a fused Triton kernel for the FFN (gate+up+SiLU+down in one pass) and a
  fused RMSNorm+QKV — the `pico-kernels` integration that actually closes the
  launch-bound gap. That's the real lever; fp16/SDPA were measured and did not help.
- Preallocate the KV cache (avoid the per-step `torch.cat` reallocation).
- Add batched/continuous prefill.

## Honest limits

- fp32 compute (slow, exact). No tensor cores in the attention path.
- KV cache reallocates on every step (`torch.cat`) rather than preallocating.
- Single architecture (Qwen2/LLaMA-style) — not a general GGUF runner.
- No chat template yet; raw completion prompts only.
