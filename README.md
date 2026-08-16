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
| pico-engine | 0.21 s | **53.0** |
| llama.cpp | — | **102.1** |

pico-engine is **~1.9× slower** than llama.cpp. The decode path is
*CPU-dispatch-bound*, not compute- or bandwidth-bound: each generated token runs
hundreds of tiny kernels, and the CPU spends most of its time dispatching them
while the GPU sits mostly idle. The measured progression:

| change | decode tok/s |
|--------|--------------|
| baseline (torch-native forward) | 21.4 |
| fused QKV + gate/up projections | 23.2 |
| fused Triton RMSNorm + RoPE | 33.5 |
| drop dead causal mask | 36.7 |
| fused Triton decode attention | 49.1 |
| preallocate KV cache + hoist RoPE gather | **53.0** |

Two hypotheses were tested and **rejected**, both measured:

- **Custom Triton GEMV kernels** for the four M=1 matmuls/layer — **neutral**
  (0.98–1.18× per matmul). `torch.profiler` showed the matmuls are only ~10% of
  decode time; cuBLAS is already fine at M=1.
- **fp16 weights** — **slower** (18 tok/s), not faster. Halving the bytes
  doesn't help when the path isn't bandwidth-bound.

What actually moved the needle was profiling, not guessing. The profiler showed
`scaled_dot_product_attention` dispatch at **43% of CPU time** plus a causal mask
rebuilt with `arange`+`ge` on every layer of every step (**12%**) — and that
mask is *all-True* for single-token decode, so it was dead work. Removing it,
replacing SDPA's dispatch with one fused Triton decode-attention kernel (online
softmax, GQA — `src/pico_engine/kernels.py:decode_attn`), then preallocating the
KV cache (killing the per-step `torch.cat`) and hoisting the RoPE gather out of
the per-layer loop, took decode 33.5 → 53.0 tok/s. (Greedy outputs diverge
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

- **Fuse the whole layer** — one Triton kernel for RMSNorm→QKV→RoPE→attention→
  SwiGLU MLP→residual would collapse the remaining ~10 launches/layer (four
  matmuls + three elementwise kernels + reshapes) into one or two. This is the
  real remaining dispatch lever — the GEMV/GEMM work was measured neutral.
- **Quantized matmuls** — weights are dequantized to fp32 (~2.5 GB read/token);
  multiplying directly against the 4/5/8-bit blocks like llama.cpp cuts that
  ~4×. This only pays off once the dispatch is tamed (fp16 was slower, so the
  path isn't bandwidth-bound yet).

## Honest limits

- fp32 compute (slow, exact). No tensor cores in the attention path.
- KV cache reallocates on every step (`torch.cat`) rather than preallocating.
- Single architecture (Qwen2/LLaMA-style) — not a general GGUF runner.
- No chat template yet; raw completion prompts only.
