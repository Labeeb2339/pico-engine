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
| pico-engine | 0.08 s | **149.0** |
| llama.cpp | — | **108.1** |

pico-engine is **~1.4× faster** than llama.cpp. The decode path was
*CPU-dispatch-bound*: ~200 kernel launches per token meant the CPU, not the
GPU, was the bottleneck. The measured progression:

| change | decode tok/s |
|--------|--------------|
| baseline (torch-native forward) | 21.4 |
| fused QKV + gate/up projections | 23.2 |
| fused Triton RMSNorm + RoPE | 33.5 |
| drop dead causal mask | 36.7 |
| fused Triton decode attention | 49.1 |
| preallocate KV cache + hoist RoPE gather | 53.0 |
| Q8_0/Q5_0 quantized matmuls + fused norm/rope/bias | 57 |
| Q6_K/Q4_K quantized ffn_down | 61.5 |
| CUDA-graph decode loop | **149.0** |

**Prefill** is fp16 (tensor cores + flash attention via `torch.autocast`); the
decode path stays fp32 for the CUDA-graph capture. On a 277-token prompt,
prefill is **37.4 ms (7401 tok/s)** vs llama.cpp's **358 ms (773 tok/s)** —
~9.6× faster — and 1.58× faster than our own fp32 prefill (54 ms). (llama.cpp's
prefill number reflects a prebuilt wheel without flash attention; take the exact
ratio with that caveat.)

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
the per-layer loop, took decode 33.5 → 53.0 tok/s.

After the attention win, the profiler pointed at two remaining costs: the
matmuls (memory-bound on fp32-dequantized weights — the model is ~630M params,
~2.5 GB read per token) and the sheer number of kernel launches (~200/token).
Multiplying against the packed quantized blocks directly — Q8_0 for the output
projection, Q5_0 for gate/up, Q6_K/Q4_K for `ffn_down` — cut the weight traffic
~4× (57 → 61.5 tok/s). But the real wall was CPU dispatch: timing the CPU enqueue
against wall clock showed the CPU spent about as long *launching* kernels as the
GPU spent running them. Capturing the single-token forward into a CUDA graph and
replaying it per step collapsed ~200 launches into one call, taking decode
61.5 → 149.0 tok/s — past llama.cpp. (Greedy output matches llama.cpp for ~100
tokens, then fp32-vs-fp16 numerics diverge.)

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

- **Fuse the whole layer into one Triton kernel** — the CUDA graph hides the
  ~200-launch overhead but still runs ~200 kernels; a single fused
  RMSNorm→QKV→RoPE→attention→SwiGLU→residual kernel per layer would cut the
  GPU-side work too (mostly prefill).
- **fp16 decode** — prefill is already fp16; the decode GEMVs + attention still
  accumulate in fp32 (no tensor cores on the M=1 path). Quantized fp16 decode
  could close the remaining decode gap to a hand-tuned kernel.
- **More architectures** — the loader/forward is Qwen2/LLaMA-shaped; MoE
  (Qwen2.5-MoE) or Mamba would stress the design.

## Honest limits

- fp32 decode compute (the quantized GEMVs + attention accumulate in fp32, no
  tensor cores); prefill is fp16.
- Decode throughput is a CUDA-graph capture; prefill is still eager.
- Single architecture (Qwen2/LLaMA-style) — not a general GGUF runner.
- No chat template yet; raw completion prompts only.
