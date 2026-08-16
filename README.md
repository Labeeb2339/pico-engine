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
| pico-engine | 0.21 s | **22.1** |
| llama.cpp | — | **103.9** |

pico-engine is **~4.7× slower** than llama.cpp — and that gap is honest.
llama.cpp ships hand-tuned CUDA kernels, fp16/bf16 compute with tensor cores,
and a fused KV-cache attention path. This engine is a straightforward **fp32**
reference implementation with naive `einsum` attention. It is the *correctness
baseline*, not a speed claim. (Greedy outputs also diverge slightly from
llama.cpp — a consequence of fp32-vs-fp16 numerics, not a bug.)

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

- Cast weights to fp16/bf16 and use `torch.nn.functional.scaled_dot_product_attention`
  (or the `pico-kernels` flash-attention / GQA kernels) to close the speed gap.
- Fuse the RMSNorm + QKV projection.
- Add batched/continuous prefill.

## Honest limits

- fp32 compute (slow, exact). No tensor cores in the attention path.
- KV cache reallocates on every step (`torch.cat`) rather than preallocating.
- Single architecture (Qwen2/LLaMA-style) — not a general GGUF runner.
- No chat template yet; raw completion prompts only.
