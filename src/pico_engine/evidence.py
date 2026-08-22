"""Produce a path-free, machine-readable pico-engine benchmark receipt.

The ordinary demo CLI optimizes for a fast human check.  This command instead
records repeated eager and CUDA-graph measurements together with the exact
model hash, software stack, GPU, Git commit, and benchmark parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .engine import Engine


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def _git_metadata(repository: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", os.fspath(repository), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "working_tree_clean": not bool(status),
    }


def _driver_version() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip().splitlines()[0]


def _run_once(
    engine: Engine,
    prompt_ids: list[int],
    n_tokens: int,
    *,
    use_graph: bool,
) -> dict[str, float | int]:
    cache = engine._empty_cache(len(prompt_ids) + n_tokens)
    graph = engine._build_decode_graph(cache) if use_graph else None
    tokens = torch.tensor(prompt_ids, device=engine.device)
    positions = torch.arange(len(prompt_ids), device=engine.device)

    torch.cuda.synchronize()
    started = time.perf_counter()
    logits = engine.model.forward(tokens, positions, cache)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started

    generated = 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(n_tokens):
        token = int(logits.argmax())
        if token == engine.tok.eos_id:
            break
        position = len(prompt_ids) + generated
        generated += 1
        logits = engine._decode_step(token, position, cache, graph)
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - started
    return {
        "prefill_tokens": len(prompt_ids),
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": len(prompt_ids) / prefill_seconds,
        "decode_tokens": generated,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": generated / decode_seconds
        if decode_seconds
        else 0.0,
    }


def _benchmark_mode(
    engine: Engine,
    prompt_ids: list[int],
    n_tokens: int,
    *,
    use_graph: bool,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        _run_once(engine, prompt_ids, n_tokens, use_graph=use_graph)
    samples = [
        _run_once(engine, prompt_ids, n_tokens, use_graph=use_graph)
        for _ in range(repetitions)
    ]
    return {
        "mode": "cuda_graph" if use_graph else "eager",
        "warmups": warmups,
        "samples": samples,
        "prefill_tokens_per_second": _summary(
            [float(sample["prefill_tokens_per_second"]) for sample in samples]
        ),
        "decode_tokens_per_second": _summary(
            [float(sample["decode_tokens_per_second"]) for sample in samples]
        ),
    }


def _profile_mode(
    engine: Engine,
    prompt_ids: list[int],
    n_tokens: int,
    *,
    use_graph: bool,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "cuda_graph" if use_graph else "eager"
    trace_path = output_dir / f"{mode}-decode-trace.json"
    table_path = output_dir / f"{mode}-decode-operators.txt"
    if trace_path.exists() or table_path.exists():
        raise FileExistsError(f"profile output already exists for {mode}")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profile:
        _run_once(engine, prompt_ids, n_tokens, use_graph=use_graph)
    profile.export_chrome_trace(os.fspath(trace_path))
    table_path.write_text(
        profile.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=40,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "trace": trace_path.name,
        "operator_table": table_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a repeated, machine-readable pico-engine benchmark receipt"
    )
    parser.add_argument("model", type=Path, help="supported dense GGUF model")
    parser.add_argument(
        "--prompt",
        default="Question: What is the capital of Sarawak? Answer:",
    )
    parser.add_argument("--n-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="optional fresh directory for eager and CUDA-graph profiler artifacts",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"model file does not exist: {args.model}")
    if args.n_tokens <= 0:
        parser.error("--n-tokens must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.output.parent.is_dir():
        parser.error(f"output parent does not exist: {args.output.parent}")
    if args.profile_dir is not None and args.profile_dir.exists():
        parser.error(f"profile directory already exists: {args.profile_dir}")
    if not torch.cuda.is_available():
        parser.error("CUDA-enabled PyTorch and an NVIDIA GPU are required")

    repository = Path(__file__).resolve().parents[2]
    source = _git_metadata(repository)
    if not source["working_tree_clean"]:
        parser.error(
            "the repository must be committed and clean before evidence capture"
        )

    import triton

    engine = Engine(args.model)
    prompt_ids = engine.tok.encode(args.prompt)
    modes = [
        _benchmark_mode(
            engine,
            prompt_ids,
            args.n_tokens,
            use_graph=False,
            warmups=args.warmups,
            repetitions=args.repetitions,
        ),
        _benchmark_mode(
            engine,
            prompt_ids,
            args.n_tokens,
            use_graph=True,
            warmups=args.warmups,
            repetitions=args.repetitions,
        ),
    ]
    profiles: dict[str, dict[str, str]] | None = None
    if args.profile_dir is not None:
        profiles = {
            "eager": _profile_mode(
                engine,
                prompt_ids,
                min(args.n_tokens, 16),
                use_graph=False,
                output_dir=args.profile_dir,
            ),
            "cuda_graph": _profile_mode(
                engine,
                prompt_ids,
                min(args.n_tokens, 16),
                use_graph=True,
                output_dir=args.profile_dir,
            ),
        }

    model_hash = _sha256(args.model)
    receipt = {
        "artifact_kind": "pico_engine_benchmark_receipt",
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "triton": triton.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "nvidia_driver": _driver_version(),
        },
        "model": {
            "filename": args.model.name,
            "size_bytes": args.model.stat().st_size,
            "sha256": model_hash,
        },
        "benchmark": {
            "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
            "prompt_token_count": len(prompt_ids),
            "requested_decode_tokens": args.n_tokens,
            "greedy": True,
            "modes": modes,
        },
        "profiles": profiles,
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")


if __name__ == "__main__":
    main()
