"""Sampling: temperature, top-k, top-p over logits."""

from __future__ import annotations

import torch


def sample(logits: torch.Tensor, temperature: float = 1.0,
           top_k: int = 0, top_p: float = 1.0) -> int:
    """Draw a single token id from ``logits`` (1-D, raw scores)."""
    logits = logits.float()
    if temperature > 0:
        logits = logits / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.numel()))
            logits[logits < v[-1]] = float("-inf")
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits[sorted_idx[remove]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
    # greedy
    return int(torch.argmax(logits).item())
