"""Pure metric & cost functions — testable without network.

Cost assumptions are documented inline below so the comparison stays honest
even when prices change. Update PRICING when OpenAI's price page moves.
"""

from __future__ import annotations

from typing import Sequence


# --------------------------------------------------------------------------- #
# Quality metrics                                                              #
# --------------------------------------------------------------------------- #


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def f1_macro(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Macro-averaged F1 — equal weight per class, no sklearn dependency."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    classes = sorted(set(y_true) | set(y_pred))
    if not classes:
        return 0.0

    f1s = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def mean_latency_ms(samples_ms: Sequence[float]) -> float:
    if not samples_ms:
        return 0.0
    return sum(samples_ms) / len(samples_ms)


# --------------------------------------------------------------------------- #
# Cost model                                                                   #
# --------------------------------------------------------------------------- #
#
# Prices in USD per 1M tokens, as of late 2025. Sources:
#   - gpt-4o-mini: https://openai.com/api/pricing
#   - text-embedding-3-small: same page
# Update these when prices change; tests pin against these constants.

PRICING_USD_PER_M_TOKENS = {
    "gpt-4o-mini_input": 0.150,
    "gpt-4o-mini_output": 0.600,
    "text-embedding-3-small": 0.020,
}

# Per-classification token assumptions used to derive cost_per_1k. These are
# placeholders, not measurements — replace with actual token counts captured
# from a real run if you want a precise number. The current values are sized
# to roughly match the prompts in llm_router.py / embed_router.py.
TOKEN_ASSUMPTIONS = {
    "llm": {  # zero-shot LLM router
        "input_tokens": 60,   # system prompt + user text
        "output_tokens": 5,    # just the intent name
    },
    "embed": {  # embeddings + LogReg
        "input_tokens": 15,    # just the user text — no system prompt
    },
    "distilbert": {
        # No API call — local CPU inference is treated as $0 per 1k for the
        # comparison. The latency column reflects the real compute cost.
    },
}


def cost_per_1k(approach: str) -> float:
    """USD per 1,000 classifications under the documented assumptions."""
    approach = approach.lower()
    if approach == "distilbert":
        return 0.0
    if approach == "llm":
        a = TOKEN_ASSUMPTIONS["llm"]
        cost_per_call = (
            a["input_tokens"] / 1_000_000 * PRICING_USD_PER_M_TOKENS["gpt-4o-mini_input"]
            + a["output_tokens"] / 1_000_000 * PRICING_USD_PER_M_TOKENS["gpt-4o-mini_output"]
        )
        return cost_per_call * 1_000
    if approach == "embed":
        a = TOKEN_ASSUMPTIONS["embed"]
        cost_per_call = (
            a["input_tokens"] / 1_000_000 * PRICING_USD_PER_M_TOKENS["text-embedding-3-small"]
        )
        return cost_per_call * 1_000
    raise ValueError(f"unknown approach: {approach!r}")
