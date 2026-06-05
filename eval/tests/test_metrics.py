"""Tests for the pure metric/cost helpers — no network."""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    PRICING_USD_PER_M_TOKENS,
    TOKEN_ASSUMPTIONS,
    accuracy,
    cost_per_1k,
    f1_macro,
    mean_latency_ms,
)


# --------------------------------------------------------------------------- #
# accuracy                                                                     #
# --------------------------------------------------------------------------- #


def test_accuracy_perfect():
    assert accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_accuracy_half():
    assert accuracy(["a", "b", "c", "d"], ["a", "b", "x", "y"]) == 0.5


def test_accuracy_zero():
    assert accuracy(["a", "b"], ["x", "y"]) == 0.0


def test_accuracy_empty_is_zero():
    assert accuracy([], []) == 0.0


def test_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(["a"], ["a", "b"])


# --------------------------------------------------------------------------- #
# f1_macro                                                                     #
# --------------------------------------------------------------------------- #


def test_f1_macro_perfect():
    assert f1_macro(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_f1_macro_all_wrong():
    assert f1_macro(["a", "b"], ["b", "a"]) == 0.0


def test_f1_macro_handles_class_with_no_predictions():
    # 'c' appears in truth but never in predictions → f1=0 for that class.
    # Expected macro = (f1_a + f1_b + 0) / 3.
    y_true = ["a", "a", "b", "b", "c"]
    y_pred = ["a", "a", "b", "b", "a"]
    score = f1_macro(y_true, y_pred)
    # f1_a: tp=2, fp=1, fn=0 → P=2/3, R=1 → F=0.8
    # f1_b: tp=2, fp=0, fn=0 → P=1, R=1 → F=1
    # f1_c: tp=0 → F=0
    expected = (0.8 + 1.0 + 0.0) / 3
    assert math.isclose(score, expected, rel_tol=1e-9)


def test_f1_macro_matches_sklearn_on_a_known_case():
    """Sanity-check our hand-rolled F1 against the canonical reference."""
    sklearn = pytest.importorskip("sklearn.metrics")
    y_true = ["a", "b", "c", "a", "b", "c", "a"]
    y_pred = ["a", "b", "a", "a", "c", "c", "b"]
    ours = f1_macro(y_true, y_pred)
    theirs = sklearn.f1_score(y_true, y_pred, average="macro", labels=["a", "b", "c"])
    assert math.isclose(ours, theirs, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# mean_latency_ms                                                              #
# --------------------------------------------------------------------------- #


def test_mean_latency_basic():
    assert mean_latency_ms([10.0, 20.0, 30.0]) == 20.0


def test_mean_latency_empty():
    assert mean_latency_ms([]) == 0.0


# --------------------------------------------------------------------------- #
# cost_per_1k                                                                  #
# --------------------------------------------------------------------------- #


def test_cost_distilbert_is_zero():
    assert cost_per_1k("distilbert") == 0.0


def test_cost_llm_uses_documented_assumptions():
    a = TOKEN_ASSUMPTIONS["llm"]
    expected_per_call = (
        a["input_tokens"] / 1_000_000 * PRICING_USD_PER_M_TOKENS["gpt-4o-mini_input"]
        + a["output_tokens"] / 1_000_000 * PRICING_USD_PER_M_TOKENS["gpt-4o-mini_output"]
    )
    assert math.isclose(cost_per_1k("llm"), expected_per_call * 1_000)


def test_cost_embed_uses_documented_assumptions():
    a = TOKEN_ASSUMPTIONS["embed"]
    expected_per_call = (
        a["input_tokens"] / 1_000_000 * PRICING_USD_PER_M_TOKENS["text-embedding-3-small"]
    )
    assert math.isclose(cost_per_1k("embed"), expected_per_call * 1_000)


def test_cost_llm_is_more_expensive_than_embed():
    # Sanity ordering — if we change pricing such that this flips, the
    # README narrative needs to flip too. Tripwire test.
    assert cost_per_1k("llm") > cost_per_1k("embed") > cost_per_1k("distilbert")


def test_cost_unknown_approach_raises():
    with pytest.raises(ValueError):
        cost_per_1k("magic")


# --------------------------------------------------------------------------- #
# Concrete numeric expectations (so a price change shows up loud in CI)        #
# --------------------------------------------------------------------------- #


def test_known_pinned_costs():
    # With the current pinned pricing & token assumptions:
    #   LLM:   60*$0.15/1M + 5*$0.60/1M = 9e-6 + 3e-6 = 1.2e-5 per call → $0.012/1k
    #   Embed: 15*$0.02/1M                          = 3e-7 per call    → $0.0003/1k
    assert math.isclose(cost_per_1k("llm"), 0.012, rel_tol=1e-6)
    assert math.isclose(cost_per_1k("embed"), 0.0003, rel_tol=1e-6)
