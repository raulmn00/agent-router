"""Tests for the synthetic dataset generation and split loader.

These tests only depend on stdlib + (for the split test) `datasets` and
`scikit-learn` — they never download a model or hit the network.
"""

from __future__ import annotations

import collections
import importlib.util
import json
from pathlib import Path

import pytest

from router import INTENTS
from router.dataset import (
    DATA_PATH,
    PER_CLASS_TARGET,
    build_dataset,
    load_jsonl,
    regenerate,
)


def test_intents_jsonl_exists_at_expected_path():
    assert DATA_PATH.exists(), (
        f"Expected dataset at {DATA_PATH}. Generate with `python -m router.dataset`."
    )


def test_jsonl_schema_is_text_and_label():
    rows = load_jsonl()
    assert rows, "dataset is empty"
    for row in rows:
        assert set(row.keys()) == {"text", "label"}, f"unexpected keys in {row}"
        assert isinstance(row["text"], str) and row["text"].strip(), "empty text"
        assert row["label"] in INTENTS, f"unknown label: {row['label']}"


def test_classes_are_balanced():
    rows = load_jsonl()
    counts = collections.Counter(r["label"] for r in rows)
    assert set(counts.keys()) == set(INTENTS), f"missing classes: {counts}"
    for label in INTENTS:
        assert counts[label] == PER_CLASS_TARGET, (
            f"class {label} has {counts[label]}, expected {PER_CLASS_TARGET}"
        )


def test_no_duplicate_texts_within_a_class():
    rows = load_jsonl()
    by_class: dict[str, list[str]] = collections.defaultdict(list)
    for r in rows:
        by_class[r["label"]].append(r["text"])
    for label, texts in by_class.items():
        assert len(texts) == len(set(texts)), f"duplicates in {label}"


def test_build_dataset_is_deterministic():
    a = build_dataset(seed=42)
    b = build_dataset(seed=42)
    assert a == b, "build_dataset must be deterministic given the same seed"


def test_regenerate_writes_jsonl(tmp_path):
    out = tmp_path / "intents.jsonl"
    regenerate(path=out, per_class=20)  # smaller so the test is fast
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 20 * len(INTENTS)
    counts = collections.Counter(r["label"] for r in rows)
    for label in INTENTS:
        assert counts[label] == 20


# Stratified-split test is skipped unless `datasets` + `sklearn` are installed —
# the schema checks should be runnable without the heavier ML deps.
_HAS_ML_DEPS = all(
    importlib.util.find_spec(m) is not None for m in ("datasets", "sklearn")
)


@pytest.mark.skipif(not _HAS_ML_DEPS, reason="needs `datasets` and `scikit-learn`")
def test_load_dataset_splits_is_stratified_80_20():
    from router.dataset import load_dataset_splits

    ds = load_dataset_splits(test_size=0.2)
    n_train = len(ds["train"])
    n_test = len(ds["test"])
    total = n_train + n_test

    # 80/20 split with stratification — allow tiny rounding tolerance.
    assert abs(n_test / total - 0.2) < 0.01
    # Each class should appear in both splits in the same proportion as the
    # source distribution (which is exactly balanced).
    for split_name in ("train", "test"):
        counts = collections.Counter(ds[split_name]["label"])
        expected = len(ds[split_name]) / len(INTENTS)
        for label, count in counts.items():
            assert abs(count - expected) <= 1, (
                f"{split_name} not stratified for {label}: {counts}"
            )
