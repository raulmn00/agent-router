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
    HARD_EXAMPLES,
    HARD_HELD_OUT_FOR_TESTSET,
    PER_CLASS_EASY,
    PER_CLASS_HARD,
    PER_CLASS_TARGET,
    build_dataset,
    load_jsonl,
    regenerate,
)


def test_intents_jsonl_exists_at_expected_path():
    assert DATA_PATH.exists(), (
        f"Expected dataset at {DATA_PATH}. Generate with `python -m router.dataset`."
    )


def test_jsonl_schema_text_label_and_optional_difficulty():
    """Every row has `text` + `label`. `difficulty` is additive (optional in the
    schema) but currently set on every row by the generator — old consumers
    that read only `text`/`label` keep working."""
    rows = load_jsonl()
    assert rows, "dataset is empty"
    required = {"text", "label"}
    allowed = required | {"difficulty"}
    valid_difficulties = {"easy", "hard"}
    for row in rows:
        keys = set(row.keys())
        assert required.issubset(keys), f"missing required keys: {row}"
        assert keys.issubset(allowed), f"unexpected extra keys in: {row}"
        assert isinstance(row["text"], str) and row["text"].strip(), "empty text"
        assert row["label"] in INTENTS, f"unknown label: {row['label']}"
        if "difficulty" in row:
            assert row["difficulty"] in valid_difficulties, (
                f"unknown difficulty: {row['difficulty']}"
            )


def test_classes_are_balanced_at_per_class_target():
    rows = load_jsonl()
    counts = collections.Counter(r["label"] for r in rows)
    assert set(counts.keys()) == set(INTENTS), f"missing classes: {counts}"
    for label in INTENTS:
        assert counts[label] == PER_CLASS_TARGET, (
            f"class {label} has {counts[label]}, expected {PER_CLASS_TARGET}"
        )


def test_easy_and_hard_counts_per_class():
    """Each class has exactly PER_CLASS_EASY easy + PER_CLASS_HARD hard rows."""
    rows = load_jsonl()
    by_pair = collections.Counter((r["label"], r["difficulty"]) for r in rows)
    for label in INTENTS:
        assert by_pair[(label, "easy")] == PER_CLASS_EASY, (
            f"class {label} easy: got {by_pair[(label, 'easy')]}, "
            f"expected {PER_CLASS_EASY}"
        )
        assert by_pair[(label, "hard")] == PER_CLASS_HARD, (
            f"class {label} hard: got {by_pair[(label, 'hard')]}, "
            f"expected {PER_CLASS_HARD}"
        )


def test_no_duplicate_texts_within_a_class():
    rows = load_jsonl()
    by_class: dict[str, list[str]] = collections.defaultdict(list)
    for r in rows:
        by_class[r["label"]].append(r["text"])
    for label, texts in by_class.items():
        assert len(texts) == len(set(texts)), f"duplicates in {label}"


def test_no_overlap_between_easy_and_hard_within_class():
    """A text shouldn't appear as both easy and hard for the same class. The
    builder also asserts this at generation time; here we lock the artifact."""
    rows = load_jsonl()
    by_class_difficulty: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for r in rows:
        by_class_difficulty[(r["label"], r["difficulty"])].add(r["text"])
    for label in INTENTS:
        easy = by_class_difficulty[(label, "easy")]
        hard = by_class_difficulty[(label, "hard")]
        overlap = easy & hard
        assert not overlap, f"{label}: easy/hard texts overlap: {overlap}"


def test_no_overlap_between_train_and_held_out_hard():
    """The HARD_HELD_OUT_FOR_TESTSET pool must NEVER leak into training. This
    is what makes the eval testset actually held-out."""
    for label in INTENTS:
        train = set(HARD_EXAMPLES[label])
        held_out = set(HARD_HELD_OUT_FOR_TESTSET[label])
        overlap = train & held_out
        assert not overlap, (
            f"{label}: training hard pool overlaps with held-out testset pool: "
            f"{overlap}"
        )


def test_hard_pools_meet_minimum_size():
    """Pool sizes are pinned so a slip in HARD_EXAMPLES (e.g. accidentally
    removing one) doesn't silently shrink the dataset."""
    for label in INTENTS:
        assert len(HARD_EXAMPLES[label]) >= PER_CLASS_HARD, (
            f"HARD_EXAMPLES[{label!r}] has {len(HARD_EXAMPLES[label])} entries; "
            f"need at least {PER_CLASS_HARD}."
        )
        assert len(HARD_HELD_OUT_FOR_TESTSET[label]) >= 3, (
            f"HARD_HELD_OUT_FOR_TESTSET[{label!r}] has "
            f"{len(HARD_HELD_OUT_FOR_TESTSET[label])} entries; need at least 3."
        )


def test_build_dataset_is_deterministic():
    a = build_dataset(seed=42)
    b = build_dataset(seed=42)
    assert a == b, "build_dataset must be deterministic given the same seed"


def test_build_dataset_emits_difficulty_field_on_every_row():
    rows = build_dataset(per_class_easy=10, per_class_hard=5)
    for row in rows:
        assert "difficulty" in row, f"missing difficulty in: {row}"
        assert row["difficulty"] in {"easy", "hard"}


def test_regenerate_writes_jsonl(tmp_path):
    out = tmp_path / "intents.jsonl"
    regenerate(path=out, per_class_easy=20, per_class_hard=5)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == (20 + 5) * len(INTENTS)
    counts = collections.Counter((r["label"], r["difficulty"]) for r in rows)
    for label in INTENTS:
        assert counts[(label, "easy")] == 20
        assert counts[(label, "hard")] == 5


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
