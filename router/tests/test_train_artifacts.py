"""Tests for `save_evaluation_artifacts` and `save_training_metadata`.

These tests deliberately don't import `train.train()` — they call the two
pure artifact-generation helpers with hand-crafted predictions, so no model
loading, no GPU, no Hugging Face download happens.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from router import INTENTS
from router.train import save_evaluation_artifacts, save_training_metadata


# A 4-class confusion case where every prediction is correct except one
# `chitchat` example that the model thinks is `simple_qa`. That gives us a
# non-trivial off-diagonal cell so the CSV and JSON tests assert real numbers.
#
# Order of INTENTS in router.intents: [simple_qa, complex_task, document_qa, chitchat]
#                                       0           1             2            3
_Y_TRUE = [0, 0, 1, 1, 2, 2, 3, 3]
_Y_PRED = [0, 0, 1, 1, 2, 2, 3, 0]  # last `chitchat` misclassified as `simple_qa`


def _expected_cm_rows():
    """Hand-computed: 2 correct per class except chitchat (1 correct, 1 → simple_qa)."""
    return {
        "simple_qa":    [2, 0, 0, 0],
        "complex_task": [0, 2, 0, 0],
        "document_qa":  [0, 0, 2, 0],
        "chitchat":     [1, 0, 0, 1],
    }


# --------------------------------------------------------------------------- #
# Confusion matrix CSV                                                         #
# --------------------------------------------------------------------------- #


def test_confusion_matrix_csv_has_header_and_rows(tmp_path):
    save_evaluation_artifacts(_Y_TRUE, _Y_PRED, results_dir=tmp_path)
    csv_path = tmp_path / "confusion_matrix.csv"
    assert csv_path.exists(), "confusion_matrix.csv must be written"

    with csv_path.open() as f:
        rows = list(csv.reader(f))

    # First row is the header — empty corner cell + intent names as columns.
    assert rows[0] == [""] + list(INTENTS)

    expected = _expected_cm_rows()
    data_rows = {row[0]: [int(v) for v in row[1:]] for row in rows[1:]}
    for intent, counts in expected.items():
        assert data_rows[intent] == counts, f"row for {intent} should be {counts}, got {data_rows[intent]}"


def test_confusion_matrix_csv_sum_equals_total_predictions(tmp_path):
    save_evaluation_artifacts(_Y_TRUE, _Y_PRED, results_dir=tmp_path)
    with (tmp_path / "confusion_matrix.csv").open() as f:
        rows = list(csv.reader(f))[1:]
    total = sum(int(v) for row in rows for v in row[1:])
    assert total == len(_Y_TRUE)


# --------------------------------------------------------------------------- #
# Confusion matrix PNG                                                         #
# --------------------------------------------------------------------------- #


_HAS_MPL = importlib.util.find_spec("matplotlib") is not None


@pytest.mark.skipif(not _HAS_MPL, reason="needs matplotlib")
def test_confusion_matrix_png_is_written_as_real_png(tmp_path):
    save_evaluation_artifacts(_Y_TRUE, _Y_PRED, results_dir=tmp_path)
    png_path = tmp_path / "confusion_matrix.png"
    assert png_path.exists()
    # Quick PNG signature check — first 8 bytes are the magic header.
    with png_path.open("rb") as f:
        magic = f.read(8)
    assert magic == b"\x89PNG\r\n\x1a\n", f"file is not a PNG: {magic!r}"


def test_artifact_generator_works_without_matplotlib(monkeypatch, tmp_path):
    """If matplotlib isn't installed, the CSV/JSON/TXT outputs must still appear
    and only the PNG is skipped."""
    import sys

    # Standard trick: setting `sys.modules["X"] = None` makes any subsequent
    # `import X` raise ImportError, even if matplotlib was already imported
    # by an earlier test in the session. monkeypatch restores on teardown.
    for mod in ("matplotlib", "matplotlib.pyplot"):
        monkeypatch.setitem(sys.modules, mod, None)

    paths = save_evaluation_artifacts(_Y_TRUE, _Y_PRED, results_dir=tmp_path)
    assert (tmp_path / "confusion_matrix.csv").exists()
    assert (tmp_path / "classification_report.json").exists()
    assert (tmp_path / "classification_report.txt").exists()
    assert not (tmp_path / "confusion_matrix.png").exists()
    assert "confusion_matrix_png" not in paths


# --------------------------------------------------------------------------- #
# Classification report                                                        #
# --------------------------------------------------------------------------- #


def test_classification_report_json_is_a_dict_with_intents_and_aggregates(tmp_path):
    save_evaluation_artifacts(_Y_TRUE, _Y_PRED, results_dir=tmp_path)
    data = json.loads((tmp_path / "classification_report.json").read_text())

    for intent in INTENTS:
        assert intent in data, f"missing per-class report for {intent}"
        per = data[intent]
        for k in ("precision", "recall", "f1-score", "support"):
            assert k in per

    assert "macro avg" in data
    assert "weighted avg" in data
    assert "accuracy" in data

    # Specific assertion: 7/8 correct → accuracy 0.875
    assert data["accuracy"] == pytest.approx(7 / 8)


def test_classification_report_txt_mentions_every_intent(tmp_path):
    save_evaluation_artifacts(_Y_TRUE, _Y_PRED, results_dir=tmp_path)
    txt = (tmp_path / "classification_report.txt").read_text()
    for intent in INTENTS:
        assert intent in txt
    assert "macro avg" in txt
    assert "weighted avg" in txt


# --------------------------------------------------------------------------- #
# Training metadata                                                            #
# --------------------------------------------------------------------------- #


def test_save_training_metadata_writes_json_with_inputs(tmp_path):
    meta = {
        "timestamp_utc": "2026-06-06T00:00:00+00:00",
        "device": "mps",
        "base_model": "distilbert-base-uncased",
        "num_train_examples": 480,
        "num_test_examples": 120,
        "num_labels": 4,
        "intents": list(INTENTS),
        "hyperparameters": {"num_train_epochs": 4, "learning_rate": 5e-5},
        "final_metrics": {"accuracy": 0.975, "f1": 0.975},
    }
    path = save_training_metadata(meta, results_dir=tmp_path)
    assert path == tmp_path / "training_meta.json"
    loaded = json.loads(path.read_text())
    assert loaded == meta


def test_save_training_metadata_creates_results_dir_if_missing(tmp_path):
    nested = tmp_path / "deep" / "results"
    save_training_metadata({"k": "v"}, results_dir=nested)
    assert (nested / "training_meta.json").exists()


# --------------------------------------------------------------------------- #
# Return-value contract                                                        #
# --------------------------------------------------------------------------- #


def test_save_evaluation_artifacts_returns_paths_dict(tmp_path):
    paths = save_evaluation_artifacts(_Y_TRUE, _Y_PRED, results_dir=tmp_path)
    assert isinstance(paths, dict)
    assert paths["confusion_matrix_csv"] == tmp_path / "confusion_matrix.csv"
    assert paths["classification_report_json"] == tmp_path / "classification_report.json"
    assert paths["classification_report_txt"] == tmp_path / "classification_report.txt"
    if _HAS_MPL:
        assert paths["confusion_matrix_png"] == tmp_path / "confusion_matrix.png"
    for p in paths.values():
        assert isinstance(p, Path)
        assert p.exists()
