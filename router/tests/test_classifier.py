"""Tests for IntentClassifier.

Strategy: monkeypatch `transformers.AutoTokenizer.from_pretrained` and
`AutoModelForSequenceClassification.from_pretrained` with deterministic fakes so
the test never touches the network, GPU, or a real model.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from router import INTENTS
from router.classifier import IntentClassifier, RouteDecision

# Tests that need torch / transformers are skipped when those aren't installed —
# the schema tests stay runnable without the heavier ML stack.
_HAS_ML_DEPS = all(
    importlib.util.find_spec(m) is not None for m in ("torch", "transformers")
)
_skipif_no_ml = pytest.mark.skipif(not _HAS_ML_DEPS, reason="needs torch + transformers")


# --------------------------------------------------------------------------- #
# RuntimeError path doesn't depend on ML deps                                 #
# --------------------------------------------------------------------------- #


def test_raises_when_model_dir_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(RuntimeError, match="python -m router.train"):
        IntentClassifier(model_path=missing)


def test_raises_when_dir_exists_but_no_config(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"")  # exists but no config.json
    with pytest.raises(RuntimeError):
        IntentClassifier(model_path=tmp_path)


# --------------------------------------------------------------------------- #
# Softmax/argmax logic — fully mocked                                          #
# --------------------------------------------------------------------------- #


def _make_fake_model_dir(tmp_path) -> "pytest.fixture":
    """Write a minimal directory shape that passes the existence check."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "distilbert"}))
    return tmp_path


@_skipif_no_ml
def test_classify_returns_highest_softmax_intent(monkeypatch, tmp_path):
    import torch

    fake_dir = _make_fake_model_dir(tmp_path)

    # Logits crafted so that ID2LABEL[2] == "document_qa" wins.
    target_idx = 2
    logits_row = [0.1, 0.2, 5.0, 0.3]  # one row, four classes
    assert len(logits_row) == len(INTENTS)

    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            # Real transformers tokenizers return a BatchEncoding (dict-like);
            # the model just needs **inputs to unpack into kwargs.
            return {"input_ids": torch.tensor([[101, 102]])}

    class FakeOutput:
        def __init__(self, logits):
            self.logits = logits

    class FakeModel:
        def __init__(self):
            self.called_with = None
            self._eval = False

        def eval(self):
            self._eval = True
            return self

        def __call__(self, **inputs):
            self.called_with = inputs
            return FakeOutput(torch.tensor([logits_row]))

    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **kw: fake_tokenizer)
    )
    monkeypatch.setattr(
        AutoModelForSequenceClassification,
        "from_pretrained",
        staticmethod(lambda *a, **kw: fake_model),
    )

    clf = IntentClassifier(model_path=fake_dir)
    decision = clf.classify("anything")

    assert isinstance(decision, RouteDecision)
    assert decision.intent == INTENTS[target_idx]
    # The third logit (5.0) dominates after softmax; the assertion below pins
    # the exact behavior. (Confidence is high but the exact value comes from
    # the deterministic softmax math, not an estimate.)
    assert 0.9 < decision.confidence <= 1.0
    assert fake_model._eval, "model.eval() must be called on load"


@_skipif_no_ml
def test_classify_rejects_empty_input(monkeypatch, tmp_path):
    import torch

    fake_dir = _make_fake_model_dir(tmp_path)

    class _Stub:
        def eval(self):
            return self

        def __call__(self, **kw):
            class _Out:
                logits = torch.zeros(1, len(INTENTS))
            return _Out()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **kw: _Stub()))
    monkeypatch.setattr(
        AutoModelForSequenceClassification,
        "from_pretrained",
        staticmethod(lambda *a, **kw: _Stub()),
    )

    clf = IntentClassifier(model_path=fake_dir)
    with pytest.raises(ValueError):
        clf.classify("")
    with pytest.raises(ValueError):
        clf.classify("   ")
