"""Runtime intent classifier.

Loads the fine-tuned model from `router/model/` once and exposes a stateless
`classify(text)` method that returns a `RouteDecision`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .intents import ID2LABEL

ROUTER_PKG = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROUTER_PKG.parent / "model"


@dataclass(frozen=True)
class RouteDecision:
    intent: str
    confidence: float


def _resolve_model_path(model_path: str | os.PathLike | None) -> Path:
    """Return the model directory path.

    Resolution order:
      1. explicit `model_path` arg (used verbatim)
      2. `ROUTER_MODEL_PATH` env var — anchored to the repo root if relative,
         so it works regardless of cwd
      3. the package default (`<repo>/router/model`, absolute)
    """
    if model_path is not None:
        return Path(model_path)
    env_override = os.getenv("ROUTER_MODEL_PATH")
    if env_override:
        p = Path(env_override)
        if p.is_absolute():
            return p
        # Anchor to repo root (parent of the `router/` package's parent dir).
        repo_root = DEFAULT_MODEL_PATH.parent.parent
        return (repo_root / p).resolve()
    return DEFAULT_MODEL_PATH


class IntentClassifier:
    """Lazy-loads the fine-tuned model and tokenizer on first construction."""

    def __init__(self, model_path: str | os.PathLike | None = None):
        path = _resolve_model_path(model_path)
        if not path.exists() or not (path / "config.json").exists():
            raise RuntimeError(
                f"Router model not found at {path}. Train it first with:\n"
                f"  python -m router.train\n"
                f"or set ROUTER_MODEL_PATH to an existing model directory."
            )

        # Imported lazily so unit tests can monkey-patch these symbols without
        # paying the import cost of transformers/torch up front.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path))
        self.model.eval()
        # The model file ships its own id2label, but we prefer the canonical
        # mapping from intents.py — if the two ever drift, that's a bug.
        self._id2label = ID2LABEL

    def classify(self, text: str) -> RouteDecision:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("classify(text): text must be a non-empty string")

        torch = self._torch
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits  # shape: (1, num_labels)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        idx = int(torch.argmax(probs).item())
        confidence = float(probs[idx].item())
        intent = self._id2label[idx]
        return RouteDecision(intent=intent, confidence=confidence)
