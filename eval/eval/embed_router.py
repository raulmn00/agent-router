"""Baseline 2: OpenAI embeddings + LogisticRegression.

Workflow:
  1. Embed every training example via text-embedding-3-small.
  2. Fit a LogisticRegression on (embedding, label) pairs.
  3. At inference: embed input, run LR.predict.

The embeddings call dominates latency; the LR step is essentially free.

Trade-offs vs the other two routers depend on the run — measure them with
`python -m eval.compare_routers` rather than asserting them here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression

DEFAULT_MODEL = "text-embedding-3-small"


@dataclass
class _Cache:
    """Optional file-backed cache so a re-run doesn't re-embed everything.

    Tests don't use this; compare_routers.py opts in via the cache_path arg.
    """

    path: str | None

    def load(self) -> dict[str, list[float]]:
        if not self.path or not os.path.exists(self.path):
            return {}
        import json

        with open(self.path) as f:
            return json.load(f)

    def save(self, data: dict[str, list[float]]) -> None:
        if not self.path:
            return
        import json

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f)


def _embed_many(texts: list[str], client, model: str, cache: _Cache) -> np.ndarray:
    cached = cache.load()
    missing = [t for t in texts if t not in cached]
    if missing:
        # OpenAI's embeddings endpoint accepts a list of inputs per call; the
        # training set here is small enough to fit in one call regardless.
        resp = client.embeddings.create(model=model, input=missing)
        for text, item in zip(missing, resp.data):
            cached[text] = item.embedding
        cache.save(cached)
    return np.array([cached[t] for t in texts], dtype=np.float32)


class EmbedRouter:
    name = "embed"

    def __init__(
        self,
        client=None,
        model: str = DEFAULT_MODEL,
        cache_path: str | None = None,
    ):
        if client is None:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY required to use EmbedRouter")
            client = OpenAI(api_key=api_key)
        self.client = client
        self.model = model
        self.cache = _Cache(cache_path)
        self.lr: LogisticRegression | None = None
        self.classes_: list[str] = []

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> "EmbedRouter":
        texts_list = list(texts)
        labels_list = list(labels)
        X = _embed_many(texts_list, self.client, self.model, self.cache)
        # NB: `multi_class` was removed in scikit-learn 1.9; the default behavior
        # now infers multinomial from the data. Don't pass that kwarg.
        self.lr = LogisticRegression(max_iter=1000)
        self.lr.fit(X, labels_list)
        self.classes_ = list(self.lr.classes_)
        return self

    def classify(self, text: str) -> str:
        if self.lr is None:
            raise RuntimeError("EmbedRouter not fitted — call .fit(...) first")
        X = _embed_many([text], self.client, self.model, self.cache)
        return str(self.lr.predict(X)[0])

    def classify_with_confidence(self, text: str) -> tuple[str, float]:
        """Like `classify`, plus the LogReg's predicted probability for the chosen class."""
        if self.lr is None:
            raise RuntimeError("EmbedRouter not fitted — call .fit(...) first")
        X = _embed_many([text], self.client, self.model, self.cache)
        probs = self.lr.predict_proba(X)[0]
        idx = int(probs.argmax())
        return str(self.lr.classes_[idx]), float(probs[idx])

    def save(self, path: str) -> None:
        """Persist the trained LogReg + class labels to disk.

        Only the fitted classifier is saved — the OpenAI client and the
        embedding cache are reconstructed at load time. Use `joblib` to match
        scikit-learn's recommended serialization for fitted estimators.
        """
        if self.lr is None:
            raise RuntimeError("nothing to save — EmbedRouter is not fitted")
        import joblib
        import os as _os

        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"lr": self.lr, "classes_": self.classes_, "model": self.model}, path)

    def load(self, path: str) -> "EmbedRouter":
        """Inverse of `save` — populates `self.lr` and `self.classes_` from disk."""
        import joblib

        payload = joblib.load(path)
        self.lr = payload["lr"]
        self.classes_ = payload["classes_"]
        # Honor the saved embedding model (avoid silent mix-ups with newer pins).
        if payload.get("model"):
            self.model = payload["model"]
        return self
