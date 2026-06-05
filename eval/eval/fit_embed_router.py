"""Fit the EmbedRouter on the training set and persist the LogReg artifact.

Run once before the backend's /compare endpoint can return embed-router
results in production. The trained LogReg is small (~50 KB); the embedding
API costs are paid here (~600 short texts × text-embedding-3-small) so the
runtime path never pays training cost again.

    python -m eval.fit_embed_router \\
        --train ../router/data/intents.jsonl \\
        --out models/embed_router.joblib

Defaults assume the standard repo layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embed_router import EmbedRouter

EVAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINSET = EVAL_ROOT.parent / "router" / "data" / "intents.jsonl"
DEFAULT_OUTPUT = EVAL_ROOT / "models" / "embed_router.joblib"
DEFAULT_CACHE = EVAL_ROOT / "results" / ".embed_cache.json"


def _load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAINSET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    texts, labels = _load_jsonl(args.train)
    print(f"fitting EmbedRouter on {len(texts)} examples from {args.train}")

    router = EmbedRouter(cache_path=str(args.cache))
    router.fit(texts=texts, labels=labels)
    router.save(str(args.out))
    print(f"saved fitted LogReg to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
