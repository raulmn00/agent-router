# router

Fine-tuned DistilBERT for 4-class intent classification. Powers the `/route` endpoint and is one of the three routers compared by `/compare`.

## Files

- `router/intents.py` — single source of truth for the intent set (`INTENTS`, `LABEL2ID`, `ID2LABEL`).
- `router/dataset.py` — synthetic dataset generator (150 per class × 4 = 600 examples) + `load_dataset_splits()` that returns a stratified 80/20 `DatasetDict`. The committed `data/intents.jsonl` is deterministic (`seed=1337`).
- `router/train.py` — CLI that fine-tunes `distilbert-base-uncased`. Saves to `model/`.
- `router/classifier.py` — `IntentClassifier` that loads `model/` once and exposes `classify(text) -> RouteDecision{intent, confidence}`. Resolves `ROUTER_MODEL_PATH` relative paths against the repo root so it works regardless of `cwd`. Raises `RuntimeError` with a clear message if the model dir is missing.

## Commands

```bash
# Regenerate the dataset (only stdlib needed).
python -m router.dataset

# Fine-tune the model. Device is auto-detected (CUDA → MPS → CPU).
python -m router.train
# optional flags: --epochs N --batch-size N --lr 5e-5 --output-dir path/

# Quick sanity check after training (from repo root).
python -c "
import sys; sys.path.insert(0, 'router')
from router.classifier import IntentClassifier
clf = IntentClassifier()
for t in ['What is the capital of France?', 'Build me a Slack bot', 'In the PDF, what is the conclusion?', 'hey there']:
    print(t, '->', clf.classify(t))
"
```

## Model artifact

`model/` is **gitignored**. The current production model is shipped via [GitHub Release v0.1.0](https://github.com/raulmn00/agent-router/releases/tag/v0.1.0) as `model.tar.gz` (~250 MB on disk, 236 MB compressed).

Three ways to make it available at runtime:

1. **Train locally** with `python -m router.train` — fastest for dev.
2. **Train on Colab T4** if you don't have a GPU — install deps, run the same command, download the directory.
3. **Ship from a release** — tar `model/`, attach to a GitHub Release, set `MODEL_RELEASE_URL` in the deployment env, and the backend container will fetch it on startup (see `backend/entrypoint.sh`).

The DistilBERT base checkpoint downloads from Hugging Face on first run (set `HF_TOKEN` to avoid the rate-limited unauthenticated path).

## Tests

```bash
python -m pytest tests/
```

**11 tests** — schema, balance, stratified split, classifier softmax/argmax with mocked transformers. No GPU, no model load required (the heavier tests use `monkeypatch` on `AutoTokenizer`/`AutoModelForSequenceClassification`).
