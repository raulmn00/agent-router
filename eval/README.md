# eval

Comparative evaluation of three router strategies. The same code is reused by the backend's `/compare` endpoint — see `backend/app/compare.py`.

## Files

- `data/routing_testset.jsonl` — 40 handcrafted held-out examples (10 per class). Zero overlap with `router/data/intents.jsonl`.
- `models/embed_router.joblib` — fitted LogReg artifact (committed, ~25 KB). Loaded by the backend at startup so production doesn't pay the training cost on every cold start.
- `eval/metrics.py` — pure functions: `accuracy`, `f1_macro`, `mean_latency_ms`, `cost_per_1k`. Cost model documented inline.
- `eval/llm_router.py` — zero-shot LLM router (`gpt-4o-mini`) with a robust output parser and `classify_with_confidence(text)` that derives confidence from OpenAI logprobs (first-token).
- `eval/embed_router.py` — `text-embedding-3-small` + `LogisticRegression`. `classify_with_confidence(text)` returns `predict_proba` max. `save(path)` / `load(path)` use `joblib`.
- `eval/fit_embed_router.py` — CLI that fits the LogReg on the training set and writes the artifact. Run once before deploying.
- `eval/compare_routers.py` — runs all three on the testset, optionally over multiple cold-cache runs, and writes `results/comparison.{md,csv}`.

## Commands

```bash
# Fit and persist the LogReg (one-time — uses the cached embeddings if available).
python -m eval.fit_embed_router
# → eval/models/embed_router.joblib

# Single run (uses warm embed cache if present).
python -m eval.compare_routers

# Multi-run for stable latency measurement — cache is cleared between runs.
python -m eval.compare_routers --runs 3
```

Outputs:

- `results/comparison.md` — human-readable table.
- `results/comparison.csv` — same data, machine-readable. Schema: `approach,accuracy,f1_macro,mean_latency_ms,stdev_latency_ms,cost_per_1k_usd,n,runs`.

## Confidence semantics

Each router returns a confidence in `[0, 1]`:

| Router | Source |
|---|---|
| DistilBERT | softmax max of the classification head's logits. |
| LLM zero-shot | `exp(first_token_logprob)` from OpenAI's `logprobs=True`. Falls back to `1.0` when the provider doesn't expose logprobs (e.g. `FakeProvider` in tests). |
| Embeddings + LogReg | `lr.predict_proba(...).max()`. |

These three are not directly comparable as probabilities — they're estimated very differently. They're useful as *within-router* signal (was this classification confident?), not across-router. The frontend's gauges reflect that.

## Preconditions per approach

- **DistilBERT (fine-tuned)** — needs `router/model/`. Train via `cd ../router && python -m router.train`.
- **LLM zero-shot (gpt-4o-mini)** — needs `OPENAI_API_KEY`. One chat completion per test example.
- **Embeddings + LogReg** — needs `OPENAI_API_KEY` and `eval/models/embed_router.joblib`. The fit step costs ~600 short-text embeddings (cheap, one-time). At test/run time, one embedding per test example.

If any precondition is missing:

- The eval CLI reports the approach under `## Skipped` in `comparison.md` and continues.
- The backend's `/compare` returns 200 with that router's row carrying the error and the other two filled normally.

## Cost assumptions

`cost_per_1k(approach)` is a function of `PRICING_USD_PER_M_TOKENS` and `TOKEN_ASSUMPTIONS` in `metrics.py`. These are documented constants — adjust if pricing changes or if you want to substitute measured token counts from a real run. The test `test_known_pinned_costs` pins the current values; it fails loudly if anyone edits the constants without updating the README's evaluation table.

## Tests

```bash
python -m pytest tests/
```

**17 tests**, all pure. No network, no LLM credits. Includes a sanity-check that compares the hand-rolled `f1_macro` against `sklearn.metrics.f1_score` on a known case.
