# eval

Comparative evaluation of three router strategies on a held-out testset.

## Files

- `data/routing_testset.jsonl` — 40 handcrafted held-out examples (10 per class). Zero overlap with `router/data/intents.jsonl`.
- `eval/metrics.py` — pure functions: `accuracy`, `f1_macro`, `mean_latency_ms`, `cost_per_1k`. Cost model documented inline.
- `eval/llm_router.py` — zero-shot LLM router (`gpt-4o-mini`) with a robust output parser.
- `eval/embed_router.py` — `text-embedding-3-small` + `LogisticRegression` fit on the training set. File-backed embedding cache so re-runs don't re-spend.
- `eval/compare_routers.py` — runs all three on the testset, optionally over multiple runs, and writes `results/comparison.{md,csv}`.

## Commands

```bash
# Single run (uses warm embed cache if present).
python -m eval.compare_routers

# Multi-run for stable latency measurement — cache is cleared between runs.
python -m eval.compare_routers --runs 3
```

Outputs:

- `results/comparison.md` — human-readable table, one row per approach.
- `results/comparison.csv` — same data, machine-readable. Schema: `approach,accuracy,f1_macro,mean_latency_ms,stdev_latency_ms,cost_per_1k_usd,n,runs`.

## Preconditions per approach

- **DistilBERT (fine-tuned)** — requires `router/model/`. Train with `cd ../router && python -m router.train` if missing.
- **LLM zero-shot (gpt-4o-mini)** — requires `OPENAI_API_KEY`. Spends one chat completion per test example (40 per run with the current testset).
- **Embeddings + LogReg** — requires `OPENAI_API_KEY`. Spends embedding calls for the training set (cached after first run) plus one per test example.

If a precondition is missing, the script reports the approach under `## Skipped` in the output and continues with the rest.

## Cost assumptions

`cost_per_1k(approach)` is a function of `PRICING_USD_PER_M_TOKENS` and `TOKEN_ASSUMPTIONS` in `metrics.py`. These are documented constants — adjust if pricing changes or if you want to substitute measured token counts from a real run. The test `test_known_pinned_costs` pins the current values; it will fail loudly if anyone edits the constants without updating the README's evaluation table.

## Tests

```bash
python -m pytest tests/
```

All tests are pure (no network, no LLM credits). Includes a sanity-check that compares the hand-rolled `f1_macro` against `sklearn.metrics.f1_score` on a known case.
