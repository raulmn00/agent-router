# Router comparison

## Results (n=3 cold-cache runs)

| Approach | Accuracy | F1 (macro) | Mean latency (ms) ± stdev | Cost / 1k (USD) | N |
|---|---:|---:|---:|---:|---:|
| DistilBERT (fine-tuned) | 0.975 | 0.975 | 25.0 ± 4.4 | $0.0000 | 40 |
| LLM zero-shot (gpt-4o-mini) | 1.000 | 1.000 | 795.7 ± 68.6 | $0.0120 | 40 |
| Embeddings + LogReg | 0.975 | 0.975 | 780.7 ± 574.9 | $0.0003 | 40 |

_Cost numbers are derived from the documented assumptions in `eval/eval/metrics.py` (PRICING_USD_PER_M_TOKENS, TOKEN_ASSUMPTIONS). Latency was measured per-classification with `time.perf_counter`._