# agent-router

Multi-agent system whose routing layer is a **DistilBERT classifier I fine-tuned**, instead of paying tokens to a frontier LLM to route each request.

Showcases, in one repo:

- **Own fine-tuning** of `distilbert-base-uncased` on a synthetic dataset for 4 intent classes (`simple_qa`, `complex_task`, `document_qa`, `chitchat`).
- **Multi-agent orchestration** (Planner / Executor / Critic) built by hand on top of a pluggable `LLMProvider` (OpenAI / Anthropic / Fake).
- **FastAPI backend** that wires the router to a 4-arm dispatch.
- **Comparative evaluation** of three router strategies: the fine-tuned model, an LLM zero-shot baseline, and an embeddings + LogisticRegression baseline — measured for accuracy, F1, latency, and cost on a held-out testset.

## Architecture

```
                       ┌─────────────────────────────┐
                       │      POST /route            │
                       │   { "input": "user text" }  │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │  DistilBERT intent router   │   <— fine-tuned
                       │   classify() -> intent      │       (this repo)
                       └──────────────┬──────────────┘
                                      │
        ┌─────────────────┬───────────┴──────────┬───────────────────┐
        ▼                 ▼                      ▼                   ▼
  simple_qa          complex_task           document_qa         chitchat
        │                 │                      │                   │
        ▼                 ▼                      ▼                   ▼
  1 direct call    Orchestrator              RAG stub            1 direct call
  to OpenAI        ┌─────────────┐           (integration        to OpenAI
  (gpt-4o-mini)    │  Planner    │            point for          (short max_tokens,
                   │     ↓       │            retriever)          warm/social tone)
                   │  Executors  │
                   │     ↓       │
                   │   Critic    │ — re-plans once if rejected
                   └─────────────┘
        │                 │                      │                   │
        └─────────────────┴──────────┬───────────┴───────────────────┘
                                     ▼
                       ┌─────────────────────────────┐
                       │      RouteResponse          │
                       │  {intent, confidence,       │
                       │   answer, path_taken,       │
                       │   trace}                    │
                       └─────────────────────────────┘
```

## Why these choices

| Choice | Why | Trade-off accepted |
|---|---|---|
| **DistilBERT local** instead of LLM-as-router | Routing is on the hot path of every request; paying an LLM call just to pick a branch is wasteful. Local inference removes that hop entirely once the model is trained. | Need to train + version + ship a 256 MB binary. |
| **Hand-rolled multi-agent loop** (no LangChain / CrewAI / etc.) | The whole point of the demo is to show I understand what an agent loop is and what `Planner → Executor → Critic → retry` looks like in plain code. Frameworks hide exactly the part the reader needs to see. | Less abstraction reuse if I later want fan-out / async tools / state machines. |
| **`LLMProvider` ABC** with OpenAI / Anthropic / Fake impls | Tests run with `FakeProvider` (no network, no spend). Production can pick the provider via env var. Both real providers go through the same call surface. | One extra layer to maintain. |
| **Synthetic dataset of 600 templated examples** for fine-tuning | A held-out, handcrafted testset of 40 lives in `eval/` and is what gets reported. The synthetic 600 is just enough signal for DistilBERT to separate the 4 classes; bigger gains would come from real labelled data. | F1 on the template-distribution test split is artificially high (the test split shares templates with training). The honest number is the one in the evaluation table below, against the handcrafted held-out set. |
| **Pure-function metrics** (`accuracy`, `f1_macro`, `cost_per_1k`) | Testable without network or training. Pricing is a documented constant — comparison stays auditable. | Cost is a function of declared token assumptions, not measured token counts. Replace with measured counts if you want a precise number. |

## How to run

### Prerequisites
- Python 3.12+ (tested on 3.13 too — the pyproject pin is conservative)
- `OPENAI_API_KEY` in `.env` (copy from `.env.example`) — required for the backend and for the eval baselines
- ML stack: `pip install -r backend/requirements.txt` (or install the per-package extras you need)

### 1. Tests (no network, no spend)

```bash
python -m pytest
```

### 2. Train the DistilBERT router

```bash
cd router
python -m router.train
```

Saves model + tokenizer to `router/model/`. Device auto-detected (CUDA → MPS → CPU). The training dataset is committed at `router/data/intents.jsonl` (regenerate with `python -m router.dataset` if you change the templates).

### 3. Run the backend

```bash
cd backend
uvicorn app.api:app --reload --port 8000
```

`POST /route` with `{"input": "..."}`. `GET /` for health. CORS is on by default; set `CORS_ALLOW_ORIGINS` env var to restrict.

### 4. Compare the three routers

```bash
cd eval
python -m eval.compare_routers --runs 3
```

Generates `eval/results/comparison.md` and `comparison.csv`. The `--runs N` flag aggregates N independent runs, clearing the embedding cache between them so the embed router stays cold-cache (production-like) for every measurement.

## Evaluation

Measured locally on 2026-06-05 against the 40-example held-out testset in `eval/data/routing_testset.jsonl`. Hardware: Apple Silicon (MPS for DistilBERT; OpenAI API for the other two).

| Approach | Accuracy | F1 (macro) | Mean latency (ms) ± stdev | Cost / 1k (USD) | N |
|---|---:|---:|---:|---:|---:|
| **DistilBERT (fine-tuned)** | 0.975 | 0.975 | 24.2 ± 0.2 | $0.0000 | 40 |
| LLM zero-shot (gpt-4o-mini) | 1.000 | 1.000 | 1786.8 ± 1893.9 | $0.0120 | 40 |
| Embeddings + LogReg | 0.975 | 0.975 | 3375.5 ± 3883.3 | $0.0003 | 40 |

Notes on the numbers:

- **Accuracy / F1 are deterministic given the trained model.** They came out identical across all 3 runs for every approach.
- **Latency is whatever the network and the API happen to be doing.** The high stdevs on the LLM and embed rows are not noise to be averaged away — they're an honest reflection of the variability that API-based routing imposes on every call. DistilBERT runs locally, has no network in the path, and its stdev (0.2 ms) confirms it.
- **Cost is calculated, not measured.** It's derived from the assumptions in `eval/eval/metrics.py` (`PRICING_USD_PER_M_TOKENS`, `TOKEN_ASSUMPTIONS`) — see those constants and adjust if pricing changes or you want to use actually-measured token counts.

Re-run on your own machine to get your numbers: `cd eval && python -m eval.compare_routers --runs 3`.

### When each approach makes sense

- **DistilBERT (fine-tuned)** wins when the set of intents is closed and stable, you have (or can synthesize) training data, and the router is on the hot path. Local inference removes the round-trip to an LLM API entirely, and the cost per request drops to local compute only. The trade-off is that you carry a trained model artifact and have to retrain when intents change.

- **LLM zero-shot** wins when intents change every week, you don't yet have labelled data, or traffic is low enough that token cost doesn't matter. It's the right baseline for an MVP — and the cheapest way to bootstrap the labelled set that the DistilBERT path needs later.

- **Embeddings + LogReg** sits in between. You need some labelled data but don't want to fine-tune a model end-to-end. Adding a class is fast (re-fit the LogReg in seconds). Whether this beats the LLM zero-shot in your setup depends on the embedding model's discriminative power for your specific intent set — measure both before committing.

For this dataset, on this run, the local DistilBERT was the fastest by a wide margin and tied the embed router on accuracy. The LLM was the only one to reach perfect accuracy, which is consistent with the literature — but it pays for it in latency and cost. Measure these trade-offs on your traffic before generalising.

## Project layout

```
agent-router/
├── router/         # DistilBERT intent classifier — train + classify
├── agents/         # Planner / Executor / Critic / Orchestrator
├── backend/        # FastAPI: router → dispatch → response
├── eval/           # Comparative evaluation (DistilBERT vs LLM vs embed)
├── pyproject.toml  # Workspace root (pytest config)
└── .env.example    # API keys + optional overrides
```

Each subpackage has its own `pyproject.toml` and `README.md`. See:

- [router/README.md](router/README.md)
- [agents/README.md](agents/README.md)
- [backend/README.md](backend/README.md)
- [eval/README.md](eval/README.md)

## Deployment (Google Cloud Run)

The model artifact is 256 MB and is **not committed to git**. The Docker image fetches it at container start from a configured URL (e.g. a GitHub Release tarball). See `backend/Dockerfile` and `backend/entrypoint.sh` for the wiring, and `backend/README.md` for the exact `gcloud run deploy` command.

Key flags:
- `--memory 1Gi` (model load + working set fit comfortably here)
- `--cpu 2` (DistilBERT inference is single-threaded; 2 vCPUs leaves headroom for the LLM dispatch path's I/O)
- `--min-instances 0` is fine; cold start pays one model load (the lazy classifier in `backend/app/api.py` defers that until the first `/route` request).

## Testing

```
router/tests/         schema, balance, stratified split, classifier softmax/argmax
agents/tests/         orchestrator happy path + critic-reproves-and-retries
backend/tests/        4 dispatch paths + 422 / 503 / 500 error handling
eval/tests/           accuracy, f1_macro, cost_per_1k pinning
```

53 tests; runs without GPU and without spending API credits. Heavy ML deps are skipif-gated so the schema and orchestration tests still run in environments that don't have `torch`/`transformers` installed.
