# backend

FastAPI service that exposes the fine-tuned router and the three-way comparison endpoint.

Live: <https://agent-router-909428365094.us-central1.run.app>

## Files

- `app/schemas.py` — Pydantic v2 models. `RouteRequest`/`RouteResponse` for `/route`; `CompareRequest`/`RouterResult`/`CompareResponse` for `/compare`.
- `app/dispatch.py` — `Dispatcher.dispatch(text)` routes to one of four arms:
  - `simple_qa` → 1 direct LLM call (`gpt-4o-mini`)
  - `complex_task` → multi-agent `Orchestrator` (Planner → Executors → Critic)
  - `document_qa` → RAG stub (integration point documented in code)
  - `chitchat` → 1 direct LLM call with small `max_tokens`
- `app/compare.py` — `CompareService.compare(text)` runs all three routers (DistilBERT, LLM zero-shot, embeddings + LogReg) with each router isolated behind try/except. A single router's failure becomes its row's `error` field; the other two still return normally.
- `app/api.py` — FastAPI app, CORS, slowapi rate limiting, lazy singletons (`_classifier_singleton`, `_llm_router_singleton`, `_embed_router_singleton`), and exception handlers for `429 / 503 / 500`.
- `Dockerfile` + `entrypoint.sh` — container that fetches the DistilBERT model from a release URL on startup. The fitted LogReg artifact for the embed router (`eval/models/embed_router.joblib`) ships baked into the image.

## Endpoints

```
GET  /         { "status": "ok", "service": "agent-router" }
POST /route    RouteRequest -> RouteResponse              (30 req/min/IP)
POST /compare  CompareRequest -> CompareResponse          (10 req/min/IP — burns tokens)
```

### Error codes

| Code | When |
|---|---|
| `400` | Malformed `Content-Length` header. |
| `413` | Request body exceeds `MAX_BODY_BYTES` (default 10 000). Rejected before Pydantic parses. |
| `422` | Pydantic validation failed (empty `input` or `input` > 2000 chars). |
| `429` | slowapi rate limit exceeded. Body: `{"detail": "too many requests — please slow down…"}`. |
| `503` | `ProviderUnavailableError` — missing credentials. Body: `{"detail": "upstream LLM provider is not configured"}` (generic; never leaks the env var name). |
| `500` | Unexpected error. Body: `{"detail": "internal server error"}` (opaque). Full traceback in server logs. |

`/compare` itself never returns 5xx on a *single router* failure — the failed router's row carries an error string drawn from a fixed allowlist (e.g. `"router artifact not available"`, `"upstream provider rate limited"`), never the raw exception text. The response stays 200. The endpoint only 5xx's if the whole request can't be processed.

### Security posture

- Container drops root via the `app` system user (UID 1000); only `/app` is writable.
- Runtime service account is `agent-router-runtime@research-agent-498415.iam.gserviceaccount.com` with `roles/secretmanager.secretAccessor` on the `openai-api-key` secret only — no other project permissions.
- `CORS_ALLOW_ORIGINS` is a comma-separated allowlist (no `*` in production). When set to `*`, `allow_credentials` is forced to `false` automatically to comply with the CORS spec.
- Per-response headers: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Strict-Transport-Security: max-age=31536000; includeSubDomains`, `Cross-Origin-Resource-Policy: same-site`.
- `/docs`, `/redoc`, `/openapi.json` return 404 when `ENABLE_API_DOCS=false`.
- Hard cap on request body via `MAX_BODY_BYTES` (default 10 000) — Pydantic validation never sees oversized payloads.

## Run locally

```bash
# from repo root
cp .env.example .env             # fill OPENAI_API_KEY

# train the DistilBERT (one-time, see router/README.md)
cd router && python -m router.train

# fit the embeddings LogReg (one-time, see eval/README.md)
cd ../eval && python -m eval.fit_embed_router

# start the server (uvicorn from anywhere; backend/app/__init__.py sets sibling paths)
cd ../backend
uvicorn app.api:app --reload --port 8000
```

If you skip the LogReg fit step, `/route` still works fully — `/compare` returns 200 with the embed-router row carrying a `FileNotFoundError`.

### Smoke tests

```bash
curl -s http://localhost:8000/ | python3 -m json.tool

# /route — single dispatched answer + agent trace
curl -s -X POST http://localhost:8000/route -H 'Content-Type: application/json' \
  -d '{"input":"What is the capital of France?"}' | python3 -m json.tool

curl -s -X POST http://localhost:8000/route -H 'Content-Type: application/json' \
  -d '{"input":"Design an end-to-end MLOps pipeline with retraining and rollback."}' | python3 -m json.tool

# /compare — all three routers on the same input
curl -s -X POST http://localhost:8000/compare -H 'Content-Type: application/json' \
  -d '{"input":"What is the capital of France?"}' | python3 -m json.tool
```

## Build & run with Docker

The image is built from the **repo root**, not from `backend/`. The Dockerfile pulls in `router/`, `agents/`, `eval/`, and `backend/`.

```bash
# from repo root
docker build -f backend/Dockerfile -t agent-router .

# run
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e MODEL_RELEASE_URL="https://github.com/raulmn00/agent-router/releases/download/v0.1.0/model.tar.gz" \
  agent-router
```

The DistilBERT model is **not** in the image — it's fetched at container start. The embeddings LogReg artifact **is** baked into the image (`eval/models/embed_router.joblib`).

## Deploy to Google Cloud Run

The live service is in project `research-agent-498415`, region `us-central1`, running under a dedicated, minimum-privilege runtime SA:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build & push via Cloud Build (faster than local push for ~3 GB images)
gcloud builds submit . \
  --config=cloudbuild.yaml \
  --project=research-agent-498415 \
  --region=us-central1

# One-time SA + IAM setup (skip if already provisioned):
gcloud iam service-accounts create agent-router-runtime --project=research-agent-498415
gcloud secrets add-iam-policy-binding openai-api-key \
  --member="serviceAccount:agent-router-runtime@research-agent-498415.iam.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor \
  --project=research-agent-498415

# Deploy. The ^@^ delimiter lets CORS_ALLOW_ORIGINS hold commas without escaping.
gcloud run deploy agent-router \
  --project research-agent-498415 \
  --image us-central1-docker.pkg.dev/research-agent-498415/agent-router/agent-router:v0.2.1 \
  --region us-central1 \
  --service-account="agent-router-runtime@research-agent-498415.iam.gserviceaccount.com" \
  --memory 2Gi --cpu 2 --timeout 300 \
  --set-env-vars='^@^MODEL_RELEASE_URL=https://github.com/raulmn00/agent-router/releases/download/v0.1.0/model.tar.gz@CORS_ALLOW_ORIGINS=https://agent-router-five.vercel.app,http://localhost:5173@ENABLE_API_DOCS=false@MAX_BODY_BYTES=10000' \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest \
  --allow-unauthenticated
```

**Always pin `--project` explicitly** — gcloud's active context can flip silently between accounts. Echo `gcloud config get-value account project` before any deploy you can't undo.

## Tests

```bash
python -m pytest tests/
```

**22 tests**, all run with `TestClient` and `app.dependency_overrides` to inject fake classifiers and `FakeProvider` factories. No network, no LLM credits, no trained model required.

- 13 `/route` tests cover the 4 dispatch arms, the base error codes (422 / 503 / 500), security headers, and the 413 body-size cap.
- 9 `/compare` tests cover: 3-way agreement, divergent intents, 1 router failing (200 response with **sanitized** error string from the safe allowlist), all 3 failing (200 with no `fastest`/`cheapest`), `Body(...)` validation, and the 429 rate limit (10/min for `/compare`, 30/min for `/route`).

`conftest.py` has an autouse fixture that resets the slowapi limiter between tests so they don't accumulate counts.
