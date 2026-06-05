# backend

FastAPI service that wires the DistilBERT router to a 4-arm dispatch.

## Files

- `app/schemas.py` — Pydantic v2 `RouteRequest` (`input: str`, 1–2000 chars) and `RouteResponse` (`intent`, `confidence`, `answer`, `path_taken`, `trace`).
- `app/dispatch.py` — `Dispatcher.dispatch(text)` routes to one of four arms:
  - `simple_qa` → 1 direct LLM call (`gpt-4o-mini`)
  - `complex_task` → multi-agent `Orchestrator`
  - `document_qa` → RAG stub (integration point documented in the code)
  - `chitchat` → 1 direct LLM call with small `max_tokens`
- `app/api.py` — FastAPI app, CORS, lazy classifier singleton, exception handlers for `422 / 503 / 500`.
- `Dockerfile` + `entrypoint.sh` — container that fetches the trained model from a release URL on startup.

## Endpoints

```
GET  /         { "status": "ok", "service": "agent-router" }
POST /route    RouteRequest -> RouteResponse
```

Error codes: `422` (Pydantic validation), `503` (`ProviderUnavailableError` — credentials missing, generic message), `500` (logged server-side, opaque body).

## Run locally

```bash
# from repo root
cp .env.example .env  # fill in OPENAI_API_KEY
cd backend
uvicorn app.api:app --reload --port 8000
```

Smoke tests:

```bash
curl -s http://localhost:8000/ | python3 -m json.tool

curl -s -X POST http://localhost:8000/route -H 'Content-Type: application/json' \
  -d '{"input":"What is the capital of France?"}' | python3 -m json.tool

curl -s -X POST http://localhost:8000/route -H 'Content-Type: application/json' \
  -d '{"input":"Design an end-to-end MLOps pipeline with retraining and rollback."}' | python3 -m json.tool

curl -s -X POST http://localhost:8000/route -H 'Content-Type: application/json' \
  -d '{"input":"In the attached PDF, what is the conclusion?"}' | python3 -m json.tool

curl -s -X POST http://localhost:8000/route -H 'Content-Type: application/json' \
  -d '{"input":"Hi! How are you?"}' | python3 -m json.tool
```

## Build & run with Docker

The image is built from the **repo root**, not from `backend/`, because the Dockerfile needs `router/` and `agents/` in the build context.

```bash
# from repo root
docker build -f backend/Dockerfile -t agent-router .
```

The container expects the model to be available at startup. Three ways to provide it:

1. **Bake it in at build time** — comment out `router/model/` from `.dockerignore`, train the model locally first.
2. **Fetch from a release URL** (recommended for portfolio deploys) — tar your trained model and host it on a GitHub Release:
   ```bash
   tar -czf model.tar.gz -C router model
   gh release create v0.1.0 model.tar.gz
   ```
   then run the container with the release URL:
   ```bash
   docker run --rm -p 8000:8000 \
     -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
     -e MODEL_RELEASE_URL="https://github.com/<you>/agent-router/releases/download/v0.1.0/model.tar.gz" \
     agent-router
   ```
3. **Mount a host directory** — `-v $(pwd)/router/model:/app/router/model`. Skips the fetch entirely.

## Deploy to Google Cloud Run

```bash
# Build & push to Artifact Registry (replace REGION/PROJECT_ID/REPO/IMAGE_TAG)
gcloud auth configure-docker REGION-docker.pkg.dev
docker build -f backend/Dockerfile \
  -t REGION-docker.pkg.dev/PROJECT_ID/REPO/agent-router:IMAGE_TAG .
docker push REGION-docker.pkg.dev/PROJECT_ID/REPO/agent-router:IMAGE_TAG

# Deploy. --memory 1Gi is enough for the model + working set; --cpu 2 leaves
# headroom for the LLM dispatch path's I/O.
gcloud run deploy agent-router \
  --project PROJECT_ID \
  --image REGION-docker.pkg.dev/PROJECT_ID/REPO/agent-router:IMAGE_TAG \
  --region REGION \
  --memory 1Gi --cpu 2 \
  --set-env-vars MODEL_RELEASE_URL="https://github.com/<you>/agent-router/releases/download/v0.1.0/model.tar.gz" \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest \
  --allow-unauthenticated
```

Pin the project flag explicitly (`--project`) — Cloud SDK's active context can flip between accounts silently. Always echo `gcloud config get-value account project` before deploys you can't undo.

## Tests

```bash
python -m pytest tests/
```

All 10 tests use `TestClient` with `app.dependency_overrides` to inject a fake classifier and a `FakeProvider` factory. Covers the 4 dispatch arms and the 3 error codes (`422 / 503 / 500`). No network.
