"""FastAPI application.

Endpoints:
  GET  /         — liveness probe
  POST /route    — classify + dispatch + respond
  POST /compare  — run all three routers (DistilBERT / LLM / Embed) on the
                   same input and aggregate (rate-limited; burns real tokens)
  GET  /stats    — in-memory observability snapshot (JSON, typed by
                   StatsResponse). Designed to be consumed by the React
                   frontend, which renders the dashboard view of this data.

Architecture: this service is a DATA API. Visualization (the live-stats
dashboard) lives in the React app at `frontend/` and fetches /stats
directly via the configured CORS allowlist. The backend serves no HTML.

Error model:
  422  Pydantic validation (automatic)
  429  Rate limit exceeded (slowapi)
  503  Provider unavailable (missing credentials — message is generic)
  500  Anything else (logged server-side, opaque to client)
"""

# NB: no `from __future__ import annotations` here — FastAPI introspects
# parameter types at runtime to wire up the request body. With PEP 563
# stringified annotations + `Body(...)` defaults, Pydantic raises
# "type adapter is not fully defined". Keep annotations evaluated eagerly.

import logging
import os
from functools import lru_cache
from pathlib import Path

from fastapi import Body, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from agents import ProviderUnavailableError, get_provider

from .compare import CompareService, build_adapters
from .dispatch import Dispatcher
from .logging_config import configure_json_logging
from .metrics_collector import MetricsCollector, get_metrics_collector
from .schemas import (
    CompareRequest,
    CompareResponse,
    RouteRequest,
    RouteResponse,
    StatsResponse,
)

# Install JSON formatter on the root logger before anything else logs. On
# Cloud Run, stdout → Cloud Logging picks up each JSON line as a structured
# entry with first-class indexed fields (intent, latency_ms, etc.).
configure_json_logging()
logger = logging.getLogger("agent_router")

EVAL_MODELS_DIR = Path(__file__).resolve().parents[2] / "eval" / "models"


# --------------------------------------------------------------------------- #
# Lazy singletons                                                              #
# --------------------------------------------------------------------------- #


class _LazyClassifier:
    """Defers DistilBERT loading until the first `classify()` call.

    Why lazy: FastAPI resolves dependencies in parallel with body validation,
    so 422-path requests would otherwise pay the full model load cost just to
    get rejected. Lazy = cheap rejections. Once loaded, the underlying
    `IntentClassifier` is shared across requests.
    """

    def __init__(self):
        self._inner = None

    def classify(self, text: str):
        if self._inner is None:
            from router.classifier import IntentClassifier  # heavy import

            self._inner = IntentClassifier()
        return self._inner.classify(text)


@lru_cache(maxsize=1)
def _classifier_singleton() -> _LazyClassifier:
    return _LazyClassifier()


@lru_cache(maxsize=1)
def _llm_router_singleton():
    """Construct the zero-shot LLM router. Raises if no OPENAI_API_KEY."""
    from eval.llm_router import LLMRouter  # noqa: WPS433 — sibling import

    return LLMRouter()


@lru_cache(maxsize=1)
def _embed_router_singleton():
    """Load the persisted EmbedRouter LogReg from disk.

    Raises `FileNotFoundError` if the artifact is missing — `compare.py`
    catches this and surfaces it as a per-router error in the response so the
    endpoint still serves the other two router results.
    """
    from eval.embed_router import EmbedRouter  # noqa: WPS433 — sibling import

    artifact_path = os.getenv(
        "EMBED_ROUTER_PATH",
        str(EVAL_MODELS_DIR / "embed_router.joblib"),
    )
    router = EmbedRouter()
    router.load(artifact_path)
    return router


def get_dispatcher() -> Dispatcher:
    """FastAPI dependency — override in tests via `app.dependency_overrides`."""
    return Dispatcher(classifier=_classifier_singleton(), provider_factory=get_provider)


def get_compare_service() -> CompareService:
    """FastAPI dependency — override in tests via `app.dependency_overrides`."""
    adapters = build_adapters(
        classifier_factory=_classifier_singleton,
        llm_router_factory=_llm_router_singleton,
        embed_router_factory=_embed_router_singleton,
    )
    return CompareService(adapters=adapters)


# --------------------------------------------------------------------------- #
# App                                                                          #
# --------------------------------------------------------------------------- #


# slowapi defaults to in-memory storage — fine for a single Cloud Run instance.
# For multi-instance with a shared limit, point `storage_uri` at Redis.
limiter = Limiter(key_func=get_remote_address)


def _env_truthy(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# Hide the FastAPI auto-generated docs in production. Defaults to enabled so
# local dev stays interactive; set ENABLE_API_DOCS=false in the Cloud Run env.
_DOCS_ENABLED = _env_truthy("ENABLE_API_DOCS", "true")

app = FastAPI(
    title="agent-router",
    version="0.2.1",
    docs_url="/docs" if _DOCS_ENABLED else None,
    redoc_url="/redoc" if _DOCS_ENABLED else None,
    openapi_url="/openapi.json" if _DOCS_ENABLED else None,
)
app.state.limiter = limiter

# CORS — never run wildcard origins together with credentials (the combo is
# a spec violation and silently rejected by some browsers). If the operator
# leaves CORS_ALLOW_ORIGINS unset, default to local-dev only.
_cors_origins = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173").split(",") if o.strip()
]
_cors_allow_credentials = "*" not in _cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["content-type"],
    max_age=600,
)
app.add_middleware(SlowAPIMiddleware)


# Hard cap on request body size — Pydantic's max_length=2000 still pays the
# cost of buffering the whole body before validating. Reject early instead.
_MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", str(10_000)))


@app.middleware("http")
async def _limit_body_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > _MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "payload too large"},
                )
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "invalid Content-Length"},
            )
    return await call_next(request)


# Conservative security headers — this is a JSON API, the values can be strict.
@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
    )
    # Disallow embedding via fetch from arbitrary pages (the API isn't designed
    # to be loaded as a subresource). CORSMiddleware still controls XHR/fetch.
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")
    return response


@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "agent-router"}


# /route is cheap (single LLM call at most) — still rate-limit it for safety.
# NB: `request: Request` is required by slowapi to derive the client IP. Body
# is annotated with `Body(...)` explicitly because slowapi's decorator wraps
# the function with `*args, **kwargs`, which makes FastAPI lose the implicit
# "Pydantic model parameter = body" inference and otherwise treat `req` as a
# query parameter.
@app.post("/route", response_model=RouteResponse)
@limiter.limit("30/minute")
def route(
    request: Request,
    req: RouteRequest = Body(...),
    dispatcher: Dispatcher = Depends(get_dispatcher),
) -> RouteResponse:
    return dispatcher.dispatch(req.input)


# /compare burns real tokens (LLM + embeddings) every call. Tight rate limit.
@app.post("/compare", response_model=CompareResponse)
@limiter.limit("10/minute")
def compare(
    request: Request,
    req: CompareRequest = Body(...),
    service: CompareService = Depends(get_compare_service),
) -> CompareResponse:
    return service.compare(req.input)


# --------------------------------------------------------------------------- #
# Error handlers                                                               #
# --------------------------------------------------------------------------- #


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(_request: Request, exc: RateLimitExceeded):
    get_metrics_collector().record_error(429)
    return JSONResponse(
        status_code=429,
        content={
            "detail": "too many requests — please slow down and try again in a minute",
            "limit": str(exc.detail) if hasattr(exc, "detail") else None,
        },
    )


@app.exception_handler(ProviderUnavailableError)
async def _provider_unavailable_handler(_request: Request, exc: ProviderUnavailableError):
    # Generic message — do NOT echo `str(exc)` (could leak the env var name).
    get_metrics_collector().record_error(503)
    logger.warning("provider unavailable: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "upstream LLM provider is not configured"},
    )


@app.exception_handler(Exception)
async def _generic_error_handler(_request: Request, exc: Exception):
    # FastAPI already maps HTTPException + Pydantic errors before this fires;
    # this is the catch-all for unexpected bugs. Log full detail, expose none.
    get_metrics_collector().record_error(500)
    logger.exception("unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error"},
    )


# --------------------------------------------------------------------------- #
# Observability endpoints                                                      #
# --------------------------------------------------------------------------- #


@app.get("/stats", response_model=StatsResponse)
def stats(collector: MetricsCollector = Depends(get_metrics_collector)) -> StatsResponse:
    """Aggregate snapshot of in-memory metrics since process start.

    Honest about the limitation: metrics reset on every Cloud Run cold start
    or revision change. The `note` field in the response repeats this so
    automated consumers don't mistake the snapshot for persistent data.
    """
    return StatsResponse(**collector.snapshot())

