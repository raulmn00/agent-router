"""FastAPI application.

Endpoints:
  GET  /         — liveness probe
  POST /route    — classify + dispatch + respond
  POST /compare  — run all three routers (DistilBERT / LLM / Embed) on the
                   same input and aggregate (rate-limited; burns real tokens)

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
from .schemas import CompareRequest, CompareResponse, RouteRequest, RouteResponse

logger = logging.getLogger("agent_router")
logging.basicConfig(level=logging.INFO)

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

app = FastAPI(title="agent-router", version="0.2.0")
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SlowAPIMiddleware)


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
    logger.warning("provider unavailable: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "upstream LLM provider is not configured"},
    )


@app.exception_handler(Exception)
async def _generic_error_handler(_request: Request, exc: Exception):
    # FastAPI already maps HTTPException + Pydantic errors before this fires;
    # this is the catch-all for unexpected bugs. Log full detail, expose none.
    logger.exception("unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error"},
    )
