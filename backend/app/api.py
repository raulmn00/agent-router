"""FastAPI application.

Endpoints:
  GET  /        — liveness probe
  POST /route   — classify + dispatch + respond

Error model:
  422  Pydantic validation (automatic)
  503  Provider unavailable (missing credentials — message is generic)
  500  Anything else (logged server-side, opaque to client)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents import ProviderUnavailableError, get_provider

from .dispatch import Dispatcher
from .schemas import RouteRequest, RouteResponse

logger = logging.getLogger("agent_router")
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------- #
# Dependency wiring                                                            #
# --------------------------------------------------------------------------- #


class _LazyClassifier:
    """Defers model loading until the first `classify()` call.

    Why lazy: FastAPI resolves dependencies in parallel with body validation,
    so 422-path tests (and real malformed requests) would otherwise pay the
    full model load cost just to get rejected. Lazy = cheap rejections.
    Once loaded, the underlying `IntentClassifier` is shared across requests.
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


def get_dispatcher() -> Dispatcher:
    """FastAPI dependency — override in tests via `app.dependency_overrides`."""
    return Dispatcher(classifier=_classifier_singleton(), provider_factory=get_provider)


# --------------------------------------------------------------------------- #
# App                                                                          #
# --------------------------------------------------------------------------- #


app = FastAPI(title="agent-router", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "agent-router"}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest, dispatcher: Dispatcher = Depends(get_dispatcher)) -> RouteResponse:
    return dispatcher.dispatch(req.input)


# --------------------------------------------------------------------------- #
# Error handlers                                                               #
# --------------------------------------------------------------------------- #


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
    logger.exception("unhandled error during /route: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error"},
    )
