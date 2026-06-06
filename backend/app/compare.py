"""/compare service — runs the three routers on the same input and aggregates.

Each router is invoked behind a try/except: if any one fails (model artifact
missing, provider credentials missing, network timeout, etc.) its RouterResult
carries the error and the other two still return normal results. The endpoint
never 500s on a single router failure.

Cost numbers come from `eval.metrics.cost_per_1k` — they are derived from
documented pricing assumptions, not measured per call. Latency is measured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Protocol

from .schemas import CompareResponse, RouterResult

logger = logging.getLogger("agent_router.compare")


# Map of known exception class names → safe, fixed messages we're willing to
# echo back to the client. Anything not in this map collapses to a generic
# "router unavailable" so internal paths, stack details, account IDs, or other
# upstream-provider chatter never leak through the per-row error field.
_SAFE_ERROR_MESSAGES: dict[str, str] = {
    "FileNotFoundError":         "router artifact not available",
    "ProviderUnavailableError":  "upstream provider not configured",
    "RuntimeError":              "router not ready",
    "RateLimitError":            "upstream provider rate limited",
    "APIConnectionError":        "upstream provider temporarily unreachable",
    "APITimeoutError":           "upstream provider timed out",
    "AuthenticationError":       "upstream provider authentication failed",
    "BadRequestError":           "upstream provider rejected the request",
    "ValueError":                "router input rejected",
}

_FALLBACK_ERROR_MESSAGE = "router unavailable"


def _safe_error(exc: Exception) -> str:
    return _SAFE_ERROR_MESSAGES.get(type(exc).__name__, _FALLBACK_ERROR_MESSAGE)


# --------------------------------------------------------------------------- #
# Protocols                                                                    #
# --------------------------------------------------------------------------- #


class ClassifierProtocol(Protocol):
    """`router.classifier.IntentClassifier` shape — `classify(text)` returns a
    `RouteDecision`-like object with `.intent: str` and `.confidence: float`."""

    def classify(self, text: str): ...  # noqa: E704  # Protocol stub


class _WithConfidence(Protocol):
    """Both `LLMRouter` and `EmbedRouter` expose `classify_with_confidence`."""

    def classify_with_confidence(self, text: str) -> tuple[str, float]: ...


# --------------------------------------------------------------------------- #
# Adapter — uniform shape for each of the three router strategies              #
# --------------------------------------------------------------------------- #


@dataclass
class _Adapter:
    name: str            # human-readable label shown in the response
    cost_key: str        # key for eval.metrics.cost_per_1k
    factory: Callable[[], object]
    classify: Callable[[object, str], tuple[str, float]]


def _classify_distilbert(clf, text: str) -> tuple[str, float]:
    d = clf.classify(text)
    return d.intent, float(d.confidence)


def _classify_with_conf(router, text: str) -> tuple[str, float]:
    return router.classify_with_confidence(text)


def build_adapters(
    classifier_factory: Callable[[], ClassifierProtocol],
    llm_router_factory: Callable[[], _WithConfidence],
    embed_router_factory: Callable[[], _WithConfidence],
) -> list[_Adapter]:
    """Order here is the order in which results appear in the response."""
    return [
        _Adapter(
            name="DistilBERT (fine-tuned)",
            cost_key="distilbert",
            factory=classifier_factory,
            classify=_classify_distilbert,
        ),
        _Adapter(
            name="LLM zero-shot (gpt-4o-mini)",
            cost_key="llm",
            factory=llm_router_factory,
            classify=_classify_with_conf,
        ),
        _Adapter(
            name="Embeddings + LogReg",
            cost_key="embed",
            factory=embed_router_factory,
            classify=_classify_with_conf,
        ),
    ]


# --------------------------------------------------------------------------- #
# Service                                                                      #
# --------------------------------------------------------------------------- #


class CompareService:
    def __init__(self, adapters: list[_Adapter]):
        self.adapters = adapters

    def compare(self, text: str) -> CompareResponse:
        # Imported lazily — `eval.metrics` lives in the sibling `eval/` package.
        from eval.metrics import cost_per_1k

        results = [self._try_router(adapter, text, cost_per_1k) for adapter in self.adapters]

        ok = [r for r in results if r.error is None]
        intents = {r.intent for r in ok}
        agreement = len(intents) == 1 if ok else False
        fastest = (
            min(ok, key=lambda r: r.latency_ms).router_name if ok else ""
        )
        cheapest = (
            min(ok, key=lambda r: r.cost_per_1k_usd).router_name if ok else ""
        )

        return CompareResponse(
            input=text,
            results=results,
            agreement=agreement,
            fastest=fastest,
            cheapest=cheapest,
        )

    @staticmethod
    def _try_router(
        adapter: _Adapter,
        text: str,
        cost_per_1k_fn: Callable[[str], float],
    ) -> RouterResult:
        # cost_per_1k is a property of the *approach*, not of this specific
        # call — report it even when the call errors out so the frontend can
        # still show "this is what it would cost".
        try:
            cost = float(cost_per_1k_fn(adapter.cost_key))
        except Exception:
            cost = 0.0

        try:
            router = adapter.factory()
            t0 = perf_counter()
            intent, confidence = adapter.classify(router, text)
            latency_ms = (perf_counter() - t0) * 1000.0
            return RouterResult(
                router_name=adapter.name,
                intent=intent,
                confidence=float(confidence),
                latency_ms=float(latency_ms),
                cost_per_1k_usd=cost,
                error=None,
            )
        except Exception as e:
            # Log the full exception for server-side debugging, but only return
            # a fixed safe message to the client (see _SAFE_ERROR_MESSAGES).
            logger.warning(
                "router %r failed: %s: %s", adapter.name, type(e).__name__, e
            )
            return RouterResult(
                router_name=adapter.name,
                intent="",
                confidence=0.0,
                latency_ms=0.0,
                cost_per_1k_usd=cost,
                error=_safe_error(e),
            )
