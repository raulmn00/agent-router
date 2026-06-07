"""Routing dispatcher.

Given a classified intent, route to the right execution strategy:

  simple_qa     → 1 direct LLM call (cheapest)
  complex_task  → multi-agent Orchestrator (Planner/Executor/Critic)
  document_qa   → honest stub (RAG would plug in here)
  chitchat      → 1 direct LLM call, very small max_tokens

A **per-class** confidence threshold gates the routing: each intent has its
own floor, and predictions below the floor for their intent land on a cheap
fallback that surfaces the uncertainty instead of routing on a guess.

Why per-class: production data shows the model's confidence distribution
is class-dependent. simple_qa / complex_task / document_qa peak around
0.81-0.94 on clear inputs, but `chitchat` saturates much lower — even
legitimate chitchat tops out around 0.52-0.56. A single global threshold
of 0.65 unfairly demoted a legitimate "Good morning!" to the fallback (see
`scripts/results/prod_routing_report.md`). Per-class thresholds let chitchat
breathe (`0.45`) without letting the genuinely ambiguous inputs through.

Configuration:
  CONFIDENCE_THRESHOLDS  (preferred) JSON object, e.g.
                         '{"chitchat": 0.45, "simple_qa": 0.65}'
  CONFIDENCE_THRESHOLD   (legacy)   scalar; treated as a uniform fallback
                         for any class not explicitly listed in
                         CONFIDENCE_THRESHOLDS. Malformed values are logged
                         and ignored.

The Dispatcher takes its dependencies via constructor injection — the API
layer wires the real ones, tests inject fakes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable, Protocol

from agents import LLMProvider, Orchestrator, get_provider

from .metrics_collector import MetricsCollector, get_metrics_collector
from .schemas import RouteResponse

logger = logging.getLogger("agent_router.dispatch")


class ClassifierProtocol(Protocol):
    """Anything that quacks like `IntentClassifier`."""

    def classify(self, text: str): ...  # returns RouteDecision-shaped object


ProviderFactory = Callable[[str], LLMProvider]


# Defaults are calibrated from production confidence distributions:
#   simple_qa / complex_task / document_qa: 0.81-0.94 on clear inputs
#   chitchat:                                0.52-0.56 on legitimate chitchat
#   genuinely ambiguous (any class):         0.39-0.59
# 0.45 for chitchat catches "what does it say about pricing?" (0.588 in a
# different class) and "build me something cool" (0.510 ⇒ landed as chitchat)
# while letting "good morning!" (0.560) through.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "simple_qa": 0.65,
    "complex_task": 0.65,
    "document_qa": 0.65,
    "chitchat": 0.45,
}

LOW_CONFIDENCE_PATH = "low_confidence_fallback"


SIMPLE_QA_SYSTEM = (
    "You are a concise assistant. Answer the user's factual question in 1-3 "
    "sentences. No preamble."
)

CHITCHAT_SYSTEM = (
    "You are warm and brief. Reply in a single short sentence, keeping the "
    "social register of the user's message."
)

DOCUMENT_QA_STUB = (
    "This question references a document, but the RAG layer isn't wired up in "
    "this demo. In production, this branch would call the retriever "
    "(embeddings → vector store → top-k chunks) and pass the augmented context "
    "to an LLM. See README.md for the planned integration point."
)

_INTENT_PATH_LABEL = {
    "simple_qa": "simple_qa:direct_llm",
    "complex_task": "complex_task:orchestrator",
    "document_qa": "document_qa:rag_stub",
    "chitchat": "chitchat:direct_llm",
}


# --------------------------------------------------------------------------- #
# Configuration helpers                                                        #
# --------------------------------------------------------------------------- #


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _read_scalar_threshold_from_env() -> float | None:
    """Legacy CONFIDENCE_THRESHOLD (scalar). None means: not set."""
    raw = os.environ.get("CONFIDENCE_THRESHOLD")
    if raw is None or raw.strip() == "":
        return None
    try:
        return _clamp01(float(raw))
    except ValueError:
        logger.warning(
            "CONFIDENCE_THRESHOLD is not a number (%r); ignoring", raw,
        )
        return None


def _parse_threshold_map_from_env() -> dict[str, float]:
    """Per-class CONFIDENCE_THRESHOLDS (JSON object). Returns {} on absence/error."""
    raw = os.environ.get("CONFIDENCE_THRESHOLDS")
    if raw is None or raw.strip() == "":
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(
            "CONFIDENCE_THRESHOLDS is not valid JSON (%s); falling back to defaults",
            e,
        )
        return {}
    if not isinstance(parsed, dict):
        logger.warning(
            "CONFIDENCE_THRESHOLDS must be a JSON object, got %s; "
            "falling back to defaults",
            type(parsed).__name__,
        )
        return {}

    cleaned: dict[str, float] = {}
    for intent, value in parsed.items():
        if intent not in DEFAULT_THRESHOLDS:
            logger.warning(
                "CONFIDENCE_THRESHOLDS contains unknown intent %r; ignoring",
                intent,
            )
            continue
        try:
            cleaned[intent] = _clamp01(float(value))
        except (TypeError, ValueError):
            logger.warning(
                "CONFIDENCE_THRESHOLDS[%r] is not a number (%r); ignoring",
                intent, value,
            )
    return cleaned


def get_thresholds() -> dict[str, float]:
    """Resolve the per-class threshold map for this process.

    Resolution order (later wins):
      1. hardcoded `DEFAULT_THRESHOLDS`
      2. legacy scalar `CONFIDENCE_THRESHOLD` — applied to every class
      3. per-class `CONFIDENCE_THRESHOLDS` JSON map — per-key override

    Always returns a dict with one entry per intent in DEFAULT_THRESHOLDS,
    so callers never have to handle KeyError. Malformed input never raises.
    """
    result = dict(DEFAULT_THRESHOLDS)

    scalar = _read_scalar_threshold_from_env()
    if scalar is not None:
        for intent in result:
            result[intent] = scalar

    overrides = _parse_threshold_map_from_env()
    for intent, value in overrides.items():
        result[intent] = value

    return result


# --------------------------------------------------------------------------- #
# Dispatcher                                                                   #
# --------------------------------------------------------------------------- #


class Dispatcher:
    def __init__(
        self,
        classifier: ClassifierProtocol,
        provider_factory: ProviderFactory = get_provider,
        confidence_thresholds: dict[str, float] | None = None,
        metrics: MetricsCollector | None = None,
    ):
        self.classifier = classifier
        self.provider_factory = provider_factory
        # None ⇒ resolve from env / defaults at construction time. Tests can
        # pass an explicit dict to skip the env lookup.
        self.confidence_thresholds = (
            dict(confidence_thresholds)
            if confidence_thresholds is not None
            else get_thresholds()
        )
        # None ⇒ use the module-level singleton. Tests inject a fresh
        # collector to isolate counts.
        self.metrics = metrics if metrics is not None else get_metrics_collector()

    def _threshold_for(self, intent: str) -> float:
        """Per-class threshold, with a strict default for unknown classes.

        An unknown intent shouldn't happen (the classifier is constrained to
        the trained label set), but if a future intent slips through we
        prefer to demote it to the fallback rather than route blindly on it.
        """
        return self.confidence_thresholds.get(intent, max(DEFAULT_THRESHOLDS.values()))

    def dispatch(self, text: str) -> RouteResponse:
        t0 = time.perf_counter()
        decision = self.classifier.classify(text)
        intent = decision.intent
        confidence = float(decision.confidence)
        threshold = self._threshold_for(intent)

        # Per-class threshold gate. The numbers and the chitchat-specific
        # value of 0.45 are empirically grounded — see the module docstring
        # and scripts/results/prod_routing_report.md.
        if confidence < threshold:
            response = self._low_confidence_fallback(intent, confidence, threshold)
        elif intent == "simple_qa":
            answer, path, trace = self._simple_qa(text)
            response = RouteResponse(
                intent=intent, confidence=confidence,
                answer=answer, path_taken=path, trace=trace,
            )
        elif intent == "complex_task":
            answer, path, trace = self._complex_task(text)
            response = RouteResponse(
                intent=intent, confidence=confidence,
                answer=answer, path_taken=path, trace=trace,
            )
        elif intent == "document_qa":
            answer, path, trace = self._document_qa(text)
            response = RouteResponse(
                intent=intent, confidence=confidence,
                answer=answer, path_taken=path, trace=trace,
            )
        elif intent == "chitchat":
            answer, path, trace = self._chitchat(text)
            response = RouteResponse(
                intent=intent, confidence=confidence,
                answer=answer, path_taken=path, trace=trace,
            )
        else:
            # Should be unreachable — the classifier is constrained to INTENTS —
            # but if a future intent is added without a dispatch arm, fail loud.
            raise RuntimeError(f"no dispatch arm for intent {intent!r}")

        latency_ms = (time.perf_counter() - t0) * 1000.0
        was_fallback = response.path_taken == LOW_CONFIDENCE_PATH

        # Observability: counters + a single structured log line. NEVER log
        # `text` itself — only its length. `intent` is what the classifier
        # PICKED, even on the fallback path (so we can see *which* class is
        # producing the most fallbacks, which is the calibration signal).
        self.metrics.record_request(
            intent=intent,
            confidence=confidence,
            path_taken=response.path_taken,
            latency_ms=latency_ms,
            was_fallback=was_fallback,
        )
        logger.info(
            "route.dispatched",
            extra={
                "input_length": len(text),
                "intent": intent,
                "confidence": round(confidence, 4),
                "threshold": round(threshold, 2),
                "path_taken": response.path_taken,
                "latency_ms": round(latency_ms, 2),
                "was_fallback": was_fallback,
            },
        )
        return response

    # ------------------------------------------------------------------- #
    # Low-confidence fallback                                              #
    # ------------------------------------------------------------------- #

    def _low_confidence_fallback(
        self,
        intent: str,
        confidence: float,
        threshold: float,
    ) -> RouteResponse:
        """Return a transparent low-confidence response instead of routing.

        Intentionally cheap: no LLM call, no orchestrator. The schema stays
        the same (RouteResponse) — `intent` still reports the model's best
        guess and `confidence` the real measurement, but `path_taken` is the
        explicit marker `low_confidence_fallback` so callers can route the
        UI accordingly.

        The threshold reported back in `answer`/`trace` is the per-class one,
        not a global value — observability for the operator.

        Future enhancement (not implemented): ask the user a clarifying
        question via an LLM, or escalate to an LLM-as-router for this single
        call.
        """
        would_have_taken = _INTENT_PATH_LABEL.get(intent, intent)
        answer = (
            "I couldn't confidently classify your request — the most likely "
            f"intent was '{intent}' with confidence {confidence:.3f}, below "
            f"the {threshold:.2f} threshold set for that class. Try "
            "rephrasing with a clearer cue (e.g. start a coding task with "
            "'build', reference a document explicitly, or just say hi)."
        )
        return RouteResponse(
            intent=intent,
            confidence=confidence,
            answer=answer,
            path_taken=LOW_CONFIDENCE_PATH,
            trace=[
                f"low confidence: intent={intent!r} confidence={confidence:.3f} "
                f"threshold={threshold:.2f} (per-class)",
                f"would have routed to: {would_have_taken}",
                "no LLM call made; future: ask clarifying question or escalate to LLM router",
            ],
        )

    # ------------------------------------------------------------------- #
    # Dispatch arms                                                        #
    # ------------------------------------------------------------------- #

    def _simple_qa(self, text: str) -> tuple[str, str, list[str]]:
        provider = self.provider_factory("openai")
        answer = provider.complete(
            [
                {"role": "system", "content": SIMPLE_QA_SYSTEM},
                {"role": "user", "content": text},
            ],
            max_tokens=200,
        ).strip()
        return answer, "simple_qa:direct_llm", [f"direct LLM call → {provider.name}"]

    def _complex_task(self, text: str) -> tuple[str, str, list[str]]:
        provider = self.provider_factory("openai")
        result = Orchestrator(provider).run_task(text)
        return result.answer, "complex_task:orchestrator", result.trace

    def _document_qa(self, text: str) -> tuple[str, str, list[str]]:
        # === RAG integration point ===
        # Replace this stub with: retrieve(text) → build_context(chunks) →
        # provider.complete([{system: ..., user: context + text}]).
        return DOCUMENT_QA_STUB, "document_qa:rag_stub", ["RAG layer not implemented"]

    def _chitchat(self, text: str) -> tuple[str, str, list[str]]:
        provider = self.provider_factory("openai")
        answer = provider.complete(
            [
                {"role": "system", "content": CHITCHAT_SYSTEM},
                {"role": "user", "content": text},
            ],
            max_tokens=60,
        ).strip()
        return answer, "chitchat:direct_llm", [f"direct LLM call → {provider.name}"]
