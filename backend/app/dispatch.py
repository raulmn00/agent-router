"""Routing dispatcher.

Given a classified intent, route to the right execution strategy:

  simple_qa     → 1 direct LLM call (cheapest)
  complex_task  → multi-agent Orchestrator (Planner/Executor/Critic)
  document_qa   → honest stub (RAG would plug in here)
  chitchat      → 1 direct LLM call, very small max_tokens

A confidence threshold (env: `CONFIDENCE_THRESHOLD`, default `0.65`) gates the
routing: predictions below the threshold land on a cheap fallback that
surfaces the uncertainty instead of routing on a low-confidence guess. The
0.65 default is empirically motivated — see `router/results/generalization_test.txt`
and the evaluation section of the root README.

The Dispatcher takes its dependencies via constructor injection — the API
layer wires the real ones, tests inject fakes.
"""

from __future__ import annotations

import os
from typing import Callable, Protocol

from agents import LLMProvider, Orchestrator, get_provider

from .schemas import RouteResponse


class ClassifierProtocol(Protocol):
    """Anything that quacks like `IntentClassifier`."""

    def classify(self, text: str): ...  # returns RouteDecision-shaped object


ProviderFactory = Callable[[str], LLMProvider]


DEFAULT_CONFIDENCE_THRESHOLD = 0.65
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

# Human-readable path label per intent — used in the fallback trace so the
# operator can see which arm would have run if the prediction had been confident.
_INTENT_PATH_LABEL = {
    "simple_qa": "simple_qa:direct_llm",
    "complex_task": "complex_task:orchestrator",
    "document_qa": "document_qa:rag_stub",
    "chitchat": "chitchat:direct_llm",
}


def _read_threshold_from_env() -> float:
    raw = os.environ.get("CONFIDENCE_THRESHOLD")
    if raw is None or raw.strip() == "":
        return DEFAULT_CONFIDENCE_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_CONFIDENCE_THRESHOLD
    # Clamp to [0, 1] — anything outside is almost certainly a typo and we
    # don't want a misconfig to either route on everything or block everything.
    return max(0.0, min(1.0, value))


class Dispatcher:
    def __init__(
        self,
        classifier: ClassifierProtocol,
        provider_factory: ProviderFactory = get_provider,
        confidence_threshold: float | None = None,
    ):
        self.classifier = classifier
        self.provider_factory = provider_factory
        # `None` means: read from CONFIDENCE_THRESHOLD env var at construction
        # time. Tests can pass an explicit float to skip the env lookup.
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else _read_threshold_from_env()
        )

    def dispatch(self, text: str) -> RouteResponse:
        decision = self.classifier.classify(text)
        intent = decision.intent
        confidence = float(decision.confidence)

        # Threshold gate — if the model isn't sure, don't pick a path on
        # its behalf. The empirical motivation is `router/results/generalization_test.txt`:
        # clear inputs land at 0.81–0.94, ambiguous ones at 0.39–0.59.
        if confidence < self.confidence_threshold:
            return self._low_confidence_fallback(intent, confidence)

        if intent == "simple_qa":
            answer, path, trace = self._simple_qa(text)
        elif intent == "complex_task":
            answer, path, trace = self._complex_task(text)
        elif intent == "document_qa":
            answer, path, trace = self._document_qa(text)
        elif intent == "chitchat":
            answer, path, trace = self._chitchat(text)
        else:
            # Should be unreachable — the classifier is constrained to INTENTS —
            # but if a future intent is added without a dispatch arm, fail loud.
            raise RuntimeError(f"no dispatch arm for intent {intent!r}")

        return RouteResponse(
            intent=intent,
            confidence=confidence,
            answer=answer,
            path_taken=path,
            trace=trace,
        )

    # ------------------------------------------------------------------- #
    # Low-confidence fallback                                              #
    # ------------------------------------------------------------------- #

    def _low_confidence_fallback(self, intent: str, confidence: float) -> RouteResponse:
        """Return a transparent low-confidence response instead of routing.

        Intentionally cheap: no LLM call, no orchestrator. The schema stays
        the same (RouteResponse) — `intent` still reports the model's best
        guess and `confidence` the real measurement, but `path_taken` is the
        explicit marker `low_confidence_fallback` so callers can route the
        UI accordingly.

        Future enhancement (not implemented): ask the user a clarifying
        question via an LLM, or escalate to an LLM-as-router for this single
        call. Either path keeps the dispatcher in charge of the *decision*
        while letting the LLM resolve the ambiguity.
        """
        threshold = self.confidence_threshold
        would_have_taken = _INTENT_PATH_LABEL.get(intent, intent)
        answer = (
            "I couldn't confidently classify your request — the most likely "
            f"intent was '{intent}' with confidence {confidence:.3f}, below "
            f"the threshold of {threshold:.2f}. Try rephrasing with a clearer "
            "cue (e.g. start a coding task with 'build', reference a document "
            "explicitly, or just say hi)."
        )
        return RouteResponse(
            intent=intent,
            confidence=confidence,
            answer=answer,
            path_taken=LOW_CONFIDENCE_PATH,
            trace=[
                f"low confidence: intent={intent!r} confidence={confidence:.3f} "
                f"threshold={threshold:.2f}",
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
