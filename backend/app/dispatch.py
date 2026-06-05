"""Routing dispatcher.

Given a classified intent, route to the right execution strategy:

  simple_qa     → 1 direct LLM call (cheapest)
  complex_task  → multi-agent Orchestrator (Planner/Executor/Critic)
  document_qa   → honest stub (RAG would plug in here)
  chitchat      → 1 direct LLM call, very small max_tokens

The Dispatcher takes its dependencies via constructor injection — the API
layer wires the real ones, tests inject fakes.
"""

from __future__ import annotations

from typing import Callable, Protocol

from agents import LLMProvider, Orchestrator, get_provider

from .schemas import RouteResponse


class ClassifierProtocol(Protocol):
    """Anything that quacks like `IntentClassifier`."""

    def classify(self, text: str): ...  # returns RouteDecision-shaped object


ProviderFactory = Callable[[str], LLMProvider]


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


class Dispatcher:
    def __init__(
        self,
        classifier: ClassifierProtocol,
        provider_factory: ProviderFactory = get_provider,
    ):
        self.classifier = classifier
        self.provider_factory = provider_factory

    def dispatch(self, text: str) -> RouteResponse:
        decision = self.classifier.classify(text)
        intent = decision.intent
        confidence = float(decision.confidence)

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
