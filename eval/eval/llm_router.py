"""Baseline 1: LLM-as-router (zero-shot).

Strategy: send the user text + the list of valid intent names to gpt-4o-mini
in a system prompt; instruct it to return ONLY one intent name. Parse robustly
(strip whitespace, lowercase, match against INTENTS, fall back to "simple_qa"
if the model goes off-script — same fallback rule for every approach so the
comparison stays fair).
"""

from __future__ import annotations

import math
import os

from agents import get_provider
from router import INTENTS

DEFAULT_FALLBACK = "simple_qa"
LLM_NO_LOGPROBS_CONFIDENCE = 1.0  # fallback when the API doesn't return logprobs

SYSTEM_PROMPT = (
    "You are an intent classifier. Given a user message, classify it into "
    "exactly ONE of these intents:\n"
    + "\n".join(f"- {i}" for i in INTENTS)
    + "\n\nDefinitions:\n"
    "- simple_qa: a short, factual question with a single-step answer.\n"
    "- complex_task: a multi-step task that requires planning or building "
    "something substantial.\n"
    "- document_qa: a question that references an attached/uploaded/provided "
    "document.\n"
    "- chitchat: small talk, greetings, social messages, jokes.\n\n"
    "Respond with ONLY the intent name. No explanation, no punctuation, no quotes."
)


def _normalize(raw: str) -> str:
    """Map a free-form LLM output to a canonical intent name (or fallback)."""
    text = raw.strip().lower()
    # First exact match — fast path.
    if text in INTENTS:
        return text
    # The model sometimes wraps the intent in quotes or adds a period.
    for delim in ('"', "'", ".", ",", ":", ";", "\n"):
        text = text.replace(delim, " ")
    for token in text.split():
        if token in INTENTS:
            return token
    return DEFAULT_FALLBACK


class LLMRouter:
    name = "llm"

    def __init__(self, provider=None):
        self.provider = provider or get_provider("openai")

    def classify(self, text: str) -> str:
        raw = self.provider.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=10,
        )
        return _normalize(raw)

    def classify_with_confidence(self, text: str) -> tuple[str, float]:
        """Like `classify`, but also returns a confidence in [0, 1].

        Confidence is derived from the OpenAI logprobs of the FIRST output
        token (the disambiguating commitment point). If the underlying
        provider doesn't expose logprobs (e.g. `FakeProvider`), confidence
        falls back to LLM_NO_LOGPROBS_CONFIDENCE (1.0) and the caller can
        treat it as "no probability information available".
        """
        # Only OpenAIProvider currently supports logprobs; fall back gracefully
        # for any provider that doesn't have a `complete_with_logprobs` method.
        complete_with_logprobs = getattr(self.provider, "complete_with_logprobs", None)
        if complete_with_logprobs is None:
            intent = self.classify(text)
            return intent, LLM_NO_LOGPROBS_CONFIDENCE

        raw, logprobs = complete_with_logprobs(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=10,
        )
        intent = _normalize(raw)
        if not logprobs:
            return intent, LLM_NO_LOGPROBS_CONFIDENCE
        confidence = math.exp(logprobs[0])  # first-token probability
        return intent, max(0.0, min(1.0, confidence))
