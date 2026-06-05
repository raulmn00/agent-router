"""LLM provider abstraction.

A single `LLMProvider.complete(messages, max_tokens) -> str` keeps the rest of
the system framework-agnostic and makes testing trivial: pass a `FakeProvider`
with canned responses.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable

Messages = list[dict]


class ProviderUnavailableError(RuntimeError):
    """Raised when a provider can't be used (e.g. missing credentials).

    Distinct from generic RuntimeError so the API layer can map it to 503
    without scanning error strings.
    """


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete(self, messages: Messages, max_tokens: int = 512) -> str:
        """Run a single chat completion and return the assistant text."""


# --------------------------------------------------------------------------- #
# OpenAI                                                                      #
# --------------------------------------------------------------------------- #


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        # Import is lazy so the rest of the system loads without `openai`
        # installed (e.g. in unit tests that only use FakeProvider).
        if not self._api_key:
            raise ProviderUnavailableError("openai credentials missing")
        from openai import OpenAI

        self._client = OpenAI(api_key=self._api_key)

    def complete(self, messages: Messages, max_tokens: int = 512) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

    def complete_with_logprobs(
        self, messages: Messages, max_tokens: int = 512
    ) -> tuple[str, list[float]]:
        """Same as `complete`, but also returns per-token logprobs of the response.

        Used by routes (e.g. the zero-shot LLM router) that want to derive a
        confidence score from the model's commitment to each output token.
        Not part of the LLMProvider ABC — providers that don't support it can
        simply omit this method; callers must check with `getattr`.
        """
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
            logprobs=True,
        )
        text = resp.choices[0].message.content or ""
        content_logprobs = resp.choices[0].logprobs
        if content_logprobs is None or not content_logprobs.content:
            return text, []
        return text, [tok.logprob for tok in content_logprobs.content]


# --------------------------------------------------------------------------- #
# Anthropic                                                                   #
# --------------------------------------------------------------------------- #


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ProviderUnavailableError("anthropic credentials missing")
        from anthropic import Anthropic

        self._client = Anthropic(api_key=self._api_key)

    def complete(self, messages: Messages, max_tokens: int = 512) -> str:
        # Anthropic separates `system` from the conversation array.
        system_chunks = [m["content"] for m in messages if m["role"] == "system"]
        convo = [m for m in messages if m["role"] != "system"]
        resp = self._client.messages.create(
            model=self.model,
            system="\n".join(system_chunks) if system_chunks else None,
            messages=convo,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        parts = [block.text for block in resp.content if getattr(block, "type", "") == "text"]
        return "".join(parts)


# --------------------------------------------------------------------------- #
# Fake — for deterministic tests                                              #
# --------------------------------------------------------------------------- #


class FakeProvider(LLMProvider):
    """Returns canned responses in FIFO order.

    Two ways to use it:
    - `FakeProvider(responses=[r1, r2, r3])`: cycles through the list, raising
      if more calls happen than responses provided.
    - Subclass and override `complete()` for custom logic.

    Every call is appended to `self.calls` so tests can assert on what each
    agent prompted.
    """

    name = "fake"

    def __init__(self, responses: Iterable[str] | None = None):
        self._queue: deque[str] = deque(responses or [])
        self.calls: list[tuple[Messages, int]] = []

    def queue(self, response: str) -> "FakeProvider":
        self._queue.append(response)
        return self

    def complete(self, messages: Messages, max_tokens: int = 512) -> str:
        self.calls.append((messages, max_tokens))
        if not self._queue:
            raise AssertionError(
                f"FakeProvider has no more queued responses but was called {len(self.calls)} times. "
                f"Last messages: {messages!r}"
            )
        return self._queue.popleft()


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def get_provider(name: str = "openai", **kwargs) -> LLMProvider:
    name = name.lower()
    if name == "openai":
        return OpenAIProvider(**kwargs)
    if name == "anthropic":
        return AnthropicProvider(**kwargs)
    if name == "fake":
        return FakeProvider(**kwargs)
    raise ValueError(f"Unknown provider: {name!r}. Use 'openai' | 'anthropic' | 'fake'.")
