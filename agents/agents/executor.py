"""Executor agent — carries out a single subtask and returns the result text."""

from __future__ import annotations

import asyncio

from .providers import LLMProvider

EXECUTOR_SYSTEM = (
    "You are an Executor agent. Carry out the given subtask and produce a "
    "concise, useful result. If the subtask requires external tools you don't "
    "have, explain what you would do and produce the best textual answer you "
    "can. Output only the result — no preamble."
)


class Executor:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def execute(self, subtask: str) -> str:
        messages = [
            {"role": "system", "content": EXECUTOR_SYSTEM},
            {"role": "user", "content": subtask},
        ]
        return self.provider.complete(messages, max_tokens=500).strip()

    async def execute_async(self, subtask: str) -> str:
        """Async wrapper around `execute()` for concurrent orchestration.

        We use `asyncio.to_thread` instead of switching to AsyncOpenAI/
        AsyncAnthropic because:

          1. `LLMProvider` is a sync ABC consumed by Planner, Executor, Critic,
             `eval/llm_router.LLMRouter`, and `eval/embed_router.EmbedRouter`.
             Switching to async would fork the entire provider hierarchy.
          2. The bottleneck is the ~500 ms-2 s LLM API roundtrip, dwarfing
             the cost of spawning a worker thread.
          3. The OpenAI/Anthropic Python SDKs use blocking `httpx` calls; the
             GIL releases during the underlying socket I/O, so threads in
             `asyncio.to_thread` actually run concurrently for I/O-bound work.

        Net effect: this method makes `Executor` parallelizable from the
        orchestrator without touching any other module.
        """
        return await asyncio.to_thread(self.execute, subtask)
