"""Executor agent — carries out a single subtask and returns the result text."""

from __future__ import annotations

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
