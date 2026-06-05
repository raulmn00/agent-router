"""Planner agent — decomposes a task into ordered subtasks."""

from __future__ import annotations

from ._json import extract_json
from .providers import LLMProvider

PLANNER_SYSTEM = (
    "You are a Planner agent. Decompose the user's task into 2-6 concrete, "
    "ordered subtasks that, executed in sequence, would accomplish the goal. "
    "Respond with strict JSON of the form: "
    '{"steps": ["...", "..."]}.  No prose outside the JSON.'
)


class Planner:
    def __init__(self, provider: LLMProvider, max_steps: int = 6):
        self.provider = provider
        self.max_steps = max_steps

    def plan(self, task: str, *, feedback: str | None = None) -> list[str]:
        user = task
        if feedback:
            user = (
                f"{task}\n\n"
                f"The previous attempt was rejected with this feedback: {feedback}\n"
                "Produce a revised plan that addresses it."
            )
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user},
        ]
        raw = self.provider.complete(messages, max_tokens=400)
        data = extract_json(raw)
        if not isinstance(data, dict) or "steps" not in data:
            raise ValueError(f"Planner: expected {{'steps': [...]}}, got {data!r}")
        steps = data["steps"]
        if not isinstance(steps, list) or not all(isinstance(s, str) and s.strip() for s in steps):
            raise ValueError(f"Planner: 'steps' must be a non-empty list of strings, got {steps!r}")
        return steps[: self.max_steps]
