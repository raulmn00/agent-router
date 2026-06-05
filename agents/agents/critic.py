"""Critic agent — judges whether an aggregated answer satisfies the task."""

from __future__ import annotations

from dataclasses import dataclass

from ._json import extract_json
from .providers import LLMProvider

CRITIC_SYSTEM = (
    "You are a Critic agent. Given the user's original task and a candidate "
    "answer, judge whether the answer adequately addresses the task. "
    'Respond with strict JSON: {"approved": true|false, "feedback": "..."}. '
    "If approved is true, feedback can be brief praise or notes. If false, "
    "feedback must explain what is missing or wrong so the next attempt can fix it."
)


@dataclass(frozen=True)
class Review:
    approved: bool
    feedback: str


class Critic:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def review(self, task: str, result: str) -> Review:
        user = (
            f"ORIGINAL TASK:\n{task}\n\n"
            f"CANDIDATE ANSWER:\n{result}\n\n"
            "Is the candidate answer adequate?"
        )
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": user},
        ]
        raw = self.provider.complete(messages, max_tokens=300)
        data = extract_json(raw)
        if not isinstance(data, dict) or "approved" not in data:
            raise ValueError(f"Critic: expected {{'approved': bool, 'feedback': str}}, got {data!r}")
        approved = bool(data["approved"])
        feedback = str(data.get("feedback", "")).strip()
        return Review(approved=approved, feedback=feedback)
