"""Orchestrator: Planner → Executors → Aggregate → Critic, retry once on rejection."""

from __future__ import annotations

from dataclasses import dataclass, field

from .critic import Critic, Review
from .executor import Executor
from .planner import Planner
from .providers import LLMProvider

MAX_STEPS = 6
MAX_ATTEMPTS = 2  # 1 initial + 1 retry, per spec ("refaz UMA vez")


@dataclass
class OrchestrationResult:
    answer: str
    trace: list[str] = field(default_factory=list)
    steps: int = 0  # total executor calls across all attempts


class Orchestrator:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.planner = Planner(provider, max_steps=MAX_STEPS)
        self.executor = Executor(provider)
        self.critic = Critic(provider)

    def run_task(self, task: str) -> OrchestrationResult:
        trace: list[str] = [f"TASK: {task}"]
        total_steps = 0
        last_answer = ""
        last_feedback: str | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            trace.append(f"--- attempt {attempt} ---")

            subtasks = self.planner.plan(task, feedback=last_feedback)
            trace.append(f"PLAN: {subtasks}")

            # Executors run sequentially. To parallelize independent subtasks,
            # swap to `asyncio.gather(*[run_async(s) for s in subtasks])` here
            # — the executor is stateless, so the only constraint is preserving
            # input order in the aggregation below.
            results: list[str] = []
            for i, subtask in enumerate(subtasks, start=1):
                result = self.executor.execute(subtask)
                total_steps += 1
                trace.append(f"EXEC[{i}] {subtask!r} -> {result[:200]!r}")
                results.append(result)

            answer = _aggregate(subtasks, results)
            last_answer = answer
            trace.append(f"AGGREGATED:\n{answer[:300]}")

            review: Review = self.critic.review(task, answer)
            trace.append(
                f"CRITIC: approved={review.approved} feedback={review.feedback!r}"
            )

            if review.approved:
                return OrchestrationResult(answer=answer, trace=trace, steps=total_steps)

            # Reproved — record feedback and retry once.
            last_feedback = review.feedback

        trace.append("MAX_ATTEMPTS reached; returning last answer.")
        return OrchestrationResult(answer=last_answer, trace=trace, steps=total_steps)


def _aggregate(subtasks: list[str], results: list[str]) -> str:
    """Combine per-subtask results into a single answer string.

    Kept dead simple on purpose — for a portfolio demo, transparency beats a
    second LLM call to synthesize. Swap for a synthesizer agent if the use
    case demands narrative output.
    """
    parts = []
    for i, (subtask, result) in enumerate(zip(subtasks, results), start=1):
        parts.append(f"{i}. {subtask}\n   -> {result}")
    return "\n".join(parts)
