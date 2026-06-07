"""Orchestrator: Planner → (parallel) Executors → Aggregate → Critic, retry once on rejection.

The execution of subtasks within an attempt is **concurrent**. The Planner is
sequential (one call), the Critic is sequential (one call), the retry loop is
sequential (initial + 1 retry). Only the N subtasks of a given attempt run in
parallel — that's where the latency was; in production a complex_task with 6
subtasks paid 6× the LLM roundtrip serially (~30 s) and could time out at 60 s.

Concurrency is bounded by `asyncio.Semaphore(max_concurrent_executors)` so we
don't fan out 20 LLM calls at once and trip the provider's per-minute rate
limits. Default 5 — configurable via `MAX_CONCURRENT_EXECUTORS` env var.

Failure isolation: subtasks run with `asyncio.gather(..., return_exceptions=True)`.
A single subtask raising doesn't lose the others — its slot is replaced by a
failure marker in the aggregated answer and a `EXEC[i] FAILED ...` line in the
trace.

Public API preserved: `run_task(task)` stays synchronous so the existing
FastAPI dispatcher (running in Starlette's threadpool) keeps working without
changes. It internally calls `asyncio.run(run_task_async(task))`. If you want
to compose this into a fully-async stack later, call `run_task_async` and
`await` it; making the dispatcher and FastAPI handler `async def` would let you
skip the `asyncio.run` wrapper entirely.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

from .critic import Critic, Review
from .executor import Executor
from .planner import Planner
from .providers import LLMProvider

MAX_STEPS = 6
MAX_ATTEMPTS = 2  # 1 initial + 1 retry, per spec ("refaz UMA vez")
DEFAULT_MAX_CONCURRENT_EXECUTORS = 5


def _max_concurrent_from_env() -> int:
    raw = os.environ.get("MAX_CONCURRENT_EXECUTORS")
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_CONCURRENT_EXECUTORS
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_MAX_CONCURRENT_EXECUTORS
    return max(1, n)


@dataclass
class OrchestrationResult:
    answer: str
    trace: list[str] = field(default_factory=list)
    steps: int = 0  # total executor calls across all attempts


class Orchestrator:
    def __init__(
        self,
        provider: LLMProvider,
        max_concurrent_executors: int | None = None,
    ):
        self.provider = provider
        self.planner = Planner(provider, max_steps=MAX_STEPS)
        self.executor = Executor(provider)
        self.critic = Critic(provider)
        self.max_concurrent_executors = (
            max_concurrent_executors
            if max_concurrent_executors is not None
            else _max_concurrent_from_env()
        )

    # ------------------------------------------------------------------ #
    # Public entrypoints                                                  #
    # ------------------------------------------------------------------ #

    def run_task(self, task: str) -> OrchestrationResult:
        """Sync wrapper for callers that aren't async-aware.

        The FastAPI handler is `def route(...)` (sync), so it runs in
        Starlette's threadpool — each worker thread has no event loop and
        `asyncio.run` is safe to call here. If the handler ever becomes
        `async def`, drop this wrapper and call `await run_task_async`.
        """
        return asyncio.run(self.run_task_async(task))

    async def run_task_async(self, task: str) -> OrchestrationResult:
        trace: list[str] = [f"TASK: {task}"]
        total_steps = 0
        last_answer = ""
        last_feedback: str | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            trace.append(f"--- attempt {attempt} ---")

            subtasks = self.planner.plan(task, feedback=last_feedback)
            trace.append(f"PLAN: {subtasks}")

            results, exec_traces, n_failures = await self._execute_concurrently(subtasks)
            total_steps += len(subtasks)
            trace.extend(exec_traces)
            trace.append(
                f"executed {len(subtasks)} subtask(s) concurrently "
                f"(max_concurrent={self.max_concurrent_executors}, failures={n_failures})"
            )

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

    # ------------------------------------------------------------------ #
    # Concurrent execution                                                #
    # ------------------------------------------------------------------ #

    async def _execute_concurrently(
        self, subtasks: list[str]
    ) -> tuple[list[str], list[str], int]:
        """Run subtasks in parallel, bounded by the configured semaphore.

        Returns (results, trace_lines, n_failures). Results are in the same
        order as subtasks; failures are surfaced as inline markers, never
        dropped silently.
        """
        sem = asyncio.Semaphore(self.max_concurrent_executors)

        async def _bounded(subtask: str) -> str:
            async with sem:
                return await self.executor.execute_async(subtask)

        # gather preserves order: results[i] corresponds to subtasks[i]. We
        # use return_exceptions=True so one failing subtask doesn't tear down
        # the rest — the orchestrator still produces an aggregated answer.
        raw = await asyncio.gather(
            *[_bounded(s) for s in subtasks],
            return_exceptions=True,
        )

        results: list[str] = []
        trace_lines: list[str] = []
        n_failures = 0
        for i, (subtask, item) in enumerate(zip(subtasks, raw), start=1):
            if isinstance(item, BaseException):
                n_failures += 1
                marker = f"[execution failed: {type(item).__name__}: {item}]"
                results.append(marker)
                trace_lines.append(
                    f"EXEC[{i}] FAILED {subtask!r} -> {type(item).__name__}: {item}"
                )
            else:
                results.append(item)
                trace_lines.append(f"EXEC[{i}] OK {subtask!r} -> {item[:200]!r}")
        return results, trace_lines, n_failures


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
