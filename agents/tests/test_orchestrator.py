"""End-to-end orchestrator tests using FakeProvider — no network, no LLM cost."""

from __future__ import annotations

import json

import pytest

from agents import (
    Critic,
    Executor,
    FakeProvider,
    OrchestrationResult,
    Orchestrator,
    Planner,
    get_provider,
)
from agents._json import extract_json


# --------------------------------------------------------------------------- #
# JSON extraction utility                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"steps": ["a", "b"]}', {"steps": ["a", "b"]}),
        ('```json\n{"steps": ["a"]}\n```', {"steps": ["a"]}),
        ('Sure! Here is the plan:\n{"steps": ["x"]}\nlet me know.', {"steps": ["x"]}),
        ('```\n{"approved": true, "feedback": "ok"}\n```', {"approved": True, "feedback": "ok"}),
    ],
)
def test_extract_json_robust(raw, expected):
    assert extract_json(raw) == expected


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("not even close")


# --------------------------------------------------------------------------- #
# Per-agent unit tests                                                        #
# --------------------------------------------------------------------------- #


def test_planner_parses_fenced_json_and_caps_steps():
    fp = FakeProvider(
        responses=[
            "```json\n" + json.dumps({"steps": [f"step {i}" for i in range(1, 9)]}) + "\n```",
        ]
    )
    planner = Planner(fp, max_steps=6)
    steps = planner.plan("do a thing")
    assert len(steps) == 6
    assert steps[0] == "step 1"


def test_planner_rejects_malformed_steps():
    fp = FakeProvider(responses=['{"steps": "not a list"}'])
    with pytest.raises(ValueError):
        Planner(fp).plan("anything")


def test_executor_strips_whitespace():
    fp = FakeProvider(responses=["   hello world   \n"])
    assert Executor(fp).execute("greet") == "hello world"


def test_critic_returns_review_dataclass():
    fp = FakeProvider(responses=['{"approved": true, "feedback": "looks good"}'])
    review = Critic(fp).review("task", "answer")
    assert review.approved is True
    assert review.feedback == "looks good"


# --------------------------------------------------------------------------- #
# Orchestrator happy path                                                     #
# --------------------------------------------------------------------------- #


def _planner_response(steps: list[str]) -> str:
    return json.dumps({"steps": steps})


def _critic_response(approved: bool, feedback: str = "") -> str:
    return json.dumps({"approved": approved, "feedback": feedback})


def test_orchestrator_happy_path_approves_on_first_attempt():
    # Sequence: planner, executor x2, critic
    responses = [
        _planner_response(["research the topic", "draft the answer"]),
        "research result A",
        "draft result B",
        _critic_response(True, "perfect"),
    ]
    fp = FakeProvider(responses=responses)
    result = Orchestrator(fp).run_task("write a blog post")

    assert isinstance(result, OrchestrationResult)
    assert result.steps == 2  # two executor calls
    assert "research result A" in result.answer
    assert "draft result B" in result.answer
    # Trace should mention the plan, each exec, and the critic's verdict
    joined = "\n".join(result.trace)
    assert "PLAN:" in joined
    assert "EXEC[1]" in joined and "EXEC[2]" in joined
    assert "approved=True" in joined


# --------------------------------------------------------------------------- #
# Orchestrator retry path                                                     #
# --------------------------------------------------------------------------- #


def test_orchestrator_retries_once_when_critic_reproves():
    # Attempt 1: 2-step plan, executor x2, critic rejects.
    # Attempt 2: 1-step plan (replanned w/ feedback), executor x1, critic approves.
    responses = [
        _planner_response(["step A", "step B"]),
        "exec 1 (bad)",
        "exec 2 (bad)",
        _critic_response(False, "needs a clearer summary"),
        _planner_response(["produce a clear summary"]),
        "exec 3 (good)",
        _critic_response(True, ""),
    ]
    fp = FakeProvider(responses=responses)
    result = Orchestrator(fp).run_task("summarize the doc")

    # 2 executors in attempt 1 + 1 executor in attempt 2 = 3 total steps
    assert result.steps == 3
    assert "exec 3 (good)" in result.answer
    assert "exec 1 (bad)" not in result.answer  # final answer is from attempt 2 only

    # Trace must show both attempts and the rejection
    joined = "\n".join(result.trace)
    assert "attempt 1" in joined and "attempt 2" in joined
    assert "approved=False" in joined
    assert "approved=True" in joined


def test_orchestrator_gives_up_after_max_attempts_with_last_answer():
    # Both attempts get rejected — orchestrator should return the last answer
    # and not call the planner a third time.
    responses = [
        _planner_response(["step A"]),
        "exec attempt 1",
        _critic_response(False, "no good"),
        _planner_response(["step A revised"]),
        "exec attempt 2",
        _critic_response(False, "still bad"),
    ]
    fp = FakeProvider(responses=responses)
    result = Orchestrator(fp).run_task("something hard")

    assert result.steps == 2
    assert "exec attempt 2" in result.answer
    assert "MAX_ATTEMPTS reached" in "\n".join(result.trace)
    # All 6 canned responses must have been used — no orphans, no extras requested.
    assert len(fp.calls) == 6


def test_orchestrator_passes_critic_feedback_to_replan():
    responses = [
        _planner_response(["initial step"]),
        "exec initial",
        _critic_response(False, "MISSING_REQUIREMENT_XYZ"),
        _planner_response(["addressed step"]),
        "exec addressed",
        _critic_response(True, ""),
    ]
    fp = FakeProvider(responses=responses)
    Orchestrator(fp).run_task("do work")

    # The replan call (4th provider call, index 3) must include the rejection feedback
    replan_messages, _ = fp.calls[3]
    user_msg = next(m for m in replan_messages if m["role"] == "user")
    assert "MISSING_REQUIREMENT_XYZ" in user_msg["content"]


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def test_get_provider_returns_fake_with_kwargs():
    p = get_provider("fake", responses=["hello"])
    assert isinstance(p, FakeProvider)
    assert p.complete([{"role": "user", "content": "hi"}]) == "hello"


def test_get_provider_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("bogus")


# --------------------------------------------------------------------------- #
# Parallel execution                                                           #
# --------------------------------------------------------------------------- #
#
# These tests use a *content-dispatched* FakeProvider responder so behavior is
# deterministic even when N executor coroutines race in parallel threads. The
# FIFO-queue mode would be order-sensitive under asyncio.gather and would make
# the assertions flaky.


import threading
import time


def _content_responder(planner_steps, exec_map, critic_approved=True, critic_feedback=""):
    """Build a responder that picks an answer based on which agent is asking.

    - Planner system prompt → return JSON with `planner_steps`.
    - Executor system prompt → look up the user message (the subtask) in
      `exec_map` and return the matching string; if exec_map is callable,
      call it on the subtask.
    - Critic system prompt → return the JSON verdict.
    """
    critic_payload = json.dumps({"approved": critic_approved, "feedback": critic_feedback})
    plan_payload = json.dumps({"steps": list(planner_steps)})

    def respond(messages):
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        if "Planner" in system:
            return plan_payload
        if "Critic" in system:
            return critic_payload
        if "Executor" in system:
            return exec_map(user) if callable(exec_map) else exec_map.get(user, "result")
        return ""

    return respond


def test_executors_run_concurrently_and_preserve_subtask_order():
    """Each subtask is paired with its OWN result in the aggregated answer,
    independent of which thread completed first."""
    subtasks = ["alpha task", "beta task", "gamma task", "delta task"]
    exec_map = {s: f"RESULT_FOR_{s.split()[0].upper()}" for s in subtasks}

    fp = FakeProvider(
        responder=_content_responder(subtasks, exec_map, critic_approved=True)
    )
    result = Orchestrator(fp, max_concurrent_executors=4).run_task("anything")

    # Each subtask MUST be aggregated with its own result, in the order it
    # was generated by the Planner. Aggregation uses the two-line shape
    # `{i}. {subtask}\n   -> {result}` — checking the full block catches both
    # missing rows and rows where parallel pops scrambled subtask→result pairs.
    for i, s in enumerate(subtasks, start=1):
        expected_block = f"{i}. {s}\n   -> {exec_map[s]}"
        assert expected_block in result.answer, (
            f"subtask #{i} not paired with its own result. expected substring:\n{expected_block!r}\n"
            f"in answer:\n{result.answer!r}"
        )

    # Trace records the parallel-execution summary line.
    joined = "\n".join(result.trace)
    assert "concurrently" in joined
    assert "max_concurrent=4" in joined
    assert "failures=0" in joined
    assert result.steps == 4


def test_one_subtask_failure_does_not_lose_the_others():
    """A subtask that raises ⇒ marker in answer + trace entry. The rest still
    run and the orchestrator returns a valid OrchestrationResult."""
    subtasks = ["good_1", "BOOM", "good_2"]

    def exec_responder(user):
        if user == "BOOM":
            raise RuntimeError("kapow")
        return f"OK::{user}"

    fp = FakeProvider(
        responder=_content_responder(subtasks, exec_responder, critic_approved=True),
    )
    result = Orchestrator(fp, max_concurrent_executors=3).run_task("anything")

    # Each surviving subtask has its real result; the broken one has a marker.
    assert "OK::good_1" in result.answer
    assert "OK::good_2" in result.answer
    assert "[execution failed: RuntimeError: kapow]" in result.answer

    joined = "\n".join(result.trace)
    assert "EXEC[2] FAILED 'BOOM'" in joined
    assert "RuntimeError: kapow" in joined
    assert "failures=1" in joined
    # All three counted as executor steps, even the failing one.
    assert result.steps == 3


def test_semaphore_caps_simultaneous_executor_calls():
    """With MAX_CONCURRENT_EXECUTORS=3, at most 3 executor calls run at once
    even if the Planner produces 6 subtasks."""
    subtasks = [f"task_{i}" for i in range(6)]

    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    def exec_responder(user):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            if in_flight > max_in_flight:
                max_in_flight = in_flight
        try:
            # Give other coroutines a chance to start so the peak overlaps.
            time.sleep(0.04)
            return f"RESULT_{user}"
        finally:
            with lock:
                in_flight -= 1

    fp = FakeProvider(
        responder=_content_responder(subtasks, exec_responder, critic_approved=True),
    )
    orch = Orchestrator(fp, max_concurrent_executors=3)
    result = orch.run_task("anything")

    # We're not asserting on timing — only on the semaphore cap.
    assert max_in_flight <= 3, f"semaphore violated: {max_in_flight} concurrent calls"
    # And the parallelism actually happened (otherwise the test is meaningless).
    assert max_in_flight >= 2, (
        f"only {max_in_flight} simultaneous call(s) — parallelism didn't engage"
    )
    assert result.steps == 6


def test_env_max_concurrent_overrides_default(monkeypatch):
    """MAX_CONCURRENT_EXECUTORS env var configures the semaphore."""
    monkeypatch.setenv("MAX_CONCURRENT_EXECUTORS", "2")
    fp = FakeProvider(
        responder=_content_responder(["a", "b"], {"a": "A", "b": "B"}, critic_approved=True)
    )
    # No explicit max_concurrent_executors arg ⇒ Dispatcher reads the env.
    orch = Orchestrator(fp)
    assert orch.max_concurrent_executors == 2
    result = orch.run_task("anything")
    assert "max_concurrent=2" in "\n".join(result.trace)


def test_env_max_concurrent_invalid_value_uses_default(monkeypatch):
    """Misconfigured env var doesn't crash the orchestrator."""
    monkeypatch.setenv("MAX_CONCURRENT_EXECUTORS", "not-a-number")
    fp = FakeProvider(
        responder=_content_responder(["a"], {"a": "A"}, critic_approved=True)
    )
    orch = Orchestrator(fp)
    assert orch.max_concurrent_executors == 5  # documented default


def test_run_task_async_is_directly_awaitable():
    """If a caller is already in an async context, they can `await` the
    coroutine instead of paying the asyncio.run wrapper cost."""
    import asyncio

    fp = FakeProvider(
        responder=_content_responder(["x", "y"], {"x": "X", "y": "Y"}, critic_approved=True)
    )
    orch = Orchestrator(fp, max_concurrent_executors=2)

    async def caller():
        return await orch.run_task_async("anything")

    result = asyncio.run(caller())
    assert "X" in result.answer and "Y" in result.answer
    assert result.steps == 2
