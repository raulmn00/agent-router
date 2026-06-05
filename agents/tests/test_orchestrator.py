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
