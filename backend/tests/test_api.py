"""API tests — TestClient + dependency overrides.

Covers:
  - GET /  → 200
  - POST /route for all 4 intents (simple_qa, complex_task, document_qa, chitchat)
  - 422 on Pydantic validation failure
  - 503 when the provider factory raises ProviderUnavailableError
  - 500 on unexpected internal error (with opaque body)

No real LLM calls, no model loading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from agents import FakeProvider, ProviderUnavailableError

from app.api import app, get_dispatcher
from app.dispatch import Dispatcher


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Decision:
    intent: str
    confidence: float


class _FakeClassifier:
    def __init__(self, intent: str, confidence: float = 0.95):
        self._decision = _Decision(intent=intent, confidence=confidence)

    def classify(self, text: str) -> _Decision:
        return self._decision


def _override_with(classifier, factory):
    """Install a Dispatcher built from the given fakes as the FastAPI dep."""
    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=classifier, provider_factory=factory
    )


@pytest.fixture(autouse=True)
def _clear_overrides_after_test():
    yield
    app.dependency_overrides.clear()


def _planner_resp(steps):
    return json.dumps({"steps": steps})


def _critic_resp(approved, feedback=""):
    return json.dumps({"approved": approved, "feedback": feedback})


# --------------------------------------------------------------------------- #
# GET /                                                                        #
# --------------------------------------------------------------------------- #


def test_health_returns_ok():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "agent-router"


# --------------------------------------------------------------------------- #
# Dispatch path: simple_qa                                                     #
# --------------------------------------------------------------------------- #


def test_route_simple_qa_uses_direct_llm():
    factory = lambda _name: FakeProvider(responses=["Paris is the capital of France."])
    _override_with(_FakeClassifier("simple_qa", 0.97), factory)

    client = TestClient(app)
    resp = client.post("/route", json={"input": "What is the capital of France?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "simple_qa"
    assert body["confidence"] == pytest.approx(0.97)
    assert "Paris" in body["answer"]
    assert body["path_taken"] == "simple_qa:direct_llm"
    assert any("direct LLM call" in t for t in body["trace"])


# --------------------------------------------------------------------------- #
# Dispatch path: complex_task                                                  #
# --------------------------------------------------------------------------- #


def test_route_complex_task_uses_orchestrator():
    # Sequence: planner → executor x2 → critic(approve)
    fake = FakeProvider(
        responses=[
            _planner_resp(["analyze the data", "draft the architecture"]),
            "data analysis result",
            "architecture draft",
            _critic_resp(True, "looks good"),
        ]
    )
    _override_with(_FakeClassifier("complex_task", 0.91), lambda _n: fake)

    client = TestClient(app)
    resp = client.post(
        "/route", json={"input": "Design an end-to-end data platform."}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "complex_task"
    assert body["path_taken"] == "complex_task:orchestrator"
    assert "data analysis result" in body["answer"]
    assert "architecture draft" in body["answer"]
    # Orchestrator trace must be passed through verbatim
    joined = "\n".join(body["trace"])
    assert "PLAN:" in joined
    assert "approved=True" in joined


# --------------------------------------------------------------------------- #
# Dispatch path: document_qa                                                   #
# --------------------------------------------------------------------------- #


def test_route_document_qa_returns_rag_stub_without_calling_llm():
    # Factory should NEVER be called for document_qa — assert on it.
    calls = []

    def factory(name):
        calls.append(name)
        raise AssertionError("provider must not be created for document_qa")

    _override_with(_FakeClassifier("document_qa", 0.88), factory)

    client = TestClient(app)
    resp = client.post(
        "/route", json={"input": "In the attached PDF, what is the conclusion?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "document_qa"
    assert body["path_taken"] == "document_qa:rag_stub"
    assert "RAG" in body["answer"]
    assert calls == []  # no provider instantiation


# --------------------------------------------------------------------------- #
# Dispatch path: chitchat                                                      #
# --------------------------------------------------------------------------- #


def test_route_chitchat_uses_direct_llm_with_short_budget():
    fake = FakeProvider(responses=["Hi there! How can I help today?"])
    _override_with(_FakeClassifier("chitchat", 0.99), lambda _n: fake)

    client = TestClient(app)
    resp = client.post("/route", json={"input": "Hello!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "chitchat"
    assert body["path_taken"] == "chitchat:direct_llm"
    assert "Hi there" in body["answer"]
    # chitchat must request a small max_tokens — assert on the actual call recorded
    _messages, max_tokens = fake.calls[0]
    assert max_tokens <= 100


# --------------------------------------------------------------------------- #
# Error: 422 Pydantic validation                                               #
# --------------------------------------------------------------------------- #


def test_empty_input_returns_422():
    client = TestClient(app)
    resp = client.post("/route", json={"input": ""})
    assert resp.status_code == 422


def test_input_over_2000_chars_returns_422():
    client = TestClient(app)
    resp = client.post("/route", json={"input": "x" * 2001})
    assert resp.status_code == 422


def test_missing_input_field_returns_422():
    client = TestClient(app)
    resp = client.post("/route", json={})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Error: 503 ProviderUnavailable                                               #
# --------------------------------------------------------------------------- #


def test_provider_unavailable_returns_503_with_generic_message():
    def factory(_name):
        raise ProviderUnavailableError("openai credentials missing")

    _override_with(_FakeClassifier("simple_qa", 0.9), factory)

    client = TestClient(app)
    resp = client.post("/route", json={"input": "anything"})
    assert resp.status_code == 503
    body = resp.json()
    # Must NOT leak the env var name or the raw exception text.
    assert "OPENAI_API_KEY" not in resp.text
    assert "openai credentials missing" not in resp.text
    assert "detail" in body and isinstance(body["detail"], str)


# --------------------------------------------------------------------------- #
# Error: 500 unexpected                                                        #
# --------------------------------------------------------------------------- #


def test_unexpected_error_returns_500_with_opaque_body():
    class _BoomClassifier:
        def classify(self, text):
            raise RuntimeError("something exploded with secret data: SECRET_TOKEN_xyz")

    _override_with(_BoomClassifier(), lambda _n: FakeProvider())

    client = TestClient(
        app, raise_server_exceptions=False
    )  # let the handler return its 500 instead of re-raising
    resp = client.post("/route", json={"input": "trigger"})
    assert resp.status_code == 500
    body = resp.json()
    assert body == {"detail": "internal server error"}
    # Sensitive details must not leak to the client.
    assert "SECRET_TOKEN_xyz" not in resp.text
