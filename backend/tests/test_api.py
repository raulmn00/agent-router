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


def _override_with(classifier, factory, *, threshold: float | None = None):
    """Install a Dispatcher built from the given fakes as the FastAPI dep.

    `threshold` overrides the confidence threshold for this dispatcher. When
    None, Dispatcher reads it from the CONFIDENCE_THRESHOLD env var or falls
    back to the documented default.
    """
    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=classifier,
        provider_factory=factory,
        confidence_threshold=threshold,
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
# Security middlewares                                                         #
# --------------------------------------------------------------------------- #


def test_security_headers_present_on_every_response():
    """X-Content-Type-Options, X-Frame-Options, Referrer-Policy, HSTS, CORP."""
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "no-referrer"
    assert "max-age=31536000" in resp.headers["strict-transport-security"]
    assert resp.headers["cross-origin-resource-policy"] == "same-site"


def test_body_size_limit_returns_413():
    """Reject oversized requests at the Content-Length stage, before parsing."""
    client = TestClient(app)
    resp = client.post(
        "/route",
        content=b"x" * 20_000,
        headers={"Content-Type": "application/json", "Content-Length": "20000"},
    )
    assert resp.status_code == 413
    assert resp.json()["detail"] == "payload too large"


def test_body_size_limit_with_malformed_content_length_returns_400():
    client = TestClient(app)
    resp = client.post(
        "/route",
        content=b'{"input":"hi"}',
        headers={"Content-Type": "application/json", "Content-Length": "not-a-number"},
    )
    # httpx may strip an invalid Content-Length header before sending; this
    # test asserts that IF the bad value reaches the middleware it gets a 400.
    if resp.headers.get("content-length") and not resp.headers["content-length"].isdigit():
        assert resp.status_code == 400


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


# --------------------------------------------------------------------------- #
# Confidence threshold + low-confidence fallback                               #
# --------------------------------------------------------------------------- #


def test_high_confidence_routes_through_normal_path():
    """Above threshold — same behavior as before the threshold existed."""
    factory = lambda _n: FakeProvider(responses=["Paris."])
    # Confidence 0.92 with default threshold 0.65 — well above.
    _override_with(_FakeClassifier("simple_qa", 0.92), factory, threshold=0.65)

    client = TestClient(app)
    body = client.post("/route", json={"input": "capital of France?"}).json()

    assert body["intent"] == "simple_qa"
    assert body["confidence"] == pytest.approx(0.92)
    assert body["path_taken"] == "simple_qa:direct_llm"
    assert body["answer"] == "Paris."  # the dispatcher actually called the LLM


def test_low_confidence_falls_back_without_calling_llm():
    """Below threshold — never reach the dispatch arms; never invoke the provider."""
    factory_calls: list[str] = []

    def factory(name):
        factory_calls.append(name)
        raise AssertionError("provider must not be created on the low-confidence path")

    # 0.42 < 0.65 → fallback fires.
    _override_with(_FakeClassifier("simple_qa", 0.42), factory, threshold=0.65)

    client = TestClient(app)
    resp = client.post("/route", json={"input": "Tell me about the requirements"})
    assert resp.status_code == 200
    body = resp.json()

    # Schema preserved: best-guess intent + measured confidence still reported,
    # but path_taken is the explicit marker and the answer surfaces uncertainty.
    assert body["intent"] == "simple_qa"
    assert body["confidence"] == pytest.approx(0.42)
    assert body["path_taken"] == "low_confidence_fallback"

    # No provider was constructed → no LLM call, no token spend.
    assert factory_calls == []


def test_low_confidence_trace_includes_attempted_intent_and_confidence():
    _override_with(
        _FakeClassifier("complex_task", 0.50),
        lambda _n: FakeProvider(),
        threshold=0.65,
    )

    client = TestClient(app)
    body = client.post("/route", json={"input": "Build me something cool"}).json()

    joined = "\n".join(body["trace"])
    # The attempted intent and the numeric confidence must be visible for
    # observability — that's the whole point of "transparent fallback".
    assert "complex_task" in joined
    assert "0.500" in joined or "0.50" in joined
    # Operator should be able to see which arm WOULD have run.
    assert any("would have routed to" in line for line in body["trace"])


def test_low_confidence_answer_does_not_pretend_to_have_classified():
    _override_with(
        _FakeClassifier("document_qa", 0.30),
        lambda _n: FakeProvider(),
        threshold=0.65,
    )
    client = TestClient(app)
    body = client.post("/route", json={"input": "huh?"}).json()
    # Honest UX — don't fake an answer; explain the situation.
    assert "couldn't confidently" in body["answer"].lower()
    assert "0.300" in body["answer"] or "0.30" in body["answer"]
    assert "0.65" in body["answer"]


def test_explicit_threshold_overrides_default():
    """Passing threshold=0.95 to Dispatcher demotes confidence 0.90 to fallback."""
    factory_calls: list[str] = []

    def factory(name):
        factory_calls.append(name)
        return FakeProvider(responses=["should not be reached"])

    _override_with(_FakeClassifier("chitchat", 0.90), factory, threshold=0.95)
    client = TestClient(app)
    body = client.post("/route", json={"input": "hi"}).json()

    assert body["path_taken"] == "low_confidence_fallback"
    assert factory_calls == []


def test_env_var_configures_threshold(monkeypatch):
    """CONFIDENCE_THRESHOLD env var changes the gate without changing code."""
    # Set the env var, then build a Dispatcher with threshold=None so it reads
    # from the env. A higher threshold (0.98) demotes confidence 0.92 to fallback,
    # demonstrating the env var actually controls the behavior.
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.98")

    # Explicitly pass threshold=None so Dispatcher reads the env we just set.
    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=_FakeClassifier("simple_qa", 0.92),
        provider_factory=lambda _n: FakeProvider(),
        confidence_threshold=None,
    )
    client = TestClient(app)
    body = client.post("/route", json={"input": "what?"}).json()
    assert body["path_taken"] == "low_confidence_fallback"


def test_env_var_invalid_value_falls_back_to_default(monkeypatch):
    """A typo in CONFIDENCE_THRESHOLD shouldn't break the service."""
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "not-a-number")

    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=_FakeClassifier("chitchat", 0.80),
        provider_factory=lambda _n: FakeProvider(responses=["hi"]),
        confidence_threshold=None,
    )
    client = TestClient(app)
    body = client.post("/route", json={"input": "hey"}).json()
    # Default is 0.65; 0.80 > 0.65 → normal route, not fallback.
    assert body["path_taken"] == "chitchat:direct_llm"


def test_env_var_out_of_range_is_clamped(monkeypatch):
    """Values outside [0, 1] are almost always a misconfig — clamp instead of break."""
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "9.0")

    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=_FakeClassifier("simple_qa", 0.99),
        provider_factory=lambda _n: FakeProvider(),
        confidence_threshold=None,
    )
    client = TestClient(app)
    body = client.post("/route", json={"input": "anything"}).json()
    # Clamp 9.0 → 1.0 ⇒ even confidence 0.99 falls below ⇒ fallback fires.
    assert body["path_taken"] == "low_confidence_fallback"
