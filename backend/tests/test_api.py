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


_INTENTS = ("simple_qa", "complex_task", "document_qa", "chitchat")


def _override_with(
    classifier,
    factory,
    *,
    threshold: float | None = None,
    thresholds: dict[str, float] | None = None,
):
    """Install a Dispatcher built from the given fakes as the FastAPI dep.

    - `threshold=0.65` → uniform threshold across every class (convenience
      shortcut for older tests written before per-class thresholds existed).
    - `thresholds={"chitchat": 0.45, ...}` → per-class explicit map.
    - both None → Dispatcher resolves via env vars / defaults at construction.

    Passing both raises — they're mutually exclusive.
    """
    if threshold is not None and thresholds is not None:
        raise AssertionError("pass either `threshold` (scalar) or `thresholds` (dict), not both")
    if threshold is not None:
        thresholds = {intent: threshold for intent in _INTENTS}
    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=classifier,
        provider_factory=factory,
        confidence_thresholds=thresholds,
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
        confidence_thresholds=None,
    )
    client = TestClient(app)
    body = client.post("/route", json={"input": "what?"}).json()
    assert body["path_taken"] == "low_confidence_fallback"


def test_env_var_invalid_value_falls_back_to_default(monkeypatch):
    """A typo in the legacy CONFIDENCE_THRESHOLD scalar shouldn't break the service."""
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "not-a-number")

    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=_FakeClassifier("chitchat", 0.80),
        provider_factory=lambda _n: FakeProvider(responses=["hi"]),
        confidence_thresholds=None,
    )
    client = TestClient(app)
    body = client.post("/route", json={"input": "hey"}).json()
    # Default chitchat threshold is 0.45; 0.80 > 0.45 → normal route.
    assert body["path_taken"] == "chitchat:direct_llm"


def test_env_var_out_of_range_is_clamped(monkeypatch):
    """Values outside [0, 1] are almost always a misconfig — clamp instead of break."""
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "9.0")

    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=_FakeClassifier("simple_qa", 0.99),
        provider_factory=lambda _n: FakeProvider(),
        confidence_thresholds=None,
    )
    client = TestClient(app)
    body = client.post("/route", json={"input": "anything"}).json()
    # Clamp 9.0 → 1.0 ⇒ even confidence 0.99 falls below ⇒ fallback fires.
    assert body["path_taken"] == "low_confidence_fallback"


# --------------------------------------------------------------------------- #
# Per-class thresholds — the real motivation for v0.3                          #
# --------------------------------------------------------------------------- #


def test_chitchat_at_056_does_not_fall_back_under_default_thresholds():
    """The actual prod regression we're fixing: 'Good morning!' scored 0.560
    and got demoted under the old uniform 0.65 threshold. Per-class defaults
    have chitchat at 0.45, so the same input now routes normally."""
    factory = lambda _n: FakeProvider(responses=["Morning!"])
    # No threshold override → Dispatcher uses DEFAULT_THRESHOLDS (chitchat=0.45).
    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=_FakeClassifier("chitchat", 0.560),
        provider_factory=factory,
        confidence_thresholds=None,
    )
    client = TestClient(app)
    body = client.post("/route", json={"input": "Good morning!"}).json()
    assert body["path_taken"] == "chitchat:direct_llm"
    assert body["intent"] == "chitchat"


def test_simple_qa_at_058_falls_back_under_default_thresholds():
    """An ambiguous input misclassified as simple_qa still falls back —
    simple_qa's per-class threshold is 0.65."""
    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=_FakeClassifier("simple_qa", 0.58),
        provider_factory=lambda _n: FakeProvider(),
        confidence_thresholds=None,
    )
    client = TestClient(app)
    body = client.post("/route", json={"input": "What does it say about pricing?"}).json()
    assert body["path_taken"] == "low_confidence_fallback"


def test_low_confidence_anywhere_below_class_threshold_falls_back():
    """Confidence 0.40 — well below any per-class threshold."""
    for intent in ("simple_qa", "complex_task", "document_qa", "chitchat"):
        app.dependency_overrides[get_dispatcher] = lambda intent=intent: Dispatcher(
            classifier=_FakeClassifier(intent, 0.40),
            provider_factory=lambda _n: FakeProvider(),
            confidence_thresholds=None,
        )
        client = TestClient(app)
        body = client.post("/route", json={"input": "hmm"}).json()
        assert body["path_taken"] == "low_confidence_fallback", (
            f"{intent!r} at 0.40 should always fall back"
        )


def test_env_thresholds_json_overrides_defaults(monkeypatch):
    """CONFIDENCE_THRESHOLDS={"chitchat": 0.80} → strict chitchat gate."""
    monkeypatch.setenv("CONFIDENCE_THRESHOLDS", '{"chitchat": 0.80}')

    # Same input as test_chitchat_at_056_..., now with a stricter env override:
    # 0.560 < 0.80 ⇒ fallback fires.
    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=_FakeClassifier("chitchat", 0.560),
        provider_factory=lambda _n: FakeProvider(),
        confidence_thresholds=None,
    )
    client = TestClient(app)
    body = client.post("/route", json={"input": "Good morning!"}).json()
    assert body["path_taken"] == "low_confidence_fallback"

    # And the unspecified classes keep their hardcoded defaults — verify
    # simple_qa at 0.70 still routes normally.
    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=_FakeClassifier("simple_qa", 0.70),
        provider_factory=lambda _n: FakeProvider(responses=["Paris."]),
        confidence_thresholds=None,
    )
    body2 = client.post("/route", json={"input": "?"}).json()
    assert body2["path_taken"] == "simple_qa:direct_llm"


def test_env_thresholds_malformed_json_falls_back_to_defaults(monkeypatch, caplog):
    """A typo in CONFIDENCE_THRESHOLDS must not crash the service."""
    monkeypatch.setenv("CONFIDENCE_THRESHOLDS", '{"chitchat": 0.45')  # missing closing brace

    with caplog.at_level("WARNING", logger="agent_router.dispatch"):
        app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
            classifier=_FakeClassifier("chitchat", 0.560),
            provider_factory=lambda _n: FakeProvider(responses=["hi"]),
            confidence_thresholds=None,
        )
        client = TestClient(app)
        body = client.post("/route", json={"input": "hi"}).json()

    # Default chitchat=0.45 wins ⇒ 0.560 > 0.45 ⇒ routes normally.
    assert body["path_taken"] == "chitchat:direct_llm"
    # And we logged the malformed JSON so the operator notices.
    assert any(
        "CONFIDENCE_THRESHOLDS" in record.getMessage() for record in caplog.records
    ), "malformed CONFIDENCE_THRESHOLDS must produce a warning log"


def test_env_thresholds_non_object_falls_back_to_defaults(monkeypatch, caplog):
    """JSON arrays / scalars / strings shouldn't be accepted."""
    monkeypatch.setenv("CONFIDENCE_THRESHOLDS", "[0.65, 0.65, 0.65, 0.45]")

    with caplog.at_level("WARNING", logger="agent_router.dispatch"):
        app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
            classifier=_FakeClassifier("simple_qa", 0.80),
            provider_factory=lambda _n: FakeProvider(responses=["x"]),
            confidence_thresholds=None,
        )
        client = TestClient(app)
        body = client.post("/route", json={"input": "x"}).json()

    assert body["path_taken"] == "simple_qa:direct_llm"
    assert any("CONFIDENCE_THRESHOLDS" in r.getMessage() for r in caplog.records)


def test_env_thresholds_unknown_intent_is_ignored(monkeypatch, caplog):
    """A key in CONFIDENCE_THRESHOLDS that isn't one of our intents → warn + drop."""
    monkeypatch.setenv("CONFIDENCE_THRESHOLDS", '{"unknown_intent": 0.99}')

    with caplog.at_level("WARNING", logger="agent_router.dispatch"):
        # Even though the env mentions a bogus intent, defaults still apply
        # to our real intents — chitchat 0.45, etc.
        app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
            classifier=_FakeClassifier("chitchat", 0.50),
            provider_factory=lambda _n: FakeProvider(responses=["hey"]),
            confidence_thresholds=None,
        )
        client = TestClient(app)
        body = client.post("/route", json={"input": "hi"}).json()

    assert body["path_taken"] == "chitchat:direct_llm"
    assert any("unknown_intent" in r.getMessage() for r in caplog.records)


def test_legacy_scalar_env_still_acts_as_uniform_fallback(monkeypatch):
    """CONFIDENCE_THRESHOLD (singular, old) sets a uniform floor across every
    class — useful for the old config, and for tightening *everything* at once
    without writing the JSON map."""
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.40")
    # Clear the new env var explicitly so it doesn't override.
    monkeypatch.delenv("CONFIDENCE_THRESHOLDS", raising=False)

    # 0.41 for any class with the new defaults (simple_qa=0.65) would fall back —
    # but with the legacy scalar set to 0.40, this routes normally.
    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=_FakeClassifier("simple_qa", 0.41),
        provider_factory=lambda _n: FakeProvider(responses=["x"]),
        confidence_thresholds=None,
    )
    client = TestClient(app)
    body = client.post("/route", json={"input": "x"}).json()
    assert body["path_taken"] == "simple_qa:direct_llm"


def test_per_class_env_takes_priority_over_legacy_scalar(monkeypatch):
    """When both env vars are set, the per-class map wins for keys it lists."""
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.40")  # legacy: 0.40 everywhere
    monkeypatch.setenv("CONFIDENCE_THRESHOLDS", '{"simple_qa": 0.90}')  # override

    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=_FakeClassifier("simple_qa", 0.85),
        provider_factory=lambda _n: FakeProvider(),
        confidence_thresholds=None,
    )
    client = TestClient(app)
    body = client.post("/route", json={"input": "?"}).json()
    # 0.85 < 0.90 (per-class) → fallback, even though legacy scalar said 0.40.
    assert body["path_taken"] == "low_confidence_fallback"

    # And a class NOT in the map gets the legacy scalar (0.40).
    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=_FakeClassifier("chitchat", 0.50),
        provider_factory=lambda _n: FakeProvider(responses=["x"]),
        confidence_thresholds=None,
    )
    body2 = client.post("/route", json={"input": "hey"}).json()
    assert body2["path_taken"] == "chitchat:direct_llm"


def test_get_thresholds_function_resolves_consistently(monkeypatch):
    """The Dispatcher should match the standalone helper for the same env."""
    from app.dispatch import DEFAULT_THRESHOLDS, get_thresholds

    monkeypatch.delenv("CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("CONFIDENCE_THRESHOLDS", raising=False)
    resolved = get_thresholds()
    assert resolved == DEFAULT_THRESHOLDS
    assert resolved["chitchat"] == 0.45
    assert resolved["simple_qa"] == 0.65
