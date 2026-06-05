"""Tests for POST /compare — all three routers mocked, no network, no model load."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from app.api import app, get_compare_service, limiter
from app.compare import CompareService, build_adapters


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
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


class _FakeWithConf:
    """Stand-in for LLMRouter / EmbedRouter — exposes classify_with_confidence."""

    def __init__(self, intent: str, confidence: float):
        self.intent = intent
        self.confidence = confidence

    def classify_with_confidence(self, text: str) -> tuple[str, float]:
        return self.intent, self.confidence


def _factories(distil, llm, embed):
    """Build factory callables that always return the same singleton, or raise."""

    def _wrap(obj):
        if isinstance(obj, BaseException):
            def _raise():
                raise obj

            return _raise
        return lambda: obj

    return _wrap(distil), _wrap(llm), _wrap(embed)


def _install_service(distil, llm, embed):
    """Override the FastAPI dep with a CompareService built from given fakes."""
    f_distil, f_llm, f_embed = _factories(distil, llm, embed)
    adapters = build_adapters(
        classifier_factory=f_distil,
        llm_router_factory=f_llm,
        embed_router_factory=f_embed,
    )
    app.dependency_overrides[get_compare_service] = lambda: CompareService(adapters)


@pytest.fixture(autouse=True)
def _isolate_each_test():
    # Each test starts with no rate-limit state and no dep overrides.
    limiter.reset()
    app.dependency_overrides.clear()
    yield
    limiter.reset()
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Happy path — all 3 succeed, all agree                                        #
# --------------------------------------------------------------------------- #


def test_compare_all_agree_returns_full_response():
    _install_service(
        distil=_FakeClassifier("simple_qa", 0.98),
        llm=_FakeWithConf("simple_qa", 0.91),
        embed=_FakeWithConf("simple_qa", 0.87),
    )

    client = TestClient(app)
    resp = client.post("/compare", json={"input": "What is the capital of France?"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["input"] == "What is the capital of France?"
    assert body["agreement"] is True
    assert len(body["results"]) == 3
    names = [r["router_name"] for r in body["results"]]
    assert names == [
        "DistilBERT (fine-tuned)",
        "LLM zero-shot (gpt-4o-mini)",
        "Embeddings + LogReg",
    ]
    for r in body["results"]:
        assert r["intent"] == "simple_qa"
        assert r["error"] is None
        assert r["latency_ms"] >= 0
        assert 0.0 <= r["confidence"] <= 1.0

    # Cost ordering is pinned by eval.metrics.cost_per_1k — DistilBERT $0,
    # embed cheaper than llm. So cheapest is DistilBERT.
    assert body["cheapest"] == "DistilBERT (fine-tuned)"
    # fastest is whoever measured fastest in this run — all are fake/local,
    # but the field must be one of the three names.
    assert body["fastest"] in names


def test_compare_cost_field_matches_pinned_assumptions():
    """The dollar values must come from eval.metrics.cost_per_1k, not invented."""
    from eval.metrics import cost_per_1k

    _install_service(
        distil=_FakeClassifier("chitchat", 0.99),
        llm=_FakeWithConf("chitchat", 0.95),
        embed=_FakeWithConf("chitchat", 0.92),
    )
    client = TestClient(app)
    body = client.post("/compare", json={"input": "hey there"}).json()

    by_name = {r["router_name"]: r for r in body["results"]}
    assert by_name["DistilBERT (fine-tuned)"]["cost_per_1k_usd"] == cost_per_1k("distilbert")
    assert by_name["LLM zero-shot (gpt-4o-mini)"]["cost_per_1k_usd"] == cost_per_1k("llm")
    assert by_name["Embeddings + LogReg"]["cost_per_1k_usd"] == cost_per_1k("embed")


# --------------------------------------------------------------------------- #
# Divergence — agreement is False                                              #
# --------------------------------------------------------------------------- #


def test_compare_divergent_intents_reports_no_agreement():
    _install_service(
        distil=_FakeClassifier("simple_qa", 0.6),
        llm=_FakeWithConf("complex_task", 0.7),
        embed=_FakeWithConf("simple_qa", 0.55),
    )

    client = TestClient(app)
    body = client.post("/compare", json={"input": "Build me a Slack bot."}).json()

    assert body["agreement"] is False
    intents = {r["intent"] for r in body["results"]}
    assert intents == {"simple_qa", "complex_task"}


# --------------------------------------------------------------------------- #
# One router fails — others still return; whole response is 200                #
# --------------------------------------------------------------------------- #


def test_compare_one_router_failure_does_not_break_response():
    _install_service(
        distil=_FakeClassifier("document_qa", 0.95),
        llm=_FakeWithConf("document_qa", 0.88),
        embed=FileNotFoundError("EmbedRouter artifact missing"),
    )

    client = TestClient(app)
    resp = client.post("/compare", json={"input": "What does the PDF say?"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 3

    by_name = {r["router_name"]: r for r in body["results"]}
    assert by_name["DistilBERT (fine-tuned)"]["error"] is None
    assert by_name["DistilBERT (fine-tuned)"]["intent"] == "document_qa"
    assert by_name["LLM zero-shot (gpt-4o-mini)"]["error"] is None
    embed_result = by_name["Embeddings + LogReg"]
    assert embed_result["error"] is not None
    assert "EmbedRouter artifact missing" in embed_result["error"]
    assert embed_result["intent"] == ""  # empty when errored
    # Cost on the approach is still reported (it's a property of the approach,
    # not of this specific call).
    assert embed_result["cost_per_1k_usd"] > 0

    # Agreement / fastest / cheapest restricted to successful ones.
    assert body["agreement"] is True  # both successful ones agreed on document_qa
    assert body["cheapest"] == "DistilBERT (fine-tuned)"
    assert body["fastest"] in (
        "DistilBERT (fine-tuned)",
        "LLM zero-shot (gpt-4o-mini)",
    )


def test_compare_all_fail_returns_200_with_empty_agreement():
    err = RuntimeError("nope")
    _install_service(distil=err, llm=err, embed=err)

    client = TestClient(app)
    resp = client.post("/compare", json={"input": "anything"})
    assert resp.status_code == 200
    body = resp.json()
    assert all(r["error"] for r in body["results"])
    assert body["agreement"] is False
    assert body["fastest"] == ""
    assert body["cheapest"] == ""


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #


def test_compare_rejects_empty_input_with_422():
    client = TestClient(app)
    resp = client.post("/compare", json={"input": ""})
    assert resp.status_code == 422


def test_compare_rejects_oversized_input_with_422():
    client = TestClient(app)
    resp = client.post("/compare", json={"input": "x" * 2001})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Rate limiting                                                                #
# --------------------------------------------------------------------------- #


def test_compare_rate_limit_returns_429_after_10_per_minute():
    _install_service(
        distil=_FakeClassifier("chitchat", 0.99),
        llm=_FakeWithConf("chitchat", 0.99),
        embed=_FakeWithConf("chitchat", 0.99),
    )

    client = TestClient(app)
    # First 10 calls — all 200.
    for i in range(10):
        resp = client.post("/compare", json={"input": "hey"})
        assert resp.status_code == 200, f"call #{i + 1} should be 200, got {resp.status_code}"

    # 11th call — must be 429 with the friendly message.
    resp = client.post("/compare", json={"input": "hey"})
    assert resp.status_code == 429
    body = resp.json()
    assert "too many" in body["detail"].lower() or "slow down" in body["detail"].lower()


def test_route_rate_limit_is_30_per_minute():
    # We only check that the limit exists and triggers — not that it's exactly 30,
    # since the existing /route tests already share this app.
    from app.api import get_dispatcher
    from app.dispatch import Dispatcher
    from agents import FakeProvider

    class _Stub:
        def classify(self, text):
            return _Decision("chitchat", 0.99)

    factory = lambda _n: FakeProvider(responses=["hi"] * 100)
    app.dependency_overrides[get_dispatcher] = lambda: Dispatcher(
        classifier=_Stub(), provider_factory=factory
    )
    client = TestClient(app)

    for i in range(30):
        resp = client.post("/route", json={"input": "hey"})
        assert resp.status_code == 200, f"/route call #{i + 1} unexpectedly {resp.status_code}"

    # 31st must be 429.
    resp = client.post("/route", json={"input": "hey"})
    assert resp.status_code == 429
