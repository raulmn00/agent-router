"""Public API schemas (Pydantic v2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    input: str = Field(min_length=1, max_length=2000)


class RouteResponse(BaseModel):
    intent: str
    confidence: float
    answer: str
    path_taken: str
    trace: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# /compare                                                                     #
# --------------------------------------------------------------------------- #


class CompareRequest(BaseModel):
    input: str = Field(min_length=1, max_length=2000)


class RouterResult(BaseModel):
    router_name: str
    intent: str
    confidence: float
    latency_ms: float
    cost_per_1k_usd: float
    error: str | None = None


class CompareResponse(BaseModel):
    input: str
    results: list[RouterResult]
    agreement: bool
    fastest: str
    cheapest: str


# --------------------------------------------------------------------------- #
# /stats                                                                       #
# --------------------------------------------------------------------------- #


class ConfidenceBucket(BaseModel):
    min: float
    max: float
    mean: float
    p50: float
    p95: float
    count: int


class IntentStats(BaseModel):
    count: int
    share: float
    confidence: ConfidenceBucket | None = None  # None when no samples yet


class LatencyBucket(BaseModel):
    p50: float
    p95: float
    mean: float
    count: int


class FallbackStats(BaseModel):
    total: int
    rate: float
    by_attempted_intent: dict[str, int] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    since: str
    uptime_seconds: float
    total_requests: int
    intents: dict[str, IntentStats] = Field(default_factory=dict)
    fallbacks: FallbackStats
    latency_ms: dict[str, LatencyBucket] = Field(default_factory=dict)
    errors: dict[str, int] = Field(default_factory=dict)
    # `ephemeral` is the programmatic flag the React frontend keys off to
    # render a "data resets on restart" disclaimer; `note` is the prose
    # version for humans. Both intentional — `note` text may evolve, the
    # `ephemeral` bool is the stable contract.
    ephemeral: bool = True
    note: str
