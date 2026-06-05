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
