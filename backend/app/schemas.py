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
