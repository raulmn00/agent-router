"""In-memory metrics collector for the /route dispatch path.

Aggregates the signals that matter for THIS system:

  - Volume + distribution by intent (simple_qa / complex_task / document_qa /
    chitchat).
  - Fallback rate: total AND broken down by the intent the dispatcher TRIED
    to route to. If a class's fallback rate trends up, its per-class
    threshold is probably wrong — that's the calibration signal.
  - Confidence distribution per intent (min / mean / max / p50 / p95).
  - Latency per `path_taken` (p50 / p95 / mean). Captures the structural
    difference between simple_qa (~ms) and complex_task (~seconds).
  - HTTP error counters (429 / 500 / 503).

Storage is **bounded**: a deque per intent for confidence samples and per
path for latency samples (default `sample_window = 1000` recent observations).
Counters grow only with the cardinality of the keyspace (4 intents + 5 paths
+ a handful of HTTP statuses). Memory is O(intents + paths) × sample_window.

This is **ephemeral** — when a Cloud Run instance scales to zero, restarts,
or gets replaced by a new revision, the in-memory state is gone. We surface
that honestly: the snapshot includes a `since` timestamp and a `note`
field flagging the in-memory caveat. Persistence would mean shipping the
snapshot to an external store; see the TO_PROMETHEUS hook below for the
escape hatch.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque, Iterable

EPHEMERAL_NOTE = (
    "Metrics are kept in process memory. They reset on every Cloud Run cold "
    "start, revision change, or service restart. For long-term retention, see "
    "the to_prometheus() hook in metrics_collector.py."
)

DEFAULT_SAMPLE_WINDOW = 1000


def _percentile(samples: Iterable[float], p: float) -> float:
    """Linear-interpolation percentile. Empty input → 0.0."""
    s = sorted(samples)
    if not s:
        return 0.0
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def _mean(samples: Iterable[float]) -> float:
    s = list(samples)
    return sum(s) / len(s) if s else 0.0


class MetricsCollector:
    """Thread-safe accumulator. All mutating ops are guarded by a single Lock.

    Snapshot reads acquire the same lock briefly to copy the underlying
    structures, then release it — percentile math runs without the lock so
    record_request() under load isn't blocked by a slow snapshot reader.
    """

    def __init__(self, sample_window: int = DEFAULT_SAMPLE_WINDOW):
        self._lock = threading.Lock()
        self._sample_window = sample_window
        self._since = datetime.now(timezone.utc)
        self._total_requests = 0
        self._intent_counts: dict[str, int] = defaultdict(int)
        self._fallback_total = 0
        # `by_attempted_intent`: the intent the classifier picked BEFORE the
        # threshold gate kicked in. If chitchat shows up here disproportionately
        # the chitchat threshold is too strict.
        self._fallback_by_attempted: dict[str, int] = defaultdict(int)
        self._confidence_samples: dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self._sample_window)
        )
        self._latency_samples: dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self._sample_window)
        )
        self._error_counts: dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------ #
    # Recording                                                           #
    # ------------------------------------------------------------------ #

    def record_request(
        self,
        intent: str,
        confidence: float,
        path_taken: str,
        latency_ms: float,
        was_fallback: bool,
    ) -> None:
        """Record one /route observation. Cheap — single lock + a few writes."""
        with self._lock:
            self._total_requests += 1
            self._intent_counts[intent] += 1
            self._confidence_samples[intent].append(float(confidence))
            self._latency_samples[path_taken].append(float(latency_ms))
            if was_fallback:
                self._fallback_total += 1
                self._fallback_by_attempted[intent] += 1

    def record_error(self, status_code: int) -> None:
        """Record an HTTP error (429 / 500 / 503 etc.)."""
        with self._lock:
            self._error_counts[int(status_code)] += 1

    def reset(self) -> None:
        """Drop all state. Used by tests; not meant for production traffic."""
        with self._lock:
            self._since = datetime.now(timezone.utc)
            self._total_requests = 0
            self._intent_counts.clear()
            self._fallback_total = 0
            self._fallback_by_attempted.clear()
            self._confidence_samples.clear()
            self._latency_samples.clear()
            self._error_counts.clear()

    # ------------------------------------------------------------------ #
    # Snapshot                                                            #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict:
        """Build an aggregate snapshot.

        Copies the underlying lists/counters inside the lock, then releases
        the lock before running the percentile math — that way record_request
        callers don't queue behind a slow reader.
        """
        with self._lock:
            since = self._since
            total = self._total_requests
            intent_counts = dict(self._intent_counts)
            fallback_total = self._fallback_total
            fallback_by_attempted = dict(self._fallback_by_attempted)
            confidence_copies = {k: list(v) for k, v in self._confidence_samples.items()}
            latency_copies = {k: list(v) for k, v in self._latency_samples.items()}
            error_counts = dict(self._error_counts)

        now = datetime.now(timezone.utc)
        uptime = (now - since).total_seconds()

        intents: dict[str, dict] = {}
        for intent, count in intent_counts.items():
            share = count / total if total else 0.0
            samples = confidence_copies.get(intent, [])
            intents[intent] = {
                "count": count,
                "share": round(share, 4),
                "confidence": _bucket_or_none(samples),
            }

        latency: dict[str, dict] = {}
        for path, samples in latency_copies.items():
            latency[path] = {
                "p50": round(_percentile(samples, 0.50), 2),
                "p95": round(_percentile(samples, 0.95), 2),
                "mean": round(_mean(samples), 2),
                "count": len(samples),
            }

        return {
            "since": since.isoformat(timespec="seconds"),
            "uptime_seconds": round(uptime, 1),
            "total_requests": total,
            "intents": intents,
            "fallbacks": {
                "total": fallback_total,
                "rate": round(fallback_total / total, 4) if total else 0.0,
                "by_attempted_intent": fallback_by_attempted,
            },
            "latency_ms": latency,
            "errors": {str(k): v for k, v in sorted(error_counts.items())},
            # The bool is the stable, machine-readable signal for the
            # frontend's "metrics reset on cold start" banner; `note` is
            # the human-readable explanation that may be reworded over time.
            "ephemeral": True,
            "note": EPHEMERAL_NOTE,
        }

    # ------------------------------------------------------------------ #
    # Future export hook                                                  #
    # ------------------------------------------------------------------ #
    #
    # TODO(observability): expose a `/metrics` endpoint in the Prometheus
    # text exposition format. The collector already has the right shape —
    # add a `to_prometheus(self) -> str` method that emits, per snapshot:
    #
    #   # HELP agent_router_requests_total Total /route requests observed
    #   # TYPE agent_router_requests_total counter
    #   agent_router_requests_total{intent="simple_qa"} 50
    #   ...
    #   # HELP agent_router_route_latency_ms Per-path latency (ms)
    #   # TYPE agent_router_route_latency_ms histogram
    #   ...
    #
    # The labels (intent, path_taken, status_code) are already the natural
    # dimensions. Scrape interval and retention move to the Prometheus side.


def _bucket_or_none(samples: list[float]) -> dict | None:
    """Confidence bucket: None if no samples yet, dict of stats otherwise."""
    if not samples:
        return None
    return {
        "min": round(min(samples), 4),
        "max": round(max(samples), 4),
        "mean": round(_mean(samples), 4),
        "p50": round(_percentile(samples, 0.50), 4),
        "p95": round(_percentile(samples, 0.95), 4),
        "count": len(samples),
    }


# --------------------------------------------------------------------- #
# Module-level singleton                                                 #
# --------------------------------------------------------------------- #
#
# The Dispatcher and the FastAPI /stats endpoint share state via this
# singleton. Tests override either by calling `_default.reset()` (via the
# autouse fixture in conftest.py) or by constructing a fresh
# MetricsCollector and injecting it through the Dispatcher's constructor.

_default = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """FastAPI dependency + the dispatcher's default. Returns the module
    singleton — overridable via `app.dependency_overrides` in tests."""
    return _default
