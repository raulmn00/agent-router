"""Tests for the in-memory MetricsCollector.

No network, no FastAPI, no tokens — just the collector with synthetic data.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.metrics_collector import (
    DEFAULT_SAMPLE_WINDOW,
    EPHEMERAL_NOTE,
    MetricsCollector,
    _percentile,
)


# --------------------------------------------------------------------------- #
# Counters                                                                     #
# --------------------------------------------------------------------------- #


def test_total_and_per_intent_counts():
    c = MetricsCollector()
    for _ in range(3):
        c.record_request("simple_qa", 0.9, "simple_qa:direct_llm", 50, False)
    for _ in range(2):
        c.record_request("complex_task", 0.92, "complex_task:orchestrator", 9000, False)
    snap = c.snapshot()
    assert snap["total_requests"] == 5
    assert snap["intents"]["simple_qa"]["count"] == 3
    assert snap["intents"]["complex_task"]["count"] == 2
    assert snap["intents"]["simple_qa"]["share"] == pytest.approx(0.6)
    assert snap["intents"]["complex_task"]["share"] == pytest.approx(0.4)


def test_intents_with_zero_requests_are_absent_from_snapshot():
    c = MetricsCollector()
    c.record_request("simple_qa", 0.9, "simple_qa:direct_llm", 50, False)
    snap = c.snapshot()
    # Don't fabricate empty rows for intents we haven't seen yet.
    assert set(snap["intents"].keys()) == {"simple_qa"}


# --------------------------------------------------------------------------- #
# Fallback rates — the calibration signal for per-class thresholds            #
# --------------------------------------------------------------------------- #


def test_fallback_total_and_rate():
    c = MetricsCollector()
    c.record_request("simple_qa", 0.92, "simple_qa:direct_llm", 50, False)
    c.record_request("chitchat", 0.40, "low_confidence_fallback", 5, True)
    c.record_request("simple_qa", 0.55, "low_confidence_fallback", 5, True)
    snap = c.snapshot()
    assert snap["fallbacks"]["total"] == 2
    assert snap["fallbacks"]["rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_fallback_attributed_to_attempted_intent():
    """The fallback count is grouped by the intent the dispatcher TRIED to
    route to (the classifier's best guess), not by `path_taken` — which is
    always the same `low_confidence_fallback` string."""
    c = MetricsCollector()
    c.record_request("chitchat", 0.40, "low_confidence_fallback", 5, True)
    c.record_request("chitchat", 0.42, "low_confidence_fallback", 5, True)
    c.record_request("simple_qa", 0.55, "low_confidence_fallback", 5, True)
    snap = c.snapshot()
    assert snap["fallbacks"]["by_attempted_intent"] == {"chitchat": 2, "simple_qa": 1}


def test_non_fallback_requests_do_not_count_toward_by_attempted():
    c = MetricsCollector()
    c.record_request("chitchat", 0.92, "chitchat:direct_llm", 200, False)
    snap = c.snapshot()
    assert snap["fallbacks"]["total"] == 0
    assert snap["fallbacks"]["by_attempted_intent"] == {}


# --------------------------------------------------------------------------- #
# Percentile math                                                              #
# --------------------------------------------------------------------------- #


def test_percentile_known_values():
    samples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert _percentile(samples, 0.50) == pytest.approx(55.0)
    assert _percentile(samples, 0.95) == pytest.approx(95.5)
    assert _percentile([], 0.5) == 0.0
    assert _percentile([42.0], 0.5) == 42.0
    assert _percentile([42.0], 0.95) == 42.0


def test_latency_percentiles_per_path():
    c = MetricsCollector()
    # Path A: tight cluster around 50ms
    for v in [40, 45, 50, 55, 60]:
        c.record_request("simple_qa", 0.9, "simple_qa:direct_llm", v, False)
    # Path B: wide cluster
    for v in [100, 500, 1000, 5000, 10000]:
        c.record_request("complex_task", 0.9, "complex_task:orchestrator", v, False)

    snap = c.snapshot()
    a = snap["latency_ms"]["simple_qa:direct_llm"]
    b = snap["latency_ms"]["complex_task:orchestrator"]
    assert a["p50"] == pytest.approx(50.0)
    assert a["count"] == 5
    assert b["p50"] == pytest.approx(1000.0)
    assert b["p95"] > a["p95"]


def test_confidence_bucket_per_intent():
    c = MetricsCollector()
    for v in [0.80, 0.85, 0.90, 0.95, 1.00]:
        c.record_request("simple_qa", v, "simple_qa:direct_llm", 50, False)
    snap = c.snapshot()
    bucket = snap["intents"]["simple_qa"]["confidence"]
    assert bucket["min"] == pytest.approx(0.80)
    assert bucket["max"] == pytest.approx(1.00)
    assert bucket["mean"] == pytest.approx(0.90)
    assert bucket["p50"] == pytest.approx(0.90)
    assert bucket["count"] == 5


def test_confidence_is_none_when_no_samples():
    c = MetricsCollector()
    # No requests recorded for any intent — but a manual record_request with
    # confidence None would never happen. Empty collector ⇒ empty intents map.
    snap = c.snapshot()
    assert snap["intents"] == {}


# --------------------------------------------------------------------------- #
# Bounded memory                                                               #
# --------------------------------------------------------------------------- #


def test_sample_window_caps_per_path_storage():
    """The deque should hold at most `sample_window` recent observations per
    (intent, path) bucket — otherwise memory grows unbounded."""
    c = MetricsCollector(sample_window=10)
    for i in range(100):
        c.record_request("simple_qa", 0.5 + i * 0.001,
                         "simple_qa:direct_llm", float(i), False)
    snap = c.snapshot()
    # 100 requests recorded BUT only the last 10 samples kept per bucket.
    assert snap["total_requests"] == 100
    assert snap["intents"]["simple_qa"]["count"] == 100  # counter, not bounded
    assert snap["intents"]["simple_qa"]["confidence"]["count"] == 10  # window-bounded
    assert snap["latency_ms"]["simple_qa:direct_llm"]["count"] == 10


def test_default_sample_window_is_documented():
    assert DEFAULT_SAMPLE_WINDOW == 1000


# --------------------------------------------------------------------------- #
# Error counters                                                               #
# --------------------------------------------------------------------------- #


def test_error_counts_by_status():
    c = MetricsCollector()
    c.record_error(429)
    c.record_error(429)
    c.record_error(503)
    c.record_error(500)
    snap = c.snapshot()
    assert snap["errors"] == {"429": 2, "500": 1, "503": 1}


def test_errors_dont_inflate_total_requests():
    c = MetricsCollector()
    c.record_error(500)
    c.record_error(503)
    snap = c.snapshot()
    assert snap["total_requests"] == 0
    assert sum(snap["errors"].values()) == 2


# --------------------------------------------------------------------------- #
# Thread safety                                                                #
# --------------------------------------------------------------------------- #


def test_record_request_is_thread_safe():
    """20 threads × 100 record_request each. Counter must end at exactly 2000."""
    c = MetricsCollector()
    n_threads = 20
    per_thread = 100

    def worker(worker_id: int):
        for _ in range(per_thread):
            c.record_request(
                "simple_qa", 0.9, "simple_qa:direct_llm", 50.0, False
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = c.snapshot()
    assert snap["total_requests"] == n_threads * per_thread
    assert snap["intents"]["simple_qa"]["count"] == n_threads * per_thread


def test_snapshot_during_concurrent_writes_is_consistent():
    """Snapshot reads should not crash when writes are happening concurrently.

    The lock-based design copies underlying lists inside the lock, so a
    snapshot taken mid-flight is consistent (it just won't include the
    writes that arrive after the copy).
    """
    c = MetricsCollector()
    stop = threading.Event()

    def writer():
        while not stop.is_set():
            c.record_request("simple_qa", 0.9, "simple_qa:direct_llm", 50.0, False)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        for _ in range(50):
            snap = c.snapshot()
            # consistency: total >= count of any single intent
            for intent_stat in snap["intents"].values():
                assert intent_stat["count"] <= snap["total_requests"]
            time.sleep(0.001)
    finally:
        stop.set()
        t.join(timeout=1)


# --------------------------------------------------------------------------- #
# Snapshot shape                                                               #
# --------------------------------------------------------------------------- #


def test_snapshot_advertises_in_memory_ephemerality_explicitly():
    """The `note` field is the only honest place to flag the limitation."""
    c = MetricsCollector()
    snap = c.snapshot()
    assert "memory" in snap["note"].lower()
    assert "reset" in snap["note"].lower()
    assert snap["note"] == EPHEMERAL_NOTE


def test_uptime_advances():
    c = MetricsCollector()
    snap1 = c.snapshot()
    time.sleep(0.01)
    snap2 = c.snapshot()
    assert snap2["uptime_seconds"] >= snap1["uptime_seconds"]
    # `since` is stable across snapshots — it's process-start, not now.
    assert snap2["since"] == snap1["since"]


def test_reset_clears_everything():
    c = MetricsCollector()
    c.record_request("simple_qa", 0.9, "simple_qa:direct_llm", 50, False)
    c.record_error(500)
    c.reset()
    snap = c.snapshot()
    assert snap["total_requests"] == 0
    assert snap["intents"] == {}
    assert snap["errors"] == {}
    assert snap["fallbacks"]["total"] == 0
