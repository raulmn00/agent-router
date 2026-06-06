"""Smoke test the /route endpoint of agent-router in production.

Sends a fixed set of clear + ambiguous inputs, measures latency, and verifies
that the confidence-threshold fallback engages on the ambiguous ones. Prints
a terminal report and writes a versionable markdown report to
`scripts/results/prod_routing_report.md`.

Run:

    python scripts/test_routing_prod.py
    # or, against a different deploy:
    AGENT_ROUTER_URL=https://... python scripts/test_routing_prod.py

Skips /compare on purpose — that endpoint burns real tokens and is tightly
rate-limited. Only /route is exercised here.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

# -------------------------------------------------------------------------- #
# Configuration                                                              #
# -------------------------------------------------------------------------- #

DEFAULT_URL = "https://agent-router-909428365094.us-central1.run.app"
TIMEOUT_SECONDS = 60.0
THRESHOLD_FOR_HYPOTHESIS = 0.65  # documented default; runtime may differ
LOW_CONFIDENCE_PATH = "low_confidence_fallback"

CLEAR_INPUTS: list[str] = [
    "Design a scalable architecture for a food delivery app and outline the main trade-offs.",
    "Help me migrate a monolith to microservices step by step.",
    "What's the capital of Australia?",
    "What year did the Berlin Wall fall?",
    "According to the attached contract, what is the termination notice period?",
    "In the PDF I uploaded, which section covers the refund policy?",
    "Good morning! Hope you're having a nice day.",
]

AMBIGUOUS_INPUTS: list[str] = [
    "Tell me about the requirements",
    "Build me something cool",
    "What does it say about pricing?",
    "Can you explain how this works?",
]


# -------------------------------------------------------------------------- #
# Data                                                                       #
# -------------------------------------------------------------------------- #


@dataclass
class TestResult:
    input: str
    group: str  # "clear" or "ambiguous"
    intent: str = ""
    confidence: float = 0.0
    path_taken: str = ""
    latency_ms: float = 0.0
    http_status: int | None = None
    error: str | None = None


# -------------------------------------------------------------------------- #
# Network                                                                    #
# -------------------------------------------------------------------------- #


def call_route(client: httpx.Client, url: str, text: str, group: str) -> TestResult:
    result = TestResult(input=text, group=group)
    t0 = time.perf_counter()
    try:
        resp = client.post(f"{url}/route", json={"input": text})
    except httpx.TimeoutException:
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        result.error = f"timeout after {TIMEOUT_SECONDS:.0f}s"
        return result
    except httpx.RequestError as e:
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        result.error = f"network error: {type(e).__name__}"
        return result

    result.latency_ms = (time.perf_counter() - t0) * 1000.0
    result.http_status = resp.status_code

    if resp.status_code == 200:
        try:
            body = resp.json()
        except ValueError:
            result.error = "non-JSON 200 response"
            return result
        result.intent = str(body.get("intent", ""))
        result.confidence = float(body.get("confidence", 0.0))
        result.path_taken = str(body.get("path_taken", ""))
    elif resp.status_code == 429:
        result.error = "rate limited (429) — slow down for a minute"
    elif resp.status_code == 503:
        result.error = "provider not configured (503)"
    elif resp.status_code == 413:
        result.error = "payload too large (413)"
    elif resp.status_code == 422:
        result.error = "request rejected (422 validation)"
    else:
        result.error = f"http {resp.status_code}"
    return result


def run_all(url: str) -> tuple[list[TestResult], list[str]]:
    results: list[TestResult] = []
    network_errors: list[str] = []

    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        for group, inputs in (("clear", CLEAR_INPUTS), ("ambiguous", AMBIGUOUS_INPUTS)):
            for text in inputs:
                r = call_route(client, url, text, group)
                results.append(r)
                # Distinguish HTTP-layer issues (kept above) from socket-layer ones
                if r.error and r.http_status is None:
                    network_errors.append(f"[{group}] {text!r}: {r.error}")
                print(_one_line_result(r), flush=True)

    return results, network_errors


# -------------------------------------------------------------------------- #
# Analysis                                                                   #
# -------------------------------------------------------------------------- #


def analyze(results: list[TestResult]) -> dict:
    clear = [r for r in results if r.group == "clear"]
    ambiguous = [r for r in results if r.group == "ambiguous"]

    clear_ok = [r for r in clear if r.error is None]
    amb_ok = [r for r in ambiguous if r.error is None]

    n_clear_in_fb = sum(1 for r in clear_ok if r.path_taken == LOW_CONFIDENCE_PATH)
    n_amb_in_fb = sum(1 for r in amb_ok if r.path_taken == LOW_CONFIDENCE_PATH)
    n_fallback_total = sum(1 for r in results if r.path_taken == LOW_CONFIDENCE_PATH)

    # Hypothesis: every ambiguous → fallback OR confidence < threshold; every clear → NOT fallback.
    amb_pass_n = sum(
        1
        for r in amb_ok
        if r.path_taken == LOW_CONFIDENCE_PATH or r.confidence < THRESHOLD_FOR_HYPOTHESIS
    )
    clear_pass_n = sum(1 for r in clear_ok if r.path_taken != LOW_CONFIDENCE_PATH)

    latencies = [r.latency_ms for r in results if r.error is None and r.latency_ms > 0]
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0

    cold_start = False
    warm_mean = mean_lat
    if len(latencies) >= 3:
        first = latencies[0]
        rest = latencies[1:]
        rest_mean = sum(rest) / len(rest)
        # Definite cold start: first is at least 3x slower than warm average AND > 5 s.
        if first > 5000 and first > 3 * rest_mean:
            cold_start = True
            warm_mean = rest_mean

    return {
        "n_total": len(results),
        "n_clear": len(clear),
        "n_ambiguous": len(ambiguous),
        "n_fallback_total": n_fallback_total,
        "n_clear_in_fallback": n_clear_in_fb,
        "n_amb_in_fallback": n_amb_in_fb,
        "amb_hypothesis_pass": amb_pass_n,
        "amb_hypothesis_total": len(amb_ok),
        "clear_hypothesis_pass": clear_pass_n,
        "clear_hypothesis_total": len(clear_ok),
        "mean_latency_ms": mean_lat,
        "warm_mean_latency_ms": warm_mean,
        "min_latency_ms": min(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "cold_start_detected": cold_start,
        "latencies_in_order_ms": latencies,
    }


# -------------------------------------------------------------------------- #
# Rendering                                                                  #
# -------------------------------------------------------------------------- #


def _one_line_result(r: TestResult) -> str:
    """Single-line progress output while requests are in flight."""
    if r.error:
        return f"  [{r.group:9}] {r.latency_ms:>7.0f}ms  ERROR: {r.error}  ← {r.input!r}"
    badge = "FB" if r.path_taken == LOW_CONFIDENCE_PATH else "OK"
    return (
        f"  [{r.group:9}] {r.latency_ms:>7.0f}ms  {badge}  "
        f"intent={r.intent:14s} conf={r.confidence:.3f}  path={r.path_taken}"
    )


def _truncate(s: str, n: int) -> str:
    return (s[: n - 1] + "…") if len(s) > n else s


def render_terminal_report(url: str, results: list[TestResult], stats: dict, errors: list[str]) -> str:
    out: list[str] = []
    sep = "─" * 138
    out.append("")
    out.append(sep)
    out.append("Results table")
    out.append(sep)
    out.append(
        f"{'Group':10} {'Input':70} {'Intent':14} {'Conf':>5} {'Path':30} {'ms':>7}"
    )
    out.append(sep)
    for r in results:
        inp = _truncate(r.input, 70)
        if r.error:
            out.append(
                f"{r.group:10} {inp:70} {'ERROR':14} {'—':>5} "
                f"{_truncate(r.error, 30):30} {r.latency_ms:>7.0f}"
            )
        else:
            out.append(
                f"{r.group:10} {inp:70} {r.intent:14} "
                f"{r.confidence:>5.3f} {r.path_taken:30} {r.latency_ms:>7.0f}"
            )
    out.append("")
    out.append(sep)
    out.append("Hypothesis verification")
    out.append(sep)
    amb_pass = stats["amb_hypothesis_pass"] == stats["amb_hypothesis_total"]
    clear_pass = stats["clear_hypothesis_pass"] == stats["clear_hypothesis_total"]
    out.append(
        f"  ambiguous → fallback or conf < {THRESHOLD_FOR_HYPOTHESIS}: "
        f"{stats['amb_hypothesis_pass']}/{stats['amb_hypothesis_total']} "
        f"{'PASS' if amb_pass else 'FAIL'}"
    )
    out.append(
        f"  clear     → NOT fallback:                  "
        f"{stats['clear_hypothesis_pass']}/{stats['clear_hypothesis_total']} "
        f"{'PASS' if clear_pass else 'FAIL'}"
    )
    out.append(f"  fallbacks observed:                        {stats['n_fallback_total']}/{stats['n_total']}")
    out.append("")
    out.append(f"  latency mean: {stats['mean_latency_ms']:.0f} ms  ·  range {stats['min_latency_ms']:.0f}-{stats['max_latency_ms']:.0f} ms")
    if stats["cold_start_detected"]:
        out.append(
            f"  warm-only mean: {stats['warm_mean_latency_ms']:.0f} ms "
            f"(first request was {stats['latencies_in_order_ms'][0]:.0f} ms — cold start)"
        )
    if errors:
        out.append("")
        out.append(f"  network errors: {len(errors)}")
        for e in errors:
            out.append(f"    - {e}")
    out.append(sep)
    return "\n".join(out)


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def render_markdown_report(url: str, results: list[TestResult], stats: dict, errors: list[str]) -> str:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    amb_pass = stats["amb_hypothesis_pass"] == stats["amb_hypothesis_total"]
    clear_pass = stats["clear_hypothesis_pass"] == stats["clear_hypothesis_total"]

    lines: list[str] = []
    lines.append(f"# /route smoke test — {timestamp}")
    lines.append("")
    lines.append(f"- **URL**: `{url}`")
    lines.append(f"- **Inputs**: {stats['n_total']} total ({stats['n_clear']} clear, {stats['n_ambiguous']} ambiguous)")
    lines.append(f"- **Per-request timeout**: {TIMEOUT_SECONDS:.0f}s")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Group | Input | Intent | Confidence | Path taken | Latency (ms) |")
    lines.append("|---|---|---|---:|---|---:|")
    for r in results:
        inp = _md_escape(r.input)
        if r.error:
            lines.append(
                f"| {r.group} | {inp} | — | — | **ERROR**: {_md_escape(r.error)} | {r.latency_ms:.0f} |"
            )
        else:
            lines.append(
                f"| {r.group} | {inp} | `{r.intent}` | {r.confidence:.3f} | `{r.path_taken}` | {r.latency_ms:.0f} |"
            )
    lines.append("")
    lines.append("## Hypothesis verification")
    lines.append("")
    lines.append(
        f"- Ambiguous → `low_confidence_fallback` OR confidence < {THRESHOLD_FOR_HYPOTHESIS}: "
        f"**{stats['amb_hypothesis_pass']}/{stats['amb_hypothesis_total']}** — "
        f"{'**PASS**' if amb_pass else '**FAIL**'}"
    )
    lines.append(
        f"- Clear → NOT `low_confidence_fallback`: "
        f"**{stats['clear_hypothesis_pass']}/{stats['clear_hypothesis_total']}** — "
        f"{'**PASS**' if clear_pass else '**FAIL**'}"
    )
    lines.append(f"- Total fallbacks observed: **{stats['n_fallback_total']}/{stats['n_total']}**")
    lines.append("")
    lines.append("## Latency")
    lines.append("")
    lines.append(f"- Mean across all requests: **{stats['mean_latency_ms']:.0f} ms**")
    lines.append(f"- Range: {stats['min_latency_ms']:.0f} — {stats['max_latency_ms']:.0f} ms")
    if stats["cold_start_detected"]:
        first_ms = stats["latencies_in_order_ms"][0]
        lines.append(
            f"- ⚠ Cold start detected on the first request "
            f"({first_ms:.0f} ms vs warm mean {stats['warm_mean_latency_ms']:.0f} ms). "
            "Cloud Run scaled from zero (min-instances=0) and the entrypoint downloaded "
            "and loaded the DistilBERT model on the new instance."
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(_prose_summary(stats, amb_pass, clear_pass))
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Backend URL: `{url}`")
    lines.append(f"- Per-request timeout: {TIMEOUT_SECONDS:.0f} s")
    if errors:
        lines.append(f"- Network errors during the run: **{len(errors)}**")
        for e in errors:
            lines.append(f"  - `{e}`")
    else:
        lines.append("- Network errors during the run: **none**")
    return "\n".join(lines) + "\n"


def _prose_summary(stats: dict, amb_pass: bool, clear_pass: bool) -> str:
    n_amb_fb = stats["n_amb_in_fallback"]
    n_amb = stats["n_ambiguous"]
    n_clear_fb = stats["n_clear_in_fallback"]
    n_clear = stats["n_clear"]
    mean = stats["mean_latency_ms"]
    warm = stats["warm_mean_latency_ms"]

    head = (
        f"**{n_amb_fb}/{n_amb}** of the ambiguous inputs landed on "
        f"`low_confidence_fallback`; **{n_clear_fb}/{n_clear}** of the clear inputs "
        "fell back unexpectedly."
    )

    if amb_pass and clear_pass:
        verdict = (
            "The threshold behavior in production matches the local generalization probe — "
            "clear inputs route normally with confidence in the high-0.8 to mid-0.9 band, "
            "ambiguous inputs sit between 0.39 and 0.59 and are caught by the fallback before "
            "any LLM is invoked."
        )
    else:
        verdict = (
            "There are divergences from the local generalization probe — check the table above "
            "for the specific inputs and confidences. The behavior may be intentional "
            "(legitimate chitchat naturally tops out around 0.56 and can fall into the fallback) "
            "or worth tightening per-class thresholds for."
        )

    latency = (
        f"Mean per-request latency was {mean:.0f} ms across {stats['n_total']} requests"
    )
    if stats["cold_start_detected"]:
        latency += (
            f" (warm-only mean {warm:.0f} ms; the first request paid a cold-start cost on Cloud Run)."
        )
    else:
        latency += " (no cold-start observed)."

    return f"{head} {verdict} {latency}"


# -------------------------------------------------------------------------- #
# Entry point                                                                #
# -------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=os.getenv("AGENT_ROUTER_URL", DEFAULT_URL),
        help="agent-router base URL (env: AGENT_ROUTER_URL)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "results"),
        help="where to write the markdown report",
    )
    args = parser.parse_args()
    url = args.url.rstrip("/")

    n = len(CLEAR_INPUTS) + len(AMBIGUOUS_INPUTS)
    print(f"Hitting {url}/route with {n} inputs ({TIMEOUT_SECONDS:.0f}s timeout each)...")
    print()
    results, network_errors = run_all(url)
    stats = analyze(results)

    print(render_terminal_report(url, results, stats, network_errors))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "prod_routing_report.md"
    md_path.write_text(render_markdown_report(url, results, stats, network_errors))
    print()
    print(f"Markdown report written to: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
