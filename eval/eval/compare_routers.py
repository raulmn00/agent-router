"""Run all three routers on the held-out testset and write a comparison table.

Usage:

    cd eval
    python -m eval.compare_routers

Skips any approach whose preconditions aren't met (e.g. no fine-tuned model,
no OPENAI_API_KEY) and reports which were skipped at the top of the table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import accuracy, cost_per_1k, f1_macro, mean_latency_ms

EVAL_ROOT = Path(__file__).resolve().parents[1]
TESTSET_PATH = EVAL_ROOT / "data" / "routing_testset.jsonl"
TRAINSET_PATH = EVAL_ROOT.parent / "router" / "data" / "intents.jsonl"
RESULTS_DIR = EVAL_ROOT / "results"
EMBED_CACHE_PATH = EVAL_ROOT / "results" / ".embed_cache.json"


# --------------------------------------------------------------------------- #
# Result container                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class ApproachResult:
    name: str
    accuracy: float
    f1: float
    mean_latency_ms: float
    cost_per_1k_usd: float
    n: int
    stdev_latency_ms: float = 0.0
    runs: int = 1
    note: str = ""
    per_class_correct: dict = field(default_factory=dict)


@dataclass
class SkippedResult:
    name: str
    reason: str


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _measure(classify_fn, texts: list[str]) -> tuple[list[str], list[float]]:
    """Time each classification individually with time.perf_counter."""
    preds: list[str] = []
    latencies_ms: list[float] = []
    for t in texts:
        t0 = time.perf_counter()
        pred = classify_fn(t)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        preds.append(pred)
    return preds, latencies_ms


# --------------------------------------------------------------------------- #
# Per-approach runners                                                         #
# --------------------------------------------------------------------------- #


def _run_distilbert(testset: list[dict]) -> ApproachResult | SkippedResult:
    try:
        from router.classifier import IntentClassifier

        clf = IntentClassifier()
    except RuntimeError as e:
        return SkippedResult("DistilBERT (fine-tuned)", str(e).splitlines()[0])
    except Exception as e:
        return SkippedResult("DistilBERT (fine-tuned)", f"{type(e).__name__}: {e}")

    texts = [r["text"] for r in testset]
    labels = [r["label"] for r in testset]
    preds, lats = _measure(lambda t: clf.classify(t).intent, texts)
    return ApproachResult(
        name="DistilBERT (fine-tuned)",
        accuracy=accuracy(labels, preds),
        f1=f1_macro(labels, preds),
        mean_latency_ms=mean_latency_ms(lats),
        cost_per_1k_usd=cost_per_1k("distilbert"),
        n=len(texts),
    )


def _run_llm(testset: list[dict]) -> ApproachResult | SkippedResult:
    if not os.getenv("OPENAI_API_KEY"):
        return SkippedResult("LLM zero-shot (gpt-4o-mini)", "OPENAI_API_KEY not set")
    try:
        from .llm_router import LLMRouter

        router = LLMRouter()
    except Exception as e:
        return SkippedResult("LLM zero-shot (gpt-4o-mini)", f"{type(e).__name__}: {e}")

    texts = [r["text"] for r in testset]
    labels = [r["label"] for r in testset]
    preds, lats = _measure(router.classify, texts)
    return ApproachResult(
        name="LLM zero-shot (gpt-4o-mini)",
        accuracy=accuracy(labels, preds),
        f1=f1_macro(labels, preds),
        mean_latency_ms=mean_latency_ms(lats),
        cost_per_1k_usd=cost_per_1k("llm"),
        n=len(texts),
    )


def _run_embed(testset: list[dict], trainset: list[dict]) -> ApproachResult | SkippedResult:
    if not os.getenv("OPENAI_API_KEY"):
        return SkippedResult("Embeddings + LogReg", "OPENAI_API_KEY not set")
    try:
        from .embed_router import EmbedRouter

        router = EmbedRouter(cache_path=str(EMBED_CACHE_PATH))
        router.fit(
            texts=[r["text"] for r in trainset],
            labels=[r["label"] for r in trainset],
        )
    except Exception as e:
        return SkippedResult("Embeddings + LogReg", f"{type(e).__name__}: {e}")

    texts = [r["text"] for r in testset]
    labels = [r["label"] for r in testset]
    preds, lats = _measure(router.classify, texts)
    return ApproachResult(
        name="Embeddings + LogReg",
        accuracy=accuracy(labels, preds),
        f1=f1_macro(labels, preds),
        mean_latency_ms=mean_latency_ms(lats),
        cost_per_1k_usd=cost_per_1k("embed"),
        n=len(texts),
    )


# --------------------------------------------------------------------------- #
# Output formatting                                                            #
# --------------------------------------------------------------------------- #


def _aggregate_runs(per_run: list[ApproachResult]) -> ApproachResult:
    """Combine N single-run results for the same approach. Accuracy/F1 should
    be identical across runs (deterministic given the model); only latency
    varies with network/system load."""
    if not per_run:
        raise ValueError("no runs to aggregate")
    if len(per_run) == 1:
        return per_run[0]
    name = per_run[0].name
    accs = {round(r.accuracy, 6) for r in per_run}
    f1s = {round(r.f1, 6) for r in per_run}
    if len(accs) > 1 or len(f1s) > 1:
        # Loud — if this fires, something is non-deterministic and the
        # comparison is suspect.
        print(f"WARN: quality varies across runs for {name}: accs={accs} f1s={f1s}")
    lats = [r.mean_latency_ms for r in per_run]
    return ApproachResult(
        name=name,
        accuracy=per_run[0].accuracy,
        f1=per_run[0].f1,
        mean_latency_ms=statistics.mean(lats),
        stdev_latency_ms=statistics.stdev(lats),
        cost_per_1k_usd=per_run[0].cost_per_1k_usd,
        n=per_run[0].n,
        runs=len(per_run),
    )


def _format_markdown(results: list, skipped: list[SkippedResult], runs: int) -> str:
    lines = ["# Router comparison\n"]
    if skipped:
        lines.append("## Skipped\n")
        for s in skipped:
            lines.append(f"- **{s.name}** — {s.reason}")
        lines.append("")

    header_lat = "Mean latency (ms) ± stdev" if runs > 1 else "Mean latency (ms)"
    suffix = f" (n={runs} cold-cache runs)" if runs > 1 else ""
    lines.append(f"## Results{suffix}\n")
    lines.append(f"| Approach | Accuracy | F1 (macro) | {header_lat} | Cost / 1k (USD) | N |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        if r.runs > 1:
            lat_cell = f"{r.mean_latency_ms:.1f} ± {r.stdev_latency_ms:.1f}"
        else:
            lat_cell = f"{r.mean_latency_ms:.1f}"
        lines.append(
            f"| {r.name} | {r.accuracy:.3f} | {r.f1:.3f} | "
            f"{lat_cell} | ${r.cost_per_1k_usd:.4f} | {r.n} |"
        )
    lines.append("")
    lines.append(
        "_Cost numbers are derived from the documented assumptions in "
        "`eval/eval/metrics.py` (PRICING_USD_PER_M_TOKENS, TOKEN_ASSUMPTIONS). "
        "Latency was measured per-classification with `time.perf_counter`._"
    )
    return "\n".join(lines)


def _write_csv(path: Path, results: list[ApproachResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["approach", "accuracy", "f1_macro", "mean_latency_ms",
             "stdev_latency_ms", "cost_per_1k_usd", "n", "runs"]
        )
        for r in results:
            w.writerow(
                [r.name, f"{r.accuracy:.4f}", f"{r.f1:.4f}",
                 f"{r.mean_latency_ms:.2f}", f"{r.stdev_latency_ms:.2f}",
                 f"{r.cost_per_1k_usd:.6f}", r.n, r.runs]
            )


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default=str(TESTSET_PATH))
    parser.add_argument("--trainset", default=str(TRAINSET_PATH))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=("Number of independent measurement runs. Latencies are aggregated "
              "as mean ± stdev across runs; embedding cache is cleared between "
              "runs so the embed router stays cold-cache."),
    )
    args = parser.parse_args()

    testset = _load_jsonl(Path(args.testset))
    trainset = _load_jsonl(Path(args.trainset))
    print(f"loaded {len(testset)} test examples, {len(trainset)} train examples")
    if args.runs > 1:
        print(f"running {args.runs} independent measurement runs")

    per_run: list[list] = []
    for run_i in range(args.runs):
        if run_i > 0 and EMBED_CACHE_PATH.exists():
            # Clearing the cache makes the embed router re-embed every test
            # text from scratch — the only honest way to compare cold-cache
            # latency to the LLM and DistilBERT paths.
            EMBED_CACHE_PATH.unlink()
        if args.runs > 1:
            print(f"\n--- run {run_i + 1}/{args.runs} ---")
        per_run.append(
            [
                _run_distilbert(testset),
                _run_llm(testset),
                _run_embed(testset, trainset),
            ]
        )

    # Transpose: per_run[run][approach] → by_approach[approach][run]
    by_approach = list(zip(*per_run))

    results: list[ApproachResult] = []
    skipped: list[SkippedResult] = []
    seen_skipped: set[tuple[str, str]] = set()
    for approach_runs in by_approach:
        ok = [r for r in approach_runs if isinstance(r, ApproachResult)]
        bad = [r for r in approach_runs if isinstance(r, SkippedResult)]
        if ok:
            results.append(_aggregate_runs(ok))
        for s in bad:
            key = (s.name, s.reason)
            if key not in seen_skipped:
                seen_skipped.add(key)
                skipped.append(s)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = _format_markdown(results, skipped, runs=args.runs)
    (out_dir / "comparison.md").write_text(md)
    _write_csv(out_dir / "comparison.csv", results)

    print("\n" + md + "\n")
    print(f"Wrote {out_dir / 'comparison.md'}")
    print(f"Wrote {out_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
