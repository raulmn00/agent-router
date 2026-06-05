"""Synthetic dataset generation + train/test split loader.

The dataset is built deterministically (fixed seed) so it can be regenerated
exactly. `intents.jsonl` is versioned in the repo; `regenerate()` overwrites it.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Iterable

from .intents import INTENTS

ROUTER_PKG = Path(__file__).resolve().parent
DATA_PATH = ROUTER_PKG.parent / "data" / "intents.jsonl"

PER_CLASS_TARGET = 150
SEED = 1337


# ---------------------------------------------------------------------------
# Template banks. Each class generates examples by combining a question/intro
# template with a subject/payload. We over-generate, dedupe, then trim to
# PER_CLASS_TARGET so every class is perfectly balanced.
# ---------------------------------------------------------------------------

SIMPLE_QA_OPENERS = [
    "What is", "What's", "Who is", "Who was", "When did", "When was",
    "Where is", "Where was", "Why is", "How many", "How much", "Which",
    "Define", "Tell me", "Name", "Give me",
]
SIMPLE_QA_SUBJECTS = [
    "the capital of France", "the boiling point of water", "the speed of light",
    "the population of Tokyo", "the tallest mountain in the world",
    "the longest river in South America", "the smallest planet in the solar system",
    "the author of Hamlet", "the inventor of the telephone",
    "the first president of the United States", "the chemical formula for water",
    "the largest desert in the world", "the deepest ocean", "the current year",
    "World War II", "the French Revolution", "the moon landing",
    "photosynthesis", "Newton's second law", "the Pythagorean theorem",
    "the square root of 144", "two plus two", "the value of pi",
    "the official language of Brazil", "the GDP of Japan",
    "the largest mammal on earth", "the fastest land animal",
    "the meaning of 'ephemeral'", "the antonym of brave", "a synonym for happy",
    "the currency of Switzerland", "the capital of Australia",
    "the discoverer of penicillin", "the founder of Microsoft",
    "the year Python was released", "the difference between TCP and UDP",
    "what HTTP stands for", "what JSON means", "the latest version of Python",
    "Einstein's most famous equation",
]

COMPLEX_TASK_OPENERS = [
    "Build a", "Design a", "Implement", "Create a", "Develop a", "Write code that",
    "Plan and build", "Architect", "Refactor my", "Set up an end-to-end",
    "Help me ship", "Walk me through building",
]
COMPLEX_TASK_PAYLOADS = [
    "web scraper that crawls news sites, dedupes by URL, and stores results in Postgres",
    "REST API in FastAPI with JWT auth, rate limiting and OpenAPI docs",
    "data pipeline that ingests CSVs from S3, validates with pydantic, and loads to BigQuery",
    "Kubernetes deployment with autoscaling, secrets via SealedSecrets and a Helm chart",
    "machine learning training loop with mixed precision, gradient accumulation and checkpointing",
    "real-time chat backend using WebSockets, Redis pub/sub and presence tracking",
    "CI/CD pipeline on GitHub Actions with test matrix, caching and preview deployments",
    "terraform module that provisions a VPC, three subnets and a NAT gateway",
    "search service with OpenSearch, ingestion via Kafka and faceted filters",
    "feature flag system with percentage rollouts, user targeting and an admin UI",
    "RAG system that indexes PDFs, embeds with bge-small and answers with citations",
    "billing service that handles Stripe webhooks, idempotent retries and reconciliation",
    "notification service that fans out to email, SMS and push with a unified template engine",
    "observability stack with OpenTelemetry, Prometheus and Grafana dashboards",
    "scheduled job runner that executes user-defined Python with sandboxing and quotas",
    "frontend dashboard with Next.js, server components, and incremental static regeneration",
    "model serving layer with batching, GPU autoscaling and request shadowing",
    "event-driven workflow with SQS, Lambda and a DLQ that retries with backoff",
    "personal finance app that imports CSVs, categorizes via ML and exports reports",
    "browser extension that captures highlights, syncs to a backend and supports OAuth",
    "multi-tenant SaaS with row-level security, per-tenant migrations and a billing portal",
    "agent that plans a trip end-to-end: flights, hotels, calendar invites, and a budget",
    "code review bot that posts inline comments based on a custom linter",
    "data labeling tool with active learning and inter-annotator agreement metrics",
    "fraud detection system with stream processing and a feedback loop",
    "vector database wrapper with hybrid search and metadata filtering",
    "load testing harness that ramps up traffic and reports p50/p95/p99",
    "image processing service that converts, resizes and watermarks with caching",
    "log analysis tool that parses NGINX access logs and surfaces anomalies",
    "scheduling agent that books meetings, resolves conflicts and emails participants",
]
COMPLEX_TASK_TAILS = [
    "", " — production ready", " with full test coverage", " step by step",
    " and document the architecture", ", with monitoring", ", including a rollback plan",
    " and benchmark it against the baseline",
]

DOCUMENT_QA_OPENERS = [
    "In the attached PDF,", "Based on the document above,", "From the text I uploaded,",
    "Looking at the contract,", "According to the report,", "In the article I shared,",
    "From the meeting notes,", "In the manual,", "From the spec,",
    "Per section 3 of the document,", "Based on chapter 2,", "In the whitepaper,",
    "From the attached spreadsheet,", "In the slide deck,", "Per the file I sent,",
]
DOCUMENT_QA_QUESTIONS = [
    "what is the main conclusion?",
    "summarize the key findings in 5 bullet points.",
    "who are the parties involved?",
    "what are the payment terms?",
    "what does the author argue?",
    "list every action item assigned to me.",
    "extract all dates and what happens on each.",
    "what is the termination clause?",
    "what are the limitations discussed?",
    "what evidence is presented?",
    "what is the recommended approach?",
    "list the prerequisites mentioned.",
    "what risks are flagged?",
    "explain figure 2 in plain English.",
    "translate the executive summary into Portuguese.",
    "what is the projected revenue for next quarter?",
    "summarize each section in one sentence.",
    "what assumptions does the model rely on?",
    "give me a TL;DR.",
    "what's the difference between option A and B as described?",
    "extract every named entity.",
    "what is the SLA?",
    "what are the supported configurations?",
    "what does paragraph 4 say?",
    "give me the methodology used.",
]
DOCUMENT_QA_STANDALONE = [
    "Summarize this document.",
    "Give me the TL;DR of the uploaded file.",
    "Extract the action items from the attached notes.",
    "Translate the document I shared into English.",
    "What does the contract say about indemnification?",
    "Pull out every number and what it refers to from the report.",
    "Compare sections 2 and 4 of the spec.",
    "From the file I just uploaded, what's the recommendation?",
    "List every name mentioned in the document.",
    "What are the takeaways from the slide deck?",
]

CHITCHAT_TEMPLATES = [
    "Hi!", "Hello!", "Hey there.", "Good morning.", "Good afternoon!", "Good evening.",
    "How are you?", "How's it going?", "What's up?", "How have you been?",
    "Thanks!", "Thank you so much.", "Appreciate it.", "Cheers.", "Awesome, thanks.",
    "lol", "haha that's funny", "nice", "cool", "ok", "got it", "sounds good",
    "Tell me a joke.", "Make me laugh.", "Say something fun.", "Tell me a fun fact.",
    "What's your name?", "Who are you?", "Are you a robot?", "What can you do?",
    "Nice to meet you.", "Pleasure to meet you.", "Talk to you later.", "Bye!",
    "See you!", "Take care.", "Have a great day.", "Goodnight.",
    "I'm bored.", "I'm tired today.", "Long day, huh?", "Mondays...",
    "Do you like music?", "What's your favorite color?", "Do you sleep?",
    "Are you having a good day?", "Hope you're well.", "Just saying hi.",
    "Random question for you.", "Quick one.", "Just wanted to chat.",
    "How's the weather where you are?", "What time is it for you?",
    "lmao", "🙂", "👋", "What a day.", "How's life?",
    "You're great.", "You're helpful.", "I like talking to you.",
    "Anyone there?", "Knock knock.", "Pop quiz: ready?", "Ping.",
    "yo", "sup", "heyo", "hiya", "hey friend", "morning!", "evening!",
    "thx", "ty", "much appreciated", "you rock",
    "smh", "rofl", "haha", "hehe", "🤣", "😂", "👍", "🙌",
    "feeling sleepy", "kinda hungry", "coffee time", "tea time",
    "i need a break", "back from lunch", "happy friday!", "happy monday",
    "tgif", "long weekend coming up", "any plans for the weekend?",
    "how's your day so far?", "how was your weekend?", "what are you up to?",
    "favorite movie?", "favorite book?", "favorite food?",
    "do you dream?", "do you have feelings?", "are you sentient?",
    "say hi back", "wave hello", "send hugs", "send vibes",
    "tell me something nice", "compliment me", "cheer me up",
    "i'm happy today", "i'm sad today", "today was great", "today was rough",
    "thanks for listening", "you're the best", "i appreciate you",
    "k", "kk", "yep", "nope", "maybe", "sure thing",
    "lemme think", "one sec", "brb", "afk for a moment",
    "back!", "i'm back", "still here?", "you there?",
    "ok i'm done for today", "logging off", "signing off", "ttyl",
]


def _gen_simple_qa(rng: random.Random) -> Iterable[str]:
    """All opener × subject combos, optionally with a '?' suffix."""
    combos = list(itertools.product(SIMPLE_QA_OPENERS, SIMPLE_QA_SUBJECTS))
    rng.shuffle(combos)
    for opener, subject in combos:
        sentence = f"{opener} {subject}"
        # Some openers (Define, Tell me, Name, Give me) read better without "?"
        if opener in {"Define", "Tell me", "Name", "Give me"}:
            yield sentence + "."
        else:
            yield sentence + "?"


def _gen_complex_task(rng: random.Random) -> Iterable[str]:
    combos = list(itertools.product(COMPLEX_TASK_OPENERS, COMPLEX_TASK_PAYLOADS, COMPLEX_TASK_TAILS))
    rng.shuffle(combos)
    for opener, payload, tail in combos:
        yield f"{opener} {payload}{tail}."


def _gen_document_qa(rng: random.Random) -> Iterable[str]:
    combos = list(itertools.product(DOCUMENT_QA_OPENERS, DOCUMENT_QA_QUESTIONS))
    rng.shuffle(combos)
    for opener, q in combos:
        yield f"{opener} {q}"
    # Sprinkle the standalone ones in too.
    for s in DOCUMENT_QA_STANDALONE:
        yield s


def _gen_chitchat(rng: random.Random) -> Iterable[str]:
    # Use templates directly; pad with simple combinations to reach the target.
    items = list(CHITCHAT_TEMPLATES)
    rng.shuffle(items)
    yield from items
    # Two-utterance combos for more variety.
    a, b = list(CHITCHAT_TEMPLATES), list(CHITCHAT_TEMPLATES)
    rng.shuffle(a)
    rng.shuffle(b)
    for x, y in zip(a, b):
        if x != y:
            yield f"{x} {y}"


GENERATORS = {
    "simple_qa": _gen_simple_qa,
    "complex_task": _gen_complex_task,
    "document_qa": _gen_document_qa,
    "chitchat": _gen_chitchat,
}


def _take_unique(stream: Iterable[str], n: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in stream:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) == n:
            break
    if len(out) < n:
        raise RuntimeError(
            f"Template bank exhausted before reaching target {n}; got {len(out)}. "
            "Add more openers/subjects."
        )
    return out


def build_dataset(per_class: int = PER_CLASS_TARGET, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for label in INTENTS:
        gen = GENERATORS[label]
        texts = _take_unique(gen(rng), per_class)
        rows.extend({"text": t, "label": label} for t in texts)
    rng.shuffle(rows)
    return rows


def regenerate(path: Path = DATA_PATH, per_class: int = PER_CLASS_TARGET, seed: int = SEED) -> Path:
    rows = build_dataset(per_class=per_class, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: Path = DATA_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python -m router.dataset` to regenerate."
        )
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dataset_splits(
    path: Path = DATA_PATH,
    test_size: float = 0.2,
    seed: int = SEED,
):
    """Return a `datasets.DatasetDict` with stratified train/test splits.

    Heavy deps (`datasets`, `scikit-learn`) are imported lazily so this module
    can be loaded — and the dataset regenerated — without them installed.
    """
    from datasets import Dataset, DatasetDict  # local import: see docstring
    from sklearn.model_selection import train_test_split

    rows = load_jsonl(path)
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return DatasetDict(
        {
            "train": Dataset.from_dict({"text": train_texts, "label": train_labels}),
            "test": Dataset.from_dict({"text": test_texts, "label": test_labels}),
        }
    )


if __name__ == "__main__":
    out = regenerate()
    print(f"Wrote {out} ({PER_CLASS_TARGET} per class × {len(INTENTS)} classes)")
