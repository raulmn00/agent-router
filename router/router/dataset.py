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

# Two-tier dataset: easy templated examples (give the model the obvious surface
# cues) + a smaller pool of HARD adversarial examples (deliberately break those
# cues so the task is realistic). Total per class = PER_CLASS_TARGET; the
# stratified split function still works since splitting is by label only.
PER_CLASS_EASY = 150
PER_CLASS_HARD = 30
PER_CLASS_TARGET = PER_CLASS_EASY + PER_CLASS_HARD  # = 180

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


# ---------------------------------------------------------------------------
# HARD examples — adversarial inputs that break the surface markers the easy
# templates rely on. Two disjoint pools per class:
#
#   HARD_EXAMPLES[label]                — included in router/data/intents.jsonl
#                                          (used for training)
#   HARD_HELD_OUT_FOR_TESTSET[label]    — committed to
#                                          eval/data/routing_testset.jsonl
#                                          and NEVER copied into the training
#                                          pool (the eval split has to stay
#                                          held-out).
#
# Zero overlap is enforced statically by these two lists being source-of-truth
# and by an assertion in build_dataset(). Don't move strings between the lists
# without re-running pytest — `test_no_overlap_between_train_and_held_out_hard`
# pins the disjointness.
# ---------------------------------------------------------------------------

HARD_EXAMPLES: dict[str, list[str]] = {
    "simple_qa": [
        # No "What is X?" framing — indirect, casual, idiomatic.
        "Any idea what time zone São Paulo is in?",
        "Remind me how many continents there are.",
        "I forget — who painted the Mona Lisa?",
        "Quick one: boiling point of water in Fahrenheit?",
        "Is a tomato technically a fruit?",
        "Remind me real quick — what year did WWII end?",
        "Hey, do you happen to know the capital of Norway?",
        "Quick fact: how far is the Moon from Earth?",
        "Any clue what the population of Tokyo is these days?",
        "Brain fart — who wrote Hamlet again?",
        "Wait, isn't the Pacific the largest ocean?",
        "Random one for you: how many bones in the human body?",
        "I forget every time — Berlin or Munich, which is the German capital?",
        "Wasn't it Edison who invented the light bulb?",
        "Tell me, how tall is the Eiffel Tower roughly?",
        "Help me settle a bet — speed of sound in m/s?",
        "Just curious: what's the smallest country in the world?",
        "Could've sworn the heart has four chambers — right?",
        "Off the top of your head, when was the first iPhone released?",
        "Refresh my memory: what's the freezing point of water?",
        "Hmm, was it Newton or Einstein who discovered gravity?",
        "Drawing a blank — official language of Switzerland?",
        "Any idea how many planets are in our solar system?",
        "Need a sanity check: how long is a marathon in km?",
        "Random thought — is honey actually bee vomit?",
        "Pop quiz: capital of Canada?",
        "Out of curiosity, what's the deepest point in the ocean?",
        "Side question: oldest university in the world?",
        "Forgot the year — when did the Berlin Wall fall?",
        "Just to confirm, is Pluto still a planet?",
    ],
    "complex_task": [
        # No leading Design/Plan/Create/Build — casual, multi-step framing.
        "I need to get our whole CI pipeline from scratch to production-ready, "
        "where do I even start?",
        "Walk me through launching a podcast end to end.",
        "We're moving 40 engineers to a new repo structure — break that down for me.",
        "Turn my messy spreadsheet of leads into an actual outreach campaign.",
        "Help me figure out everything involved in adopting a rescue dog.",
        "Our onboarding flow is a mess — give me a full plan to fix it.",
        "I want to launch a side business selling handmade candles. Take me through it.",
        "What do I need to think about to migrate our SaaS from Heroku to AWS?",
        "Got a small team that needs to ship a mobile app in 6 weeks — how do we do it?",
        "I'm overwhelmed planning my wedding. Where do we even begin?",
        "Need to redesign our marketing site from scratch. Walk me through the steps.",
        "Our customer support is drowning — break down everything we'd need to set up a help center.",
        "I want to grow my newsletter from 100 to 10k subscribers. Lay out the strategy.",
        "We're going from on-prem to multi-cloud. Map out everything.",
        "Just started a YouTube channel. Walk me through getting to 1k subscribers.",
        "Help me plan opening a small bookstore in my neighborhood, soup to nuts.",
        "Need to take our open source library from 50 stars to 10k. Roadmap, please.",
        "I'm thinking of becoming a freelance consultant — break down the whole transition.",
        "Walk me through setting up a Series A fundraise from scratch.",
        "I want to organize a 200-person tech conference. What's the whole list?",
        "Got asked to lead a team of 6 — talk me through the first 90 days.",
        "We need to build a research lab from zero. What does that look like?",
        "Take me through everything I'd need to launch a SaaS in a regulated industry.",
        "I want to make our product accessible to screen reader users — walk me through it.",
        "Trying to overhaul our annual review process for 200 employees. Lay it out.",
        "Need to bootstrap a developer relations program at our startup. Start to finish.",
        "Walk me through what's involved in opening a coffee shop.",
        "I want to make our docs world-class. Break down the whole project.",
        "Just inherited a 10-year-old codebase that nobody understands. How do I tackle it?",
        "Help me think through migrating our org from Slack to Teams without chaos.",
    ],
    "document_qa": [
        # Subtle document reference — no "PDF/document/attached", uses pronouns
        # (it/this/these) or section references.
        "What's the deadline mentioned in section 4?",
        "Does it say anything about late payment penalties?",
        "Pull out the main argument from what I just shared.",
        "Which clauses talk about data retention?",
        "According to this, who's liable if the shipment is damaged?",
        "Did it list the dependencies anywhere?",
        "Is there a non-compete clause? If yes, how long?",
        "Where does it say the meeting is scheduled?",
        "Pull out every phone number it mentions.",
        "Summarize what they said about Q3 in two lines.",
        "Which paragraph covers refunds?",
        "It mentions three priorities — which ones?",
        "Does this lay out the timeline anywhere?",
        "Were the metrics for success defined?",
        "Find me the part about parental leave.",
        "What does it say in the conclusion?",
        "Pull the table of contents.",
        "Was there a deadline mentioned anywhere?",
        "Who signed off on this?",
        "Does this cover the data deletion process?",
        "What's the bottom line in plain English?",
        "Which page has the pricing?",
        "It cites a study — give me the citation.",
        "What's listed under acceptance criteria?",
        "Does it allow remote work?",
        "Where does it define 'confidential information'?",
        "Pull the requirements for hiring senior engineers.",
        "Talks about a probation period — for how long?",
        "Spot anything that contradicts what was agreed yesterday?",
        "Highlight everything related to security.",
    ],
    "chitchat": [
        # Longer / reactive conversational turns — not just greetings/thanks.
        "Ugh, Mondays. How do you even function this early?",
        "That makes total sense, appreciate you breaking it down.",
        "Honestly you're way more patient than my last assistant lol",
        "No worries, take your time!",
        "Wow, didn't expect that answer — neat.",
        "OK that was helpful, you're a lifesaver.",
        "Lmao you have a sense of humor, who knew.",
        "Honestly impressive how fast you came back with that.",
        "Sometimes I wonder if you actually enjoy these conversations.",
        "Not gonna lie, that was a satisfying explanation.",
        "Tough crowd today, sorry I'm cranky.",
        "I always forget you don't actually sleep — must be peaceful tbh.",
        "Coffee's kicking in, I should be more useful soon.",
        "Ha, fair point. I'll concede that one.",
        "If you were a person we'd grab coffee.",
        "Sorry, I rambled. Where were we?",
        "Always weird talking to AI but you're chill.",
        "OK that one made me laugh out loud at my desk.",
        "Damn, you're efficient today.",
        "Hold on, my dog's barking at the mailman.",
        "Wait wait wait, that's actually genius.",
        "Eh, not your fault, I phrased it badly.",
        "Random thought: I bet you're tired of dumb questions.",
        "Cheers, that was easier than I expected.",
        "Honestly that's the most useful conversation I've had all week.",
        "OK that's a brain-melter, gonna need a minute.",
        "Tea break, bbiab.",
        "Hah, you and me both.",
        "Gonna miss this when I have to actually go work.",
        "You know, you'd be a great manager.",
    ],
}

# Reserved for eval/data/routing_testset.jsonl — DO NOT include in training.
# Three per class, deliberately different from anything in HARD_EXAMPLES above.
HARD_HELD_OUT_FOR_TESTSET: dict[str, list[str]] = {
    "simple_qa": [
        "Funny how I never remember — diameter of Earth?",
        "Genuine question: how many time zones in Russia?",
        "Curious thought — what bird has the longest wingspan?",
    ],
    "complex_task": [
        "Take me through what's involved in becoming a notary in my state.",
        "Our backups have never been tested — work me through making this bulletproof.",
        "Walk me through the full process of moving across the country with a family of four.",
    ],
    "document_qa": [
        "Anything in there about intellectual property assignment?",
        "It references appendix B — what's in it?",
        "Find me the part where they mention the warranty terms.",
    ],
    "chitchat": [
        "Solid advice, I'll think about it on my walk.",
        "Lmao that's exactly what my wife said.",
        "OK now I'm curious what you do for fun in there.",
    ],
}


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


def build_dataset(
    per_class_easy: int = PER_CLASS_EASY,
    per_class_hard: int = PER_CLASS_HARD,
    seed: int = SEED,
) -> list[dict]:
    """Generate the full balanced dataset.

    Each row has the shape:
        {"text": str, "label": str, "difficulty": "easy" | "hard"}

    `difficulty` is additive — old consumers that read only `text`/`label`
    keep working. The retrocompatibility test in test_dataset.py pins this.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    for label in INTENTS:
        # 1. Easy templated examples (preserves the legacy generation path).
        easy_gen = GENERATORS[label]
        easy_texts = _take_unique(easy_gen(rng), per_class_easy)

        # 2. Hard adversarial examples — deterministic slice of the pool.
        hard_pool = HARD_EXAMPLES[label]
        if len(hard_pool) < per_class_hard:
            raise RuntimeError(
                f"HARD_EXAMPLES[{label!r}] has {len(hard_pool)} entries; "
                f"need at least {per_class_hard}."
            )
        hard_texts = hard_pool[:per_class_hard]

        # 3. Belt-and-suspenders zero-overlap checks. The static lists make
        # collisions unlikely, but a future edit could introduce one.
        overlap_easy_hard = set(easy_texts) & set(hard_texts)
        if overlap_easy_hard:
            raise RuntimeError(
                f"{label}: hard examples collide with easy ones: {overlap_easy_hard}"
            )
        held_out = set(HARD_HELD_OUT_FOR_TESTSET.get(label, []))
        overlap_train_test = set(hard_texts) & held_out
        if overlap_train_test:
            raise RuntimeError(
                f"{label}: training hard pool overlaps with held-out testset: "
                f"{overlap_train_test}"
            )

        for t in easy_texts:
            rows.append({"text": t, "label": label, "difficulty": "easy"})
        for t in hard_texts:
            rows.append({"text": t, "label": label, "difficulty": "hard"})

    rng.shuffle(rows)
    return rows


def regenerate(
    path: Path = DATA_PATH,
    per_class_easy: int = PER_CLASS_EASY,
    per_class_hard: int = PER_CLASS_HARD,
    seed: int = SEED,
) -> Path:
    rows = build_dataset(
        per_class_easy=per_class_easy, per_class_hard=per_class_hard, seed=seed,
    )
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
    print(
        f"Wrote {out} "
        f"({PER_CLASS_EASY} easy + {PER_CLASS_HARD} hard = {PER_CLASS_TARGET} "
        f"per class × {len(INTENTS)} classes)"
    )
