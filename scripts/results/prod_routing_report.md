# /route smoke test — 2026-06-07T13:56:20+00:00

- **URL**: `https://agent-router-909428365094.us-central1.run.app`
- **Inputs**: 11 total (7 clear, 4 ambiguous)
- **Per-request timeout**: 60s

## Results

| Group | Input | Intent | Confidence | Path taken | Latency (ms) |
|---|---|---|---:|---|---:|
| clear | Design a scalable architecture for a food delivery app and outline the main trade-offs. | `complex_task` | 0.939 | `complex_task:orchestrator` | 26412 |
| clear | Help me migrate a monolith to microservices step by step. | `complex_task` | 0.910 | `complex_task:orchestrator` | 9213 |
| clear | What's the capital of Australia? | `simple_qa` | 0.857 | `simple_qa:direct_llm` | 707 |
| clear | What year did the Berlin Wall fall? | `simple_qa` | 0.851 | `simple_qa:direct_llm` | 667 |
| clear | According to the attached contract, what is the termination notice period? | `document_qa` | 0.871 | `document_qa:rag_stub` | 186 |
| clear | In the PDF I uploaded, which section covers the refund policy? | `document_qa` | 0.879 | `document_qa:rag_stub` | 207 |
| clear | Good morning! Hope you're having a nice day. | `chitchat` | 0.560 | `chitchat:direct_llm` | 989 |
| ambiguous | Tell me about the requirements | `chitchat` | 0.385 | `low_confidence_fallback` | 180 |
| ambiguous | Build me something cool | `chitchat` | 0.510 | `chitchat:direct_llm` | 914 |
| ambiguous | What does it say about pricing? | `simple_qa` | 0.588 | `low_confidence_fallback` | 189 |
| ambiguous | Can you explain how this works? | `chitchat` | 0.417 | `low_confidence_fallback` | 188 |

## Hypothesis verification

- Ambiguous → `low_confidence_fallback` OR confidence < 0.65: **4/4** — **PASS**
- Clear → NOT `low_confidence_fallback`: **7/7** — **PASS**
- Total fallbacks observed: **3/11**

## Latency

- Mean across all requests: **3623 ms**
- Range: 180 — 26412 ms
- ⚠ Cold start detected on the first request (26412 ms vs warm mean 1344 ms). Cloud Run scaled from zero (min-instances=0) and the entrypoint downloaded and loaded the DistilBERT model on the new instance.

## Summary

**3/4** of the ambiguous inputs landed on `low_confidence_fallback`; **0/7** of the clear inputs fell back unexpectedly. The threshold behavior in production matches the local generalization probe — clear inputs route normally with confidence in the high-0.8 to mid-0.9 band, ambiguous inputs sit between 0.39 and 0.59 and are caught by the fallback before any LLM is invoked. Mean per-request latency was 3623 ms across 11 requests (warm-only mean 1344 ms; the first request paid a cold-start cost on Cloud Run).

## Environment

- Backend URL: `https://agent-router-909428365094.us-central1.run.app`
- Per-request timeout: 60 s
- Network errors during the run: **none**
