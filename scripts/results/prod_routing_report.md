# /route smoke test — 2026-06-06T18:47:09+00:00

- **URL**: `https://agent-router-909428365094.us-central1.run.app`
- **Inputs**: 11 total (7 clear, 4 ambiguous)
- **Per-request timeout**: 60s

## Results

| Group | Input | Intent | Confidence | Path taken | Latency (ms) |
|---|---|---|---:|---|---:|
| clear | Design a scalable architecture for a food delivery app and outline the main trade-offs. | — | — | **ERROR**: timeout after 60s | 60179 |
| clear | Help me migrate a monolith to microservices step by step. | `complex_task` | 0.910 | `complex_task:orchestrator` | 31735 |
| clear | What's the capital of Australia? | `simple_qa` | 0.857 | `simple_qa:direct_llm` | 720 |
| clear | What year did the Berlin Wall fall? | `simple_qa` | 0.851 | `simple_qa:direct_llm` | 627 |
| clear | According to the attached contract, what is the termination notice period? | `document_qa` | 0.871 | `document_qa:rag_stub` | 210 |
| clear | In the PDF I uploaded, which section covers the refund policy? | `document_qa` | 0.879 | `document_qa:rag_stub` | 230 |
| clear | Good morning! Hope you're having a nice day. | `chitchat` | 0.560 | `low_confidence_fallback` | 223 |
| ambiguous | Tell me about the requirements | `chitchat` | 0.385 | `low_confidence_fallback` | 205 |
| ambiguous | Build me something cool | `chitchat` | 0.510 | `low_confidence_fallback` | 214 |
| ambiguous | What does it say about pricing? | `simple_qa` | 0.588 | `low_confidence_fallback` | 214 |
| ambiguous | Can you explain how this works? | `chitchat` | 0.417 | `low_confidence_fallback` | 223 |

## Hypothesis verification

- Ambiguous → `low_confidence_fallback` OR confidence < 0.65: **4/4** — **PASS**
- Clear → NOT `low_confidence_fallback`: **5/6** — **FAIL**
- Total fallbacks observed: **5/11**

## Latency

- Mean across all requests: **3460 ms**
- Range: 205 — 31735 ms
- ⚠ Cold start detected on the first request (31735 ms vs warm mean 319 ms). Cloud Run scaled from zero (min-instances=0) and the entrypoint downloaded and loaded the DistilBERT model on the new instance.

## Summary

**4/4** of the ambiguous inputs landed on `low_confidence_fallback`; **1/7** of the clear inputs fell back unexpectedly. There are divergences from the local generalization probe — check the table above for the specific inputs and confidences. The behavior may be intentional (legitimate chitchat naturally tops out around 0.56 and can fall into the fallback) or worth tightening per-class thresholds for. Mean per-request latency was 3460 ms across 11 requests (warm-only mean 319 ms; the first request paid a cold-start cost on Cloud Run).

## Environment

- Backend URL: `https://agent-router-909428365094.us-central1.run.app`
- Per-request timeout: 60 s
- Network errors during the run: **1**
  - `[clear] 'Design a scalable architecture for a food delivery app and outline the main trade-offs.': timeout after 60s`
