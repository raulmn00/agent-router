# agents

Multi-agent orchestration (Planner / Executor / Critic) on top of a pluggable LLM provider abstraction.

## Files

- `agents/providers.py`
  - `LLMProvider` ABC with `complete(messages, max_tokens) -> str`.
  - `OpenAIProvider`, `AnthropicProvider`, `FakeProvider` implementations + `get_provider(name)` factory.
  - `ProviderUnavailableError` — typed exception raised when credentials are missing. The backend maps it to HTTP 503 with a generic message that never leaks the env var name.
  - `OpenAIProvider.complete_with_logprobs(messages, max_tokens) -> tuple[str, list[float]]` — optional method used by `eval.llm_router.LLMRouter` to derive a confidence score from per-token logprobs. Providers that don't expose logprobs can simply omit this method; callers check with `getattr`.
- `agents/_json.py` — tolerant JSON extractor for LLM outputs (handles fenced ```json``` blocks, JSON embedded in prose, etc.).
- `agents/planner.py` — `Planner.plan(task, feedback=None) -> list[str]`.
- `agents/executor.py` — `Executor.execute(subtask) -> str`.
- `agents/critic.py` — `Critic.review(task, result) -> Review{approved, feedback}`.
- `agents/orchestrator.py` — `Orchestrator.run_task(task) -> OrchestrationResult{answer, trace, steps}`. Loop: Planner → Executors sequential → aggregate → Critic. Retries once with the Critic's feedback if rejected.

## Constants

- `MAX_STEPS = 6` — caps the Planner's output so a verbose model can't produce 30 nano-steps.
- `MAX_ATTEMPTS = 2` — initial attempt + one retry, per design.
- `DEFAULT_MAX_CONCURRENT_EXECUTORS = 5` — `asyncio.Semaphore` cap on parallel Executor calls within a single attempt. Configurable via the `MAX_CONCURRENT_EXECUTORS` env var.

## Example

```python
from agents import get_provider, Orchestrator

provider = get_provider("openai")  # uses OPENAI_MODEL env var or "gpt-4o-mini"
result = Orchestrator(provider).run_task("Design an end-to-end MLOps pipeline.")
print(result.answer)
print(result.trace)
print(result.steps)
```

For testing, use `FakeProvider`:

```python
from agents import FakeProvider, Orchestrator

fake = FakeProvider(responses=[
    '{"steps": ["analyze", "draft"]}',
    "analysis text",
    "draft text",
    '{"approved": true, "feedback": ""}',
])
result = Orchestrator(fake).run_task("anything")
```

The fake provider records every call in `self.calls`, which lets tests assert on what each agent prompted (including that the Critic's feedback was propagated to the Planner on replan).

## Parallel execution

`Orchestrator.run_task_async` runs the Executors of a single attempt **concurrently** via `asyncio.gather` + `asyncio.Semaphore`. The Planner and Critic remain sequential (one call each), and the retry loop is still sequential — only the N subtasks inside an attempt fan out.

Concurrency mechanism: each Executor call goes through `Executor.execute_async`, which wraps the sync `execute()` in `asyncio.to_thread`. We chose `to_thread` over `AsyncOpenAI` / `AsyncAnthropic` deliberately — the `LLMProvider` ABC stays sync (no fork of the provider hierarchy), and the GIL releases during the SDK's underlying `httpx` socket I/O, so threads actually run concurrently for the LLM roundtrip. The thread overhead is irrelevant next to ~500 ms-2 s of API latency.

Failure isolation: subtasks run with `return_exceptions=True`. A single subtask raising doesn't lose the others — its slot in the aggregated answer carries an `[execution failed: ...]` marker and the trace records `EXEC[i] FAILED ...`.

The public `Orchestrator.run_task(task)` keeps its sync signature so the existing FastAPI dispatcher (running in Starlette's threadpool) keeps working without changes — it internally calls `asyncio.run(run_task_async(task))`. If you compose this into a fully-async stack, call `await orchestrator.run_task_async(task)` and drop the sync wrapper cost.

## Tests

```bash
python -m pytest tests/
```

**15 tests**, all using `FakeProvider`. No network, no LLM credits.
