# agents

Multi-agent orchestration on top of a pluggable LLM provider.

## Files

- `agents/providers.py` — `LLMProvider` ABC + `OpenAIProvider`, `AnthropicProvider`, `FakeProvider`. Factory `get_provider(name)`. Defines `ProviderUnavailableError` for missing-credentials (mapped to HTTP 503 by the backend).
- `agents/_json.py` — tolerant JSON extractor for LLM outputs (handles fenced ```json``` blocks, JSON embedded in prose, etc.).
- `agents/planner.py` — `Planner.plan(task, feedback=None) -> list[str]`. Returns subtasks. Robust JSON parsing.
- `agents/executor.py` — `Executor.execute(subtask) -> str`.
- `agents/critic.py` — `Critic.review(task, result) -> Review{approved, feedback}`.
- `agents/orchestrator.py` — `Orchestrator.run_task(task) -> OrchestrationResult{answer, trace, steps}`. Loop: Planner → Executors sequential → aggregate → Critic. Retries once with the Critic's feedback if the first attempt is rejected.

## Constants

- `MAX_STEPS = 6` — caps the Planner's output so a verbose model can't produce 30 nano-steps.
- `MAX_ATTEMPTS = 2` — initial attempt + one retry, per the brief.

## Example

```python
from agents import get_provider, Orchestrator

provider = get_provider("openai")   # uses OPENAI_MODEL env var or "gpt-4o-mini"
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

The fake provider records every call in `self.calls`, which lets tests assert on what each agent prompted (including whether the Critic's feedback was propagated to the Planner on replan).

## Where to parallelize

`Orchestrator._run_task` runs Executors sequentially. The point where to swap in `asyncio.gather(*[run_async(s) for s in subtasks])` is marked in `orchestrator.py` — the Executor is stateless, so the only constraint is preserving input order in the aggregation step.

## Tests

```bash
python -m pytest tests/
```

All tests use `FakeProvider`. No network, no LLM credits spent.
