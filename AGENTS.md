# DeepResearch

## Project

DeepResearch is an evidence-driven multi-hop web research agent.

```text
Question
→ Initialization
→ Planner
→ SEARCH / OPEN / LOCATE
→ Evidence Update
→ ANSWER
→ Deterministic Guard
```

## Sources of Truth

- Architecture design: `specs/deepresearch-v1.md`
- Current code takes precedence over stale documentation.

## Development Rules

- Implement the smallest complete change that satisfies the current requirement.
- Keep each change focused on one capability or bug.
- Inspect the relevant existing code path before editing.
- Do not perform unrelated refactors or dependency upgrades.
- Prefer clear data contracts; avoid abstractions for hypothetical future needs.
- Keep important research decisions observable.
- Do not silently swallow unexpected exceptions.
- Verify behavior with the narrowest meaningful test or smoke test.
- For bugs, reproduce before fixing.
- Once the requested behavior works and verification passes, stop making unrelated changes.
- Do not add LLM calls, orchestration layers, retrieval components, or abstractions unless explicitly required.
- Drive Agent or Prompt behavior changes from observed traces or reproducible bad cases.
- Before architecture or data-contract changes, present the minimal design and inspect existing contracts first.

## Verification

Pytest is configured to use `tests/` as its test path.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```
