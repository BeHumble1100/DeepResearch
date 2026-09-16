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
- Inspect existing code paths before editing.
- Do not perform unrelated refactors.
- Do not perform unrelated dependency upgrades.
- Prefer explicit data contracts over abstractions for hypothetical future needs.
- Keep important research decisions observable.
- Do not silently swallow unexpected exceptions.
- Reproduce bugs before fixing.
- Verify behavior with the narrowest meaningful test or smoke test.
- Once requested behavior works and verification passes, stop.
- Do not add new LLM calls, agents, orchestration layers, retrieval layers, databases, queues, or infrastructure dependencies unless explicitly required by the task.
- Do not expand the SEARCH / OPEN / LOCATE / ANSWER action space without explicit approval.
- Do not substantially rewrite production prompts without observed Bad Case or Trace evidence.
- Drive Agent, Prompt, Guard, and Search optimization from reproducible failures, not hypothetical improvement.
- Do not weaken evidence, provenance, or deterministic guardrails to improve apparent benchmark performance.
- If spec and implementation disagree, inspect the current code path first and report the discrepancy before changing architecture.

## Autonomous Iteration

For implementation, debugging, and evaluation tasks, work iteratively without waiting for confirmation after every small step.

Default loop:

1. Inspect current implementation and relevant contracts.
2. Reproduce the issue or establish a baseline.
3. Form the smallest testable hypothesis.
4. Implement the smallest complete fix.
5. Run the narrowest meaningful test.
6. Run relevant regression tests.
7. Inspect traces or results.
8. If acceptance criteria are not met, repeat from step 3.
9. Once behavior works and regressions pass, stop.

Ordinary implementation and debugging iterations do not require user confirmation.

Stop and request confirmation before:

- a core architecture change;
- a major data-contract change outside the current task;
- a new LLM call, agent, retrieval layer, storage, or infrastructure dependency;
- an action-space change;
- a substantial production-prompt rewrite;
- a benchmark definition, metric, or ground-truth change;
- a workaround that weakens provenance or evidence guarantees;
- an unrelated refactor.

## Bad Case Optimization

Use this loop:

```text
reproduce
→ inspect Trace
→ classify failure layer
→ identify the smallest responsible contract or component
→ minimal fix
→ replay the same case
→ broader regression
```

Classify failures as one of:

- search miss;
- query or rewrite;
- source selection or OPEN;
- document retrieval or LOCATE;
- passage rerank;
- fact extraction;
- entity or candidate resolution;
- planner or action choice;
- evidence-state contract;
- answer guard;
- format or output;
- external backend failure.

Do not:

- add special rules for one question;
- keep overfitting the same case after a local fix;
- add further heuristics before running broader regression.

## Verification

Pytest is configured with `tests/` as its test path.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q tests/test_graph.py
git diff --check
```

There is no committed canonical real smoke or regression runner yet. Historical scripts and outputs under `eval/answers/` are experiment artifacts, not standard project commands.

## Documentation

- Do not put temporary experiment results, Bad Cases, or benchmark outputs in this file.
