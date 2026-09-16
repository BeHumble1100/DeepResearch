# Current State

## Project Status

DeepResearch V1 has an implemented, test-covered research harness. Phase 8 offline evaluation has not started.

## Completed

- Core LangGraph Research Loop: `Question → Initialization → Planner → SEARCH / OPEN / LOCATE → Evidence Update → ANSWER → Guard`.
- Structured initialization and provider-neutral LLM planner contracts.
- SearXNG Search Gateway with query rewrite, asynchronous requests, retry/timeout, normalization, and URL deduplication.
- HTML/PDF document opening, with parsed content retained outside Planner context.
- Document-local retrieval: chunking, BM25 recall, and LLM semantic reranking.
- Structured Fact Extraction, resolved-entity provenance, and deterministic Answer Guard.
- Lightweight serializable Research Trace, including planner context snapshots, bounded search results, action rejections, and Guard outcomes.
- DeepSeek OpenAI-compatible adapter and real structured API smoke coverage.
- Prompt Pack V1 and Calibration V2.
- Discovered-URL allow-list, recoverable action rejection, exact duplicate SEARCH/LOCATE rejection, and step-budget enforcement.
- Candidate Evidence Scope Contract V1: deterministic candidate scopes, scoped document/fact provenance, mutually exclusive Guard modes, final-only global constraint projection, and scope-creation Trace records.

## Current Focus

Evidence Scope Contract V1 is implemented. The next validation task is a real regression comparison for `question.jsonl` `id=1` and `id=6`.

The contract addresses candidate cross-contamination:

- A candidate-local contradiction does not permanently mark a global required constraint as `contradicted`.
- Positive evidence from different candidate scopes cannot be combined into one accepted answer.
- Guard accepts evidence only from the selected candidate scope, or only unscoped evidence in global/direct mode.

## Next Steps

1. Run real `id=1` and `id=6` regression smoke tests with the scope contract.
2. Compare traces with prior runs, especially candidate binding, OPEN/LOCATE behavior, constraint evidence, and Guard outcomes.
3. Begin a small Phase 8 evaluation pilot only after the regressions are understood.
4. Build the reproducible 100-question evaluation runner and trace-derived diagnostics.

## Notes

- `specs/deepresearch-v1.md` remains the V1 design baseline.
- Implemented contracts, especially Candidate Evidence Scope V1, refine the original spec to address observed bad cases.
- Do not report benchmark accuracy until Phase 8 produces measured results.
