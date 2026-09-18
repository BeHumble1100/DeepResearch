# Current State

## Project Status

DeepResearch V1 has an implemented, test-covered research harness and a completed Phase 8 baseline evaluation.

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
- Initialization requires every proposed constraint to be explicitly classified and keeps acceptance constraints answer-neutral, preventing a guessed target value from becoming its own evidence requirement.
- Discovered-URL allow-list, recoverable action rejection, exact duplicate SEARCH/LOCATE rejection, and step-budget enforcement.
- Candidate Evidence Scope Contract V1: deterministic candidate scopes, scoped document/fact provenance, mutually exclusive Guard modes, final-only global constraint projection, and scope-creation Trace records.
- Phase 8 evaluation runner: failure-isolated sequential execution, per-question state/Trace artifacts, conservative normalized exact-match scoring, and trace-derived aggregate metrics.
- Phase 9 API Layer: FastAPI `POST /research` endpoint wired to the existing production graph, with a compact terminal-state response that omits fetched document text and fact passages.
- Recovery for source-side OPEN failures and deterministic exact duplicate-OPEN rejection.
- Source-aware OPEN recovery: structured fetch/parse failure categories in Trace, access-denied host avoidance, and deterministic candidate ordering that demotes common non-documentary material without filtering URLs.
- Evidence relation calibration: rank or comparison evidence about a different entity no longer contradicts an unresolved answer constraint; Planner receives each known Fact's compact per-constraint evidence status.
- Candidate visibility boundary: Planner context excludes URLs already attempted or belonging to an access-denied host, and the graph deterministically rejects further OPEN attempts to those hosts without making another request.
- Query-progress control: Planner sees the current consecutive no-evidence SEARCH streak; Query Rewrite receives prior planner queries and deterministically excludes exact historical rewrites.
- Document-local retrieval context keeps every opened document locatable with its prior retrieval progress, so a productive LOCATE can be refined with a new query instead of causing a duplicate OPEN attempt.
- Real 100-question baseline run with DeepSeek, local SearXNG, and `max_steps=10`: 96 completed runs, 4 isolated `LLMClientError` runs, and 0/100 exact or normalized matches. This is the measured baseline, not a target claim.

## Current Focus

The baseline evaluation shows a systemic research-efficiency problem, not a finish-control bypass: every completed question exhausted its budget and no answer was Guard-accepted.

The Candidate Evidence Scope contract addresses cross-contamination:

- A candidate-local contradiction does not permanently mark a global required constraint as `contradicted`.
- Positive evidence from different candidate scopes cannot be combined into one accepted answer.
- Guard accepts evidence only from the selected candidate scope, or only unscoped evidence in global/direct mode.

Observed Phase 8 Trace patterns requiring evidence-driven follow-up:

- SEARCH is overused relative to LOCATE (563 SEARCH vs. 22 LOCATE actions across the full run).
- The Planner repeatedly proposes already-attempted OPEN URLs despite deterministic rejection (99 duplicate-OPEN rejections).
- Unsupported ANSWER proposals and missing scoped constraint support remain common; raising the budget from 10 to 15 steps on the same 10-question pilot did not improve accuracy.
- Real-web availability remains variable: an otherwise usable candidate can still fail due to upstream access denial, timeout, or non-document content.

## Next Steps

1. Classify recurrent Planner/source-selection and evidence-progression failures from the saved Phase 8 traces.
2. Propose a minimal, trace-backed Calibration V3 only if it addresses a repeated failure without weakening provenance or Guard rules.
3. Replay the same pilot and full benchmark only after a validated systemic change; do not tune individual questions.

## Notes

- `specs/deepresearch-v1.md` remains the V1 design baseline.
- Implemented contracts, especially Candidate Evidence Scope V1, refine the original spec to address observed bad cases.
- The baseline full-run artifact is an ignored experiment directory under `eval/answers/`; rerun it with `python -m eval.runner` and an explicit output directory rather than overwriting prior results.
