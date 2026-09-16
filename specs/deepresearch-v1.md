# DeepResearch V1 Design Spec

**Status:** Design Freeze  
**Version:** V1  
**Purpose:** Implementation source of truth for the first complete version of DeepResearch.

---

## 1. Overview

DeepResearch is an **Evidence-driven Multi-hop Web Research Agent** evolved from an earlier ResearchAgent built for the Alibaba Cloud PAI developer competition.

The original version used a relatively fixed:

```text
Planner -> Research -> Answer
```

flow.

V1 is redesigned for factual research tasks where key entities are often unknown at the start and must be resolved progressively through search, document reading, and evidence verification.

Typical task characteristics:

- cross-source Web search;
- multi-hop entity resolution;
- multi-condition disambiguation;
- HTML/PDF deep reading;
- evidence-backed answer generation;
- exact or short-form factual answers rather than long research reports.

This project is **not** positioned as:

- a traditional knowledge-base RAG system;
- a generic report-writing DeepResearch workflow;
- a Multi-Agent system.

---

## 2. Goals

V1 should support:

1. parsing a question into a target and verifiable constraints;
2. maintaining explicit research state across multiple hops;
3. dynamically choosing the next research action from current evidence;
4. open-Web retrieval through SearXNG;
5. deep reading of retrieved HTML/PDF documents;
6. document-local retrieval for long documents;
7. structured evidence extraction;
8. deterministic finish checks before accepting an answer;
9. complete research tracing;
10. offline evaluation on a 100-question Chinese/English multi-hop benchmark.

---

## 3. Non-goals

Do **not** add the following to V1 unless later evaluation proves they are necessary:

- Multi-Agent architecture;
- GraphRAG;
- Neo4j or another graph database;
- vector database;
- embedding retrieval pipeline;
- hybrid retrieval / RRF;
- dedicated cross-encoder reranker;
- MCP;
- Redis;
- Celery;
- Kafka;
- long-term memory;
- human-in-the-loop;
- complex LangGraph checkpoint/resume infrastructure;
- microservice decomposition.

V1 should remain strong, testable, and explainable.

---

## 4. Key Design Decisions

### D1. Dynamic planning instead of one-shot planning

Do not generate the full search route at the beginning.

The system should repeatedly:

```text
inspect current state
-> choose the most useful unresolved goal
-> execute one action
-> update evidence and constraints
-> plan again
```

Reason: many later queries depend on entities discovered in earlier hops.

---

### D2. Research state is explicit

The agent should not rely only on conversation history.

The planner should receive a compact structured state containing:

- target;
- constraints;
- resolved entities;
- known facts;
- available documents;
- recent actions;
- remaining budget.

---

### D3. Planner action space stays small

V1 exposes only:

```text
SEARCH
OPEN
LOCATE
ANSWER
```

Fact extraction is an internal capability triggered after OPEN/LOCATE and is not a separate planner action.

---

### D4. Evidence is represented as structured Fact objects

V1 uses `list[Fact]` as the evidence model.

Do not introduce a separate evidence graph database.

Facts should retain source URL and supporting passage so they remain traceable.

---

### D5. Long-document retrieval uses BM25 + LLM semantic rerank

For an opened HTML/PDF document:

```text
parse
-> chunk
-> BM25 Top-K recall
-> LLM semantic rerank
-> Top-N passages
-> structured fact extraction
```

This is an on-demand document retrieval stage, not a pre-built knowledge-base RAG system.

---

### D6. Code owns termination

The model may propose `ANSWER`, but deterministic code decides whether the research loop may end.

Core rule:

> The model proposes the answer; code controls termination.

---

## 5. High-level Architecture

```text
User Question
    |
    v
Question Initialization
# Target + Constraints
    |
    v
ResearchState
    |
    v
Planner
    |
    +------ SEARCH ------> SearXNG
    |
    +------ OPEN --------> HTML / PDF parse
    |
    +------ LOCATE ------> BM25 -> LLM rerank -> Passage
    |
    +------ ANSWER ------> Answer Guard
                              |
                      reject  |  accept
                              |
                          Planner / END
```

### Responsibility boundary

**Research Harness** owns domain logic:

- ResearchState;
- Target / Constraint / Fact;
- planner semantics;
- action contracts;
- retrieval behavior;
- finish control;
- tracing;
- evaluation.

**LangGraph** provides orchestration:

- state transitions;
- nodes;
- conditional edges;
- research loop execution.

LangGraph is an implementation mechanism, not the project's core identity.

---

## 6. Core Data Contracts

### 6.1 Target

Represents what the system must finally answer.

```python
class Target:
    description: str
    answer_type: str
    format_instruction: str | None
```

Example:

```text
description: dissertation author's son's first name
answer_type: person_name
format_instruction: first name only
```

---

### 6.2 Constraint

Represents a condition used to identify or verify the answer path.

```python
class Constraint:
    id: str
    description: str

    subject: str | None
    predicate: str | None
    object: str | None

    required: bool

    status: Literal["unknown", "supported", "contradicted"]

    supporting_fact_ids: list[str]
```

Keep `description` even when structured fields are available because not every clue requires a formal ontology.

---

### 6.3 Fact

Represents a verifiable claim plus its source evidence.

```python
class Fact:
    id: str

    statement: str

    subject: str | None
    predicate: str | None
    object: str | None

    source_url: str
    document_id: str | None

    passage: str

    confidence: float
    supports_constraints: list[str]
```

A fact should remain inspectable and traceable to source content.

---

### 6.4 DocumentRef

Represents an opened resource.

```python
class DocumentRef:
    id: str

    url: str
    title: str | None
    content_type: str

    local_path: str | None
    summary: str | None
```

Do not repeatedly inject full document text into planner prompts.

---

### 6.5 ResearchState

Keep the state small.

```python
class ResearchState:
    question: str

    target: Target | None
    constraints: list[Constraint]

    resolved_entities: dict[str, str]
    facts: list[Fact]

    documents: list[DocumentRef]
    executed_queries: list[str]
    visited_urls: list[str]

    current_goal: str | None

    step_count: int
    max_steps: int

    answer: str | None
    status: str
```

Do not add HypothesisStore / CandidateManager / EvidenceGraph / MemoryManager in V1.

---

## 7. Planner Contract

The planner has two responsibilities.

### 7.1 Initialization

Input:

```text
Question
```

Output:

```text
Target
+
Constraints
```

Initialization should **not** produce a complete multi-step search plan.

---

### 7.2 Next-action planning

The planner receives a compact state view.

Recommended planner context:

```text
Original Question
Target
Resolved Entities
Constraint Status
Known Facts
Available Documents
Recent Actions
Remaining Step Budget
```

Do not send all fetched page content or the full historical trace every round.

---

## 8. Action Contracts

### 8.1 SEARCH

Purpose: discover Web resources.

```python
class SearchAction:
    type: Literal["search"]
    goal: str
    query: str
```

---

### 8.2 OPEN

Purpose: fetch and parse a known URL.

```python
class OpenAction:
    type: Literal["open"]
    goal: str
    url: str
```

Guardrail:

The URL must come from search results or already known documents.

Do not trust arbitrary model-generated URLs.

---

### 8.3 LOCATE

Purpose: retrieve relevant passages inside an opened document.

```python
class LocateAction:
    type: Literal["locate"]
    goal: str
    document_id: str
    query: str
```

LOCATE is document-local retrieval, not Web search.

---

### 8.4 ANSWER

Purpose: propose the final answer.

```python
class AnswerAction:
    type: Literal["answer"]
    answer: str
    supporting_fact_ids: list[str]
    supporting_constraint_ids: list[str]
```

ANSWER is a proposal, not an automatic termination signal.

---

## 9. Search Design

### 9.1 Web search pipeline

Use SearXNG as the unified Search Gateway.

```text
Research Goal
-> Query Rewrite
-> SearXNG
-> Multi-engine Results
-> Normalize
-> URL Dedup
-> Planner selects what to OPEN
```

V1 Search Gateway should support:

- query rewrite;
- multi-engine aggregation;
- async concurrency;
- timeout;
- retry;
- result normalization;
- URL deduplication.

Do not add a dedicated Web-level reranker in V1.

Search snippets may guide which result to OPEN but should not be treated as final high-confidence evidence.

---

## 10. Document-local Retrieval

For opened HTML/PDF documents:

```text
HTML / PDF
-> Parse
-> Chunk
-> BM25 Top-K
-> LLM Semantic Rerank
-> Top-N Passage
-> Fact Extraction
```

### BM25 role

BM25 is the low-cost lexical recall stage.

### LLM rerank role

The LLM reranks only a small candidate set against the current Research Goal.

This covers semantic mismatch cases without adding embedding/index infrastructure in V1.

---

## 11. Fact Extraction and State Update

OPEN and LOCATE may trigger structured fact extraction automatically.

New facts can update:

- constraint status;
- resolved entities;
- planner context.

Example:

```text
Constraint:
author -- degree_from --> University of San Diego
status = unknown

Fact:
Matt Wytock -- degree_from --> University of San Diego

After validation:
constraint status -> supported
```

The planner should prefer unresolved constraints and avoid repeatedly researching already-supported ones.

---

## 12. Answer Guard

Before accepting `ANSWER`, validate at least:

1. every referenced `supporting_fact_id` exists;
2. key constraints have sufficient support;
3. no required constraint is fatally contradicted;
4. supporting evidence comes from opened/located content, not only search snippets;
5. answer format matches the Target.

If validation fails:

```text
ANSWER rejected
-> Planner
-> continue research
```

If validation passes:

```text
END
```

---

## 13. Research Trace

Record each research step in structured form.

Minimum fields:

```text
step
current_goal
action
action_input
observation_summary
new_facts
constraint_changes
resolved_entities
```

Use the trace for:

- debugging;
- replay;
- bad-case analysis;
- evaluation.

Keep one trace model rather than multiple recorder abstractions.

---

## 14. Evaluation

Dataset:

- 100 Chinese/English multi-hop Web research questions.

Evaluation flow:

```text
Question
-> Research Harness
-> Final Answer
-> Normalize
-> Exact / normalized match
```

### Primary metric

```text
End-to-End Answer Accuracy
```

### Auxiliary metrics

Derive from Research Trace where useful:

- average research steps;
- search count;
- open count;
- locate count;
- constraint coverage;
- failure category.

Suggested failure taxonomy:

- Search miss;
- Entity resolution error;
- Document locate error;
- Evidence extraction error;
- Planning/reasoning error;
- Early stop;
- Answer formatting error.

---

## 15. LLM Boundary

Business logic must not depend on one provider.

Recommended abstraction:

```python
class LLMClient:
    async def structured(self, messages, schema, **kwargs):
        ...

    async def text(self, messages, **kwargs):
        ...
```

V1 may use one model for all LLM calls.

Model specialization can be introduced later only if evaluation shows a clear need.

---

## 16. Recommended Repository Structure

```text
app/
├── api/
│   └── research.py
│
├── research/
│   ├── graph.py
│   ├── planner.py
│   ├── state.py
│   ├── schemas.py
│   ├── guard.py
│   └── trace.py
│
├── tools/
│   ├── search.py
│   ├── document.py
│   └── retrieval.py
│
├── llm/
│   └── client.py
│
├── config.py
└── main.py

eval/
├── runner.py
├── metrics.py
├── analyze.py
└── datasets/
    └── question.jsonl

tests/
```

Avoid adding generic layers such as:

```text
strategies/
judges/
verifiers/
repositories/
managers/
processors/
```

unless concrete complexity justifies them.

---

## 17. Implementation Plan

### Phase 1 — State & Graph Skeleton

Deliver:

- minimal project skeleton;
- Pydantic schemas;
- ResearchState;
- Action schemas;
- LangGraph skeleton;
- mock planner;
- basic unit tests.

Exit condition:

A mocked question can move through the graph with valid state transitions.

---

### Phase 2 — Planner & Minimal Research Loop

Deliver:

- structured planner output;
- SEARCH / OPEN / ANSWER loop;
- minimal state updates;
- mock end-to-end research flow.

Exit condition:

The planner can repeatedly choose actions from current state without one-shot planning.

---

### Phase 3 — SearXNG Search Gateway

Deliver:

- SearXNG integration;
- query rewrite;
- async search;
- timeout/retry;
- normalization;
- URL dedup.

Exit condition:

SEARCH returns stable normalized results.

---

### Phase 4 — Document Opening

Deliver:

- HTML fetch/parse;
- PDF fetch/parse;
- DocumentRef;
- document cache/storage.

Exit condition:

OPEN can produce a reusable parsed document representation.

---

### Phase 5 — Document-local Retrieval

Deliver:

- chunking;
- BM25 recall;
- LLM semantic rerank;
- LOCATE action.

Exit condition:

LOCATE can return goal-relevant passages from a long document.

---

### Phase 6 — Evidence & Finish Control

Deliver:

- structured Fact extraction;
- constraint updates;
- resolved entity updates;
- Answer Guard.

Exit condition:

An unsupported ANSWER is rejected and a supported ANSWER may terminate the graph.

---

### Phase 7 — Research Trace

Deliver:

- structured step trace;
- replay/debug-friendly storage.

Exit condition:

A complete research path can be inspected after execution.

---

### Phase 8 — Offline Evaluation

Deliver:

- benchmark runner;
- answer normalization;
- primary accuracy metric;
- trace-derived diagnostics;
- failure analysis.

Exit condition:

The 100-question benchmark can run reproducibly and produce real metrics.

---

### Phase 9 — API Layer

Deliver:

- FastAPI research endpoint;
- optional progress streaming.

Exit condition:

The complete research workflow is accessible through an API.

---

## 18. Change Control

This file is the V1 source of truth.

If implementation reveals a real design problem:

1. describe the problem;
2. explain why the current design fails;
3. propose the smallest change;
4. approve the change;
5. update this spec;
6. then update code.

Do not allow implementation convenience to silently redefine architecture.

---

