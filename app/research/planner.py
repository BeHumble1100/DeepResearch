"""Planner interfaces and a deterministic Phase 1 mock implementation."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, Field

from app.llm.client import LLMClient, Message

from .schemas import (
    Action,
    ActionDecision,
    AnswerAction,
    Constraint,
    ConstraintProposal,
    OpenAction,
    LocateAction,
    SearchAction,
    Target,
)
from .state import ResearchState


class Initialization(BaseModel):
    target: Target
    constraints: list[Constraint] = Field(default_factory=list)


class InitializationProposal(BaseModel):
    """LLM initialization output before deterministic state normalization."""

    target: Target
    constraints: list[ConstraintProposal] = Field(default_factory=list)


class Planner(Protocol):
    async def initialize(self, question: str) -> Initialization: ...

    async def next_action(self, state: ResearchState) -> Action: ...


class MockPlanner:
    """Deterministic planner that chooses one action from the current state."""

    async def initialize(self, question: str) -> Initialization:
        return Initialization(target=Target(description=question, answer_type="text"))

    async def next_action(self, state: ResearchState) -> Action:
        if not state.executed_queries:
            return SearchAction(
                goal="Find an initial source for the research question.",
                query=state.question,
            )
        if state.search_results and not state.documents:
            result = state.search_results[0]
            return OpenAction(
                goal="Inspect the first discovered source.",
                url=result.url,
            )
        if state.documents and not state.located_passages:
            return LocateAction(
                goal="Locate the most relevant passages in the opened source.",
                document_id=state.documents[0].id,
                query=state.question,
            )
        if state.located_passages:
            return AnswerAction(
                answer="Mock answer",
                supporting_fact_ids=[fact.id for fact in state.facts],
                supporting_constraint_ids=[],
            )
        raise ValueError("Mock planner has no available action for the current state.")


class LLMPlanner:
    """Planner that requests one validated action from a provider-neutral client."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def initialize(self, question: str) -> Initialization:
        messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    "Only model the question. Do not search, answer, infer the final answer, or "
                    "create a research plan. Return one target and atomic, independently "
                    "verifiable constraints. Each constraint must describe one checkable condition."
                ),
            },
            {"role": "user", "content": question},
        ]
        proposal = await self._client.structured(messages=messages, schema=InitializationProposal)
        return _materialize_initialization(proposal)

    async def next_action(self, state: ResearchState) -> Action:
        messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    "Choose exactly one next research action from the supplied state. Prioritize "
                    "unresolved required constraints and avoid repeated queries, URLs, or document "
                    "locates. SEARCH only when a useful resource is not yet available. OPEN only a "
                    "URL from search results or known documents. LOCATE only an opened document when "
                    "document evidence is needed. ANSWER only when required constraints are supported "
                    "by existing facts and cite their IDs. Search snippets are not final evidence. Do "
                    "not create a multi-step plan. Return one structured SEARCH, OPEN, LOCATE, or "
                    "ANSWER proposal."
                ),
            },
            {"role": "user", "content": _compact_state_view(state)},
        ]
        decision = await self._client.structured(messages=messages, schema=ActionDecision)
        return decision.action


def _compact_state_view(state: ResearchState) -> str:
    """Serialize only the planner context allowed by the V1 specification."""

    context = {
        "original_question": state.question,
        "current_goal": state.current_goal,
        "target": state.target.model_dump() if state.target else None,
        "resolved_entities": state.resolved_entities,
        "constraints": [
            {
                "id": constraint.id,
                "description": constraint.description,
                "required": constraint.required,
                "status": constraint.status,
            }
            for constraint in state.constraints
        ],
        "known_facts": [
            {
                "id": fact.id,
                "statement": fact.statement,
                "supports_constraints": fact.supports_constraints,
            }
            for fact in state.facts
        ],
        "available_documents": [
            {
                "id": document.id,
                "url": document.url,
                "title": document.title,
                "content_type": document.content_type,
                "summary": document.summary,
            }
            for document in state.documents
        ],
        "search_results": [
            {"url": result.url, "title": result.title, "snippet": result.snippet}
            for result in state.search_results
        ],
        "recent_actions": {
            "executed_queries": state.executed_queries,
            "visited_urls": state.visited_urls,
        },
        "remaining_step_budget": state.max_steps - state.step_count,
    }
    return json.dumps(context, ensure_ascii=False)


def _materialize_initialization(proposal: InitializationProposal) -> Initialization:
    """Assign deterministic state-owned IDs and initial fields to unique constraints."""
    constraints: list[Constraint] = []
    seen: set[tuple[str, str | None, str | None, str | None, bool]] = set()
    for candidate in proposal.constraints:
        normalized = _normalize_constraint(candidate)
        key = (
            normalized.description,
            normalized.subject,
            normalized.predicate,
            normalized.object,
            normalized.required,
        )
        if key in seen:
            continue
        seen.add(key)
        constraints.append(
            Constraint(
                id=f"c{len(constraints) + 1}",
                description=normalized.description,
                subject=normalized.subject,
                predicate=normalized.predicate,
                object=normalized.object,
                required=normalized.required,
                status="unknown",
                supporting_fact_ids=[],
            )
        )
    return Initialization(target=proposal.target, constraints=constraints)


def _normalize_constraint(candidate: ConstraintProposal) -> ConstraintProposal:
    return ConstraintProposal(
        description=candidate.description.strip(),
        subject=_normalize_optional(candidate.subject),
        predicate=_normalize_optional(candidate.predicate),
        object=_normalize_optional(candidate.object),
        required=candidate.required,
    )


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip() or None
