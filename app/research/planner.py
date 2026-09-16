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
                    "Model only the research target and evidence requirements implied by the question.\n\n"
                    "Do not search, answer the question, infer the final answer, or create a multi-step "
                    "research plan.\n\n"
                    "Return exactly one target and a minimal set of atomic, independently verifiable "
                    "constraints. Each constraint must represent one checkable condition that would help "
                    "establish the final answer.\n\n"
                    "Avoid duplicate constraints, bundled conditions, speculative entities, and facts "
                    "not supported by the wording of the question.\n\n"
                    "Do not assign constraint IDs, statuses, or supporting fact IDs. Those are owned "
                    "by deterministic code."
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
                    "Choose exactly one next research action from the supplied research state.\n\n"
                    "Focus on the highest-value unresolved required constraint. Do not create a "
                    "multi-step plan.\n\n"
                    "Research progress is not the number of collected facts. Treat progress primarily "
                    "as:\n"
                    "- a required constraint becoming supported or contradicted by evidence,\n"
                    "- a new supporting fact relevant to an unresolved constraint,\n"
                    "- or an entity being explicitly resolved from opened or located evidence.\n\n"
                    "Action policy:\n\n"
                    "SEARCH:\n"
                    "Use SEARCH when the current goal still lacks a useful source or candidate. "
                    "Avoid repeating the same retrieval angle with superficial query paraphrases.\n\n"
                    "OPEN:\n"
                    "When current search results contain a plausibly relevant source for the current "
                    "goal, prefer OPEN over another SEARCH so the candidate can be verified. "
                    "Search-result titles and snippets are only for source selection and are not "
                    "evidence.\n\n"
                    "OPEN only URLs supplied by the research state. Never invent a URL.\n\n"
                    "For a new candidate hypothesis, provide new_candidate_label and leave "
                    "candidate_scope_id empty. For an existing hypothesis, use only a "
                    "candidate_scope_id listed in candidate_scopes. Never invent a scope ID or "
                    "provide both fields.\n\n"
                    "LOCATE:\n"
                    "Use LOCATE only on an already opened document when the needed evidence is likely "
                    "to be inside that document.\n\n"
                    "If the previous LOCATE on the same document produced no constraint progress, "
                    "no new supporting fact, and no resolved entity, do not keep probing that document "
                    "with minor query variations. Prefer searching for another source.\n\n"
                    "HYPOTHESES:\n"
                    "A specific person, work, institution, date, or other entity that has not been "
                    "supported by OPEN or LOCATE facts is only a hypothesis.\n\n"
                    "You may search to verify or falsify a hypothesis, but do not treat it as an "
                    "established fact in subsequent reasoning.\n\n"
                    "ANSWER:\n"
                    "Propose ANSWER only when the required constraints needed for the answer are "
                    "supported by existing evidence.\n\n"
                    "supporting_fact_ids must contain only IDs that appear in known_facts. "
                    "supporting_constraint_ids must contain only IDs that appear in constraints. "
                    "Never place a constraint ID in supporting_fact_ids or a fact ID in "
                    "supporting_constraint_ids.\n\n"
                    "Cite only existing IDs. Do not invent facts, constraints, documents, passages, "
                    "or URLs.\n\n"
                    "When candidate-scoped evidence supports an answer, select its existing "
                    "candidate_scope_id.\n\n"
                    "Use the remaining step budget efficiently and return exactly one structured "
                    "SEARCH, OPEN, LOCATE, or ANSWER action."
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
                "evidence_scope_id": fact.evidence_scope_id,
            }
            for fact in state.facts
        ],
        "candidate_scopes": _candidate_scope_summary(state),
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
            "last_locate_outcome": _last_locate_outcome(state),
        },
        "remaining_step_budget": state.max_steps - state.step_count,
    }
    return json.dumps(context, ensure_ascii=False)


def _last_locate_outcome(state: ResearchState) -> dict[str, object] | None:
    """Expose only the latest LOCATE's compact evidence outcome to the planner."""
    if not state.trace or state.trace[-1].action != "locate":
        return None
    entry = state.trace[-1]
    new_supporting_fact_ids = [fact.id for fact in entry.new_facts if fact.supports_constraints]
    return {
        "document_id": entry.action_input.get("document_id"),
        "query": entry.action_input.get("query"),
        "constraint_progress": bool(new_supporting_fact_ids),
        "new_supporting_fact_ids": new_supporting_fact_ids,
        "resolved_entity_keys": sorted(entry.resolved_entities),
    }


def _candidate_scope_summary(state: ResearchState) -> list[dict[str, object]]:
    """Expose deterministic per-scope relation state without passage text."""
    summary = []
    for scope in state.candidate_scopes:
        statuses: dict[str, set[str]] = {}
        for fact in state.facts:
            if fact.evidence_scope_id != scope.id:
                continue
            for relation in fact.constraint_evidence:
                statuses.setdefault(relation.constraint_id, set()).add(relation.status)
        summary.append(
            {
                "id": scope.id,
                "label": scope.label,
                "constraint_evidence": {
                    constraint_id: (
                        next(iter(values)) if len(values) == 1 else "mixed"
                    )
                    for constraint_id, values in statuses.items()
                },
            }
        )
    return summary


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
