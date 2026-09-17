"""Planner interfaces and a deterministic Phase 1 mock implementation."""

from __future__ import annotations

import json
from typing import Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from app.llm.client import LLMClient, Message
from app.tools.search import source_quality_score

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
                    "Classify each constraint by kind. Use acceptance only for a condition the final "
                    "answer must directly satisfy or be directly verified by. Use research_clue for "
                    "historical, contextual, entity-bridging, or disambiguation clues that guide research "
                    "but should not independently block a final answer. Keep acceptance constraints few.\n\n"
                    "The Target already owns answer type and format. Do not create an acceptance "
                    "constraint that merely restates a generic type or format, such as 'the answer "
                    "identifies a person' or 'the answer is an ocean'; acceptance constraints must "
                    "describe a factual relation that source evidence can directly verify.\n\n"
                    "Return at least one acceptance constraint. Deterministic code will add a target "
                    "verification constraint if you fail to provide one.\n\n"
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
                    "Focus on the highest-value unresolved acceptance constraint or research clue. Do not create a "
                    "multi-step plan.\n\n"
                    "Research progress is not the number of collected facts. Treat progress primarily "
                    "as:\n"
                    "- an acceptance constraint becoming supported or contradicted by evidence,\n"
                    "- a new supporting fact relevant to an unresolved acceptance constraint or clue,\n"
                    "- or an entity being explicitly resolved from opened or located evidence.\n\n"
                    "Action policy:\n\n"
                    "ACTION PRIORITY:\n"
                    "Before SEARCH, inspect openable_search_results and documents_requiring_locate. "
                    "If openable_search_results is non-empty, OPEN one relevant unattempted source. "
                    "Those candidates are ordered by source quality and relevance; prefer the lowest "
                    "source_rank when candidates are similarly relevant. "
                    "If documents_requiring_locate is non-empty and it is likely to contain the needed "
                    "evidence, LOCATE that document. Use SEARCH only when neither path can advance the "
                    "current goal, or after they have been exhausted.\n\n"
                    "SEARCH:\n"
                    "Use SEARCH when the current goal still lacks a useful source or candidate. "
                    "Avoid repeating the same retrieval angle with superficial query paraphrases.\n\n"
                    "OPEN:\n"
                    "When current search results contain a plausibly relevant source for the current "
                    "goal, prefer OPEN over another SEARCH so the candidate can be verified. "
                    "Search-result titles and snippets are only for source selection and are not "
                    "evidence.\n\n"
                    "OPEN only a URL listed in openable_search_results. Never invent a URL, reopen an "
                    "attempted URL, or retry a URL listed in failed_open_urls.\n\n"
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
                    "Propose ANSWER only when all acceptance constraints needed for the answer are "
                    "supported by existing evidence.\n\n"
                    "supporting_fact_ids must contain only IDs that appear in known_facts. "
                    "supporting_constraint_ids must contain only IDs that appear in constraints. "
                    "Cite a Fact for a constraint only when its constraint_evidence marks that "
                    "constraint supported; never cite a Fact that marks it contradicted.\n\n"
                    "The answer field must contain only the requested target value. Do not append "
                    "aliases, parenthetical notes, explanations, citations, or labels unless the "
                    "target format instruction explicitly requests them.\n\n"
                    "Never place a constraint ID in supporting_fact_ids or a fact ID in "
                    "supporting_constraint_ids.\n\n"
                    "Cite only existing IDs. Do not invent facts, constraints, documents, passages, "
                    "or URLs.\n\n"
                    "When candidate-scoped evidence supports an answer, select its existing "
                    "candidate_scope_id.\n\n"
                    "If recent_actions.last_guard_rejection is present, do not repeat that rejected "
                    "ANSWER without new supporting evidence. Use its reject reasons to seek the "
                    "missing evidence instead.\n\n"
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

    openable_search_results = _openable_search_results(state)
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
                "kind": constraint.kind,
                "status": constraint.status,
            }
            for constraint in state.constraints
        ],
        "known_facts": [
            {
                "id": fact.id,
                "statement": fact.statement,
                "supports_constraints": fact.supports_constraints,
                "constraint_evidence": [
                    {"constraint_id": relation.constraint_id, "status": relation.status}
                    for relation in fact.constraint_evidence
                ],
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
        "openable_search_results": openable_search_results,
        "documents_requiring_locate": _documents_requiring_locate(state),
        "recent_actions": {
            "executed_queries": state.executed_queries,
            "visited_urls": state.visited_urls,
            "failed_open_urls": _failed_open_urls(state),
            "failed_open_hosts": _failed_open_hosts(state),
            "failed_open_sources": _failed_open_sources(state),
            "last_locate_outcome": _last_locate_outcome(state),
            "last_guard_rejection": _last_guard_rejection(state),
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


def _failed_open_urls(state: ResearchState) -> list[str]:
    """Expose prior source-opening failures without adding document content to context."""
    return list(
        dict.fromkeys(
            str(entry.action_input["url"])
            for entry in state.trace
            if entry.action == "open"
            and entry.validation_rejection_reason
            and entry.validation_rejection_reason.startswith("OPEN document fetch or parse failed:")
            and isinstance(entry.action_input.get("url"), str)
        )
    )


def _failed_open_hosts(state: ResearchState) -> list[str]:
    """Avoid another URL from a host that explicitly denied this run's requests."""
    hosts: list[str] = []
    for entry in state.trace:
        if entry.action != "open" or not entry.validation_rejection_reason:
            continue
        if entry.source_failure_category != "access_denied" and not (
            entry.validation_rejection_reason.startswith(
                "OPEN document fetch or parse failed: Document request returned HTTP "
            )
            and (
                "HTTP 401:" in entry.validation_rejection_reason
                or "HTTP 403:" in entry.validation_rejection_reason
            )
        ):
            continue
        url = entry.action_input.get("url")
        host = urlsplit(url).netloc.lower() if isinstance(url, str) else ""
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def _failed_open_sources(state: ResearchState) -> list[dict[str, str]]:
    """Keep one concise failure classification per attempted source for replanning."""
    sources: list[dict[str, str]] = []
    for entry in state.trace:
        url = entry.action_input.get("url")
        if (
            entry.action != "open"
            or not isinstance(url, str)
            or entry.source_failure_category is None
        ):
            continue
        sources.append(
            {
                "url": url,
                "host": urlsplit(url).netloc.lower(),
                "category": entry.source_failure_category,
            }
        )
    return sources


def _last_guard_rejection(state: ResearchState) -> dict[str, object] | None:
    """Expose unmet evidence requirements until a later action adds new evidence."""
    for entry in reversed(state.trace):
        if entry.new_facts or entry.resolved_entities:
            return None
        if entry.action == "answer" and entry.guard_result and not entry.guard_result.accepted:
            return {
                "answer": entry.action_input.get("answer"),
                "supporting_fact_ids": entry.answer_supporting_fact_ids,
                "supporting_constraint_ids": entry.answer_supporting_constraint_ids,
                "candidate_scope_id": entry.action_input.get("candidate_scope_id"),
                "reject_reasons": entry.guard_result.reject_reasons,
            }
    return None


def _openable_search_results(state: ResearchState) -> list[dict[str, str | int | None]]:
    """Expose current search candidates that have not been attempted in this run."""
    attempted_urls = {
        str(entry.action_input["url"])
        for entry in state.trace
        if entry.action == "open" and isinstance(entry.action_input.get("url"), str)
    }
    failed_hosts = set(_failed_open_hosts(state))
    candidates = [
        result
        for result in state.search_results
        if result.url not in attempted_urls and urlsplit(result.url).netloc.lower() not in failed_hosts
    ]
    candidates.sort(key=source_quality_score, reverse=True)
    return [
        {
            "source_rank": index,
            "url": result.url,
            "title": result.title,
            "snippet": result.snippet,
        }
        for index, result in enumerate(candidates, start=1)
    ]


def _documents_requiring_locate(state: ResearchState) -> list[dict[str, str | None]]:
    """Expose opened documents without a prior document-local retrieval attempt."""
    located_document_ids = {
        str(entry.action_input["document_id"])
        for entry in state.trace
        if entry.action == "locate" and isinstance(entry.action_input.get("document_id"), str)
    }
    return [
        {"id": document.id, "url": document.url, "title": document.title}
        for document in state.documents
        if document.id not in located_document_ids
    ]


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
    seen: set[tuple[str, str | None, str | None, str | None, bool, str]] = set()
    for candidate in proposal.constraints:
        normalized = _normalize_constraint(candidate)
        key = (
            normalized.description,
            normalized.subject,
            normalized.predicate,
            normalized.object,
            normalized.required,
            normalized.kind,
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
                kind=normalized.kind,
                status="unknown",
                supporting_fact_ids=[],
            )
        )
    if not any(constraint.required and constraint.kind == "acceptance" for constraint in constraints):
        constraints.append(
            Constraint(
                id=f"c{len(constraints) + 1}",
                description=proposal.target.description,
                required=True,
                kind="acceptance",
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
        kind=candidate.kind,
    )


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip() or None
