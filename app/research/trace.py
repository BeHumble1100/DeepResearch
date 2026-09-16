"""Compact, deterministic trace records for one research run."""

from __future__ import annotations

from typing import Any

from .schemas import Action, ConstraintChange, ResearchTraceEntry, TraceFact
from .state import ResearchState


def append_action_trace(
    before: ResearchState,
    after: ResearchState,
    *,
    action: Action,
    observation_summary: str,
) -> ResearchState:
    """Append one action record using only state deltas and compact metadata."""
    entry = ResearchTraceEntry(
        step=after.step_count,
        current_goal=after.current_goal,
        action=action.type,
        action_input=action.model_dump(),
        observation_summary=observation_summary,
        new_facts=_new_facts(before, after),
        constraint_changes=_constraint_changes(before, after),
        resolved_entities=_entity_changes(before, after),
    )
    return after.model_copy(update={"trace": [*after.trace, entry]})


def append_event_trace(
    research: ResearchState,
    *,
    action: str,
    action_input: dict[str, Any],
    observation_summary: str,
) -> ResearchState:
    """Append a terminal control-flow event without copying evidence payloads."""
    entry = ResearchTraceEntry(
        step=research.step_count,
        current_goal=research.current_goal,
        action=action,
        action_input=action_input,
        observation_summary=observation_summary,
    )
    return research.model_copy(update={"trace": [*research.trace, entry]})


def _new_facts(before: ResearchState, after: ResearchState) -> list[TraceFact]:
    existing_ids = {fact.id for fact in before.facts}
    return [
        TraceFact(
            id=fact.id,
            statement=fact.statement,
            source_url=fact.source_url,
            document_id=fact.document_id,
            passage_id=fact.passage_id,
            evidence_kind=fact.evidence_kind,
            supports_constraints=fact.supports_constraints,
        )
        for fact in after.facts
        if fact.id not in existing_ids
    ]


def _constraint_changes(before: ResearchState, after: ResearchState) -> list[ConstraintChange]:
    previous = {constraint.id: constraint for constraint in before.constraints}
    changes: list[ConstraintChange] = []
    for constraint in after.constraints:
        old = previous.get(constraint.id)
        if old is None:
            continue
        old_ids = set(old.supporting_fact_ids)
        added_ids = [fact_id for fact_id in constraint.supporting_fact_ids if fact_id not in old_ids]
        if old.status != constraint.status or added_ids:
            changes.append(
                ConstraintChange(
                    constraint_id=constraint.id,
                    previous_status=old.status,
                    current_status=constraint.status,
                    added_supporting_fact_ids=added_ids,
                )
            )
    return changes


def _entity_changes(before: ResearchState, after: ResearchState) -> dict[str, str]:
    return {
        key: value
        for key, value in after.resolved_entities.items()
        if before.resolved_entities.get(key) != value
    }
