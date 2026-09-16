"""Deterministic finish checks for answer proposals."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import AnswerAction, DocumentRef, Fact, Target
from .state import ResearchState


@dataclass(frozen=True)
class GuardResult:
    accepted: bool
    reasons: tuple[str, ...] = ()


class AnswerGuard:
    """Accepts only answers backed by opened or located document evidence."""

    def check(self, *, research: ResearchState, proposal: AnswerAction) -> GuardResult:
        reasons: list[str] = []
        facts = {fact.id: fact for fact in research.facts}
        constraints = {constraint.id: constraint for constraint in research.constraints}
        documents = {document.id: document for document in research.documents}

        if not _matches_basic_format(proposal.answer, research.target):
            reasons.append("Answer does not satisfy the target's basic format requirement.")
        selected_facts = _facts_in_scope(research.facts, proposal.candidate_scope_id)
        selected_fact_ids = {fact.id for fact in selected_facts}
        if proposal.candidate_scope_id is not None and proposal.candidate_scope_id not in {
            scope.id for scope in research.candidate_scopes
        }:
            reasons.append(f"Unknown candidate scope: {proposal.candidate_scope_id}")

        for fact_id in proposal.supporting_fact_ids:
            fact = facts.get(fact_id)
            if fact is None:
                reasons.append(f"Unknown supporting fact: {fact_id}")
            elif fact_id not in selected_fact_ids:
                reasons.append(f"Supporting fact is outside the selected evidence scope: {fact_id}")
            elif not _is_opened_evidence(fact, documents):
                reasons.append(f"Fact lacks opened or located document provenance: {fact_id}")
        for constraint_id in proposal.supporting_constraint_ids:
            if constraint_id not in constraints:
                reasons.append(f"Unknown supporting constraint: {constraint_id}")

        for constraint in constraints.values():
            if not constraint.required:
                continue
            scoped_relations = [
                (fact, relation.status)
                for fact in selected_facts
                if _is_opened_evidence(fact, documents)
                for relation in fact.constraint_evidence
                if relation.constraint_id == constraint.id
            ]
            if any(status == "contradicted" for _, status in scoped_relations):
                reasons.append(
                    f"Required constraint is contradicted in the selected evidence scope: {constraint.id}"
                )
            elif not any(status == "supported" for _, status in scoped_relations):
                reasons.append(f"Required constraint lacks sufficient scoped support: {constraint.id}")
            elif constraint.id not in proposal.supporting_constraint_ids:
                reasons.append(f"Required constraint is missing from the answer proposal: {constraint.id}")
        if not proposal.supporting_fact_ids:
            reasons.append("Answer proposal has no supporting facts.")
        return GuardResult(accepted=not reasons, reasons=tuple(reasons))


def _facts_in_scope(facts: list[Fact], evidence_scope_id: str | None) -> list[Fact]:
    """Keep candidate and direct/global evidence modes mutually exclusive."""
    return [fact for fact in facts if fact.evidence_scope_id == evidence_scope_id]


def _matches_basic_format(answer: str, target: Target | None) -> bool:
    if not answer.strip() or "\n" in answer:
        return False
    if target and target.format_instruction and target.format_instruction.strip().lower() == "first name only":
        return len(answer.strip().split()) == 1
    return True


def _is_opened_evidence(fact: Fact, documents: dict[str, DocumentRef]) -> bool:
    document = documents.get(fact.document_id)
    return bool(
        document
        and fact.source_url == document.url
        and fact.passage_id
        and fact.evidence_kind in {"open", "locate"}
        and fact.passage.strip()
    )
