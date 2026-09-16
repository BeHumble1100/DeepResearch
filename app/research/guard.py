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
        for fact_id in proposal.supporting_fact_ids:
            fact = facts.get(fact_id)
            if fact is None:
                reasons.append(f"Unknown supporting fact: {fact_id}")
            elif not _is_opened_evidence(fact, documents):
                reasons.append(f"Fact lacks opened or located document provenance: {fact_id}")
        for constraint_id in proposal.supporting_constraint_ids:
            if constraint_id not in constraints:
                reasons.append(f"Unknown supporting constraint: {constraint_id}")

        for constraint in constraints.values():
            if not constraint.required:
                continue
            if constraint.status == "contradicted":
                reasons.append(f"Required constraint is contradicted: {constraint.id}")
            elif constraint.status != "supported" or not constraint.supporting_fact_ids:
                reasons.append(f"Required constraint lacks sufficient support: {constraint.id}")
            elif not any(
                fact is not None
                and constraint.id in fact.supports_constraints
                and _is_opened_evidence(fact, documents)
                for fact_id in constraint.supporting_fact_ids
                if (fact := facts.get(fact_id))
            ):
                reasons.append(f"Required constraint lacks valid evidence: {constraint.id}")
            elif constraint.id not in proposal.supporting_constraint_ids:
                reasons.append(f"Required constraint is missing from the answer proposal: {constraint.id}")
        if not proposal.supporting_fact_ids:
            reasons.append("Answer proposal has no supporting facts.")
        return GuardResult(accepted=not reasons, reasons=tuple(reasons))


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
