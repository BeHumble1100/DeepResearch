from app.research.guard import AnswerGuard
from app.research.schemas import (
    AnswerAction,
    CandidateScope,
    Constraint,
    ConstraintEvidence,
    DocumentRef,
    Fact,
    Target,
)
from app.research.state import ResearchState


def _fact(
    fact_id: str, constraint_id: str, status: str, *, scope_id: str | None
) -> Fact:
    return Fact(
        id=fact_id,
        statement=f"Evidence for {constraint_id}",
        source_url="https://example.com/source",
        document_id=f"doc-{fact_id}",
        passage_id=f"doc-{fact_id}:chunk:0",
        evidence_kind="locate",
        passage="Explicit source evidence.",
        confidence=0.9,
        evidence_scope_id=scope_id,
        constraint_evidence=[ConstraintEvidence(constraint_id=constraint_id, status=status)],
        supports_constraints=[constraint_id] if status == "supported" else [],
    )


def _state(*facts: Fact, constraints: list[Constraint] | None = None) -> ResearchState:
    return ResearchState(
        question="Who is the author?",
        target=Target(
            description="Author", answer_type="person_name", format_instruction="first name only"
        ),
        constraints=constraints or [Constraint(id="c-1", description="Identify the author")],
        candidate_scopes=[
            CandidateScope(id="cand_1", label="Candidate A"),
            CandidateScope(id="cand_2", label="Candidate B"),
        ],
        facts=list(facts),
        documents=[
            DocumentRef(
                id=f"doc-{fact.id}",
                url="https://example.com/source",
                content_type="text/html",
                evidence_scope_id=fact.evidence_scope_id,
            )
            for fact in facts
        ],
    )


def test_guard_accepts_global_direct_mode_with_only_unscoped_evidence() -> None:
    fact = _fact("fact-1", "c-1", "supported", scope_id=None)
    result = AnswerGuard().check(
        research=_state(fact),
        proposal=AnswerAction(
            answer="Ada", supporting_fact_ids=[fact.id], supporting_constraint_ids=["c-1"]
        ),
    )

    assert result.accepted


def test_guard_candidate_mode_uses_only_selected_scope() -> None:
    first = _fact("fact-a", "c-1", "supported", scope_id="cand_1")
    second = _fact("fact-b", "c-1", "supported", scope_id="cand_2")
    result = AnswerGuard().check(
        research=_state(first, second),
        proposal=AnswerAction(
            answer="Ada",
            supporting_fact_ids=[first.id],
            supporting_constraint_ids=["c-1"],
            candidate_scope_id="cand_1",
        ),
    )

    assert result.accepted


def test_guard_does_not_combine_positive_evidence_across_candidate_scopes() -> None:
    constraints = [Constraint(id="c1", description="First"), Constraint(id="c2", description="Second")]
    first = _fact("fact-a", "c1", "supported", scope_id="cand_1")
    second = _fact("fact-b", "c2", "supported", scope_id="cand_2")
    result = AnswerGuard().check(
        research=_state(first, second, constraints=constraints),
        proposal=AnswerAction(
            answer="Ada",
            supporting_fact_ids=[first.id],
            supporting_constraint_ids=["c1", "c2"],
            candidate_scope_id="cand_1",
        ),
    )

    assert not result.accepted
    assert any("c2" in reason and "scoped support" in reason for reason in result.reasons)


def test_guard_accepts_all_required_constraints_supported_in_one_scope() -> None:
    constraints = [Constraint(id="c1", description="First"), Constraint(id="c2", description="Second")]
    first = _fact("fact-a", "c1", "supported", scope_id="cand_1")
    second = _fact("fact-b", "c2", "supported", scope_id="cand_1")
    result = AnswerGuard().check(
        research=_state(first, second, constraints=constraints),
        proposal=AnswerAction(
            answer="Ada",
            supporting_fact_ids=[first.id, second.id],
            supporting_constraint_ids=["c1", "c2"],
            candidate_scope_id="cand_1",
        ),
    )

    assert result.accepted


def test_old_candidate_contradiction_does_not_block_new_candidate() -> None:
    old = _fact("fact-old", "c-1", "contradicted", scope_id="cand_1")
    new = _fact("fact-new", "c-1", "supported", scope_id="cand_2")
    result = AnswerGuard().check(
        research=_state(old, new),
        proposal=AnswerAction(
            answer="Ada",
            supporting_fact_ids=[new.id],
            supporting_constraint_ids=["c-1"],
            candidate_scope_id="cand_2",
        ),
    )

    assert result.accepted


def test_guard_rejects_a_contradicted_selected_scope_and_out_of_scope_fact() -> None:
    contradicted = _fact("fact-a", "c-1", "contradicted", scope_id="cand_1")
    supported = _fact("fact-b", "c-1", "supported", scope_id="cand_2")
    result = AnswerGuard().check(
        research=_state(contradicted, supported),
        proposal=AnswerAction(
            answer="Ada",
            supporting_fact_ids=[supported.id],
            supporting_constraint_ids=["c-1"],
            candidate_scope_id="cand_1",
        ),
    )

    assert not result.accepted
    assert any("outside the selected" in reason for reason in result.reasons)
    assert any("contradicted" in reason for reason in result.reasons)


def test_guard_rejects_unknown_ids_missing_support_and_bad_format() -> None:
    fact = _fact("fact-1", "c-1", "supported", scope_id=None)
    result = AnswerGuard().check(
        research=_state(fact),
        proposal=AnswerAction(
            answer="Ada Lovelace", supporting_fact_ids=["unknown"], supporting_constraint_ids=["unknown"]
        ),
    )

    assert not result.accepted
    assert any("Unknown supporting fact" in reason for reason in result.reasons)
    assert any("Unknown supporting constraint" in reason for reason in result.reasons)
    assert any("basic format" in reason for reason in result.reasons)
