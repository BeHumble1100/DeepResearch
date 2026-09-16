from app.research.guard import AnswerGuard
from app.research.schemas import AnswerAction, Constraint, DocumentRef, Fact, Target
from app.research.state import ResearchState


def _supported_state() -> ResearchState:
    fact = Fact(
        id="fact-1",
        statement="Ada is the author.",
        source_url="https://example.com/source",
        document_id="doc-1",
        passage_id="doc-1:chunk:0",
        evidence_kind="locate",
        passage="Ada is the author.",
        confidence=0.9,
        supports_constraints=["c-1"],
    )
    return ResearchState(
        question="Who is the author?",
        target=Target(description="Author", answer_type="person_name", format_instruction="first name only"),
        constraints=[
            Constraint(
                id="c-1",
                description="Identify the author",
                status="supported",
                supporting_fact_ids=["fact-1"],
            )
        ],
        facts=[fact],
        documents=[DocumentRef(id="doc-1", url="https://example.com/source", content_type="text/html")],
    )


def test_guard_accepts_supported_answer() -> None:
    result = AnswerGuard().check(
        research=_supported_state(),
        proposal=AnswerAction(
            answer="Ada",
            supporting_fact_ids=["fact-1"],
            supporting_constraint_ids=["c-1"],
        ),
    )

    assert result.accepted


def test_guard_rejects_unknown_ids_missing_support_and_bad_format() -> None:
    result = AnswerGuard().check(
        research=_supported_state(),
        proposal=AnswerAction(
            answer="Ada Lovelace",
            supporting_fact_ids=["unknown"],
            supporting_constraint_ids=["unknown"],
        ),
    )

    assert not result.accepted
    assert any("Unknown supporting fact" in reason for reason in result.reasons)
    assert any("Unknown supporting constraint" in reason for reason in result.reasons)
    assert any("basic format" in reason for reason in result.reasons)


def test_guard_rejects_contradicted_or_unopened_evidence() -> None:
    state = _supported_state()
    state = state.model_copy(
        update={
            "constraints": [state.constraints[0].model_copy(update={"status": "contradicted"})],
            "facts": [state.facts[0].model_copy(update={"document_id": "missing"})],
        }
    )
    result = AnswerGuard().check(
        research=state,
        proposal=AnswerAction(
            answer="Ada",
            supporting_fact_ids=["fact-1"],
            supporting_constraint_ids=["c-1"],
        ),
    )

    assert not result.accepted
    assert any("provenance" in reason for reason in result.reasons)
    assert any("contradicted" in reason for reason in result.reasons)
