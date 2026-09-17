import pytest
from pydantic import TypeAdapter, ValidationError

from app.research.schemas import Action, ConstraintProposal, Fact, SearchAction


def test_action_is_discriminated_by_type() -> None:
    action = TypeAdapter(Action).validate_python(
        {"type": "search", "goal": "Find a source", "query": "deep research"}
    )

    assert isinstance(action, SearchAction)


def test_fact_confidence_must_be_a_probability() -> None:
    with pytest.raises(ValidationError):
        Fact(
            id="fact-1",
            statement="A claim",
            source_url="https://example.com",
            passage="Supporting text",
            confidence=1.1,
        )


def test_constraint_proposal_requires_explicit_kind() -> None:
    with pytest.raises(ValidationError):
        ConstraintProposal(description="A condition")
