import pytest
from pydantic import ValidationError

from app.research.state import ResearchState


def test_research_state_starts_with_compact_empty_collections() -> None:
    state = ResearchState(question="Who wrote this work?")

    assert state.status == "initialized"
    assert state.constraints == []
    assert state.facts == []
    assert state.search_results == []
    assert state.located_passages == []
    assert state.max_steps == 8


def test_research_state_requires_positive_step_budget() -> None:
    with pytest.raises(ValidationError):
        ResearchState(question="Question", max_steps=0)
