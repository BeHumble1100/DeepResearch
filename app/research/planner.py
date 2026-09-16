"""Planner interfaces and a deterministic Phase 1 mock implementation."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from .schemas import Action, Constraint, SearchAction, Target
from .state import ResearchState


class Initialization(BaseModel):
    target: Target
    constraints: list[Constraint] = Field(default_factory=list)


class Planner(Protocol):
    def initialize(self, question: str) -> Initialization: ...

    def next_action(self, state: ResearchState) -> Action: ...


class MockPlanner:
    """Deterministic planner used only to exercise graph transitions."""

    def initialize(self, question: str) -> Initialization:
        return Initialization(target=Target(description=question, answer_type="text"))

    def next_action(self, state: ResearchState) -> Action:
        return SearchAction(
            goal="Find an initial source for the research question.",
            query=state.question,
        )
