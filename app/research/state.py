"""Explicit, compact state for one research run."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .schemas import Constraint, DocumentRef, Fact, Target


class ResearchState(BaseModel):
    question: str
    target: Target | None = None
    constraints: list[Constraint] = Field(default_factory=list)
    resolved_entities: dict[str, str] = Field(default_factory=dict)
    facts: list[Fact] = Field(default_factory=list)
    documents: list[DocumentRef] = Field(default_factory=list)
    executed_queries: list[str] = Field(default_factory=list)
    visited_urls: list[str] = Field(default_factory=list)
    current_goal: str | None = None
    step_count: int = Field(default=0, ge=0)
    max_steps: int = Field(default=8, gt=0)
    answer: str | None = None
    status: str = "initialized"
