"""Planner interfaces and a deterministic Phase 1 mock implementation."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, Field

from app.llm.client import LLMClient, Message

from .schemas import (
    Action,
    ActionDecision,
    AnswerAction,
    Constraint,
    OpenAction,
    SearchAction,
    Target,
)
from .state import ResearchState


class Initialization(BaseModel):
    target: Target
    constraints: list[Constraint] = Field(default_factory=list)


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
        if state.documents:
            return AnswerAction(
                answer="Mock answer",
                supporting_fact_ids=[],
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
                "content": "Extract a research target and verifiable constraints. Do not plan steps.",
            },
            {"role": "user", "content": question},
        ]
        return await self._client.structured(messages=messages, schema=Initialization)

    async def next_action(self, state: ResearchState) -> Action:
        messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    "Choose exactly one next research action. Return a structured SEARCH, OPEN, "
                    "LOCATE, or ANSWER proposal from the supplied state."
                ),
            },
            {"role": "user", "content": _compact_state_view(state)},
        ]
        decision = await self._client.structured(messages=messages, schema=ActionDecision)
        return decision.action


def _compact_state_view(state: ResearchState) -> str:
    """Serialize only the planner context allowed by the V1 specification."""

    context = {
        "original_question": state.question,
        "target": state.target.model_dump() if state.target else None,
        "resolved_entities": state.resolved_entities,
        "constraints": [
            {
                "id": constraint.id,
                "description": constraint.description,
                "required": constraint.required,
                "status": constraint.status,
            }
            for constraint in state.constraints
        ],
        "known_facts": [
            {
                "id": fact.id,
                "statement": fact.statement,
                "supports_constraints": fact.supports_constraints,
            }
            for fact in state.facts
        ],
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
        "recent_actions": {
            "executed_queries": state.executed_queries,
            "visited_urls": state.visited_urls,
        },
        "remaining_step_budget": state.max_steps - state.step_count,
    }
    return json.dumps(context, ensure_ascii=False)
