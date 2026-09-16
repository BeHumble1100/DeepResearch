"""Structured contracts shared by the research harness."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Target(BaseModel):
    """The final value the research process must answer."""

    description: str
    answer_type: str
    format_instruction: str | None = None


class Constraint(BaseModel):
    """A condition used to identify or verify an answer path."""

    id: str
    description: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    required: bool = True
    status: Literal["unknown", "supported", "contradicted"] = "unknown"
    supporting_fact_ids: list[str] = Field(default_factory=list)


class Fact(BaseModel):
    """A verifiable claim with traceable source evidence."""

    id: str
    statement: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    source_url: str
    document_id: str | None = None
    passage: str
    confidence: float = Field(ge=0, le=1)
    supports_constraints: list[str] = Field(default_factory=list)


class DocumentRef(BaseModel):
    """An opened document kept out of planner prompt payloads."""

    id: str
    url: str
    title: str | None = None
    content_type: str
    local_path: str | None = None
    summary: str | None = None


class SearchAction(BaseModel):
    type: Literal["search"] = "search"
    goal: str
    query: str


class OpenAction(BaseModel):
    type: Literal["open"] = "open"
    goal: str
    url: str


class LocateAction(BaseModel):
    type: Literal["locate"] = "locate"
    goal: str
    document_id: str
    query: str


class AnswerAction(BaseModel):
    type: Literal["answer"] = "answer"
    answer: str
    supporting_fact_ids: list[str]
    supporting_constraint_ids: list[str]


Action = Annotated[
    SearchAction | OpenAction | LocateAction | AnswerAction,
    Field(discriminator="type"),
]
