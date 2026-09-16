"""Structured contracts shared by the research harness."""

from __future__ import annotations

from typing import Annotated, Any, Literal

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


class CandidateScope(BaseModel):
    """A state-owned evidence namespace for one candidate hypothesis."""

    id: str
    label: str


class ConstraintProposal(BaseModel):
    """A constraint proposed by initialization before code assigns state fields."""

    description: str = Field(min_length=1)
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    required: bool = True


class Fact(BaseModel):
    """A verifiable claim with traceable source evidence."""

    id: str
    statement: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    source_url: str
    document_id: str | None = None
    passage_id: str | None = None
    evidence_kind: Literal["open", "locate"] | None = None
    passage: str
    confidence: float = Field(ge=0, le=1)
    supports_constraints: list[str] = Field(default_factory=list)
    evidence_scope_id: str | None = None
    constraint_evidence: list[ConstraintEvidence] = Field(default_factory=list)


class DocumentRef(BaseModel):
    """An opened document kept out of planner prompt payloads."""

    id: str
    url: str
    title: str | None = None
    content_type: str
    local_path: str | None = None
    summary: str | None = None
    evidence_scope_id: str | None = None


class SearchResult(BaseModel):
    """A candidate resource returned by a search action."""

    url: str
    title: str | None = None
    snippet: str | None = None


class QueryRewrite(BaseModel):
    """One or more search queries derived from the current research goal."""

    queries: list[str] = Field(min_length=1)


class Passage(BaseModel):
    """A bounded passage selected from one locally stored document."""

    id: str
    document_id: str
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    bm25_score: float | None = None
    rank: int | None = None


class PassageRanking(BaseModel):
    """Structured LLM output that ranks only supplied candidate passage IDs."""

    passage_ids: list[str] = Field(default_factory=list)


class ConstraintEvidence(BaseModel):
    """One extracted fact's relationship to a known research constraint."""

    constraint_id: str
    status: Literal["supported", "contradicted"]


class ExtractedFact(BaseModel):
    """A fact candidate grounded in one supplied, bounded passage."""

    statement: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    confidence: float = Field(ge=0, le=1)
    passage_id: str
    constraint_evidence: list[ConstraintEvidence] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    """An entity explicitly resolved in one supplied passage."""

    key: str = Field(min_length=1)
    value: str = Field(min_length=1)
    passage_id: str


class EntityProvenance(BaseModel):
    """Compact source metadata for one resolved entity."""

    source_url: str
    document_id: str
    passage_id: str
    evidence_kind: Literal["open", "locate"]


class FactExtraction(BaseModel):
    """Structured output returned by a fact extractor for bounded source passages."""

    facts: list[ExtractedFact] = Field(default_factory=list)
    resolved_entities: list[ExtractedEntity] = Field(default_factory=list)


class TraceFact(BaseModel):
    """Compact fact metadata retained in a research trace."""

    id: str
    statement: str
    source_url: str
    document_id: str | None = None
    passage_id: str | None = None
    evidence_kind: Literal["open", "locate"] | None = None
    supports_constraints: list[str] = Field(default_factory=list)
    evidence_scope_id: str | None = None
    constraint_evidence: list[ConstraintEvidence] = Field(default_factory=list)


class TraceCandidateScope(BaseModel):
    """The minimal scope-creation record needed to replay a hypothesis namespace."""

    id: str
    label: str
    originating_document_id: str


class ConstraintChange(BaseModel):
    """A deterministic constraint delta caused by one research action."""

    constraint_id: str
    previous_status: Literal["unknown", "supported", "contradicted"]
    current_status: Literal["unknown", "supported", "contradicted"]
    added_supporting_fact_ids: list[str] = Field(default_factory=list)


class TraceSearchResult(BaseModel):
    """A bounded search-result snapshot retained for action replay."""

    url: str
    title: str | None = None
    snippet: str | None = None


class TraceGuardResult(BaseModel):
    """The deterministic guard outcome for one answer proposal."""

    accepted: bool
    reject_reasons: list[str] = Field(default_factory=list)


class ResearchTraceEntry(BaseModel):
    """One compact, serializable research action record."""

    step: int = Field(ge=0)
    current_goal: str | None = None
    action: str
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation_summary: str
    remaining_step_budget: int = Field(ge=0)
    planner_context: dict[str, Any] | None = None
    search_results: list[TraceSearchResult] = Field(default_factory=list)
    answer_supporting_fact_ids: list[str] = Field(default_factory=list)
    answer_supporting_constraint_ids: list[str] = Field(default_factory=list)
    guard_result: TraceGuardResult | None = None
    validation_rejection_reason: str | None = None
    new_facts: list[TraceFact] = Field(default_factory=list)
    constraint_changes: list[ConstraintChange] = Field(default_factory=list)
    resolved_entities: dict[str, str] = Field(default_factory=dict)
    created_candidate_scope: TraceCandidateScope | None = None


class SearchAction(BaseModel):
    type: Literal["search"] = "search"
    goal: str
    query: str


class OpenAction(BaseModel):
    type: Literal["open"] = "open"
    goal: str
    url: str
    candidate_scope_id: str | None = None
    new_candidate_label: str | None = None


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
    candidate_scope_id: str | None = None


Action = Annotated[
    SearchAction | OpenAction | LocateAction | AnswerAction,
    Field(discriminator="type"),
]


class ActionDecision(BaseModel):
    """One structured next-action proposal from the planner."""

    action: Action
