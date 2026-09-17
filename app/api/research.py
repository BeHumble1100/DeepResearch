"""Minimal FastAPI boundary for one complete research run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from app.config import Settings
from app.llm.client import OpenAICompatibleClient
from app.research.evidence import LLMFactExtractor
from app.research.graph import build_research_graph
from app.research.guard import AnswerGuard
from app.research.planner import LLMPlanner
from app.research.schemas import Constraint, ResearchTraceEntry, Target
from app.research.state import ResearchState
from app.tools.document import HttpDocumentOpener
from app.tools.retrieval import DocumentRetriever, LLMPassageReranker
from app.tools.search import LLMQueryRewriter, SearXNGSearchGateway

ResearchExecutor = Callable[[str, int], Awaitable[ResearchState]]


class ResearchRequest(BaseModel):
    """A single question and its bounded research budget."""

    question: str = Field(min_length=1)
    max_steps: int = Field(default=10, gt=0)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class FactResponse(BaseModel):
    """Traceable fact metadata without replaying source passage content."""

    id: str
    statement: str
    source_url: str
    document_id: str | None
    passage_id: str | None
    evidence_kind: str | None
    confidence: float
    evidence_scope_id: str | None
    supports_constraints: list[str]


class ResearchResponse(BaseModel):
    """Compact, serializable terminal state for an API research request."""

    question: str
    answer: str | None
    status: str
    step_count: int
    target: Target | None
    constraints: list[Constraint]
    resolved_entities: dict[str, str]
    facts: list[FactResponse]
    trace: list[ResearchTraceEntry]


def build_production_executor(*, settings: Settings) -> ResearchExecutor:
    """Wire existing production components without adding a second research loop."""
    client = OpenAICompatibleClient(settings)
    graph = build_research_graph(
        LLMPlanner(client),
        SearXNGSearchGateway(settings, query_rewriter=LLMQueryRewriter(client)),
        HttpDocumentOpener(settings),
        DocumentRetriever(settings, LLMPassageReranker(client)),
        LLMFactExtractor(client),
        AnswerGuard(),
    )

    async def execute(question: str, max_steps: int) -> ResearchState:
        result = await graph.ainvoke(
            {"research": ResearchState(question=question, max_steps=max_steps), "action": None}
        )
        return result["research"]

    return execute


def create_research_router(*, executor: ResearchExecutor) -> APIRouter:
    router = APIRouter(tags=["research"])

    @router.post("/research", response_model=ResearchResponse)
    async def run_research(request: ResearchRequest) -> ResearchResponse:
        research = await executor(request.question, request.max_steps)
        return ResearchResponse(
            question=research.question,
            answer=research.answer,
            status=research.status,
            step_count=research.step_count,
            target=research.target,
            constraints=research.constraints,
            resolved_entities=research.resolved_entities,
            facts=[
                FactResponse(
                    id=fact.id,
                    statement=fact.statement,
                    source_url=fact.source_url,
                    document_id=fact.document_id,
                    passage_id=fact.passage_id,
                    evidence_kind=fact.evidence_kind,
                    confidence=fact.confidence,
                    evidence_scope_id=fact.evidence_scope_id,
                    supports_constraints=fact.supports_constraints,
                )
                for fact in research.facts
            ],
            trace=research.trace,
        )

    return router
