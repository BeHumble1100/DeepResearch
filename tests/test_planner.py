import asyncio
import json

import pytest
from pydantic import BaseModel

from app.llm.client import Message
from app.research.graph import build_research_graph
from app.research.guard import AnswerGuard
from app.research.planner import Initialization, InitializationProposal, LLMPlanner
from app.research.schemas import (
    ActionDecision,
    AnswerAction,
    ConstraintProposal,
    DocumentRef,
    Fact,
    LocateAction,
    OpenAction,
    Passage,
    SearchAction,
    Target,
    ResearchTraceEntry,
)
from app.research.state import ResearchState
from app.research.schemas import SearchResult


class FakeLLMClient:
    def __init__(self, responses: list[BaseModel]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[Message], type[BaseModel]]] = []

    async def structured(
        self,
        *,
        messages: list[Message],
        schema: type[BaseModel],
        **kwargs: object,
    ) -> BaseModel:
        self.calls.append((messages, schema))
        return self.responses.pop(0)

    async def text(self, *, messages: list[Message], **kwargs: object) -> str:
        return "unused"


class FakeSearchGateway:
    async def search(self, *, goal: str, query: str) -> list[SearchResult]:
        return [SearchResult(url="https://example.com/mock-source", title="Mock source")]


class FakeDocumentOpener:
    async def open(self, *, url: str) -> DocumentRef:
        return DocumentRef(
            id="mock-document",
            url=url,
            content_type="text/html",
            local_path=__file__,
        )


class FakeDocumentRetriever:
    async def retrieve(self, *, document: DocumentRef, goal: str, query: str) -> list[Passage]:
        return [
            Passage(
                id=f"{document.id}:chunk:0",
                document_id=document.id,
                text="Relevant passage",
                start_char=0,
                end_char=16,
                bm25_score=1.0,
                rank=1,
            )
        ]


class FakeFactExtractor:
    async def extract(self, *, document, passages, constraints):
        from app.research.schemas import FactExtraction

        return FactExtraction()


def test_llm_planner_initializes_with_a_structured_contract() -> None:
    client = FakeLLMClient(
        [
            InitializationProposal(
                target=Target(description="Find an author", answer_type="person_name"),
                constraints=[
                    ConstraintProposal(description=" Author wrote the work ", subject=" Author "),
                    ConstraintProposal(description="Author wrote the work", subject="Author"),
                    ConstraintProposal(description="Work was published", required=False),
                ],
            )
        ]
    )

    result = asyncio.run(LLMPlanner(client).initialize("Who wrote this work?"))

    assert result.target.answer_type == "person_name"
    assert client.calls[0][1] is InitializationProposal
    assert [constraint.id for constraint in result.constraints] == ["c1", "c2"]
    assert result.constraints[0].status == "unknown"
    assert result.constraints[0].supporting_fact_ids == []
    assert "Do not search, answer" in client.calls[0][0][0]["content"]


@pytest.mark.parametrize(
    "action",
    [
        SearchAction(goal="Find a source", query="research"),
        OpenAction(goal="Read a source", url="https://example.com/source"),
        LocateAction(goal="Locate support", document_id="doc-1", query="evidence"),
        AnswerAction(answer="Ada", supporting_fact_ids=["fact-1"], supporting_constraint_ids=[]),
    ],
)
def test_llm_planner_accepts_every_structured_action(action: object) -> None:
    client = FakeLLMClient([ActionDecision(action=action)])
    planner = LLMPlanner(client)
    state = ResearchState(question="Question")

    result = asyncio.run(planner.next_action(state))

    assert result == action
    assert client.calls[0][1] is ActionDecision


def test_llm_planner_sends_a_compact_state_without_fact_passages() -> None:
    hidden_passage = "this full document passage must not reach the planner"
    client = FakeLLMClient([ActionDecision(action=SearchAction(goal="Find", query="Question"))])
    planner = LLMPlanner(client)
    state = ResearchState(
        question="Question",
        target=Target(description="Answer", answer_type="text"),
        facts=[
            Fact(
                id="fact-1",
                statement="Known claim",
                source_url="https://example.com",
                passage=hidden_passage,
                confidence=0.8,
            )
        ],
        documents=[
            DocumentRef(
                id="doc-1",
                url="https://example.com",
                content_type="text/html",
                summary="Short summary",
            )
        ],
        search_results=[
            SearchResult(
                url="https://example.com/search-result",
                title="Search result title",
                snippet="Search snippet",
            )
        ],
        located_passages=[
            Passage(
                id="doc-1:chunk:0",
                document_id="doc-1",
                text="This passage must not reach the planner",
                start_char=0,
                end_char=40,
            )
        ],
        current_goal="Resolve the author",
        trace=[
            ResearchTraceEntry(
                step=1,
                action="locate",
                action_input={"document_id": "doc-1", "query": "author evidence"},
                observation_summary="No supporting evidence found.",
                remaining_step_budget=7,
            )
        ],
    )

    asyncio.run(planner.next_action(state))

    context = client.calls[0][0][1]["content"]
    assert "Known claim" in context
    assert "Short summary" in context
    assert "Search result title" in context
    assert hidden_passage not in context
    assert "This passage must not reach the planner" not in context
    assert "Resolve the author" in context
    assert "SEARCH only" in client.calls[0][0][0]["content"]
    assert "Fact count alone is not research progress" in client.calls[0][0][0]["content"]
    compact_context = json.loads(context)
    assert compact_context["recent_actions"]["last_locate_outcome"] == {
        "document_id": "doc-1",
        "query": "author evidence",
        "constraint_progress": False,
        "new_supporting_fact_ids": [],
        "resolved_entity_keys": [],
    }


def test_fake_llm_client_drives_the_phase_2_graph_loop() -> None:
    client = FakeLLMClient(
        [
            Initialization(target=Target(description="Answer", answer_type="text")),
            ActionDecision(action=SearchAction(goal="Find", query="Question")),
            ActionDecision(
                action=OpenAction(goal="Read", url="https://example.com/mock-source")
            ),
            ActionDecision(
                action=AnswerAction(
                    answer="Mock answer",
                    supporting_fact_ids=[],
                    supporting_constraint_ids=[],
                )
            ),
        ]
    )
    graph = build_research_graph(
        LLMPlanner(client),
        FakeSearchGateway(),
        FakeDocumentOpener(),
        FakeDocumentRetriever(),
        FakeFactExtractor(),
        AnswerGuard(),
    )

    result = asyncio.run(
        graph.ainvoke(
            {"research": ResearchState(question="Question", max_steps=3), "action": None}
        )
    )

    assert result["research"].status == "budget_exhausted"
    assert result["research"].answer is None
