import asyncio

import pytest
from pydantic import BaseModel

from app.llm.client import Message
from app.research.graph import build_research_graph
from app.research.planner import Initialization, LLMPlanner
from app.research.schemas import (
    ActionDecision,
    AnswerAction,
    DocumentRef,
    Fact,
    LocateAction,
    OpenAction,
    SearchAction,
    Target,
)
from app.research.state import ResearchState
from app.tools.document import StubDocumentOpener
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


def test_llm_planner_initializes_with_a_structured_contract() -> None:
    client = FakeLLMClient(
        [Initialization(target=Target(description="Find an author", answer_type="person_name"))]
    )

    result = asyncio.run(LLMPlanner(client).initialize("Who wrote this work?"))

    assert result.target.answer_type == "person_name"
    assert client.calls[0][1] is Initialization


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
    )

    asyncio.run(planner.next_action(state))

    context = client.calls[0][0][1]["content"]
    assert "Known claim" in context
    assert "Short summary" in context
    assert "Search result title" in context
    assert hidden_passage not in context


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
        StubDocumentOpener(),
    )

    result = asyncio.run(
        graph.ainvoke({"research": ResearchState(question="Question"), "action": None})
    )

    assert result["research"].status == "completed"
    assert result["research"].answer == "Mock answer"
