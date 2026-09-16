import asyncio

import pytest

from app.research.graph import build_research_graph
from app.research.planner import MockPlanner
from app.research.schemas import OpenAction, Passage, SearchResult
from app.research.state import ResearchState


class FakeSearchGateway:
    async def search(self, *, goal: str, query: str) -> list[SearchResult]:
        return [SearchResult(url="https://example.com/mock-source", title="Mock source")]


class FakeDocumentOpener:
    async def open(self, *, url: str):
        from app.research.schemas import DocumentRef

        return DocumentRef(
            id="mock-document",
            url=url,
            content_type="text/html",
            local_path=".deepresearch/documents/mock-document/content.txt",
        )


class FakeDocumentRetriever:
    async def retrieve(self, *, document, goal: str, query: str) -> list[Passage]:
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


def make_graph(planner: MockPlanner) -> object:
    return build_research_graph(
        planner,
        FakeSearchGateway(),
        FakeDocumentOpener(),
        FakeDocumentRetriever(),
    )


def test_mock_question_moves_through_search_open_and_answer() -> None:
    graph = make_graph(MockPlanner())

    result = asyncio.run(
        graph.ainvoke(
            {"research": ResearchState(question="Who wrote this work?"), "action": None}
        )
    )

    research = result["research"]
    assert research.target is not None
    assert research.target.description == "Who wrote this work?"
    assert research.status == "completed"
    assert research.step_count == 4
    assert research.executed_queries == ["Who wrote this work?"]
    assert research.visited_urls == ["https://example.com/mock-source"]
    assert len(research.documents) == 1
    assert research.located_passages[0].text == "Relevant passage"
    assert research.answer == "Mock answer"


class InvalidOpenPlanner(MockPlanner):
    async def next_action(self, state: ResearchState) -> OpenAction:
        return OpenAction(goal="Open an unknown URL", url="https://example.com/unknown")


def test_open_rejects_urls_not_returned_by_search() -> None:
    graph = make_graph(InvalidOpenPlanner())

    with pytest.raises(ValueError, match="prior search result"):
        asyncio.run(graph.ainvoke({"research": ResearchState(question="Question"), "action": None}))


def test_loop_stops_when_the_step_budget_is_exhausted() -> None:
    graph = make_graph(MockPlanner())

    result = asyncio.run(
        graph.ainvoke(
            {"research": ResearchState(question="Question", max_steps=2), "action": None}
        )
    )

    research = result["research"]
    assert research.status == "budget_exhausted"
    assert research.answer is None
    assert research.step_count == 2
