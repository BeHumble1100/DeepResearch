import asyncio

import pytest

from app.research.graph import build_research_graph
from app.research.planner import MockPlanner
from app.research.schemas import OpenAction
from app.research.state import ResearchState
from app.tools.document import StubDocumentOpener
from app.tools.search import StubSearchGateway


def make_graph(planner: MockPlanner) -> object:
    return build_research_graph(planner, StubSearchGateway(), StubDocumentOpener())


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
    assert research.step_count == 3
    assert research.executed_queries == ["Who wrote this work?"]
    assert research.visited_urls == ["https://example.com/mock-source"]
    assert len(research.documents) == 1
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
