from app.research.graph import build_research_graph
from app.research.planner import MockPlanner
from app.research.schemas import SearchAction
from app.research.state import ResearchState


def test_mock_question_moves_through_the_graph() -> None:
    graph = build_research_graph(MockPlanner())

    result = graph.invoke(
        {"research": ResearchState(question="Who wrote this work?"), "action": None}
    )

    research = result["research"]
    assert research.target is not None
    assert research.target.description == "Who wrote this work?"
    assert research.status == "awaiting_tool_execution"
    assert research.step_count == 1
    assert isinstance(result["action"], SearchAction)
