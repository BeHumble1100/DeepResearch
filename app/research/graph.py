"""Phase 1 LangGraph skeleton for controlled research-state transitions."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .planner import Planner
from .schemas import Action
from .state import ResearchState


class GraphState(TypedDict):
    research: ResearchState
    action: Action | None


def build_research_graph(planner: Planner):
    """Build the skeleton graph; action execution is deferred to later phases."""

    def initialize(state: GraphState) -> dict[str, ResearchState]:
        research = state["research"]
        initialization = planner.initialize(research.question)
        return {
            "research": research.model_copy(
                update={
                    "target": initialization.target,
                    "constraints": initialization.constraints,
                    "status": "planning",
                }
            )
        }

    def plan(state: GraphState) -> dict[str, Action]:
        return {"action": planner.next_action(state["research"])}

    def mark_action_pending(state: GraphState) -> dict[str, ResearchState]:
        research = state["research"]
        action = state["action"]
        assert action is not None
        return {
            "research": research.model_copy(
                update={
                    "current_goal": getattr(action, "goal", None),
                    "step_count": research.step_count + 1,
                    "status": "awaiting_tool_execution",
                }
            )
        }

    def mark_answer_pending(state: GraphState) -> dict[str, ResearchState]:
        research = state["research"]
        return {
            "research": research.model_copy(
                update={
                    "step_count": research.step_count + 1,
                    "status": "awaiting_answer_guard",
                }
            )
        }

    def route_action(state: GraphState) -> str:
        action = state["action"]
        if action is None:
            raise ValueError("Planner must return an action before routing.")
        return action.type

    graph = StateGraph(GraphState)
    graph.add_node("initialize", initialize)
    graph.add_node("plan", plan)
    graph.add_node("search", mark_action_pending)
    graph.add_node("open", mark_action_pending)
    graph.add_node("locate", mark_action_pending)
    graph.add_node("answer", mark_answer_pending)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "plan")
    graph.add_conditional_edges(
        "plan",
        route_action,
        {"search": "search", "open": "open", "locate": "locate", "answer": "answer"},
    )
    graph.add_edge("search", END)
    graph.add_edge("open", END)
    graph.add_edge("locate", END)
    graph.add_edge("answer", END)
    return graph.compile()
