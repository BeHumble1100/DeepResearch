"""Phase 2 minimal SEARCH / OPEN / ANSWER research loop."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .planner import Planner
from .schemas import Action, AnswerAction, OpenAction, SearchAction
from .state import ResearchState
from app.tools.document import DocumentOpener
from app.tools.search import SearchGateway


class GraphState(TypedDict):
    research: ResearchState
    action: Action | None


def build_research_graph(
    planner: Planner,
    search_gateway: SearchGateway,
    document_opener: DocumentOpener,
):
    """Build the minimal research loop with injected search and document boundaries."""

    async def initialize(state: GraphState) -> dict[str, ResearchState]:
        research = state["research"]
        initialization = await planner.initialize(research.question)
        return {
            "research": research.model_copy(
                update={
                    "target": initialization.target,
                    "constraints": initialization.constraints,
                    "status": "planning",
                }
            )
        }

    async def plan(state: GraphState) -> dict[str, Action]:
        return {"action": await planner.next_action(state["research"])}

    async def search(state: GraphState) -> dict[str, ResearchState]:
        research = state["research"]
        action = state["action"]
        if not isinstance(action, SearchAction):
            raise ValueError("Search node requires a SearchAction.")
        results = await search_gateway.search(goal=action.goal, query=action.query)
        return {
            "research": research.model_copy(
                update={
                    "current_goal": action.goal,
                    "executed_queries": [*research.executed_queries, action.query],
                    "search_results": results,
                    "step_count": research.step_count + 1,
                    "status": "planning",
                }
            )
        }

    async def open_document(state: GraphState) -> dict[str, ResearchState]:
        research = state["research"]
        action = state["action"]
        if not isinstance(action, OpenAction):
            raise ValueError("Open node requires an OpenAction.")
        known_urls = {result.url for result in research.search_results}
        if action.url not in known_urls:
            raise ValueError("OpenAction URL must come from a prior search result.")
        document = await document_opener.open(url=action.url)
        return {
            "research": research.model_copy(
                update={
                    "current_goal": action.goal,
                    "documents": [*research.documents, document],
                    "visited_urls": [*research.visited_urls, action.url],
                    "step_count": research.step_count + 1,
                    "status": "planning",
                }
            )
        }

    def propose_answer(state: GraphState) -> dict[str, ResearchState]:
        research = state["research"]
        action = state["action"]
        if not isinstance(action, AnswerAction):
            raise ValueError("Answer node requires an AnswerAction.")
        return {
            "research": research.model_copy(
                update={
                    "answer": action.answer,
                    "step_count": research.step_count + 1,
                    "status": "answer_proposed",
                }
            )
        }

    def finish(state: GraphState) -> dict[str, ResearchState]:
        research = state["research"]
        if research.answer is None:
            raise ValueError("Cannot finish without an answer proposal.")
        return {"research": research.model_copy(update={"status": "completed"})}

    def exhaust_budget(state: GraphState) -> dict[str, ResearchState]:
        research = state["research"]
        return {"research": research.model_copy(update={"status": "budget_exhausted"})}

    def route_budget(state: GraphState) -> str:
        return "exhausted" if state["research"].step_count >= state["research"].max_steps else "plan"

    def route_action(state: GraphState) -> str:
        action = state["action"]
        if action is None:
            raise ValueError("Planner must return an action before routing.")
        if action.type == "locate":
            raise ValueError("LOCATE is not enabled until Phase 5.")
        return action.type

    graph = StateGraph(GraphState)
    graph.add_node("initialize", initialize)
    graph.add_node("budget", lambda state: {})
    graph.add_node("plan", plan)
    graph.add_node("search", search)
    graph.add_node("open", open_document)
    graph.add_node("answer", propose_answer)
    graph.add_node("finish", finish)
    graph.add_node("exhausted", exhaust_budget)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "budget")
    graph.add_conditional_edges("budget", route_budget, {"plan": "plan", "exhausted": "exhausted"})
    graph.add_conditional_edges(
        "plan",
        route_action,
        {"search": "search", "open": "open", "answer": "answer"},
    )
    graph.add_edge("search", "budget")
    graph.add_edge("open", "budget")
    graph.add_edge("answer", "finish")
    graph.add_edge("finish", END)
    graph.add_edge("exhausted", END)
    return graph.compile()
