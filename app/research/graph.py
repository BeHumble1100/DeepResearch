"""Phase 2 minimal SEARCH / OPEN / ANSWER research loop."""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .planner import Planner, _compact_state_view
from .schemas import (
    Action,
    AnswerAction,
    LocateAction,
    OpenAction,
    SearchAction,
    TraceGuardResult,
)
from .state import ResearchState
from .evidence import FactExtractor, apply_fact_extraction, opening_passage
from .guard import AnswerGuard
from .trace import append_action_trace, append_event_trace
from app.tools.document import DocumentOpener
from app.tools.search import SearchGateway
from app.tools.retrieval import DocumentRetriever


class GraphState(TypedDict):
    research: ResearchState
    action: Action | None
    planner_context: dict[str, object] | None


def build_research_graph(
    planner: Planner,
    search_gateway: SearchGateway,
    document_opener: DocumentOpener,
    document_retriever: DocumentRetriever,
    fact_extractor: FactExtractor,
    answer_guard: AnswerGuard,
    *,
    fact_extraction_open_max_chars: int = 4000,
):
    """Build the minimal research loop with injected search and document boundaries."""
    if fact_extraction_open_max_chars <= 0:
        raise ValueError("Fact extraction opening length must be positive.")

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

    async def plan(state: GraphState) -> dict[str, object]:
        research = state["research"]
        planner_context = json.loads(_compact_state_view(research))
        return {
            "action": await planner.next_action(research),
            "planner_context": planner_context,
        }

    async def search(state: GraphState) -> dict[str, ResearchState]:
        research = state["research"]
        action = state["action"]
        if not isinstance(action, SearchAction):
            raise ValueError("Search node requires a SearchAction.")
        results = await search_gateway.search(goal=action.goal, query=action.query)
        updated = research.model_copy(
            update={
                "current_goal": action.goal,
                "executed_queries": [*research.executed_queries, action.query],
                "search_results": results,
                "step_count": research.step_count + 1,
                "status": "planning",
            }
        )
        return {
            "research": append_action_trace(
                research,
                updated,
                action=action,
                observation_summary=f"Search returned {len(results)} normalized results.",
                planner_context=state.get("planner_context"),
                search_results=results,
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
        passage = await opening_passage(
            document, max_chars=fact_extraction_open_max_chars
        )
        extraction = await fact_extractor.extract(
            document=document,
            passages=[passage],
            constraints=research.constraints,
        )
        updated = apply_fact_extraction(
            research,
            document=document,
            passages=[passage],
            extraction=extraction,
            evidence_kind="open",
        )
        completed = updated.model_copy(
            update={
                "current_goal": action.goal,
                "documents": [*updated.documents, document],
                "visited_urls": [*updated.visited_urls, action.url],
                "step_count": updated.step_count + 1,
                "status": "planning",
            }
        )
        return {
            "research": append_action_trace(
                research,
                completed,
                action=action,
                observation_summary=(
                    f"Opened {document.content_type} document {document.id} and extracted facts "
                    "from its bounded opening excerpt."
                ),
                planner_context=state.get("planner_context"),
            )
        }

    async def locate(state: GraphState) -> dict[str, ResearchState]:
        research = state["research"]
        action = state["action"]
        if not isinstance(action, LocateAction):
            raise ValueError("Locate node requires a LocateAction.")
        document = next((item for item in research.documents if item.id == action.document_id), None)
        if document is None:
            raise ValueError("LocateAction document_id must refer to an opened document.")
        passages = await document_retriever.retrieve(
            document=document,
            goal=action.goal,
            query=action.query,
        )
        extraction = await fact_extractor.extract(
            document=document,
            passages=passages,
            constraints=research.constraints,
        )
        updated = apply_fact_extraction(
            research,
            document=document,
            passages=passages,
            extraction=extraction,
            evidence_kind="locate",
        )
        completed = updated.model_copy(
            update={
                "current_goal": action.goal,
                "located_passages": passages,
                "step_count": updated.step_count + 1,
                "status": "planning",
            }
        )
        return {
            "research": append_action_trace(
                research,
                completed,
                action=action,
                observation_summary=(
                    f"Located {len(passages)} relevant passages in document {document.id}."
                ),
                planner_context=state.get("planner_context"),
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

    def guard_answer(state: GraphState) -> dict[str, ResearchState]:
        research = state["research"]
        action = state["action"]
        if not isinstance(action, AnswerAction):
            raise ValueError("Answer guard requires an AnswerAction.")
        result = answer_guard.check(research=research, proposal=action)
        if result.accepted:
            accepted = research.model_copy(update={"status": "answer_accepted"})
            return {
                "research": append_action_trace(
                    research,
                    accepted,
                    action=action,
                    observation_summary="Answer guard accepted the evidence-backed proposal.",
                    planner_context=state.get("planner_context"),
                    guard_result=TraceGuardResult(accepted=True),
                )
            }
        rejected = research.model_copy(update={"answer": None, "status": "planning"})
        return {
            "research": append_action_trace(
                research,
                rejected,
                action=action,
                observation_summary="Answer guard rejected the proposal: " + "; ".join(result.reasons),
                planner_context=state.get("planner_context"),
                guard_result=TraceGuardResult(
                    accepted=False, reject_reasons=list(result.reasons)
                ),
            )
        }

    def exhaust_budget(state: GraphState) -> dict[str, ResearchState]:
        research = state["research"]
        exhausted = research.model_copy(update={"status": "budget_exhausted"})
        return {
            "research": append_event_trace(
                exhausted,
                action="budget_exhausted",
                action_input={"max_steps": exhausted.max_steps, "step_count": exhausted.step_count},
                observation_summary="Research step budget was exhausted.",
            )
        }

    def route_budget(state: GraphState) -> str:
        return "exhausted" if state["research"].step_count >= state["research"].max_steps else "plan"

    def route_action(state: GraphState) -> str:
        action = state["action"]
        if action is None:
            raise ValueError("Planner must return an action before routing.")
        return action.type

    def route_answer_guard(state: GraphState) -> str:
        return "accept" if state["research"].status == "answer_accepted" else "reject"

    graph = StateGraph(GraphState)
    graph.add_node("initialize", initialize)
    graph.add_node("budget", lambda state: {})
    graph.add_node("plan", plan)
    graph.add_node("search", search)
    graph.add_node("open", open_document)
    graph.add_node("locate", locate)
    graph.add_node("answer", propose_answer)
    graph.add_node("guard", guard_answer)
    graph.add_node("finish", finish)
    graph.add_node("exhausted", exhaust_budget)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "budget")
    graph.add_conditional_edges("budget", route_budget, {"plan": "plan", "exhausted": "exhausted"})
    graph.add_conditional_edges(
        "plan",
        route_action,
        {"search": "search", "open": "open", "locate": "locate", "answer": "answer"},
    )
    graph.add_edge("search", "budget")
    graph.add_edge("open", "budget")
    graph.add_edge("locate", "budget")
    graph.add_edge("answer", "guard")
    graph.add_conditional_edges("guard", route_answer_guard, {"accept": "finish", "reject": "budget"})
    graph.add_edge("finish", END)
    graph.add_edge("exhausted", END)
    return graph.compile()
