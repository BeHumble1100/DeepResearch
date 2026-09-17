"""Phase 2 minimal SEARCH / OPEN / ANSWER research loop."""

from __future__ import annotations

import json
from typing import TypedDict
from urllib.parse import urlsplit

from langgraph.graph import END, START, StateGraph

from .planner import Planner, _compact_state_view, _failed_open_hosts
from .schemas import (
    Action,
    AnswerAction,
    CandidateScope,
    LocateAction,
    OpenAction,
    SearchAction,
    TraceGuardResult,
    TraceCandidateScope,
    SourceFailureCategory,
)
from .state import ResearchState
from .evidence import FactExtractor, apply_fact_extraction, opening_passage
from .guard import AnswerGuard
from .trace import append_action_trace, append_event_trace
from app.tools.document import DocumentOpenError, DocumentOpener
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
        if action.query in research.executed_queries:
            return {
                "research": _reject_action(
                    research,
                    action=action,
                    planner_context=state.get("planner_context"),
                    reason="SEARCH query exactly duplicates an earlier query in this research run.",
                )
            }
        required_constraints = [
            {
                "id": constraint.id,
                "description": constraint.description,
                "kind": constraint.kind,
            }
            for constraint in research.constraints
            if constraint.required
        ]
        results = await search_gateway.search(
            goal=action.goal,
            query=action.query,
            unresolved_required_constraints=required_constraints,
            resolved_entities=research.resolved_entities,
        )
        updated = research.model_copy(
            update={
                "current_goal": action.goal,
                "executed_queries": [*research.executed_queries, action.query],
                "search_results": results,
                "discovered_urls": list(
                    dict.fromkeys([*research.discovered_urls, *(result.url for result in results)])
                ),
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
        if _is_duplicate_open(research, action):
            return {
                "research": _reject_action(
                    research,
                    action=action,
                    planner_context=state.get("planner_context"),
                    reason="OPEN URL exactly duplicates an earlier OPEN attempt in this research run.",
                )
            }
        scope_rejection = _validate_open_scope(research, action)
        if scope_rejection:
            return {
                "research": _reject_action(
                    research,
                    action=action,
                    planner_context=state.get("planner_context"),
                    reason=scope_rejection,
                )
            }
        known_urls = set(research.discovered_urls)
        if action.url not in known_urls:
            return {
                "research": _reject_action(
                    research,
                    action=action,
                    planner_context=state.get("planner_context"),
                    reason="OPEN URL was not discovered by a prior SEARCH in this research run.",
                )
            }
        if urlsplit(action.url).netloc.lower() in set(_failed_open_hosts(research)):
            return {
                "research": _reject_action(
                    research,
                    action=action,
                    planner_context=state.get("planner_context"),
                    reason="OPEN URL host was denied by an earlier document request in this research run.",
                )
            }
        try:
            document = await document_opener.open(url=action.url)
        except DocumentOpenError as error:
            return {
                "research": _reject_action(
                    research,
                    action=action,
                    planner_context=state.get("planner_context"),
                    reason=f"OPEN document fetch or parse failed: {error}",
                    source_failure_category=error.category,
                )
            }
        created_scope = None
        if action.new_candidate_label is not None:
            created_scope = CandidateScope(
                id=f"cand_{len(research.candidate_scopes) + 1}",
                label=action.new_candidate_label.strip(),
            )
            document = document.model_copy(update={"evidence_scope_id": created_scope.id})
        elif action.candidate_scope_id is not None:
            document = document.model_copy(update={"evidence_scope_id": action.candidate_scope_id})
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
                "candidate_scopes": (
                    [*updated.candidate_scopes, created_scope]
                    if created_scope is not None
                    else updated.candidate_scopes
                ),
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
                created_candidate_scope=(
                    TraceCandidateScope(
                        id=created_scope.id,
                        label=created_scope.label,
                        originating_document_id=document.id,
                    )
                    if created_scope is not None
                    else None
                ),
            )
        }

    async def locate(state: GraphState) -> dict[str, ResearchState]:
        research = state["research"]
        action = state["action"]
        if not isinstance(action, LocateAction):
            raise ValueError("Locate node requires a LocateAction.")
        if _is_duplicate_locate(research, action):
            return {
                "research": _reject_action(
                    research,
                    action=action,
                    planner_context=state.get("planner_context"),
                    reason=(
                        "LOCATE document_id and query exactly duplicate an earlier LOCATE "
                        "in this research run."
                    ),
                )
            }
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
        if _is_duplicate_rejected_answer(research, action):
            return {
                "research": _reject_action(
                    research,
                    action=action,
                    planner_context=state.get("planner_context"),
                    reason=(
                        "ANSWER exactly duplicates a prior guard-rejected proposal without "
                        "new evidence."
                    ),
                )
            }
        if action.candidate_scope_id is not None and action.candidate_scope_id not in {
            scope.id for scope in research.candidate_scopes
        }:
            return {
                "research": _reject_action(
                    research,
                    action=action,
                    planner_context=state.get("planner_context"),
                    reason="ANSWER candidate_scope_id does not refer to an existing candidate scope.",
                )
            }
        if not any(
            fact_id in {fact.id for fact in research.facts}
            for fact_id in action.supporting_fact_ids
        ):
            return {
                "research": _reject_action(
                    research,
                    action=action,
                    planner_context=state.get("planner_context"),
                    reason="ANSWER proposal does not cite any real supporting Fact from this research run.",
                )
            }
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
            accepted = _project_accepted_constraints(research, action).model_copy(
                update={"status": "answer_accepted"}
            )
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

    def route_answer(state: GraphState) -> str:
        return "guard" if state["research"].status == "answer_proposed" else "budget"

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
    graph.add_conditional_edges("answer", route_answer, {"guard": "guard", "budget": "budget"})
    graph.add_conditional_edges("guard", route_answer_guard, {"accept": "finish", "reject": "budget"})
    graph.add_edge("finish", END)
    graph.add_edge("exhausted", END)
    return graph.compile()


def _reject_action(
    research: ResearchState,
    *,
    action: Action,
    planner_context: dict[str, object] | None,
    reason: str,
    source_failure_category: SourceFailureCategory | None = None,
) -> ResearchState:
    """Record a safe deterministic action rejection and return to planning."""
    rejected = research.model_copy(
        update={
            "current_goal": getattr(action, "goal", research.current_goal),
            "step_count": research.step_count + 1,
            "status": "planning",
        }
    )
    return append_action_trace(
        research,
        rejected,
        action=action,
        observation_summary="Rejected action validation: " + reason,
        planner_context=planner_context,
        validation_rejection_reason=reason,
        source_failure_category=source_failure_category,
    )


def _is_duplicate_locate(research: ResearchState, action: LocateAction) -> bool:
    """Match only exact document/query pairs; semantic similarity is intentionally out of scope."""
    return any(
        entry.action == "locate"
        and entry.action_input.get("document_id") == action.document_id
        and entry.action_input.get("query") == action.query
        for entry in research.trace
    )


def _is_duplicate_open(research: ResearchState, action: OpenAction) -> bool:
    """Prevent repeated network access to the exact same URL within one run."""
    return any(
        entry.action == "open" and entry.action_input.get("url") == action.url
        for entry in research.trace
    )


def _is_duplicate_rejected_answer(research: ResearchState, action: AnswerAction) -> bool:
    """Reject only an unchanged proposal after Guard rejection and before new evidence."""
    candidate = _answer_signature(action.model_dump())
    for entry in reversed(research.trace):
        if entry.new_facts or entry.resolved_entities:
            return False
        if entry.action != "answer" or entry.guard_result is None:
            continue
        if entry.guard_result.accepted:
            return False
        return candidate == _answer_signature(entry.action_input)
    return False


def _answer_signature(payload: dict[str, object]) -> tuple[object, ...]:
    return (
        payload.get("answer"),
        tuple(payload.get("supporting_fact_ids", [])),
        tuple(payload.get("supporting_constraint_ids", [])),
        payload.get("candidate_scope_id"),
    )


def _validate_open_scope(research: ResearchState, action: OpenAction) -> str | None:
    """Validate state-owned candidate namespaces before any URL is opened."""
    if action.candidate_scope_id is not None and action.new_candidate_label is not None:
        return "OPEN must provide either candidate_scope_id or new_candidate_label, not both."
    if action.new_candidate_label is not None and not action.new_candidate_label.strip():
        return "OPEN new_candidate_label must not be blank."
    if action.candidate_scope_id is not None and action.candidate_scope_id not in {
        scope.id for scope in research.candidate_scopes
    }:
        return "OPEN candidate_scope_id does not refer to an existing candidate scope."
    return None


def _project_accepted_constraints(research: ResearchState, action: AnswerAction) -> ResearchState:
    """Project only the accepted answer's scoped support into global final state."""
    constraints = []
    for constraint in research.constraints:
        if not (constraint.required and constraint.kind == "acceptance"):
            constraints.append(constraint)
            continue
        supporting_fact_ids = [
            fact.id
            for fact in research.facts
            if fact.evidence_scope_id == action.candidate_scope_id
            and any(
                relation.constraint_id == constraint.id and relation.status == "supported"
                for relation in fact.constraint_evidence
            )
        ]
        constraints.append(
            constraint.model_copy(
                update={"status": "supported", "supporting_fact_ids": supporting_fact_ids}
            )
        )
    return research.model_copy(update={"constraints": constraints})
