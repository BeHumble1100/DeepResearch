import asyncio
import json

import pytest
from pydantic import BaseModel

from app.llm.client import Message
from app.research.graph import build_research_graph
from app.research.guard import AnswerGuard
from app.research.planner import (
    Initialization,
    InitializationProposal,
    LLMPlanner,
    _compact_state_view,
    _materialize_initialization,
)
from app.research.schemas import (
    ActionDecision,
    AnswerAction,
    CandidateScope,
    Constraint,
    ConstraintEvidence,
    ConstraintProposal,
    DocumentRef,
    Fact,
    LocateAction,
    OpenAction,
    Passage,
    SearchAction,
    Target,
    TraceGuardResult,
    ConstraintEvidence,
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
    async def search(
        self,
        *,
        goal: str,
        query: str,
        unresolved_required_constraints=None,
        resolved_entities=None,
    ) -> list[SearchResult]:
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
                    ConstraintProposal(
                        description=" Author wrote the work ", subject=" Author ", kind="acceptance"
                    ),
                    ConstraintProposal(
                        description="Author wrote the work", subject="Author", kind="acceptance"
                    ),
                    ConstraintProposal(
                        description="Work was published", required=False, kind="research_clue"
                    ),
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
                constraint_evidence=[
                    ConstraintEvidence(constraint_id="c1", status="supported"),
                    ConstraintEvidence(constraint_id="c2", status="contradicted"),
                ],
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
                step=0,
                action="open",
                action_input={"url": "https://example.com/blocked"},
                observation_summary="Source rejected the request.",
                remaining_step_budget=8,
                validation_rejection_reason=(
                    "OPEN document fetch or parse failed: "
                    "Document request failed: https://example.com/blocked"
                ),
            ),
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
    assert "Action policy" in client.calls[0][0][0]["content"]
    assert "Research progress is not the number of collected facts" in client.calls[0][0][0]["content"]
    assert "ACTION PRIORITY" in client.calls[0][0][0]["content"]
    compact_context = json.loads(context)
    assert compact_context["recent_actions"]["last_locate_outcome"] == {
        "document_id": "doc-1",
        "query": "author evidence",
        "constraint_progress": False,
        "new_supporting_fact_ids": [],
        "resolved_entity_keys": [],
    }
    assert compact_context["recent_actions"]["failed_open_urls"] == [
        "https://example.com/blocked"
    ]
    assert compact_context["recent_actions"]["last_guard_rejection"] is None
    assert compact_context["openable_search_results"] == [
        {
            "source_rank": 1,
            "url": "https://example.com/search-result",
            "title": "Search result title",
            "snippet": "Search snippet",
        }
    ]
    assert compact_context["documents_requiring_locate"] == []
    assert compact_context["known_facts"][0]["constraint_evidence"] == [
        {"constraint_id": "c1", "status": "supported"},
        {"constraint_id": "c2", "status": "contradicted"},
    ]


def test_compact_context_exposes_latest_guard_rejection_until_new_evidence() -> None:
    state = ResearchState(
        question="Question",
        trace=[
            ResearchTraceEntry(
                step=1,
                action="answer",
                action_input={"answer": "Ada", "candidate_scope_id": None},
                observation_summary="Guard rejected.",
                remaining_step_budget=3,
                answer_supporting_fact_ids=["fact-1"],
                answer_supporting_constraint_ids=["c1"],
                guard_result=TraceGuardResult(
                    accepted=False, reject_reasons=["Required constraint lacks support: c1"]
                ),
            )
        ],
    )

    context = json.loads(_compact_state_view(state))

    assert context["recent_actions"]["last_guard_rejection"] == {
        "answer": "Ada",
        "supporting_fact_ids": ["fact-1"],
        "supporting_constraint_ids": ["c1"],
        "candidate_scope_id": None,
        "reject_reasons": ["Required constraint lacks support: c1"],
    }


def test_compact_context_excludes_hosts_that_explicitly_denied_opening() -> None:
    state = ResearchState(
        question="Question",
        search_results=[
            SearchResult(url="https://blocked.example/second", title="Blocked again"),
            SearchResult(url="https://available.example/source", title="Available source"),
        ],
        trace=[
            ResearchTraceEntry(
                step=1,
                action="open",
                action_input={"url": "https://blocked.example/first"},
                observation_summary="Blocked.",
                remaining_step_budget=3,
                validation_rejection_reason=(
                    "OPEN document fetch or parse failed: Document request returned HTTP 403: "
                    "https://blocked.example/first"
                ),
            )
        ],
    )

    context = json.loads(_compact_state_view(state))

    assert context["recent_actions"]["failed_open_hosts"] == ["blocked.example"]
    assert "search_results" not in context
    assert context["openable_search_results"] == [
        {
            "source_rank": 1,
            "url": "https://available.example/source",
            "title": "Available source",
            "snippet": None,
        }
    ]


def test_compact_context_exposes_structured_open_failure_categories() -> None:
    state = ResearchState(
        question="Question",
        trace=[
            ResearchTraceEntry(
                step=1,
                action="open",
                action_input={"url": "https://example.com/blocked"},
                observation_summary="Blocked.",
                remaining_step_budget=3,
                validation_rejection_reason="OPEN document fetch or parse failed: Document request returned HTTP 403.",
                source_failure_category="access_denied",
            )
        ],
    )

    context = json.loads(_compact_state_view(state))

    assert context["recent_actions"]["failed_open_sources"] == [
        {
            "url": "https://example.com/blocked",
            "host": "example.com",
            "category": "access_denied",
        }
    ]


def test_llm_planner_includes_deterministic_candidate_scope_summary() -> None:
    client = FakeLLMClient([ActionDecision(action=SearchAction(goal="Find", query="Question"))])
    state = ResearchState(
        question="Question",
        candidate_scopes=[CandidateScope(id="cand_1", label="Candidate A")],
        facts=[
            Fact(
                id="fact-1",
                statement="Candidate A meets a requirement.",
                source_url="https://example.com",
                passage="Hidden passage",
                confidence=0.8,
                evidence_scope_id="cand_1",
                constraint_evidence=[ConstraintEvidence(constraint_id="c1", status="supported")],
            )
        ],
    )

    asyncio.run(LLMPlanner(client).next_action(state))

    context = json.loads(client.calls[0][0][1]["content"])
    assert context["candidate_scopes"] == [
        {"id": "cand_1", "label": "Candidate A", "constraint_evidence": {"c1": "supported"}}
    ]


def test_initialization_preserves_constraint_kinds_and_compact_context_exposes_them() -> None:
    initialization = InitializationProposal(
        target=Target(description="Answer", answer_type="text"),
        constraints=[
            ConstraintProposal(description="Final condition", kind="acceptance"),
            ConstraintProposal(description="Bridge clue", kind="research_clue"),
        ],
    )

    state = ResearchState(
        question="Question",
        constraints=_materialize_initialization(initialization).constraints,
    )
    client = FakeLLMClient([ActionDecision(action=SearchAction(goal="Find", query="Question"))])

    asyncio.run(LLMPlanner(client).next_action(state))

    context = json.loads(client.calls[0][0][1]["content"])
    assert [(item["id"], item["kind"]) for item in context["constraints"]] == [
        ("c1", "acceptance"),
        ("c2", "research_clue"),
    ]


def test_initialization_adds_target_acceptance_constraint_when_model_returns_only_clues() -> None:
    initialization = InitializationProposal(
        target=Target(description="Identify the final role", answer_type="role"),
        constraints=[ConstraintProposal(description="Historical broadcast clue", kind="research_clue")],
    )

    result = _materialize_initialization(initialization)

    assert [(item.id, item.kind, item.description) for item in result.constraints] == [
        ("c1", "research_clue", "Historical broadcast clue"),
        ("c2", "acceptance", "Identify the final role"),
    ]


def test_initialization_prompt_keeps_target_type_out_of_acceptance_constraints() -> None:
    client = FakeLLMClient(
        [
            InitializationProposal(
                target=Target(description="Author", answer_type="person"),
                constraints=[ConstraintProposal(description="Author wrote the work", kind="acceptance")],
            )
        ]
    )

    asyncio.run(LLMPlanner(client).initialize("Who wrote the work?"))

    prompt = client.calls[0][0][0]["content"]
    assert "Target already owns answer type and format" in prompt
    assert "Keep acceptance constraints answer-neutral" in prompt


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
