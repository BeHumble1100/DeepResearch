import asyncio
from pathlib import Path

from app.research.graph import build_research_graph
from app.research.planner import MockPlanner
from app.research.guard import AnswerGuard
from app.research.schemas import (
    AnswerAction,
    ConstraintEvidence,
    Constraint,
    ExtractedFact,
    FactExtraction,
    LocateAction,
    OpenAction,
    Passage,
    SearchAction,
    SearchResult,
)
from app.research.state import ResearchState
from app.tools.document import DocumentOpenError


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
    def __init__(self) -> None:
        self.opened_urls: list[str] = []

    async def open(self, *, url: str):
        from app.research.schemas import DocumentRef

        self.opened_urls.append(url)
        return DocumentRef(
            id="mock-document",
            url=url,
            content_type="text/html",
            local_path=str(Path(__file__)),
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


class FakeFactExtractor:
    async def extract(self, *, document, passages, constraints) -> FactExtraction:
        return FactExtraction(
            facts=[
                ExtractedFact(
                    statement="Mock source supports the answer.",
                    confidence=0.9,
                    passage_id=passages[0].id,
                )
            ]
        )


def make_graph(
    planner: MockPlanner,
    *,
    search_gateway: FakeSearchGateway | None = None,
    document_opener: FakeDocumentOpener | None = None,
) -> object:
    return build_research_graph(
        planner,
        search_gateway or FakeSearchGateway(),
        document_opener or FakeDocumentOpener(),
        FakeDocumentRetriever(),
        FakeFactExtractor(),
        AnswerGuard(),
    )


def test_mock_question_moves_through_search_open_and_answer() -> None:
    graph = make_graph(MockPlanner())

    result = asyncio.run(
        graph.ainvoke(
            {
                "research": ResearchState(question="Who wrote this work?", max_steps=4),
                "action": None,
            }
        )
    )

    research = result["research"]
    assert research.target is not None
    assert research.target.description == "Who wrote this work?"
    assert research.status == "completed"
    assert research.step_count == 4
    assert research.executed_queries == ["Who wrote this work?"]
    assert research.discovered_urls == ["https://example.com/mock-source"]
    assert research.visited_urls == ["https://example.com/mock-source"]
    assert len(research.documents) == 1
    assert research.located_passages[0].text == "Relevant passage"
    assert research.answer == "Mock answer"
    assert [entry.action for entry in research.trace] == ["search", "open", "locate", "answer"]
    assert [entry.step for entry in research.trace] == [1, 2, 3, 4]
    assert [entry.remaining_step_budget for entry in research.trace] == [4, 3, 2, 1]
    assert research.trace[0].planner_context is not None
    assert research.trace[0].planner_context["original_question"] == "Who wrote this work?"
    assert [result.model_dump() for result in research.trace[0].search_results] == [
        {"url": "https://example.com/mock-source", "title": "Mock source", "snippet": None}
    ]
    assert research.trace[-1].answer_supporting_fact_ids
    assert research.trace[-1].answer_supporting_constraint_ids == []
    assert research.trace[-1].guard_result is not None
    assert research.trace[-1].guard_result.accepted
    assert research.trace[-1].guard_result.reject_reasons == []
    assert research.trace[1].new_facts
    assert not hasattr(research.trace[1].new_facts[0], "passage")


class InvalidOpenPlanner(MockPlanner):
    async def next_action(self, state: ResearchState) -> OpenAction:
        return OpenAction(goal="Open an unknown URL", url="https://example.com/unknown")


def test_open_rejects_undiscovered_urls_without_calling_opener() -> None:
    opener = FakeDocumentOpener()
    graph = make_graph(InvalidOpenPlanner(), document_opener=opener)

    result = asyncio.run(
        graph.ainvoke(
            {"research": ResearchState(question="Question", max_steps=1), "action": None}
        )
    )

    research = result["research"]
    rejection = research.trace[0]
    assert research.status == "budget_exhausted"
    assert opener.opened_urls == []
    assert rejection.action == "open"
    assert rejection.action_input["url"] == "https://example.com/unknown"
    assert rejection.planner_context is not None
    assert rejection.remaining_step_budget == 1
    assert rejection.validation_rejection_reason == (
        "OPEN URL was not discovered by a prior SEARCH in this research run."
    )


class SearchThenOpenPlanner(MockPlanner):
    def __init__(self) -> None:
        self._actions = [
            SearchAction(goal="Discover source", query="source"),
            OpenAction(goal="Open source", url="https://example.com/mock-source"),
        ]

    async def next_action(self, state: ResearchState):
        return self._actions.pop(0)


class FailingDocumentOpener(FakeDocumentOpener):
    async def open(self, *, url: str):
        self.opened_urls.append(url)
        raise DocumentOpenError("Document request failed: https://example.com/mock-source")


def test_open_backend_failure_is_recoverable_and_traced() -> None:
    opener = FailingDocumentOpener()
    graph = make_graph(SearchThenOpenPlanner(), document_opener=opener)

    result = asyncio.run(
        graph.ainvoke(
            {"research": ResearchState(question="Question", max_steps=2), "action": None}
        )
    )

    research = result["research"]
    rejection = research.trace[1]
    assert research.status == "budget_exhausted"
    assert opener.opened_urls == ["https://example.com/mock-source"]
    assert research.documents == []
    assert rejection.action == "open"
    assert rejection.planner_context is not None
    assert rejection.remaining_step_budget == 1
    assert rejection.validation_rejection_reason == (
        "OPEN document fetch or parse failed: "
        "Document request failed: https://example.com/mock-source"
    )
    assert rejection.source_failure_category == "request_failed"


class SearchThenRepeatedOpenPlanner(MockPlanner):
    def __init__(self) -> None:
        self._actions = [
            SearchAction(goal="Discover source", query="source"),
            OpenAction(goal="Open source", url="https://example.com/mock-source"),
            OpenAction(goal="Retry source", url="https://example.com/mock-source"),
        ]

    async def next_action(self, state: ResearchState):
        return self._actions.pop(0)


def test_duplicate_open_is_rejected_without_second_network_attempt() -> None:
    opener = FailingDocumentOpener()
    graph = make_graph(SearchThenRepeatedOpenPlanner(), document_opener=opener)

    result = asyncio.run(
        graph.ainvoke(
            {"research": ResearchState(question="Question", max_steps=3), "action": None}
        )
    )

    research = result["research"]
    assert opener.opened_urls == ["https://example.com/mock-source"]
    assert research.trace[2].validation_rejection_reason == (
        "OPEN URL exactly duplicates an earlier OPEN attempt in this research run."
    )
    assert research.trace[2].planner_context is not None
    assert research.trace[2].remaining_step_budget == 1


class TwoSearchThenOpenPlanner(MockPlanner):
    def __init__(self) -> None:
        self._actions = [
            SearchAction(goal="Discover first", query="first"),
            SearchAction(goal="Discover second", query="second"),
            OpenAction(goal="Open earlier result", url="https://example.com/first"),
        ]

    async def next_action(self, state: ResearchState):
        return self._actions.pop(0)


class TwoResultSearchGateway:
    async def search(
        self,
        *,
        goal: str,
        query: str,
        unresolved_required_constraints=None,
        resolved_entities=None,
    ) -> list[SearchResult]:
        return [SearchResult(url=f"https://example.com/{query}", title=query)]


def test_open_accepts_url_discovered_before_most_recent_search() -> None:
    opener = FakeDocumentOpener()
    graph = make_graph(
        TwoSearchThenOpenPlanner(),
        search_gateway=TwoResultSearchGateway(),
        document_opener=opener,
    )

    result = asyncio.run(
        graph.ainvoke(
            {"research": ResearchState(question="Question", max_steps=3), "action": None}
        )
    )

    research = result["research"]
    assert research.discovered_urls == [
        "https://example.com/first",
        "https://example.com/second",
    ]
    assert research.search_results == [
        SearchResult(url="https://example.com/second", title="second")
    ]
    assert opener.opened_urls == ["https://example.com/first"]
    assert research.documents[0].url == "https://example.com/first"


def test_repeated_invalid_open_actions_are_limited_by_step_budget() -> None:
    opener = FakeDocumentOpener()
    graph = make_graph(InvalidOpenPlanner(), document_opener=opener)

    result = asyncio.run(
        graph.ainvoke(
            {"research": ResearchState(question="Question", max_steps=2), "action": None}
        )
    )

    research = result["research"]
    assert research.status == "budget_exhausted"
    assert research.step_count == 2
    assert [entry.action for entry in research.trace] == ["open", "open", "budget_exhausted"]
    assert all(entry.validation_rejection_reason for entry in research.trace[:2])
    assert opener.opened_urls == []


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
    assert research.trace[-1].action == "budget_exhausted"


class RejectThenSearchPlanner(MockPlanner):
    def __init__(self) -> None:
        self._actions = [
            SearchAction(goal="Find", query="Question"),
            OpenAction(goal="Read", url="https://example.com/mock-source"),
            AnswerAction(answer="Unsupported", supporting_fact_ids=[], supporting_constraint_ids=[]),
            SearchAction(goal="Try another source", query="Question evidence"),
        ]

    async def next_action(self, state: ResearchState):
        return self._actions.pop(0)


def test_rejected_answer_returns_to_planner() -> None:
    graph = make_graph(RejectThenSearchPlanner())

    result = asyncio.run(
        graph.ainvoke(
            {"research": ResearchState(question="Question", max_steps=4), "action": None}
        )
    )

    research = result["research"]
    assert research.status == "budget_exhausted"
    assert research.answer is None
    assert research.executed_queries == ["Question", "Question evidence"]
    assert research.trace[2].action == "answer"
    assert "Rejected action validation" in research.trace[2].observation_summary
    assert research.trace[2].guard_result is None
    assert research.trace[2].validation_rejection_reason
    assert research.trace[3].action == "search"


class ScriptedPlanner(MockPlanner):
    def __init__(self, actions, *, constraints=None):
        self._actions = actions
        self._constraints = constraints

    async def initialize(self, question: str):
        initialization = await super().initialize(question)
        if self._constraints is None:
            return initialization
        return initialization.model_copy(update={"constraints": self._constraints})

    async def next_action(self, state: ResearchState):
        return self._actions.pop(0)


class CountingSearchGateway(FakeSearchGateway):
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(
        self,
        *,
        goal: str,
        query: str,
        unresolved_required_constraints=None,
        resolved_entities=None,
    ) -> list[SearchResult]:
        self.queries.append(query)
        return await super().search(goal=goal, query=query)


class ContextSearchGateway(FakeSearchGateway):
    def __init__(self) -> None:
        self.unresolved_required_constraints = None
        self.resolved_entities = None

    async def search(
        self,
        *,
        goal: str,
        query: str,
        unresolved_required_constraints=None,
        resolved_entities=None,
    ) -> list[SearchResult]:
        self.unresolved_required_constraints = unresolved_required_constraints
        self.resolved_entities = resolved_entities
        return await super().search(goal=goal, query=query)


class CountingDocumentRetriever(FakeDocumentRetriever):
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, *, document, goal: str, query: str) -> list[Passage]:
        self.calls += 1
        return await super().retrieve(document=document, goal=goal, query=query)


class RecordingGuard(AnswerGuard):
    def __init__(self) -> None:
        self.calls = 0

    def check(self, *, research: ResearchState, proposal: AnswerAction):
        self.calls += 1
        return super().check(research=research, proposal=proposal)


def test_duplicate_search_is_rejected_without_calling_gateway() -> None:
    gateway = CountingSearchGateway()
    graph = make_graph(
        ScriptedPlanner(
            [
                SearchAction(goal="Find", query="first"),
                SearchAction(goal="Repeat", query="first"),
                SearchAction(goal="Find another", query="second"),
            ]
        ),
        search_gateway=gateway,
    )

    result = asyncio.run(
        graph.ainvoke({"research": ResearchState(question="Question", max_steps=3), "action": None})
    )

    research = result["research"]
    assert gateway.queries == ["first", "second"]
    assert research.trace[1].action == "search"
    assert research.trace[1].validation_rejection_reason == (
        "SEARCH query exactly duplicates an earlier query in this research run."
    )
    assert research.step_count == 3


def test_search_passes_only_unresolved_constraints_and_entities_to_gateway() -> None:
    gateway = ContextSearchGateway()
    constraints = [
        Constraint(id="c1", description="Unresolved", status="unknown"),
        Constraint(id="c2", description="Supported", status="supported"),
        Constraint(id="c3", description="Optional", required=False),
    ]
    graph = make_graph(
        ScriptedPlanner([SearchAction(goal="Find", query="query")], constraints=constraints),
        search_gateway=gateway,
    )

    asyncio.run(
        graph.ainvoke(
            {
                "research": ResearchState(
                    question="Question",
                    max_steps=1,
                    resolved_entities={"author": "Ada"},
                ),
                "action": None,
            }
        )
    )

    assert gateway.unresolved_required_constraints == [
        {"id": "c1", "description": "Unresolved", "kind": "acceptance"},
        {"id": "c2", "description": "Supported", "kind": "acceptance"},
    ]
    assert gateway.resolved_entities == {"author": "Ada"}


def test_duplicate_locate_is_rejected_without_calling_retriever() -> None:
    retriever = CountingDocumentRetriever()
    graph = build_research_graph(
        ScriptedPlanner(
            [
                SearchAction(goal="Find", query="source"),
                OpenAction(goal="Open", url="https://example.com/mock-source"),
                LocateAction(goal="Locate", document_id="mock-document", query="evidence"),
                LocateAction(goal="Repeat", document_id="mock-document", query="evidence"),
            ]
        ),
        FakeSearchGateway(),
        FakeDocumentOpener(),
        retriever,
        FakeFactExtractor(),
        AnswerGuard(),
    )

    result = asyncio.run(
        graph.ainvoke({"research": ResearchState(question="Question", max_steps=4), "action": None})
    )

    research = result["research"]
    assert retriever.calls == 1
    assert research.trace[3].action == "locate"
    assert research.trace[3].validation_rejection_reason == (
        "LOCATE document_id and query exactly duplicate an earlier LOCATE in this research run."
    )


def test_answer_without_real_fact_is_rejected_before_guard() -> None:
    guard = RecordingGuard()
    graph = build_research_graph(
        ScriptedPlanner(
            [AnswerAction(answer="Unsupported", supporting_fact_ids=[], supporting_constraint_ids=[])]
        ),
        FakeSearchGateway(),
        FakeDocumentOpener(),
        FakeDocumentRetriever(),
        FakeFactExtractor(),
        guard,
    )

    result = asyncio.run(
        graph.ainvoke({"research": ResearchState(question="Question", max_steps=1), "action": None})
    )

    research = result["research"]
    assert guard.calls == 0
    assert research.answer is None
    assert research.trace[0].action == "answer"
    assert research.trace[0].validation_rejection_reason == (
        "ANSWER proposal does not cite any real supporting Fact from this research run."
    )


def test_duplicate_guard_rejected_answer_is_rejected_until_new_evidence() -> None:
    class RepeatingAnswerPlanner(MockPlanner):
        async def next_action(self, state):
            if not state.executed_queries:
                return SearchAction(goal="Find", query="Question")
            if not state.documents:
                return OpenAction(goal="Open", url="https://example.com/mock-source")
            return AnswerAction(
                answer="Unsupported",
                supporting_fact_ids=[state.facts[0].id],
                supporting_constraint_ids=["c1"],
            )

    graph = make_graph(RepeatingAnswerPlanner())
    result = asyncio.run(
        graph.ainvoke(
            {
                "research": ResearchState(
                    question="Question",
                    constraints=[Constraint(id="c1", description="Needs explicit support")],
                    max_steps=4,
                ),
                "action": None,
            }
        )
    )

    research = result["research"]
    assert research.trace[2].guard_result is not None
    assert not research.trace[2].guard_result.accepted
    assert research.trace[3].validation_rejection_reason == (
        "ANSWER exactly duplicates a prior guard-rejected proposal without new evidence."
    )


class ScopedFactExtractor(FakeFactExtractor):
    async def extract(self, *, document, passages, constraints) -> FactExtraction:
        return FactExtraction(
            facts=[
                ExtractedFact(
                    statement="The candidate satisfies the requirement.",
                    confidence=0.9,
                    passage_id=passages[0].id,
                    constraint_evidence=[
                        ConstraintEvidence(constraint_id="c1", status="supported")
                    ],
                )
            ]
        )


def test_open_new_candidate_creates_scope_and_projects_only_after_guard_accept() -> None:
    # The final action needs the deterministic fact ID, which is only available after OPEN.
    # Use a planner that obtains it from state instead of predicting an opaque ID.
    class AnswerAfterOpenPlanner(ScriptedPlanner):
        async def next_action(self, state):
            if not state.executed_queries:
                return SearchAction(goal="Find", query="candidate")
            if not state.documents:
                return OpenAction(
                    goal="Verify candidate",
                    url="https://example.com/mock-source",
                    new_candidate_label="Candidate A",
                )
            return AnswerAction(
                answer="Answer",
                supporting_fact_ids=[state.facts[0].id],
                supporting_constraint_ids=["c1"],
                candidate_scope_id="cand_1",
            )

    graph = build_research_graph(
        AnswerAfterOpenPlanner([], constraints=[Constraint(id="c1", description="Requirement")]),
        FakeSearchGateway(),
        FakeDocumentOpener(),
        FakeDocumentRetriever(),
        ScopedFactExtractor(),
        AnswerGuard(),
    )
    result = asyncio.run(
        graph.ainvoke({"research": ResearchState(question="Question", max_steps=3), "action": None})
    )

    research = result["research"]
    assert research.status == "completed"
    assert research.candidate_scopes[0].id == "cand_1"
    assert research.candidate_scopes[0].label == "Candidate A"
    assert research.documents[0].evidence_scope_id == "cand_1"
    assert research.facts[0].evidence_scope_id == "cand_1"
    assert research.constraints[0].status == "supported"
    created = research.trace[1].created_candidate_scope
    assert created is not None
    assert created.model_dump() == {
        "id": "cand_1",
        "label": "Candidate A",
        "originating_document_id": "mock-document",
    }


def test_open_rejects_unknown_candidate_scope_without_calling_opener() -> None:
    opener = FakeDocumentOpener()
    graph = make_graph(
        ScriptedPlanner(
            [
                SearchAction(goal="Find", query="candidate"),
                OpenAction(
                    goal="Verify", url="https://example.com/mock-source", candidate_scope_id="cand_99"
                ),
            ]
        ),
        document_opener=opener,
    )

    result = asyncio.run(
        graph.ainvoke({"research": ResearchState(question="Question", max_steps=2), "action": None})
    )

    research = result["research"]
    assert opener.opened_urls == []
    assert research.trace[1].validation_rejection_reason == (
        "OPEN candidate_scope_id does not refer to an existing candidate scope."
    )


def test_locate_reuses_the_target_documents_candidate_scope() -> None:
    graph = build_research_graph(
        ScriptedPlanner(
            [
                SearchAction(goal="Find", query="candidate"),
                OpenAction(
                    goal="Verify",
                    url="https://example.com/mock-source",
                    new_candidate_label="Candidate A",
                ),
                LocateAction(goal="Find evidence", document_id="mock-document", query="requirement"),
            ],
            constraints=[Constraint(id="c1", description="Requirement")],
        ),
        FakeSearchGateway(),
        FakeDocumentOpener(),
        FakeDocumentRetriever(),
        ScopedFactExtractor(),
        AnswerGuard(),
    )

    result = asyncio.run(
        graph.ainvoke({"research": ResearchState(question="Question", max_steps=3), "action": None})
    )

    research = result["research"]
    assert research.documents[0].evidence_scope_id == "cand_1"
    assert research.located_passages[0].document_id == "mock-document"
    assert {fact.evidence_scope_id for fact in research.facts} == {"cand_1"}
