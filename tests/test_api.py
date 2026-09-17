from fastapi.testclient import TestClient

from app.main import create_app
from app.research.schemas import Constraint, Fact, Target
from app.research.state import ResearchState


def _completed_research(question: str) -> ResearchState:
    return ResearchState(
        question=question,
        target=Target(description="A short answer", answer_type="name"),
        constraints=[Constraint(id="c1", description="Supported", status="supported")],
        facts=[
            Fact(
                id="fact:1",
                statement="Evidence supports the answer.",
                source_url="https://example.test/source",
                document_id="doc:1",
                passage_id="passage:1",
                evidence_kind="locate",
                passage="Private source passage must not be returned by the API.",
                confidence=0.9,
            )
        ],
        answer="Ada",
        status="completed",
        step_count=3,
    )


def test_research_endpoint_runs_injected_production_contract() -> None:
    calls: list[tuple[str, int]] = []

    async def execute(question: str, max_steps: int) -> ResearchState:
        calls.append((question, max_steps))
        return _completed_research(question)

    response = TestClient(create_app(executor=execute)).post(
        "/research", json={"question": "Who is Ada?", "max_steps": 6}
    )

    assert response.status_code == 200
    assert calls == [("Who is Ada?", 6)]
    assert response.json() == {
        "question": "Who is Ada?",
        "answer": "Ada",
        "status": "completed",
        "step_count": 3,
        "target": {"description": "A short answer", "answer_type": "name", "format_instruction": None},
        "constraints": [
            {
                "id": "c1",
                "description": "Supported",
                "subject": None,
                "predicate": None,
                "object": None,
                "required": True,
                "kind": "acceptance",
                "status": "supported",
                "supporting_fact_ids": [],
            }
        ],
        "resolved_entities": {},
        "facts": [
            {
                "id": "fact:1",
                "statement": "Evidence supports the answer.",
                "source_url": "https://example.test/source",
                "document_id": "doc:1",
                "passage_id": "passage:1",
                "evidence_kind": "locate",
                "confidence": 0.9,
                "evidence_scope_id": None,
                "supports_constraints": [],
            }
        ],
        "trace": [],
    }


def test_research_endpoint_rejects_blank_question_and_invalid_budget() -> None:
    async def execute(question: str, max_steps: int) -> ResearchState:
        return _completed_research(question)

    client = TestClient(create_app(executor=execute))

    assert client.post("/research", json={"question": "   ", "max_steps": 1}).status_code == 422
    assert client.post("/research", json={"question": "Question", "max_steps": 0}).status_code == 422
