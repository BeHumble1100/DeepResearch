from app.research.schemas import Constraint, Fact, SearchAction
from app.research.state import ResearchState
from app.research.trace import append_action_trace


def test_trace_is_serializable_and_does_not_copy_fact_passages() -> None:
    before = ResearchState(
        question="Question",
        constraints=[Constraint(id="c-1", description="Constraint")],
    )
    fact = Fact(
        id="fact-1",
        statement="Ada is the author.",
        source_url="https://example.com/source",
        document_id="doc-1",
        passage_id="doc-1:chunk:0",
        evidence_kind="locate",
        passage="This full passage must not be copied into the trace.",
        confidence=0.9,
        supports_constraints=["c-1"],
    )
    after = before.model_copy(
        update={
            "facts": [fact],
            "constraints": [
                before.constraints[0].model_copy(
                    update={"status": "supported", "supporting_fact_ids": ["fact-1"]}
                )
            ],
            "resolved_entities": {"author": "Ada"},
            "current_goal": "Find the author",
            "step_count": 1,
        }
    )

    traced = append_action_trace(
        before,
        after,
        action=SearchAction(goal="Find the author", query="Question"),
        observation_summary="Search completed.",
    )

    entry = traced.trace[0]
    assert entry.new_facts[0].statement == "Ada is the author."
    assert entry.constraint_changes[0].previous_status == "unknown"
    assert entry.constraint_changes[0].current_status == "supported"
    assert entry.resolved_entities == {"author": "Ada"}
    encoded = entry.model_dump_json()
    assert "This full passage" not in encoded
    assert "passage\":\"" not in encoded
