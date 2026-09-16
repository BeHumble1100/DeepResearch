import asyncio
from pathlib import Path

import pytest

from app.llm.client import Message
from app.research.evidence import (
    FactExtractionError,
    LLMFactExtractor,
    apply_fact_extraction,
    opening_passage,
)
from app.research.schemas import (
    Constraint,
    ConstraintEvidence,
    DocumentRef,
    ExtractedFact,
    FactExtraction,
    Passage,
)
from app.research.state import ResearchState


def _document() -> DocumentRef:
    return DocumentRef(id="doc-1", url="https://example.com/source", content_type="text/html")


def _passage() -> Passage:
    return Passage(
        id="doc-1:chunk:0",
        document_id="doc-1",
        text="Ada graduated from Example University.",
        start_char=0,
        end_char=36,
    )


def test_extraction_materializes_trusted_provenance_and_updates_state() -> None:
    state = ResearchState(
        question="Question",
        constraints=[Constraint(id="degree", description="Degree institution")],
    )
    updated = apply_fact_extraction(
        state,
        document=_document(),
        passages=[_passage()],
        evidence_kind="locate",
        extraction=FactExtraction(
            facts=[
                ExtractedFact(
                    statement="Ada graduated from Example University.",
                    subject="Ada",
                    predicate="graduated_from",
                    object="Example University",
                    confidence=0.9,
                    passage_id="doc-1:chunk:0",
                    constraint_evidence=[
                        ConstraintEvidence(constraint_id="degree", status="supported")
                    ],
                )
            ],
            resolved_entities={"author": "Ada"},
        ),
    )

    fact = updated.facts[0]
    assert fact.source_url == "https://example.com/source"
    assert fact.document_id == "doc-1"
    assert fact.passage_id == "doc-1:chunk:0"
    assert fact.evidence_kind == "locate"
    assert fact.passage == _passage().text
    assert updated.constraints[0].status == "supported"
    assert updated.constraints[0].supporting_fact_ids == [fact.id]
    assert updated.resolved_entities == {"author": "Ada"}


def test_contradiction_takes_precedence_and_existing_entities_are_not_overwritten() -> None:
    state = ResearchState(
        question="Question",
        constraints=[Constraint(id="c-1", description="Required fact", status="supported")],
        resolved_entities={"author": "Ada"},
    )
    updated = apply_fact_extraction(
        state,
        document=_document(),
        passages=[_passage()],
        evidence_kind="open",
        extraction=FactExtraction(
            facts=[
                ExtractedFact(
                    statement="The requirement is false.",
                    confidence=0.8,
                    passage_id="doc-1:chunk:0",
                    constraint_evidence=[
                        ConstraintEvidence(constraint_id="c-1", status="contradicted")
                    ],
                )
            ],
            resolved_entities={"author": "Grace"},
        ),
    )

    assert updated.constraints[0].status == "contradicted"
    assert updated.resolved_entities == {"author": "Ada"}


def test_extraction_rejects_unknown_passage_or_constraint() -> None:
    state = ResearchState(question="Question")
    with pytest.raises(FactExtractionError, match="unknown passage"):
        apply_fact_extraction(
            state,
            document=_document(),
            passages=[_passage()],
            evidence_kind="open",
            extraction=FactExtraction(
                facts=[
                    ExtractedFact(statement="Claim", confidence=0.5, passage_id="not-supplied")
                ]
            ),
        )


def test_opening_passage_is_bounded(tmp_path: Path) -> None:
    content_path = tmp_path / "content.txt"
    content_path.write_text("  abcdefghij", encoding="utf-8")
    document = _document().model_copy(update={"local_path": str(content_path)})

    result = asyncio.run(opening_passage(document, max_chars=6))

    assert result.id == "doc-1:open:0"
    assert result.text == "abcd"
    assert result.start_char == 2


class FakeLLMClient:
    def __init__(self) -> None:
        self.messages: list[Message] = []

    async def structured(self, *, messages, schema, **kwargs):
        self.messages = messages
        return FactExtraction()

    async def text(self, *, messages, **kwargs):
        return "unused"


def test_llm_extractor_receives_only_supplied_passages() -> None:
    client = FakeLLMClient()
    asyncio.run(
        LLMFactExtractor(client).extract(
            document=_document(),
            passages=[_passage()],
            constraints=[Constraint(id="c-1", description="Constraint")],
        )
    )

    payload = client.messages[1]["content"]
    assert "Ada graduated" in payload
    assert "c-1" in payload
