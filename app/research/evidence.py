"""Bounded fact extraction and deterministic research-state evidence updates."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Literal, Protocol

from app.llm.client import LLMClient, Message

from .schemas import (
    Constraint,
    DocumentRef,
    EntityProvenance,
    Fact,
    FactExtraction,
    Passage,
)
from .state import ResearchState

EvidenceKind = Literal["open", "locate"]


class FactExtractionError(RuntimeError):
    """Raised when an extractor returns references outside the supplied evidence."""


class FactExtractor(Protocol):
    async def extract(
        self,
        *,
        document: DocumentRef,
        passages: list[Passage],
        constraints: list[Constraint],
    ) -> FactExtraction: ...


class LLMFactExtractor:
    """Extracts structured claims solely from the supplied bounded passages."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def extract(
        self,
        *,
        document: DocumentRef,
        passages: list[Passage],
        constraints: list[Constraint],
    ) -> FactExtraction:
        if not passages:
            return FactExtraction()
        messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    "Extract only atomic facts explicitly and unambiguously supported by one supplied "
                    "passage. Each fact and resolved entity must cite exactly one supplied passage_id. "
                    "You may resolve an explicit local reference only within that same passage. Do not "
                    "combine passages, use outside knowledge, guess uncertain references, or infer a "
                    "conclusion. Return empty lists when no fact or entity is explicit. Mark a constraint "
                    "contradicted only when a passage explicitly negates it; otherwise do not mark it "
                    "contradicted. Do not use search snippets."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "document": {"id": document.id, "url": document.url},
                        "constraints": [
                            {
                                "id": item.id,
                                "description": item.description,
                                "subject": item.subject,
                                "predicate": item.predicate,
                                "object": item.object,
                            }
                            for item in constraints
                        ],
                        "passages": [{"id": item.id, "text": item.text} for item in passages],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return await self._client.structured(messages=messages, schema=FactExtraction)


async def opening_passage(document: DocumentRef, *, max_chars: int) -> Passage:
    """Load a bounded leading excerpt from an opened document for fact extraction."""
    if not document.local_path:
        raise FactExtractionError(f"Document has no local parsed content: {document.id}")
    content = await asyncio.to_thread(Path(document.local_path).read_text, encoding="utf-8")
    text = content[:max_chars].strip()
    if not text:
        raise FactExtractionError(f"Document contains no extractable opening content: {document.id}")
    start = len(content[:max_chars]) - len(content[:max_chars].lstrip())
    return Passage(
        id=f"{document.id}:open:0",
        document_id=document.id,
        text=text,
        start_char=start,
        end_char=start + len(text),
    )


def apply_fact_extraction(
    research: ResearchState,
    *,
    document: DocumentRef,
    passages: list[Passage],
    extraction: FactExtraction,
    evidence_kind: EvidenceKind,
) -> ResearchState:
    """Materialize trusted provenance and apply deterministic constraint/entity updates."""
    passage_by_id = {passage.id: passage for passage in passages}
    constraint_by_id = {constraint.id: constraint for constraint in research.constraints}
    facts = list(research.facts)
    existing_fact_ids = {fact.id for fact in facts}
    constraints = list(research.constraints)
    entities = dict(research.resolved_entities)
    entity_provenance = dict(research.resolved_entity_provenance)

    for candidate in extraction.facts:
        passage = passage_by_id.get(candidate.passage_id)
        if passage is None:
            raise FactExtractionError("Fact extractor cited an unknown passage ID.")
        if passage.document_id != document.id:
            raise FactExtractionError("Fact extractor cited a passage from another document.")
        for relation in candidate.constraint_evidence:
            if relation.constraint_id not in constraint_by_id:
                raise FactExtractionError("Fact extractor cited an unknown constraint ID.")

        fact_id = _fact_id(document.id, passage.id, candidate.statement)
        if fact_id in existing_fact_ids:
            continue
        fact = Fact(
            id=fact_id,
            statement=candidate.statement,
            subject=candidate.subject,
            predicate=candidate.predicate,
            object=candidate.object,
            source_url=document.url,
            document_id=document.id,
            passage_id=passage.id,
            evidence_kind=evidence_kind,
            passage=passage.text,
            confidence=candidate.confidence,
            supports_constraints=[item.constraint_id for item in candidate.constraint_evidence],
        )
        facts.append(fact)
        existing_fact_ids.add(fact_id)
        for relation in candidate.constraint_evidence:
            index = next(
                index for index, item in enumerate(constraints) if item.id == relation.constraint_id
            )
            constraint = constraints[index]
            fact_ids = [*constraint.supporting_fact_ids, fact.id]
            status = (
                "contradicted"
                if relation.status == "contradicted" or constraint.status == "contradicted"
                else "supported"
            )
            constraints[index] = constraint.model_copy(
                update={"status": status, "supporting_fact_ids": list(dict.fromkeys(fact_ids))}
            )

    for entity in extraction.resolved_entities:
        passage = passage_by_id.get(entity.passage_id)
        if passage is None:
            raise FactExtractionError("Fact extractor cited an unknown entity passage ID.")
        if passage.document_id != document.id:
            raise FactExtractionError("Fact extractor cited an entity passage from another document.")
        if entity.key not in entities:
            entities[entity.key] = entity.value
            entity_provenance[entity.key] = EntityProvenance(
                source_url=document.url,
                document_id=document.id,
                passage_id=passage.id,
                evidence_kind=evidence_kind,
            )
    return research.model_copy(
        update={
            "facts": facts,
            "constraints": constraints,
            "resolved_entities": entities,
            "resolved_entity_provenance": entity_provenance,
        }
    )


def _fact_id(document_id: str, passage_id: str, statement: str) -> str:
    material = f"{document_id}\0{passage_id}\0{statement}".encode("utf-8")
    return f"fact:{hashlib.sha256(material).hexdigest()[:16]}"
