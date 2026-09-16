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
                    "Extract only atomic facts that are explicitly and unambiguously supported by "
                    "one supplied passage.\n\n"
                    "Every extracted fact must cite exactly one supplied passage_id.\n\n"
                    "Do not combine evidence across passages, use outside knowledge, guess uncertain "
                    "references, or infer a conclusion that the passage itself does not establish.\n\n"
                    "You may resolve a clear and unambiguous local reference only within the same "
                    "passage.\n\n"
                    "A fact may list a constraint in supports_constraints only when that single "
                    "passage directly provides evidence for that constraint. Mere topical relevance "
                    "is not enough.\n\n"
                    "Mark a constraint contradicted only when the supplied passage contains explicit "
                    "evidence incompatible with it. Absence of evidence is not contradiction.\n\n"
                    "Resolved entities must also be explicitly stated or unambiguously resolved within "
                    "one supplied passage and must cite that passage_id.\n\n"
                    "Do not use search-result snippets as evidence.\n\n"
                    "If no explicit fact or entity can be extracted, return empty lists."
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
    """Materialize trusted provenance without projecting local evidence globally."""
    passage_by_id = {passage.id: passage for passage in passages}
    constraint_by_id = {constraint.id: constraint for constraint in research.constraints}
    facts = list(research.facts)
    existing_fact_ids = {fact.id for fact in facts}
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
            supports_constraints=[
                item.constraint_id
                for item in candidate.constraint_evidence
                if item.status == "supported"
            ],
            evidence_scope_id=document.evidence_scope_id,
            constraint_evidence=list(candidate.constraint_evidence),
        )
        facts.append(fact)
        existing_fact_ids.add(fact_id)

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
            "resolved_entities": entities,
            "resolved_entity_provenance": entity_provenance,
        }
    )


def _fact_id(document_id: str, passage_id: str, statement: str) -> str:
    material = f"{document_id}\0{passage_id}\0{statement}".encode("utf-8")
    return f"fact:{hashlib.sha256(material).hexdigest()[:16]}"
