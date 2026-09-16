"""Document-local lexical retrieval followed by small-set LLM reranking."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Protocol

from rank_bm25 import BM25Okapi

from app.config import Settings
from app.llm.client import LLMClient, Message
from app.research.schemas import DocumentRef, Passage, PassageRanking
from app.tools.document import LocalDocumentStore


class RetrievalError(RuntimeError):
    """Raised when document-local retrieval cannot produce valid passages."""


class PassageReranker(Protocol):
    async def rerank(
        self,
        *,
        goal: str,
        query: str,
        candidates: list[Passage],
        top_n: int,
    ) -> list[str]: ...


class LLMPassageReranker:
    """Reranks only BM25 candidates via the provider-neutral LLM client."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def rerank(
        self,
        *,
        goal: str,
        query: str,
        candidates: list[Passage],
        top_n: int,
    ) -> list[str]:
        messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    "Rank only the supplied candidate passage IDs by their value as direct evidence "
                    "for the current research goal and query.\n\n"
                    "Prefer passages that directly state facts needed to verify or falsify the target "
                    "claim, such as names, dates, relationships, titles, quotations, events, or other "
                    "discriminating details.\n\n"
                    "Do not rank a passage highly merely because it is topically related.\n\n"
                    "Return only unique IDs from the supplied candidates, ordered from strongest "
                    "direct evidence to weakest.\n\n"
                    "Do not answer the question, extract facts, explain the ranking, or create new "
                    "passage IDs.\n\n"
                    "Return an empty list when none of the supplied passages provides useful evidence."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "goal": goal,
                        "query": query,
                        "top_n": top_n,
                        "candidates": [
                            {"id": candidate.id, "text": candidate.text}
                            for candidate in candidates
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        ranking = await self._client.structured(messages=messages, schema=PassageRanking)
        return ranking.passage_ids


class DocumentRetriever:
    """Reads one parsed document, recalls with BM25, then reranks a small set."""

    def __init__(
        self,
        settings: Settings,
        reranker: PassageReranker,
        *,
        store: LocalDocumentStore | None = None,
    ) -> None:
        if settings.retrieval_chunk_overlap_chars >= settings.retrieval_chunk_size_chars:
            raise ValueError("Chunk overlap must be smaller than chunk size.")
        self._settings = settings
        self._reranker = reranker
        self._store = store or LocalDocumentStore(settings.document_store_dir)

    async def retrieve(
        self,
        *,
        document: DocumentRef,
        goal: str,
        query: str,
    ) -> list[Passage]:
        if not document.local_path:
            raise RetrievalError(f"Document has no local parsed content: {document.id}")
        if not Path(document.local_path).is_file():
            raise RetrievalError(f"Document content is unavailable: {document.id}")

        content = await asyncio.to_thread(self._store.read_text, local_path=document.local_path)
        chunks = await asyncio.to_thread(
            chunk_document,
            document_id=document.id,
            content=content,
            chunk_size=self._settings.retrieval_chunk_size_chars,
            overlap=self._settings.retrieval_chunk_overlap_chars,
        )
        if not chunks:
            raise RetrievalError(f"Document contains no retrievable text: {document.id}")

        candidates = await asyncio.to_thread(
            bm25_recall,
            chunks=chunks,
            query=query,
            top_k=self._settings.retrieval_bm25_top_k,
        )
        ranked_ids = await self._reranker.rerank(
            goal=goal,
            query=query,
            candidates=candidates,
            top_n=self._settings.retrieval_top_n,
        )
        return _select_ranked_passages(
            candidates=candidates,
            ranked_ids=ranked_ids,
            top_n=self._settings.retrieval_top_n,
        )


def chunk_document(
    *, document_id: str, content: str, chunk_size: int, overlap: int
) -> list[Passage]:
    """Create bounded, overlap-aware chunks while preserving source offsets."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than size.")

    chunks: list[Passage] = []
    start = 0
    chunk_index = 0
    content_length = len(content)
    while start < content_length:
        tentative_end = min(start + chunk_size, content_length)
        end = _preferred_boundary(content, start, tentative_end)
        if end <= start:
            end = tentative_end
        text = content[start:end].strip()
        if text:
            left_trim = len(content[start:end]) - len(content[start:end].lstrip())
            right_trim = len(content[start:end]) - len(content[start:end].rstrip())
            chunk_start = start + left_trim
            chunk_end = end - right_trim
            chunks.append(
                Passage(
                    id=f"{document_id}:chunk:{chunk_index}",
                    document_id=document_id,
                    text=text,
                    start_char=chunk_start,
                    end_char=chunk_end,
                )
            )
            chunk_index += 1
        if end >= content_length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def bm25_recall(*, chunks: list[Passage], query: str, top_k: int) -> list[Passage]:
    query_tokens = tokenize(query)
    if not query_tokens:
        raise RetrievalError("LOCATE query contains no searchable tokens.")
    tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
    scores = BM25Okapi(tokenized_chunks).get_scores(query_tokens)
    ordered_indices = sorted(range(len(chunks)), key=lambda index: (-float(scores[index]), index))
    return [
        chunks[index].model_copy(update={"bm25_score": float(scores[index])})
        for index in ordered_indices[:top_k]
    ]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def _preferred_boundary(content: str, start: int, tentative_end: int) -> int:
    if tentative_end == len(content):
        return tentative_end
    lower_bound = start + (tentative_end - start) // 2
    paragraph_boundary = content.rfind("\n\n", lower_bound, tentative_end)
    if paragraph_boundary > start:
        return paragraph_boundary
    whitespace_boundary = content.rfind(" ", lower_bound, tentative_end)
    return whitespace_boundary if whitespace_boundary > start else tentative_end


def _select_ranked_passages(
    *, candidates: list[Passage], ranked_ids: list[str], top_n: int
) -> list[Passage]:
    candidate_by_id = {candidate.id: candidate for candidate in candidates}
    selected: list[Passage] = []
    seen_ids: set[str] = set()
    for passage_id in ranked_ids:
        if passage_id in seen_ids:
            raise RetrievalError("LLM reranker returned duplicate passage IDs.")
        if passage_id not in candidate_by_id:
            raise RetrievalError("LLM reranker returned an unknown passage ID.")
        seen_ids.add(passage_id)
        selected.append(candidate_by_id[passage_id].model_copy(update={"rank": len(selected) + 1}))
        if len(selected) == top_n:
            break
    return selected
