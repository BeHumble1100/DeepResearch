import asyncio
import json
from pathlib import Path

import pytest

from app.config import Settings
from app.research.schemas import DocumentRef, Passage, PassageRanking
from app.tools.document import LocalDocumentStore
from app.tools.retrieval import (
    DocumentRetriever,
    LLMPassageReranker,
    RetrievalError,
    chunk_document,
    tokenize,
)


class FakeReranker:
    def __init__(self, ranked_ids: list[str] | None = None) -> None:
        self.ranked_ids = ranked_ids
        self.candidates: list[Passage] = []

    async def rerank(
        self,
        *,
        goal: str,
        query: str,
        candidates: list[Passage],
        top_n: int,
    ) -> list[str]:
        self.candidates = candidates
        return self.ranked_ids or [candidate.id for candidate in candidates]


class FakeLLMClient:
    def __init__(self, ranking: PassageRanking) -> None:
        self.ranking = ranking
        self.messages: list[dict[str, str]] | None = None
        self.schema: object | None = None

    async def structured(self, *, messages, schema, **kwargs):
        self.messages = messages
        self.schema = schema
        return self.ranking

    async def text(self, *, messages, **kwargs) -> str:
        return "unused"


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        searxng_base_url="http://localhost:8080",
        document_store_dir=tmp_path,
        retrieval_chunk_size_chars=80,
        retrieval_chunk_overlap_chars=10,
        retrieval_bm25_top_k=2,
        retrieval_top_n=1,
    )


def make_document(tmp_path: Path, content: str) -> DocumentRef:
    store = LocalDocumentStore(tmp_path)
    local_path = store.write_text(document_id="doc-1", content=content)
    return DocumentRef(
        id="doc-1",
        url="https://example.com/document",
        content_type="text/html",
        local_path=local_path,
    )


def test_chunking_preserves_offsets_and_tokenizes_chinese_and_english() -> None:
    content = "Alpha beta gamma.\n\n中文检索测试内容。"
    chunks = chunk_document(document_id="doc-1", content=content, chunk_size=20, overlap=5)

    assert chunks
    assert all(content[chunk.start_char : chunk.end_char] == chunk.text for chunk in chunks)
    assert tokenize("English 中文") == ["english", "中", "文"]


def test_document_retriever_uses_bm25_candidates_then_reranks(tmp_path: Path) -> None:
    document = make_document(
        tmp_path,
        "General background text without the term.\n\n"
        "The moonstone artifact was discovered in the archive.\n\n"
        "Another unrelated conclusion.",
    )
    reranker = FakeReranker()
    retriever = DocumentRetriever(make_settings(tmp_path), reranker)

    passages = asyncio.run(
        retriever.retrieve(document=document, goal="Find the artifact", query="moonstone artifact")
    )

    assert reranker.candidates
    assert len(reranker.candidates) <= 2
    assert "moonstone" in reranker.candidates[0].text.lower()
    assert len(passages) == 1
    assert passages[0].rank == 1
    assert passages[0].bm25_score is not None


def test_document_retriever_rejects_unknown_reranked_passage(tmp_path: Path) -> None:
    document = make_document(tmp_path, "Relevant source text.")
    retriever = DocumentRetriever(make_settings(tmp_path), FakeReranker(["unknown-chunk"]))

    with pytest.raises(RetrievalError, match="unknown passage"):
        asyncio.run(retriever.retrieve(document=document, goal="Find", query="source"))


def test_llm_reranker_receives_only_bm25_candidates() -> None:
    candidates = [
        Passage(
            id="doc-1:chunk:0",
            document_id="doc-1",
            text="Candidate passage",
            start_char=0,
            end_char=17,
        )
    ]
    client = FakeLLMClient(PassageRanking(passage_ids=["doc-1:chunk:0"]))

    ranked_ids = asyncio.run(
        LLMPassageReranker(client).rerank(
            goal="Find support", query="candidate", candidates=candidates, top_n=1
        )
    )

    assert ranked_ids == ["doc-1:chunk:0"]
    assert client.schema is PassageRanking
    assert json.loads(client.messages[1]["content"])["candidates"] == [
        {"id": "doc-1:chunk:0", "text": "Candidate passage"}
    ]
