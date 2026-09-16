"""Document-opening contract and deterministic Phase 2 stub."""

from __future__ import annotations

from typing import Protocol

from app.research.schemas import DocumentRef


class DocumentOpener(Protocol):
    def open(self, *, url: str) -> DocumentRef: ...


class StubDocumentOpener:
    """Returns a fixed document reference without fetching the URL."""

    def open(self, *, url: str) -> DocumentRef:
        return DocumentRef(
            id="mock-document",
            url=url,
            title="Mock source",
            content_type="text/html",
            summary="Stub document; no network fetch or parsing was performed.",
        )
