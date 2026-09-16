"""Search gateway contract and deterministic Phase 2 stub."""

from __future__ import annotations

from typing import Protocol

from app.research.schemas import SearchResult


class SearchGateway(Protocol):
    def search(self, *, goal: str, query: str) -> list[SearchResult]: ...


class StubSearchGateway:
    """Returns fixed results without making an external request."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results or [
            SearchResult(
                url="https://example.com/mock-source",
                title="Mock source",
                snippet="A deterministic result for Phase 2 tests.",
            )
        ]

    def search(self, *, goal: str, query: str) -> list[SearchResult]:
        return list(self._results)
