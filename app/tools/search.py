"""Async SearXNG search gateway with query rewrite and normalized results."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.config import Settings
from app.llm.client import LLMClient, Message
from app.research.schemas import QueryRewrite, SearchResult


class SearchGateway(Protocol):
    async def search(self, *, goal: str, query: str) -> list[SearchResult]: ...


class QueryRewriter(Protocol):
    async def rewrite(self, *, goal: str, query: str) -> list[str]: ...


class IdentityQueryRewriter:
    """Preserves the original query when no LLM rewriter is configured."""

    async def rewrite(self, *, goal: str, query: str) -> list[str]:
        return [query]


class LLMQueryRewriter:
    """Uses the existing provider-neutral LLM boundary for query rewriting."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def rewrite(self, *, goal: str, query: str) -> list[str]:
        messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    "Generate 1 to 3 concise, complementary web-search queries for the current "
                    "goal. Preserve known entities and discriminating constraints. Do not answer the "
                    "question, explain your choices, or generate generic paraphrases."
                ),
            },
            {"role": "user", "content": f"Goal: {goal}\nQuery: {query}"},
        ]
        rewrite = await self._client.structured(messages=messages, schema=QueryRewrite)
        return rewrite.queries


class SearchGatewayError(RuntimeError):
    """Raised when SearXNG cannot produce a complete search response."""


class SearXNGSearchGateway:
    """Queries a configured SearXNG instance without treating snippets as evidence."""

    _RETRIABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
    _MAX_REWRITTEN_QUERIES = 3

    def __init__(
        self,
        settings: Settings,
        *,
        query_rewriter: QueryRewriter | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._query_rewriter = query_rewriter or IdentityQueryRewriter()
        self._transport = transport

    async def search(self, *, goal: str, query: str) -> list[SearchResult]:
        queries = _unique_nonempty(
            await self._query_rewriter.rewrite(goal=goal, query=query),
            limit=self._MAX_REWRITTEN_QUERIES,
        )
        if not queries:
            raise SearchGatewayError("Query rewriter returned no usable queries.")

        timeout = httpx.Timeout(self._settings.searxng_timeout_seconds)
        async with httpx.AsyncClient(
            base_url=self._settings.searxng_base_url,
            timeout=timeout,
            transport=self._transport,
        ) as client:
            batches = await asyncio.gather(
                *(self._search_one(client, rewritten_query) for rewritten_query in queries)
            )

        flattened = [result for batch in batches for result in batch]
        return _deduplicate_results(flattened)[: self._settings.searxng_max_results]

    async def _search_one(
        self, client: httpx.AsyncClient, query: str
    ) -> list[SearchResult]:
        last_error: Exception | None = None
        for attempt in range(self._settings.searxng_max_retries + 1):
            try:
                response = await client.get("/search", params={"q": query, "format": "json"})
                if response.status_code in self._RETRIABLE_STATUS_CODES:
                    raise _RetriableResponseError(response.status_code)
                response.raise_for_status()
                return _normalize_results(response.json())
            except (_RetriableResponseError, httpx.TimeoutException, httpx.TransportError) as error:
                last_error = error
                if attempt == self._settings.searxng_max_retries:
                    break
                await asyncio.sleep(self._settings.searxng_retry_backoff_seconds * (2**attempt))
            except ValueError as error:
                raise SearchGatewayError("SearXNG returned invalid JSON.") from error

        raise SearchGatewayError(f"SearXNG request failed for query: {query}") from last_error


class _RetriableResponseError(RuntimeError):
    pass


def _normalize_results(payload: object) -> list[SearchResult]:
    if not isinstance(payload, Mapping):
        raise SearchGatewayError("SearXNG JSON response must be an object.")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise SearchGatewayError("SearXNG JSON response must contain a results list.")

    normalized: list[SearchResult] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, Mapping):
            continue
        url = raw_result.get("url")
        if not isinstance(url, str):
            continue
        canonical_url = _canonicalize_url(url)
        if canonical_url is None:
            continue
        title = raw_result.get("title")
        snippet = raw_result.get("content")
        normalized.append(
            SearchResult(
                url=canonical_url,
                title=title.strip() if isinstance(title, str) and title.strip() else None,
                snippet=snippet.strip() if isinstance(snippet, str) and snippet.strip() else None,
            )
        )
    return normalized


def _deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    unique: list[SearchResult] = []
    seen_urls: set[str] = set()
    for result in results:
        if result.url in seen_urls:
            continue
        seen_urls.add(result.url)
        unique.append(result)
    return unique


def _unique_nonempty(queries: list[str], *, limit: int | None = None) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = query.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
            if limit is not None and len(unique) == limit:
                break
    return unique


def _canonicalize_url(url: str) -> str | None:
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.query,
            "",
        )
    )
