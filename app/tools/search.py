"""Async SearXNG search gateway with query rewrite and normalized results."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.config import Settings
from app.llm.client import LLMClient, Message
from app.research.schemas import QueryRewrite, SearchResult


class SearchGateway(Protocol):
    async def search(
        self,
        *,
        goal: str,
        query: str,
        unresolved_required_constraints: list[dict[str, str]] | None = None,
        resolved_entities: dict[str, str] | None = None,
        prior_queries: list[str] | None = None,
    ) -> list[SearchResult]: ...


class QueryRewriter(Protocol):
    async def rewrite(
        self,
        *,
        goal: str,
        query: str,
        unresolved_required_constraints: list[dict[str, str]] | None = None,
        resolved_entities: dict[str, str] | None = None,
        prior_queries: list[str] | None = None,
    ) -> list[str]: ...


class IdentityQueryRewriter:
    """Preserves the original query when no LLM rewriter is configured."""

    async def rewrite(
        self,
        *,
        goal: str,
        query: str,
        unresolved_required_constraints: list[dict[str, str]] | None = None,
        resolved_entities: dict[str, str] | None = None,
        prior_queries: list[str] | None = None,
    ) -> list[str]:
        return [query]


class LLMQueryRewriter:
    """Uses the existing provider-neutral LLM boundary for query rewriting."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def rewrite(
        self,
        *,
        goal: str,
        query: str,
        unresolved_required_constraints: list[dict[str, str]] | None = None,
        resolved_entities: dict[str, str] | None = None,
        prior_queries: list[str] | None = None,
    ) -> list[str]:
        messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    "Generate 1 to 3 concise and complementary web-search queries for the current "
                    "research goal.\n\n"
                    "Use unresolved required constraints as discriminating search clues. "
                    "Use resolved entities only as established anchors.\n\n"
                    "Preserve exact names, dates, quoted phrases, titles, organizations, and other "
                    "high-value identifiers when they are available.\n\n"
                    "Each query should pursue a meaningfully different retrieval angle. Do not return "
                    "several superficial paraphrases of the same query.\n\n"
                    "When prior_planner_queries is non-empty, do not repeat one or merely reorder its "
                    "words. Choose a different discriminating clue from the goal or constraints, such "
                    "as a relation, date, organization, title, or historical anchor.\n\n"
                    "Do not introduce an unsupported entity as if it were established fact. If a "
                    "possible entity is only a hypothesis, phrase the query so that it verifies or "
                    "falsifies that hypothesis.\n\n"
                    "Do not answer the research question, explain the queries, or add commentary. "
                    "Return only the structured query list."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "goal": goal,
                        "planner_query": query,
                        "unresolved_required_constraints": unresolved_required_constraints or [],
                        "resolved_entities": resolved_entities or {},
                        "prior_planner_queries": prior_queries or [],
                    },
                    ensure_ascii=False,
                ),
            },
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

    async def search(
        self,
        *,
        goal: str,
        query: str,
        unresolved_required_constraints: list[dict[str, str]] | None = None,
        resolved_entities: dict[str, str] | None = None,
        prior_queries: list[str] | None = None,
    ) -> list[SearchResult]:
        prior_query_set = set(_unique_nonempty(prior_queries or []))
        rewritten_queries = _unique_nonempty(
            await self._query_rewriter.rewrite(
                goal=goal,
                query=query,
                unresolved_required_constraints=unresolved_required_constraints,
                resolved_entities=resolved_entities,
                prior_queries=prior_queries,
            ),
            limit=self._MAX_REWRITTEN_QUERIES,
        )
        if not rewritten_queries:
            raise SearchGatewayError("Query rewriter returned no usable queries.")
        queries = [
            rewritten_query
            for rewritten_query in rewritten_queries
            if rewritten_query not in prior_query_set
        ]
        if not queries and query.strip() not in prior_query_set:
            queries = [query.strip()]
        if not queries:
            raise SearchGatewayError("Query rewriter returned no new usable queries.")

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
        unique_results = _deduplicate_results(flattened)
        return _rank_by_lexical_relevance(
            unique_results,
            query=" ".join(queries),
            goal=goal,
        )[: self._settings.searxng_max_results]

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


def _rank_by_lexical_relevance(
    results: list[SearchResult], *, query: str, goal: str
) -> list[SearchResult]:
    """Demote off-topic engine noise while preserving SearXNG order for ties.

    This is candidate selection only: snippets remain non-evidence and no source is
    excluded solely because it is from an unfamiliar domain.
    """

    terms = set(_relevance_terms(query)) | set(_relevance_terms(goal))
    if not terms:
        return results

    def score(result: SearchResult) -> int:
        title_terms = set(_relevance_terms(result.title or ""))
        snippet_terms = set(_relevance_terms(result.snippet or ""))
        return (2 * len(terms & title_terms)) + len(terms & snippet_terms)

    return sorted(
        results,
        key=lambda result: (source_quality_score(result), score(result)),
        reverse=True,
    )


_LOW_QUALITY_SOURCE_HOSTS = {
    "answers.com",
    "askfilo.com",
    "brainly.com",
    "brainly.in",
    "gauthmath.com",
    "pinterest.com",
    "questionai.com",
    "studyx.ai",
    "tiktok.com",
}


_NON_DOCUMENTARY_MARKERS = (
    "royalty free",
    "stock photo",
    "stock image",
    "stock footage",
    "vector",
    "illustration",
    "wallpaper",
    "clipart",
)


def source_quality_score(result: SearchResult) -> int:
    """Keep commonly non-documentary source categories behind document candidates.

    This is an ordering signal, not an allow-list: every normalized URL remains
    available when no better candidate exists.
    """

    host = urlsplit(result.url).netloc.casefold()
    text = " ".join(part for part in (result.title, result.snippet) if part).casefold()
    if any(host == item or host.endswith(f".{item}") for item in _LOW_QUALITY_SOURCE_HOSTS):
        return -2
    if any(marker in text for marker in _NON_DOCUMENTARY_MARKERS):
        return -1
    labels = host.split(".")
    if "gov" in labels or "edu" in labels or "ac" in labels:
        return 1
    return 0


def _relevance_terms(value: str) -> list[str]:
    return [term for term in re.findall(r"[\w]+", value.casefold()) if len(term) >= 3]


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
