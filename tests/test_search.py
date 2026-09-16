import asyncio

import httpx
import pytest

from app.config import Settings
from app.research.schemas import QueryRewrite, SearchResult
from app.tools.search import (
    LLMQueryRewriter,
    SearchGatewayError,
    SearXNGSearchGateway,
)


class FixedQueryRewriter:
    def __init__(self, queries: list[str]) -> None:
        self._queries = queries

    async def rewrite(self, *, goal: str, query: str) -> list[str]:
        return self._queries


class FakeLLMClient:
    def __init__(self, response: QueryRewrite) -> None:
        self.response = response
        self.messages: list[dict[str, str]] = []

    async def structured(self, *, messages, schema, **kwargs):
        self.messages = messages
        return self.response

    async def text(self, *, messages, **kwargs) -> str:
        return "unused"


def make_settings(**overrides: object) -> Settings:
    return Settings(
        searxng_base_url="https://searxng.test",
        searxng_timeout_seconds=1,
        searxng_max_retries=1,
        searxng_retry_backoff_seconds=0,
        searxng_max_results=10,
        **overrides,
    )


def test_gateway_rewrites_queries_and_normalizes_deduplicated_results() -> None:
    requested_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_queries.append(request.url.params["q"])
        assert request.url.path == "/search"
        assert request.url.params["format"] == "json"
        if request.url.params["q"] == "first query":
            results = [
                {
                    "url": "https://EXAMPLE.com/source#section",
                    "title": " First source ",
                    "content": " First snippet ",
                },
                {"url": "not-a-url", "title": "Ignored"},
            ]
        else:
            results = [
                {"url": "https://example.com/source", "title": "Duplicate"},
                {"url": "https://example.com/second", "content": "Second snippet"},
            ]
        return httpx.Response(200, json={"results": results})

    gateway = SearXNGSearchGateway(
        make_settings(),
        query_rewriter=FixedQueryRewriter(["first query", "second query"]),
        transport=httpx.MockTransport(handler),
    )

    results = asyncio.run(gateway.search(goal="Find sources", query="original query"))

    assert requested_queries == ["first query", "second query"]
    assert results == [
        SearchResult(
            url="https://example.com/source",
            title="First source",
            snippet="First snippet",
        ),
        SearchResult(
            url="https://example.com/second",
            title=None,
            snippet="Second snippet",
        ),
    ]


def test_gateway_retries_transient_response_before_succeeding() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"results": [{"url": "https://example.com"}]})

    gateway = SearXNGSearchGateway(
        make_settings(),
        transport=httpx.MockTransport(handler),
    )

    results = asyncio.run(gateway.search(goal="Find", query="query"))

    assert attempts == 2
    assert results == [SearchResult(url="https://example.com", title=None, snippet=None)]


def test_gateway_retries_a_timeout_before_succeeding() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"results": []})

    gateway = SearXNGSearchGateway(
        make_settings(),
        transport=httpx.MockTransport(handler),
    )

    assert asyncio.run(gateway.search(goal="Find", query="query")) == []
    assert attempts == 2


def test_gateway_rejects_an_empty_rewrite() -> None:
    gateway = SearXNGSearchGateway(
        make_settings(),
        query_rewriter=FixedQueryRewriter(["", "  "]),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"results": []})),
    )

    with pytest.raises(SearchGatewayError, match="no usable queries"):
        asyncio.run(gateway.search(goal="Find", query="query"))


def test_gateway_filters_deduplicates_and_limits_rewrites() -> None:
    requested_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_queries.append(request.url.params["q"])
        return httpx.Response(200, json={"results": []})

    gateway = SearXNGSearchGateway(
        make_settings(),
        query_rewriter=FixedQueryRewriter([" first ", "", "first", "second", "third", "fourth"]),
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(gateway.search(goal="Find", query="query"))

    assert requested_queries == ["first", "second", "third"]


def test_llm_query_rewriter_prompt_is_goal_focused() -> None:
    client = FakeLLMClient(QueryRewrite(queries=["author work"]))

    assert asyncio.run(LLMQueryRewriter(client).rewrite(goal="Resolve author", query="work")) == [
        "author work"
    ]
    assert "1 to 3" in client.messages[0]["content"]
    assert "current goal" in client.messages[0]["content"]
