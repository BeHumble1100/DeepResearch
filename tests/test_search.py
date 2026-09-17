import asyncio
import json

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

    async def rewrite(
        self,
        *,
        goal: str,
        query: str,
        unresolved_required_constraints=None,
        resolved_entities=None,
    ) -> list[str]:
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


def test_gateway_demotes_off_topic_results_without_domain_filtering() -> None:
    gateway = SearXNGSearchGateway(
        make_settings(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://noise.example/translate",
                            "title": "Translation service",
                            "content": "Translate text between languages.",
                        },
                        {
                            "url": "https://source.example/gold",
                            "title": "Gold chemical symbol Au",
                            "content": "Gold has the chemical symbol Au.",
                        },
                    ]
                },
            )
        ),
    )

    results = asyncio.run(
        gateway.search(goal="Find the chemical symbol for gold", query="chemical symbol gold")
    )

    assert [result.url for result in results] == [
        "https://source.example/gold",
        "https://noise.example/translate",
    ]


def test_gateway_demotes_common_non_documentary_sources_even_when_lexically_relevant() -> None:
    gateway = SearXNGSearchGateway(
        make_settings(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://www.tiktok.com/example",
                            "title": "Tokyo capital city of Japan Kyoto",
                            "content": "Tokyo is the capital city of Japan.",
                        },
                        {
                            "url": "https://source.example/japan",
                            "title": "Japan country profile",
                            "content": "Information about Japan and Kyoto.",
                        },
                    ]
                },
            )
        ),
    )

    results = asyncio.run(
        gateway.search(goal="Find the capital city of the country containing Kyoto", query="Kyoto capital")
    )

    assert [result.url for result in results] == [
        "https://source.example/japan",
        "https://www.tiktok.com/example",
    ]


def test_gateway_prefers_institutional_sources_and_demotes_stock_material() -> None:
    gateway = SearXNGSearchGateway(
        make_settings(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://photos.example/canada",
                            "title": "Canada maple leaf stock photo",
                            "content": "Royalty free stock photo of the Canadian flag.",
                        },
                        {
                            "url": "https://example.gov/canada-flag",
                            "title": "National flag of Canada",
                            "content": "Official information about the maple leaf flag.",
                        },
                    ]
                },
            )
        ),
    )

    results = asyncio.run(
        gateway.search(goal="Verify Canada's maple leaf flag", query="Canada maple leaf flag")
    )

    assert [result.url for result in results] == [
        "https://example.gov/canada-flag",
        "https://photos.example/canada",
    ]


def test_llm_query_rewriter_receives_compact_research_context() -> None:
    client = FakeLLMClient(QueryRewrite(queries=["author work"]))

    assert asyncio.run(
        LLMQueryRewriter(client).rewrite(
            goal="Resolve author",
            query="work",
            unresolved_required_constraints=[{"id": "c1", "description": "Author wrote work"}],
            resolved_entities={"author": "Ada"},
        )
    ) == ["author work"]
    assert "1 to 3" in client.messages[0]["content"]
    assert "unresolved required constraints" in client.messages[0]["content"]
    assert json.loads(client.messages[1]["content"]) == {
        "goal": "Resolve author",
        "planner_query": "work",
        "unresolved_required_constraints": [{"id": "c1", "description": "Author wrote work"}],
        "resolved_entities": {"author": "Ada"},
    }
