import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.llm.client import LLMClientError, OpenAICompatibleClient
from app.research.schemas import QueryRewrite


def make_settings() -> Settings:
    return Settings(
        searxng_base_url="http://localhost:8080",
        llm_api_key="test-key",
        llm_base_url="https://llm.test",
        llm_model="test-model",
    )


def test_structured_request_uses_json_mode_and_validates_response() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"queries":["LangGraph"]}'}}]},
        )

    client = OpenAICompatibleClient(
        make_settings(), transport=httpx.MockTransport(handler)
    )
    result = asyncio.run(
        client.structured(
            messages=[{"role": "user", "content": "Return a query."}], schema=QueryRewrite
        )
    )

    assert result == QueryRewrite(queries=["LangGraph"])
    assert captured["model"] == "test-model"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["thinking"] == {"type": "disabled"}
    assert "JSON object" in captured["messages"][0]["content"]
    assert '"queries"' in captured["messages"][0]["content"]


def test_structured_request_rejects_schema_mismatch() -> None:
    client = OpenAICompatibleClient(
        make_settings(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"choices": [{"message": {"content": "{}"}}]}
            )
        ),
    )

    with pytest.raises(LLMClientError, match="schema"):
        asyncio.run(
            client.structured(messages=[{"role": "user", "content": "Return JSON."}], schema=QueryRewrite)
        )


def test_text_request_returns_provider_content() -> None:
    client = OpenAICompatibleClient(
        make_settings(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"choices": [{"message": {"content": "hello"}}]}
            )
        ),
    )

    assert asyncio.run(client.text(messages=[{"role": "user", "content": "Hello"}])) == "hello"
