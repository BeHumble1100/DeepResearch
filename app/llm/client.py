"""Minimal provider-neutral contract for LLM calls."""

from __future__ import annotations

import json
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings

SchemaT = TypeVar("SchemaT", bound=BaseModel)
Message = dict[str, str]


class LLMClient(Protocol):
    async def structured(
        self,
        *,
        messages: list[Message],
        schema: type[SchemaT],
        **kwargs: object,
    ) -> SchemaT: ...

    async def text(self, *, messages: list[Message], **kwargs: object) -> str: ...


class LLMClientError(RuntimeError):
    """Raised when an OpenAI-compatible provider cannot return a usable response."""


class OpenAICompatibleClient:
    """Minimal async client for OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if settings.llm_api_key is None:
            raise ValueError("LLM_API_KEY is required to create an OpenAI-compatible client.")
        self._api_key = settings.llm_api_key.get_secret_value()
        self._base_url = settings.llm_base_url
        self._model = settings.llm_model
        self._timeout_seconds = settings.llm_timeout_seconds
        self._thinking_mode = settings.llm_thinking_mode
        self._transport = transport

    async def structured(
        self,
        *,
        messages: list[Message],
        schema: type[SchemaT],
        **kwargs: object,
    ) -> SchemaT:
        response_text = await self._complete(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return a valid JSON object matching this requested structured schema exactly: "
                        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
                    ),
                },
                *messages,
            ],
            response_format={"type": "json_object"},
            **kwargs,
        )
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise LLMClientError("Provider returned invalid JSON for a structured response.") from error
        try:
            return schema.model_validate(payload)
        except ValidationError as error:
            raise LLMClientError("Provider JSON did not match the requested schema.") from error

    async def text(self, *, messages: list[Message], **kwargs: object) -> str:
        return await self._complete(messages=messages, **kwargs)

    async def _complete(
        self,
        *,
        messages: list[Message],
        response_format: dict[str, str] | None = None,
        **kwargs: object,
    ) -> str:
        payload: dict[str, object] = {"model": self._model, "messages": messages, **kwargs}
        if response_format is not None:
            payload["response_format"] = response_format
        if self._thinking_mode is not None:
            payload["thinking"] = {"type": self._thinking_mode}
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_seconds),
                transport=self._transport,
            ) as client:
                response = await client.post(
                    "/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as error:
            raise LLMClientError("OpenAI-compatible chat completion request failed.") from error
        except ValueError as error:
            raise LLMClientError("Provider returned invalid JSON.") from error

        try:
            content = body["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as error:
            raise LLMClientError("Provider response did not contain message content.") from error
        if not isinstance(content, str) or not content.strip():
            raise LLMClientError("Provider returned empty message content.")
        return content
