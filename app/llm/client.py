"""Minimal provider-neutral contract for LLM calls."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

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
