"""ASGI application entry point for DeepResearch."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.research import ResearchExecutor, build_production_executor, create_research_router
from app.config import Settings


def create_app(
    *,
    executor: ResearchExecutor | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create the API, allowing tests to supply a no-network research executor."""
    if executor is None:
        executor = build_production_executor(settings=settings or Settings())

    app = FastAPI(title="DeepResearch", version="0.1.0")
    app.include_router(create_research_router(executor=executor))
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:create_app", factory=True)
