"""Command-line entry point for one complete DeepResearch run."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from app.api.research import build_production_executor
from app.config import Settings
from app.research.state import ResearchState


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one evidence-driven research question.")
    parser.add_argument("question", help="Question to research.")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum research actions (default: 10).")
    args = parser.parse_args(argv)
    if not args.question.strip():
        parser.error("question must not be blank")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    return args


def _format_result(research: ResearchState) -> str:
    lines = [
        f"Question: {research.question}",
        f"Status: {research.status}",
        f"Steps: {research.step_count}/{research.max_steps}",
        "Answer:",
        research.answer or "No evidence-supported answer was accepted.",
        "Evidence:",
    ]
    if not research.facts:
        lines.append("- None")
    for fact in research.facts:
        lines.extend((f"- [{fact.id}] {fact.statement}", f"  Source: {fact.source_url}"))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the production research executor and print a compact terminal result."""
    args = _parse_args(argv)
    executor = build_production_executor(settings=Settings())
    research = asyncio.run(executor(args.question, args.max_steps))
    print(_format_result(research))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
