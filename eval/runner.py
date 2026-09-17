"""Reproducible, failure-isolated Phase 8 benchmark runner."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.llm.client import OpenAICompatibleClient
from app.research.evidence import LLMFactExtractor
from app.research.graph import build_research_graph
from app.research.guard import AnswerGuard
from app.research.planner import LLMPlanner
from app.research.state import ResearchState
from app.tools.document import HttpDocumentOpener
from app.tools.retrieval import DocumentRetriever, LLMPassageReranker
from app.tools.search import LLMQueryRewriter, SearXNGSearchGateway
from eval.metrics import is_correct, summarize_records

ResearchExecutor = Callable[[str, int], Awaitable[ResearchState]]


@dataclass(frozen=True)
class BenchmarkCase:
    id: int
    question: str
    expected_answer: str


def load_cases(*, question_path: Path, answer_path: Path) -> list[BenchmarkCase]:
    questions = _load_jsonl_by_id(question_path, required_key="question")
    answers = _load_jsonl_by_id(answer_path, required_key="answer")
    if set(questions) != set(answers):
        raise ValueError("Question and answer datasets must contain the same IDs.")
    return [
        BenchmarkCase(id=item_id, question=questions[item_id], expected_answer=answers[item_id])
        for item_id in sorted(questions)
    ]


async def run_cases(
    *,
    cases: list[BenchmarkCase],
    executor: ResearchExecutor,
    output_dir: Path,
    max_steps: int,
) -> list[dict[str, Any]]:
    """Run sequentially; one provider or backend failure never stops later cases."""
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for case in cases:
        try:
            research = await executor(case.question, max_steps)
        except Exception as error:
            record = {
                **asdict(case),
                "actual_answer": None,
                "correct": False,
                "status": None,
                "step_count": 0,
                "trace": [],
                "error_type": type(error).__name__,
                "error": str(error),
            }
        else:
            state_path = output_dir / f"id-{case.id}.state.json"
            state_path.write_text(research.model_dump_json(indent=2), encoding="utf-8")
            record = {
                **asdict(case),
                "actual_answer": research.answer,
                "correct": is_correct(expected=case.expected_answer, actual=research.answer),
                "status": research.status,
                "step_count": research.step_count,
                "trace": [entry.model_dump() for entry in research.trace],
                "error_type": None,
                "error": None,
            }
        records.append(record)
        _append_jsonl(output_dir / "results.jsonl", record)
    (output_dir / "metrics.json").write_text(
        json.dumps(summarize_records(records), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records


def build_real_executor(*, searxng_base_url: str) -> ResearchExecutor:
    """Wire the production dependencies without adding a separate execution path."""
    settings = Settings(searxng_base_url=searxng_base_url)
    client = OpenAICompatibleClient(settings)
    graph = build_research_graph(
        LLMPlanner(client),
        SearXNGSearchGateway(settings, query_rewriter=LLMQueryRewriter(client)),
        HttpDocumentOpener(settings),
        DocumentRetriever(settings, LLMPassageReranker(client)),
        LLMFactExtractor(client),
        AnswerGuard(),
    )

    async def execute(question: str, max_steps: int) -> ResearchState:
        return (
            await graph.ainvoke(
                {"research": ResearchState(question=question, max_steps=max_steps), "action": None}
            )
        )["research"]

    return execute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DeepResearch Phase 8 benchmark.")
    parser.add_argument("--questions", type=Path, default=Path("eval/datasets/question.jsonl"))
    parser.add_argument("--answers", type=Path, default=Path("eval/datasets/answer.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--searxng-base-url", required=True)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", type=int, nargs="*")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    cases = load_cases(question_path=args.questions, answer_path=args.answers)
    if args.ids is not None:
        selected_ids = set(args.ids)
        cases = [case for case in cases if case.id in selected_ids]
        if len(cases) != len(selected_ids):
            raise ValueError("One or more --ids values are absent from the dataset.")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("No benchmark cases selected.")
    records = await run_cases(
        cases=cases,
        executor=build_real_executor(searxng_base_url=args.searxng_base_url),
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )
    print(json.dumps(summarize_records(records), ensure_ascii=False, indent=2))


def _load_jsonl_by_id(path: Path, *, required_key: str) -> dict[int, str]:
    rows: dict[int, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            item_id = payload["id"]
            value = payload[required_key]
        except (TypeError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid dataset row {line_number} in {path}.") from error
        if not isinstance(item_id, int) or not isinstance(value, str) or item_id in rows:
            raise ValueError(f"Invalid dataset row {line_number} in {path}.")
        rows[item_id] = value
    return rows


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
