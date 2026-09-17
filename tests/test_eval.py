from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.research.state import ResearchState
from eval.metrics import is_correct, normalize_answer, summarize_records
from eval.runner import BenchmarkCase, load_cases, run_cases


def test_normalize_answer_is_conservative() -> None:
    assert normalize_answer("  James\u3000Lockhart ") == "james lockhart"
    assert is_correct(expected="James Lockhart", actual="  james lockhart ")
    assert not is_correct(expected="James Lockhart", actual="James E. Lockhart")


def test_load_cases_requires_matching_unique_dataset_ids(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    answers = tmp_path / "answers.jsonl"
    questions.write_text(json.dumps({"id": 2, "question": "q"}) + "\n", encoding="utf-8")
    answers.write_text(json.dumps({"id": 2, "answer": "a"}) + "\n", encoding="utf-8")
    assert load_cases(question_path=questions, answer_path=answers) == [
        BenchmarkCase(id=2, question="q", expected_answer="a")
    ]


def test_simple_v1_dataset_has_eight_matching_cases() -> None:
    cases = load_cases(
        question_path=Path("eval/datasets/simple-v1/questions.jsonl"),
        answer_path=Path("eval/datasets/simple-v1/answers.jsonl"),
    )

    assert [case.id for case in cases] == list(range(1, 9))
    assert all(case.question and case.expected_answer for case in cases)


def test_simple_multihop_v1_dataset_has_five_matching_cases() -> None:
    cases = load_cases(
        question_path=Path("eval/datasets/simple-multihop-v1/questions.jsonl"),
        answer_path=Path("eval/datasets/simple-multihop-v1/answers.jsonl"),
    )

    assert [case.id for case in cases] == list(range(1, 6))
    assert all(case.question and case.expected_answer for case in cases)


def test_runner_isolates_question_failures_and_writes_metrics(tmp_path: Path) -> None:
    async def execute(question: str, max_steps: int) -> ResearchState:
        if question == "broken":
            raise RuntimeError("backend unavailable")
        return ResearchState(
            question=question,
            answer="Answer",
            status="answer_accepted",
            step_count=max_steps,
        )

    records = asyncio.run(
        run_cases(
            cases=[
                BenchmarkCase(id=1, question="works", expected_answer="answer"),
                BenchmarkCase(id=2, question="broken", expected_answer="none"),
            ],
            executor=execute,
            output_dir=tmp_path / "run",
            max_steps=3,
        )
    )
    assert [record["correct"] for record in records] == [True, False]
    assert records[1]["error_type"] == "RuntimeError"
    assert (tmp_path / "run" / "id-1.state.json").is_file()
    assert not (tmp_path / "run" / "id-2.state.json").exists()
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["question_count"] == 2
    assert metrics["completed_question_count"] == 1
    assert metrics["accuracy"] == 0.5


def test_trace_metrics_are_derived_from_saved_records() -> None:
    metrics = summarize_records(
        [
            {
                "correct": False,
                "status": "budget_exhausted",
                "step_count": 4,
                "error": None,
                "error_type": None,
                "trace": [
                    {"action": "search", "guard_result": None, "validation_rejection_reason": None},
                    {
                        "action": "answer",
                        "guard_result": {"accepted": False},
                        "validation_rejection_reason": "No evidence",
                    },
                ],
            }
        ]
    )
    assert metrics["action_counts"] == {"answer": 1, "search": 1}
    assert metrics["guard_rejection_count"] == 1
    assert metrics["validation_rejection_count"] == 1
