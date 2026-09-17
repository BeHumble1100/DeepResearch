"""Conservative answer scoring and trace-derived aggregate metrics."""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Any


def normalize_answer(answer: str | None) -> str | None:
    """Normalize only Unicode form, whitespace, and case for benchmark comparison."""
    if answer is None:
        return None
    normalized = " ".join(unicodedata.normalize("NFKC", answer).split()).casefold()
    return normalized or None


def is_correct(*, expected: str, actual: str | None) -> bool:
    return normalize_answer(expected) == normalize_answer(actual)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce reproducible primary and trace-derived metrics from run records."""
    completed = [record for record in records if record.get("error") is None]
    correct = sum(bool(record.get("correct")) for record in records)
    action_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    guard_rejections = 0
    validation_rejections = 0
    total_steps = 0
    for record in records:
        if record.get("status") is not None:
            status_counts[str(record["status"])] += 1
        if record.get("error_type") is not None:
            error_types[str(record["error_type"])] += 1
        total_steps += int(record.get("step_count") or 0)
        for entry in record.get("trace", []):
            action_counts[str(entry["action"])] += 1
            guard_result = entry.get("guard_result") or {}
            if guard_result.get("accepted") is False:
                guard_rejections += 1
            if entry.get("validation_rejection_reason"):
                validation_rejections += 1
    return {
        "question_count": len(records),
        "completed_question_count": len(completed),
        "failed_question_count": len(records) - len(completed),
        "correct_count": correct,
        "accuracy": correct / len(records) if records else 0.0,
        "average_steps": total_steps / len(completed) if completed else 0.0,
        "status_counts": dict(sorted(status_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "guard_rejection_count": guard_rejections,
        "validation_rejection_count": validation_rejections,
        "error_type_counts": dict(sorted(error_types.items())),
    }
