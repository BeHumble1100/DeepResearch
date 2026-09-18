from app import cli
from app.research.schemas import Fact
from app.research.state import ResearchState


def test_cli_runs_the_production_executor_and_prints_compact_result(monkeypatch, capsys) -> None:
    calls: list[tuple[str, int]] = []

    async def execute(question: str, max_steps: int) -> ResearchState:
        calls.append((question, max_steps))
        return ResearchState(
            question=question,
            max_steps=max_steps,
            status="completed",
            step_count=3,
            answer="Ada",
            facts=[
                Fact(
                    id="fact:1",
                    statement="Evidence supports Ada.",
                    source_url="https://example.test/source",
                    passage="Private passage.",
                    confidence=0.9,
                )
            ],
    )

    monkeypatch.setattr(cli, "build_production_executor", lambda *, settings: execute)
    monkeypatch.setattr(cli, "Settings", lambda: object())

    assert cli.main(["Who is Ada?", "--max-steps", "6"]) == 0
    assert calls == [("Who is Ada?", 6)]
    assert capsys.readouterr().out == (
        "Question: Who is Ada?\n"
        "Status: completed\n"
        "Steps: 3/6\n"
        "Answer:\n"
        "Ada\n"
        "Evidence:\n"
        "- [fact:1] Evidence supports Ada.\n"
        "  Source: https://example.test/source\n"
    )
