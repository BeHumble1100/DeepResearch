"""Print compact metrics from one completed Phase 8 experiment directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import summarize_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DeepResearch benchmark results.")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    results_path = args.output_dir / "results.jsonl"
    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(json.dumps(summarize_records(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
