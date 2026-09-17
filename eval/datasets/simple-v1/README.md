# Simple V1

`simple-v1` is a diagnostic dataset for the shortest evidence-backed research
loop. Every question has one stable, single-hop factual answer and should be
answerable in global/direct evidence mode without candidate scopes or multi-hop
entity resolution.

It is intentionally separate from the 100-question Phase 8 dataset. A failure
must be classified from its saved Trace as one of search, OPEN, Fact Extraction,
Fact-to-constraint binding, Planner/ANSWER decision, or Guard before changing
the production system.

Run it against real dependencies with a new ignored experiment directory:

```powershell
.\.venv\Scripts\python.exe -m eval.runner `
  --questions eval\datasets\simple-v1\questions.jsonl `
  --answers eval\datasets\simple-v1\answers.jsonl `
  --output-dir eval\answers\simple-v1-YYYYMMDD-run-N `
  --searxng-base-url http://127.0.0.1:8080 `
  --max-steps 6
```

Promotion gate: all eight questions must be correct and complete in two
independent real runs, with no infrastructure failures. Only then should a
change be evaluated on the multi-hop benchmark.
