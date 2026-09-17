# Simple Multi-hop V1

This diagnostic set contains five short, stable questions with one explicit
intermediate entity or relation. It is not a replacement for the Phase 8
benchmark: it isolates the transition from a research clue to an evidence-backed
final answer without complex disambiguation or candidate elimination.

Run with real dependencies and a fresh ignored output directory:

```powershell
.\.venv\Scripts\python.exe -m eval.runner `
  --questions eval\datasets\simple-multihop-v1\questions.jsonl `
  --answers eval\datasets\simple-multihop-v1\answers.jsonl `
  --output-dir eval\answers\simple-multihop-v1-YYYYMMDD-run-N `
  --searxng-base-url http://127.0.0.1:8080 `
  --max-steps 8
```

Promotion gate: all five questions must be correct in two independent real
runs. Use the saved Trace to classify any failure before changing production
logic.
