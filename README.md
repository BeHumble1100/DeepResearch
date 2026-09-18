# DeepResearch

DeepResearch is an evidence-driven multi-hop web research agent based on search + LLM synthesis.

## Run a question

With `.env` configured, run one question directly in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m app.cli "Who wrote Pride and Prejudice?"
```

Use `--max-steps` to set the action budget (default: `10`):

```powershell
.\.venv\Scripts\python.exe -m app.cli "Who wrote Pride and Prejudice?" --max-steps 12
```

The command prints the final status, accepted answer (if any), and traceable Fact sources.

## API

With `.env` configured, start the Phase 9 API with:

```powershell
.\.venv\Scripts\python.exe -m app.main
```

Send `POST /research` with `{"question": "...", "max_steps": 10}`. The endpoint returns the terminal research state in compact form, including traceable fact metadata and the lightweight research trace; it does not return fetched document text or fact passages.
