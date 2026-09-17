# DeepResearch

DeepResearch is an evidence-driven multi-hop web research agent based on search + LLM synthesis.

## API

With `.env` configured, start the Phase 9 API with:

```powershell
.\.venv\Scripts\python.exe -m app.main
```

Send `POST /research` with `{"question": "...", "max_steps": 10}`. The endpoint returns the terminal research state in compact form, including traceable fact metadata and the lightweight research trace; it does not return fetched document text or fact passages.
