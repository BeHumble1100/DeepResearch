# DeepResearch

[English](#english) · [中文](#中文)

---

## English

DeepResearch is an evidence-driven multi-hop web research system evolved from a ResearchAgent prototype developed for the Alibaba Cloud PAI Developer Competition. It targets complex factual questions through cross-source retrieval, candidate-entity and answer-hypothesis verification, long-document passage retrieval, and deterministic evidence validation.

### Research loop

```text
Question
  → Initialization
  → Planner
  → SEARCH / OPEN / LOCATE
  → Evidence Update
  → ANSWER
  → Deterministic Guard
```

### What is implemented

- **Structured planning:** an LLM selects `SEARCH`, `OPEN`, `LOCATE`, or `ANSWER` from a compact research-state view.
- **Web search:** an asynchronous SearXNG gateway with query rewrite, timeout/retry handling, result normalization, and URL deduplication.
- **Source opening:** HTTP retrieval and parsing for HTML and PDF sources. Parsed document content is retained locally rather than injected into Planner context.
- **Document-local retrieval:** chunking, BM25 lexical recall, and LLM semantic reranking over a small candidate set.
- **Evidence contracts:** structured Facts retain source URL, document ID, passage ID, and constraint relations.
- **Candidate-scoped evidence:** positive or negative evidence from separate answer hypotheses cannot be silently combined.
- **Finish control:** an Answer Guard validates evidence IDs, required constraints, source provenance, scope consistency, and target format before ending a run.
- **Observability:** a lightweight serializable Research Trace records decisions, compact Planner context, bounded search observations, rejections, evidence updates, and Guard outcomes.
- **Interfaces:** a FastAPI endpoint and a one-command CLI both use the same production research graph.

Search snippets are used for source selection only; they are not accepted as answer evidence.

### Requirements

- Python 3.10+
- A reachable SearXNG instance
- An OpenAI-compatible LLM API. The default configuration targets DeepSeek.

### Setup

```powershell
git clone https://github.com/<your-user>/DeepResearch.git
cd DeepResearch
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env
```

Set at least the following values in `.env`:

```env
LLM_API_KEY=your_api_key
SEARXNG_BASE_URL=http://localhost:8080
```

Adjust `LLM_BASE_URL`, `LLM_MODEL`, and the SearXNG URL as needed. Never commit `.env`.

### Run one question from the terminal

```powershell
.\.venv\Scripts\python.exe -m app.cli "Who wrote Pride and Prejudice?"
```

Set a bounded action budget when needed (default: `10`):

```powershell
.\.venv\Scripts\python.exe -m app.cli "Who wrote Pride and Prejudice?" --max-steps 12
```

The CLI prints the terminal status, accepted answer when available, step usage, and traceable Fact sources. A run may end without an accepted answer when the available evidence does not satisfy the Guard.

### Run the API

```powershell
.\.venv\Scripts\python.exe -m app.main
```

The API is available at `http://127.0.0.1:8000` and interactive documentation at `http://127.0.0.1:8000/docs`.

```http
POST /research
Content-Type: application/json

{
  "question": "Who wrote Pride and Prejudice?",
  "max_steps": 10
}
```

The response contains terminal state, constraints, compact Fact metadata, and the lightweight Trace. It intentionally excludes fetched document text and Fact passage text.

### Verify

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```

### Scope and limitations

This repository focuses on an inspectable V1 research harness rather than claiming reliable answers for every open-web question. Real-web performance depends on source availability, SearXNG engine quality, document accessibility, and model behavior. The Trace is intended to support reproducible Bad Case analysis and iterative improvement.

For architecture and implementation phases, see [`specs/deepresearch-v1.md`](specs/deepresearch-v1.md).

---

## 中文

DeepResearch 是一个基于阿里云 PAI 开发者大赛 ResearchAgent 原型持续迭代的 **evidence-driven multi-hop Web Research System**。它面向复杂事实问题，通过跨来源检索、候选实体与答案假设验证、长文档 Passage 定位和确定性证据校验，生成可追溯的研究结论。

### 研究主链

```text
Question
  → Initialization
  → Planner
  → SEARCH / OPEN / LOCATE
  → Evidence Update
  → ANSWER
  → Deterministic Guard
```

### 已实现机制

- **结构化 Planner：** LLM 基于压缩后的 ResearchState，在 `SEARCH`、`OPEN`、`LOCATE`、`ANSWER` 间选择下一步。
- **Web Search：** 通过异步 SearXNG Gateway 执行 Query Rewrite、超时/重试、结果归一化与 URL 去重。
- **Document Opening：** 真实抓取并解析 HTML/PDF；完整正文保存在本地，不直接进入 Planner context。
- **Document-local Retrieval：** 对文档分块后，使用 BM25 进行低成本召回，再用 LLM 对小规模候选 Passage 语义重排。
- **Evidence Contract：** Fact 保留 source URL、document ID、passage ID 及其与 Constraint 的关系。
- **Candidate Evidence Scope：** 不同候选答案假设的正负证据不会被静默拼接或互相污染。
- **Answer Guard：** 在结束前校验证据 ID、required constraints、来源 provenance、scope 一致性与 Target 格式。
- **Research Trace：** 轻量且可序列化地记录 Planner 决策、压缩 context、有限搜索结果、动作拒绝、证据更新与 Guard 结果。
- **调用入口：** CLI 和 FastAPI API 共用同一条 production research graph。

Search snippet 仅用于选择候选来源，不能直接作为最终答案的证据。

### 环境要求

- Python 3.10+
- 可访问的 SearXNG 实例
- OpenAI-compatible LLM API；默认配置为 DeepSeek

### 配置与运行

按英文部分的 **Setup** 创建虚拟环境、安装依赖，并复制 `.env.example` 为 `.env`。至少配置：

```env
LLM_API_KEY=你的API密钥
SEARXNG_BASE_URL=http://localhost:8080
```

确认本地 SearXNG 已启动后，直接在 PowerShell 中提问：

```powershell
.\.venv\Scripts\python.exe -m app.cli "《傲慢与偏见》的作者是谁？"
```

设置最大研究步数：

```powershell
.\.venv\Scripts\python.exe -m app.cli "《傲慢与偏见》的作者是谁？" --max-steps 12
```

CLI 会输出最终状态、通过 Guard 的答案（如有）、实际步数，以及每条 Fact 的来源。证据不足时，系统可能以没有 accepted answer 的状态结束，这是一项证据约束而非静默猜测。

### API

启动服务：

```powershell
.\.venv\Scripts\python.exe -m app.main
```

然后访问：

- API 文档：`http://127.0.0.1:8000/docs`
- 研究接口：`POST http://127.0.0.1:8000/research`

请求示例：

```json
{
  "question": "《傲慢与偏见》的作者是谁？",
  "max_steps": 10
}
```

响应包含终态、constraints、轻量 Fact metadata 与 Trace；不会返回抓取的完整网页/PDF 正文或 Fact passage 原文。

### 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```

### 边界

本仓库聚焦于可检查、可追溯的 V1 研究框架，不将其表述为对所有开放网络问题都稳定可靠的答案系统。实际表现会受到搜索引擎质量、网页可访问性、文档解析结果和模型行为影响。Research Trace 用于复现 Bad Case，并驱动后续迭代。

架构与阶段设计见 [`specs/deepresearch-v1.md`](specs/deepresearch-v1.md)。
