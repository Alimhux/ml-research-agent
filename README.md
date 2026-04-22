# ml-research-agent

A ReAct-style research assistant that answers ML/AI questions in natural
language, grounded in arxiv papers.

You ask a question — the agent reformulates it into a technical query,
searches arxiv, reads the returned abstracts, and composes a direct
answer with cited papers and per-paper key ideas extracted from their
abstracts.

Built on LangGraph + LangChain + Yandex AI Studio (OpenAI-compatible
endpoint).

## Setup

Requires Python 3.11+.

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Fill .env with your Yandex Cloud API key and folder id.
```

## Run

```bash
python -m agent.main "What is GRPO and how does it differ from PPO?"
python -m agent.main -v "Что такое трансформер?"   # verbose logs on stderr
```

Output on stdout is a JSON-serialized `FinalAnswer`:

```json
{
  "question": "...",
  "answer": "A direct answer grounded in the retrieved papers.",
  "papers": [
    {
      "arxiv_id": "1706.03762",
      "title": "Attention Is All You Need",
      "url": "https://arxiv.org/abs/1706.03762",
      "abstract": "...",
      "key_ideas": [
        "Self-attention replaces recurrence for sequence modeling.",
        "..."
      ]
    }
  ]
}
```

## Architecture

Three-node LangGraph state machine with a ReAct loop:

```
         ┌─────────┐       tool_calls        ┌───────┐
START ──▶│  agent  │ ──────────────────────▶ │ tools │
         └─────────┘ ◀────────────────────── └───────┘
              │              messages
              │  no tool_calls / step cap
              ▼
         ┌─────────┐
         │ compose │ ──▶ END
         └─────────┘
```

1. **`agent`** — LLM with `arxiv_search` bound as a tool. Decides whether
   to issue another search or stop. Capped at `_MAX_AGENT_STEPS = 6`.
2. **`tools`** — executes tool calls, appends `ToolMessage`s, deduplicates
   retrieved papers into `state.papers_found`. Tool errors are surfaced
   back to the LLM as `ToolMessage(status="error")` rather than crashing
   the graph.
3. **`compose`** — a separate LLM call that takes the dialog + all
   retrieved papers and produces a structured `FinalAnswer` via
   `PydanticOutputParser`. The parser is used instead of
   `with_structured_output` because YandexGPT's native tool-call format
   does not always roundtrip cleanly through LangChain's structured
   output wrappers on longer prompts.

See [`src/agent/graph.py`](src/agent/graph.py) and
[`src/agent/schemas.py`](src/agent/schemas.py).

## Tests

```bash
pytest                       # unit tests only
pytest -m integration        # hits the live arxiv API
```

## Status

Work in progress. Known limitations and planned iterations:

- **Search quality.** arxiv's built-in relevance ranking does not surface
  canonical papers well (e.g. querying "PPO" does not return Schulman et
  al. 2017 on top). Planned: migrate `arxiv_search` to the Semantic
  Scholar API, which exposes citation counts and behaves closer to
  Google Scholar, and add a category filter to keep results within
  ML/math (`cs.LG`, `cs.AI`, `cs.CL`, `stat.ML`, `math.OC`, ...).
- Reflection / self-critique over retrieved papers.
- Paper caching between runs.
- Full-PDF reading via `pymupdf4llm` for questions that need method
  details beyond the abstract.
- Evaluation harness and cross-model comparison.
- HTTP interface (FastAPI) and Docker packaging.
