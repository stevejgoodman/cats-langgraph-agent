# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Serve graphs locally (API at http://localhost:2024)
uv run langgraph dev

# Test the running server (requires langgraph dev to be running)
uv run test_served_graph.py
```

## Environment Setup

Copy `.env.example` to `.env` and fill in:
- `OPENAI_API_KEY` (required)
- `TAVILY_API_KEY` (required)
- `LANGSMITH_API_KEY` (optional, for LangSmith Studio)

Optional overrides: `OPENAI_MODEL`, `OPENAI_CHAT_MODEL` (default: `gpt-4.1-nano`), `RAG_DATA_DIR` (default: `data`).

## Architecture

This project serves LangGraph agent graphs via the LangGraph Platform (local dev server).

**`langgraph.json`** — Platform config file. Declares which graphs to serve (as `graphs`) and exposes them as named assistants (as `assistants`). To add a new graph, register it here.

**`app/` package:**
- `models.py` — `get_chat_model()` factory; reads `OPENAI_MODEL` env var, defaults to `gpt-4.1-nano`
- `state.py` — Re-exports `MessagesState` from LangGraph (standard message list state)
- `tools.py` — `get_tool_belt()` assembles the agent's tools: Tavily search, Arxiv, and the RAG tool
- `rag.py` — Builds an in-memory RAG pipeline (Qdrant) from PDFs in `RAG_DATA_DIR`. The pipeline is cached via `lru_cache` and exposed as a `@tool` called `retrieve_information` (domain: feline health)
- `graphs/simple_agent.py` — Minimal tool-using agent: `START → agent → (tools_condition) → action → agent → END`
- `graphs/agent_with_helpfulness.py` — Extends simple agent with a post-response `helpfulness` node that uses structured output (`HelpfulnessResult`) to decide whether to loop back to `agent` or end. Hard limit: terminates after 10 messages to prevent infinite loops.

**Adding a new graph:**
1. Create `app/graphs/your_graph.py` with a compiled `graph` object
2. Add an entry to `langgraph.json` under `graphs` (`"your_id": "app.graphs.your_graph:graph"`) and optionally under `assistants`

**LangSmith Studio:** When `langgraph dev` is running, visit `https://smith.langchain.com/studio?baseUrl=http://localhost:2024` to visualize and debug graphs.
