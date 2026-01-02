# GIA Agentic Systems

LangGraph-based agentic systems using **Anthropic Claude** with LangSmith observability.

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Required keys:
- `ANTHROPIC_API_KEY` - Get from [Anthropic Console](https://console.anthropic.com/)
- `LANGSMITH_API_KEY` - Get from [LangSmith](https://smith.langchain.com/)
- `TAVILY_API_KEY` - Get from [Tavily](https://tavily.com/)

### 3. Run the Agent

```bash
# Interactive CLI
uv run python -m src.main

# Or directly
uv run python src/main.py
```

## LangGraph Studio

Visualize and debug agents with LangGraph Studio:

```bash
cd studio
langgraph dev
```

Then open: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

## Project Structure

```
gia-agentic-short-v2/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── base.py       # ReAct agent with LangGraph
│   │   └── research.py   # Research-focused agent
│   ├── tools/            # Tool definitions
│   │   ├── search.py     # Tavily web search
│   │   └── basic.py      # Utility tools
│   ├── config/           # Configuration
│   │   └── settings.py   # Environment settings
│   └── main.py           # CLI entrypoint
├── studio/               # LangGraph Studio config
│   ├── langgraph.json
│   └── graphs.py
├── pyproject.toml        # Dependencies
└── .env                  # API keys (not in git)
```

## Available Agents

### ReAct Agent
General-purpose agent with reasoning and tool use:
- Web search (Tavily)
- Calculator
- Current time

### Research Agent  
Specialized for information gathering:
- Enhanced web search
- Source citation
- Structured responses

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Format code
uv run ruff format .
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | ✅ |
| `LANGSMITH_API_KEY` | LangSmith API key for tracing | ✅ |
| `TAVILY_API_KEY` | Tavily API key for web search | ✅ |
| `LANGSMITH_TRACING` | Enable tracing (default: true) | ❌ |
| `LANGSMITH_PROJECT` | Project name in LangSmith | ❌ |
