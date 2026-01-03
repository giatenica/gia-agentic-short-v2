# GIA Agentic v2 - Copilot Instructions

## Architecture Overview

- **LangGraph** for agent orchestration and state management
- **Claude 4.5 Family** via `langchain-anthropic` with task-based model selection
- **LangSmith** for tracing, debugging, and evaluation
- **Tavily** for web search capabilities
- **DuckDB** for high-performance data analysis (handles 46M+ rows)
- **Memory**: Checkpointers (SQLite/MemorySaver) for conversation persistence

## Complete Project Structure

```
src/
├── agents/              # Agent implementations
│   ├── base.py          # ReAct agent with tools
│   ├── research.py      # Research-focused agent
│   └── data_analyst.py  # Data analysis agent
├── nodes/               # LangGraph workflow nodes
│   ├── intake.py        # Research intake processing
│   ├── data_explorer.py # Dataset analysis (DuckDB, parallel loading)
│   ├── literature_reviewer.py    # Academic search (Semantic Scholar, arXiv)
│   ├── literature_synthesizer.py # Theme extraction, gap identification
│   ├── gap_identifier.py         # Gap analysis with human approval
│   ├── planner.py                # Methodology planning with approval
│   ├── data_analyst.py           # Statistical analysis execution
│   ├── conceptual_synthesizer.py # Theoretical framework building
│   └── writer.py                 # Paper section composition
├── tools/               # 17 tool modules with 35+ tools
│   ├── academic_search.py    # Semantic Scholar, arXiv, Tavily
│   ├── citation_analysis.py  # Citation metrics, network analysis
│   ├── data_loading.py       # Load CSV, Parquet, Excel, Stata, SPSS, ZIP
│   ├── data_profiling.py     # Column stats, distributions, outliers
│   ├── data_transformation.py # Filter, join, aggregate, create variables
│   ├── data_analysis.py      # Regression, correlation, hypothesis tests
│   ├── data_interpretation.py # Insights, recommendations
│   ├── gap_analysis.py       # Literature gap detection
│   ├── methodology.py        # Research design tools
│   └── contribution.py       # Contribution framing
├── state/               # State management
│   ├── schema.py        # WorkflowState TypedDict (30+ fields)
│   ├── models.py        # 50+ Pydantic models
│   └── enums.py         # Status and type enumerations
├── cache/               # SQLite-based LLM response caching
│   └── __init__.py      # get_cache(), get_cache_policy()
├── citations/           # Citation management
│   ├── manager.py       # CitationManager class
│   ├── formatter.py     # APA formatting
│   └── reference_list.py
├── style/               # Academic writing style enforcement
│   ├── banned_words.py  # 100+ flagged words
│   ├── academic_tone.py # Tone analysis
│   ├── hedging.py       # Hedging language detection
│   └── enforcer.py      # StyleEnforcer with auto-fix
├── writers/             # Section-specific paper writers
│   ├── abstract.py
│   ├── introduction.py
│   ├── literature_review.py
│   ├── methods.py
│   ├── results.py
│   ├── discussion.py
│   └── conclusion.py
├── memory/              # Persistence layer
│   ├── checkpointer.py  # Thread-based conversation memory
│   └── store.py         # Long-term cross-session memory
├── config/              # Environment and settings
│   └── settings.py      # All config from env vars
└── server.py            # Flask intake form server

studio/                  # LangGraph Studio configuration
├── graphs.py            # Workflow graph definition
└── langgraph.json
```

## Research Workflow Architecture

```
INTAKE → DATA_EXPLORER → LITERATURE_REVIEWER → LITERATURE_SYNTHESIZER
                              ↓
                    GAP_IDENTIFIER (interrupt for approval)
                              ↓
                    PLANNER (interrupt for approval)
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
       DATA_ANALYST              CONCEPTUAL_SYNTHESIZER
       (empirical)                   (theoretical)
              ↓                               ↓
              └───────────────┬───────────────┘
                              ↓
                           WRITER → END
```

### Node Details

| Node | Purpose | Caching |
|------|---------|---------|
| `intake` | Parse form, validate inputs, process uploads | Never |
| `data_explorer` | Parallel dataset loading, schema detection | Never |
| `literature_reviewer` | Academic search (Semantic Scholar, arXiv) | 1 hour |
| `literature_synthesizer` | Extract themes, identify gaps | 30 min |
| `gap_identifier` | Analyze gaps, human approval checkpoint | 30 min |
| `planner` | Design methodology, human approval | Never |
| `data_analyst` | Execute regressions, correlations, tests | 30 min |
| `conceptual_synthesizer` | Build theoretical frameworks | 30 min |
| `writer` | Generate paper sections with style enforcement | 10 min |

## Agent Model Configuration

| Task Type | Model | Use Case |
|-----------|-------|----------|
| Complex Reasoning | `claude-opus-4-5-20251101` | Research, scientific analysis, academic writing |
| General/Coding | `claude-sonnet-4-5-20250929` | Default for most tasks, agents, data analysis |
| High-Volume | `claude-haiku-4-5-20251001` | Classification, summarization, extraction |

Default model is configured in `src/config/settings.py`.

## Creating New Agents (REQUIRED PATTERN)

All new agents MUST follow this LangGraph pattern:

```python
from typing import Annotated, Literal
import operator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from src.config import settings


class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[list[AnyMessage], operator.add]


def create_my_agent(
    tools: list | None = None,
    model_name: str | None = None,
    system_prompt: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
):
    """Create agent with memory support."""
    
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=4096,
        api_key=settings.anthropic_api_key,
    )
    
    # Bind tools if provided
    if tools:
        model = model.bind_tools(tools)
    
    # ALWAYS include current date in system prompt
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    system_content = f"""Current date: {current_date}

{system_prompt or 'You are a helpful AI assistant.'}

IMPORTANT: If you need current information beyond your knowledge cutoff, 
indicate that web search would be needed."""

    system_message = SystemMessage(content=system_content)
    
    def call_model(state: AgentState) -> dict:
        messages = [system_message] + state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END
    
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    if tools:
        graph.add_node("tools", ToolNode(tools))
        graph.add_conditional_edges("agent", should_continue, ["tools", END])
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", END)
    graph.add_edge(START, "agent")
    
    return graph.compile(checkpointer=checkpointer, store=store)
```

## Memory System

### Conversation Persistence (Checkpointer)
```python
from src.memory import get_checkpointer

# In-memory (development)
checkpointer = get_checkpointer(persistent=False)

# SQLite (persistent)
checkpointer = get_checkpointer(persistent=True)

# Use with thread_id
config = {"configurable": {"thread_id": "user-123"}}
result = agent.invoke({"messages": [...]}, config=config)
```

### Long-term Memory (Store)
```python
from src.memory import get_memory_store, MemoryManager

store = get_memory_store()
memory = MemoryManager(store)

# Store user facts
memory.store_user_fact("user-123", "name", "Alice")
memory.store_preference("user-123", "language", "Python")

# Retrieve
facts = memory.get_user_facts("user-123")
prefs = memory.get_preferences("user-123")
```

## Tool Creation

```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """
    Tool description (shown to the model).
    
    Args:
        param: Parameter description.
    
    Returns:
        Result description.
    """
    # Implementation
    return result
```

## LangGraph Studio

For local development and debugging:

```bash
cd studio
uv run langgraph dev
```

**Important**: Don't pass `checkpointer` or `store` to agents in `studio/graphs.py` - LangGraph API handles persistence automatically.

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | ✅ |
| `LANGSMITH_API_KEY` | LangSmith API key for tracing | ✅ |
| `TAVILY_API_KEY` | Tavily API key for web search | ✅ |
| `LANGSMITH_TRACING` | Enable tracing (default: true) | ❌ |
| `LANGSMITH_PROJECT` | Project name in LangSmith | ❌ |
| `CACHE_ENABLED` | Enable LLM response caching (default: true) | ❌ |
| `CACHE_PATH` | SQLite cache location | ❌ |
| `CACHE_TTL_LITERATURE` | Literature cache TTL in seconds (default: 3600) | ❌ |
| `CACHE_TTL_SYNTHESIS` | Synthesis cache TTL in seconds (default: 1800) | ❌ |
| `CACHE_TTL_WRITER` | Writer cache TTL in seconds (default: 600) | ❌ |

## Critical Rules for All Agents

1. **NEVER make up data, statistics, numbers, or facts**
2. **NEVER use emojis**
3. **NEVER use em dashes** (use semicolons, colons, or periods)
4. **ALWAYS include current date** in system prompts
5. **ALWAYS flag outdated knowledge** that needs web search
6. **ALWAYS cite sources** for quantitative claims

## Banned Words (NEVER USE UNLESS IN TECHNICAL CONTEXT)

delve, realm, harness, unlock, tapestry, paradigm, cutting-edge, revolutionize,
landscape, potential, findings, intricate, showcasing, crucial, pivotal, surpass,
meticulously, vibrant, unparalleled, underscore, leverage, synergy, innovative,
game-changer, testament, commendable, meticulous, highlight, emphasize, boast,
groundbreaking, align, foster, showcase, enhance, holistic, garner, accentuate,
pioneering, trailblazing, unleash, versatile, transformative, redefine, seamless,
optimize, scalable, robust (non-statistical), breakthrough, empower, streamline,
intelligent, smart, next-gen, frictionless, elevate, adaptive, effortless,
data-driven, insightful, proactive, mission-critical, visionary, disruptive,
reimagine, agile, customizable, personalized, unprecedented, intuitive,
leading-edge, synergize, democratize, automate, accelerate, state-of-the-art,
dynamic (non-technical), reliable, efficient, cloud-native, immersive, predictive,
transparent, proprietary, integrated, plug-and-play, turnkey, future-proof,
open-ended, AI-powered, next-generation, always-on, hyper-personalized,
results-driven, machine-first, paradigm-shifting, novel, unique, utilize, impactful

## Development Guidelines

- Use `uv` for dependency management (`uv sync`, `uv add`, `uv run`)
- Use async/await patterns where appropriate
- Add LangSmith tracing for debugging multi-agent interactions
- Test agents independently before integration
- Keep secrets in `.env` (gitignored)

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_agents.py -v
```

When testing agents:
- Mock the Anthropic client for unit tests
- Use `MemorySaver` (in-memory) for test isolation
- Test tool bindings separately from agent logic

## Running the Application

```bash
# Interactive CLI
uv run python -m src.main

# LangGraph Studio
cd studio && uv run langgraph dev
```

## Git Configuration

- user.name: giatenica
- user.email: me@giatenica.com
- Keep `.env` out of git
- Run tests before committing

## Authorship

The Author of all academic writing is always **Gia Tenica*** (me@giatenica.com)

*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher. For more information see: https://giatenica.com
