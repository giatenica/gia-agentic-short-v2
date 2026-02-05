# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install dependencies (uses uv, not pip)
uv sync
uv sync --all-extras          # include dev dependencies

# Run all tests (~622 tests)
uv run pytest tests/ -v

# Run unit or integration tests
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v

# Run a single test file or function
uv run pytest tests/unit/test_reviewer.py -v
uv run pytest tests/unit/test_reviewer.py::TestReviewerNode::test_approval -v

# Coverage
uv run pytest --cov=src tests/

# Linting and type checking
uv run ruff check src/
uv run mypy src/

# Run the CLI
uv run python -m src.main

# Run LangGraph Studio (visual debugger)
cd studio && uv run langgraph dev

# Run evaluation suite
python -m evaluation.run_evaluation
python -m evaluation.run_evaluation --query-id crypto-001 --mock --dry-run
```

## Architecture

This is a LangGraph-based autonomous academic research system. A single `WorkflowState` TypedDict (30+ fields) flows through an 11-node directed graph, where each node is an async function that reads from state and returns a partial dict update.

### Workflow Graph

```
START -> intake -> [data_explorer?] -> literature_reviewer -> literature_synthesizer
  -> gap_identifier (HITL) -> planner (HITL) -> data_acquisition
  -> [data_analyst | conceptual_synthesizer] -> writer -> reviewer (HITL)
  -> [output | writer(revision, max 3)] -> END
  Any node -> fallback -> END  (on 3+ errors or unrecoverable error)
```

- **HITL gates**: `gap_identifier` and `planner` have `interrupt_before`; `reviewer` has `interrupt_after`
- **Routing**: `route_after_planner` sends theoretical research directly to `conceptual_synthesizer`, empirical to `data_acquisition` then `data_analyst`
- **Revision loop**: Reviewer scores on 5 dimensions; >= 7.0 approves, 4.0-6.9 revises (max 3 cycles), < 4.0 rejects
- **Fallback**: Triggers on 3+ errors, any unrecoverable error, or `FAILED` status

### Key Source Locations

- **Graph assembly**: `src/graphs/research_workflow.py` -- `create_research_workflow()` factory
- **Routing logic**: `src/graphs/routers.py` -- all conditional edge functions
- **State schema**: `src/state/schema.py` -- `WorkflowState` TypedDict, `create_initial_state()`
- **State models**: `src/state/models.py` -- 50+ Pydantic models (DataFile, ResearchPlan, ReviewerOutput, etc.)
- **State enums**: `src/state/enums.py` -- 23+ enums (ResearchStatus, ResearchType, ReviewDecision, etc.)
- **Node implementations**: `src/nodes/` -- one file per node, all follow the same async pattern
- **Tools**: `src/tools/` -- 19 modules, 35+ `@tool`-decorated functions
- **Config**: `src/config/settings.py` -- singleton loaded from env vars
- **Data sources**: `src/data_sources/base.py` (protocol + registry), `finance.py` (yfinance, FRED, CoinGecko)

### Node Pattern

Every node follows this structure:

```python
async def my_node(state: WorkflowState) -> dict:
    # 1. Read from state
    query = state.get("original_query")
    # 2. Process (LLM calls, tool use, computation)
    result = await do_work(query)
    # 3. Return partial state update
    return {"field_name": result, "status": ResearchStatus.NEXT_PHASE}
```

### Tool Pattern

Tools use `@tool` from `langchain_core.tools` and are grouped into lists per module:

```python
@tool
def my_tool(param: str) -> str:
    """Description shown to the model."""
    return result

MY_TOOLS = [my_tool]
```

### Routing Pattern

Each `route_after_*` function in `routers.py` checks `_should_fallback(state)` first (3+ errors or unrecoverable), then applies node-specific logic returning a `Literal` of valid next-node names.

## Critical Rules

- **NEVER fabricate data, statistics, or citations** -- all claims must be sourced
- **NEVER use emojis or em dashes** -- use semicolons, colons, or periods instead
- **ALWAYS use timezone-aware datetimes** -- `datetime.now(timezone.utc)`, never `datetime.utcnow()`
- **ALWAYS include current date in system prompts** for any new agents

### Banned Words

Never use these in generated text (unless quoting or in a strictly technical context):
delve, realm, harness, unlock, tapestry, paradigm, cutting-edge, revolutionize, landscape, potential, findings, intricate, showcasing, crucial, pivotal, surpass, meticulously, vibrant, unparalleled, underscore, leverage, synergy, innovative, game-changer, testament, commendable, meticulous, highlight, emphasize, boast, groundbreaking, align, foster, showcase, enhance, holistic, garner, accentuate, pioneering, trailblazing, unleash, versatile, transformative, redefine, seamless, optimize, scalable, robust (non-statistical), breakthrough, empower, streamline, novel, unique, utilize, impactful

Full list in `src/style/banned_words.py` and `.github/copilot-instructions.md`.

## Environment Variables

Required: `ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY`, `TAVILY_API_KEY`
Optional: `FRED_API_KEY`, `CACHE_ENABLED`, `CACHE_PATH`, `CACHE_TTL_LITERATURE` (3600), `CACHE_TTL_SYNTHESIS` (1800), `CACHE_TTL_WRITER` (600)

Config loaded via `src/config/settings.py`. Copy `.env.example` to `.env`.

## Model Selection

| Task | Model |
|------|-------|
| Complex reasoning / academic writing | `claude-opus-4-5-20251101` |
| General / default | `claude-sonnet-4-5-20250929` |
| High-volume classification / extraction | `claude-haiku-4-5-20251001` |

Default model is set in `src/config/settings.py`.

## Testing Conventions

- pytest with `pytest-asyncio` for async tests
- Mock external API calls (Anthropic, Semantic Scholar, arXiv)
- Use `MemorySaver` (in-memory) for test isolation, not SQLite
- Test fixtures for `WorkflowState` in `tests/conftest.py`
- Sprint-specific test files follow `test_sprint{N}_*.py` naming

## Authorship

Author of all academic output: **Gia Tenica** (me@giatenica.com). Gia Tenica is an anagram for Agentic AI.

## Git

- user.name: giatenica
- user.email: me@giatenica.com
