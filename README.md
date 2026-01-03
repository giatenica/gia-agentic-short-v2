# GIA Agentic Research System v2

[![Tests](https://img.shields.io/badge/tests-622%20passed-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-61%25-yellow)](tests/)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](pyproject.toml)
[![LangGraph](https://img.shields.io/badge/langgraph-0.2+-purple)](https://github.com/langchain-ai/langgraph)

LangGraph-based autonomous research system using **Anthropic Claude** for academic research automation. Features a multi-node workflow that takes a research question through literature review, gap analysis, methodology planning, data analysis, paper writing, and critical review.

## Overview

GIA (Gia Tenica) is an AI-powered research assistant that automates the complete academic research workflow:

```
INTAKE → DATA_EXPLORER → LITERATURE_REVIEWER → LITERATURE_SYNTHESIZER
                              ↓
              GAP_IDENTIFIER (human approval checkpoint)
                              ↓
                  PLANNER (human approval checkpoint)
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
       DATA_ANALYST              CONCEPTUAL_SYNTHESIZER
       (empirical)                   (theoretical)
              ↓                               ↓
              └───────────────┬───────────────┘
                              ↓
                           WRITER
                              ↓
                          REVIEWER
                        ↙         ↘
                   APPROVE       REVISE (max 3)
                      ↓            ↓
                   OUTPUT ←───────┘
                              ↓
                          FALLBACK (error recovery)
```

### Key Features

- **10 Workflow Nodes** - Complete research pipeline from intake to publication
- **Human-in-the-Loop** - Approval checkpoints at gap analysis and planning
- **Multi-Source Search** - Semantic Scholar, arXiv, and Tavily integration
- **Data Analysis** - DuckDB backend handles 46M+ rows with 35+ analysis tools
- **Academic Writing** - Style enforcement with 100+ banned words filter
- **Self-Critique Loop** - REVIEWER node with revision cycles (max 3)
- **Error Recovery** - Graceful degradation with fallback node
- **LangSmith Tracing** - Full observability and debugging

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Install with dev dependencies
uv sync --all-extras
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

### 3. Run with LangGraph Studio

```bash
cd studio
uv run langgraph dev
```

Open: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

### 4. Run CLI

```bash
uv run python -m src.main
```

## Workflow Nodes

| Node | Purpose | HITL |
|------|---------|------|
| `intake` | Parse form data, validate research question, process uploads | - |
| `data_explorer` | Parallel dataset loading, schema detection, quality checks | - |
| `literature_reviewer` | Multi-source academic search (Semantic Scholar, arXiv) | - |
| `literature_synthesizer` | Theme extraction, findings synthesis, gap identification | - |
| `gap_identifier` | Analyze gaps, generate refined research question | ✓ |
| `planner` | Design methodology, assess feasibility | ✓ |
| `data_analyst` | Execute regressions, correlations, hypothesis tests | - |
| `conceptual_synthesizer` | Build theoretical frameworks (non-empirical) | - |
| `writer` | Generate paper sections with style enforcement | - |
| `reviewer` | Critical evaluation with 5-dimension scoring | - |
| `output` | Prepare final paper for delivery | - |
| `fallback` | Generate partial output on errors | - |

### Review Decision Thresholds

| Score Range | Decision | Action |
|-------------|----------|--------|
| ≥ 7.0 | APPROVE | Route to OUTPUT |
| 4.0 - 6.9 | REVISE | Route to WRITER (max 3 iterations) |
| < 4.0 | REJECT | Route to OUTPUT with rejection note |

## Project Structure

```
gia-agentic-short-v2/
├── src/
│   ├── agents/              # Agent implementations
│   │   ├── base.py          # ReAct agent with tools
│   │   ├── research.py      # Research-focused agent
│   │   └── data_analyst.py  # Data analysis agent
│   ├── nodes/               # LangGraph workflow nodes (10 nodes)
│   │   ├── intake.py        # Research intake processing
│   │   ├── data_explorer.py # Dataset analysis (DuckDB)
│   │   ├── literature_reviewer.py    # Academic search
│   │   ├── literature_synthesizer.py # Theme extraction
│   │   ├── gap_identifier.py         # Gap analysis (HITL)
│   │   ├── planner.py                # Methodology planning (HITL)
│   │   ├── data_analyst.py           # Statistical analysis
│   │   ├── conceptual_synthesizer.py # Theoretical research
│   │   ├── writer.py                 # Paper composition
│   │   ├── reviewer.py               # Critical review
│   │   └── fallback.py               # Error recovery
│   ├── graphs/              # Workflow assembly
│   │   ├── research_workflow.py  # Main workflow factory
│   │   ├── routers.py       # Routing functions
│   │   ├── streaming.py     # Streaming utilities
│   │   ├── debug.py         # Time travel debugging
│   │   └── subgraphs.py     # Modular subgraphs
│   ├── tools/               # 17 tool modules with 35+ tools
│   │   ├── academic_search.py    # Semantic Scholar, arXiv, Tavily
│   │   ├── citation_analysis.py  # Citation metrics
│   │   ├── data_loading.py       # CSV, Parquet, Excel, Stata, SPSS
│   │   ├── data_profiling.py     # Column statistics, distributions
│   │   ├── data_transformation.py # Filter, join, aggregate
│   │   ├── analysis.py           # Regression, correlation
│   │   ├── llm_interpretation.py # Insights, recommendations
│   │   ├── gap_analysis.py       # Literature gap detection
│   │   ├── methodology.py        # Research design
│   │   ├── contribution.py       # Contribution framing
│   │   └── synthesis.py          # Conceptual framework
│   ├── state/               # State management
│   │   ├── schema.py        # WorkflowState TypedDict (30+ fields)
│   │   ├── models.py        # 50+ Pydantic models
│   │   └── enums.py         # Research status enums
│   ├── errors/              # Error handling
│   │   ├── exceptions.py    # Custom exception hierarchy
│   │   ├── policies.py      # RetryPolicy configurations
│   │   ├── handlers.py      # Error handler functions
│   │   └── recovery.py      # Recovery strategies
│   ├── review/              # Review criteria
│   │   └── criteria.py      # 5-dimension scoring
│   ├── cache/               # SQLite-based LLM response caching
│   ├── citations/           # Citation management (APA style)
│   ├── style/               # Academic writing style enforcement
│   │   ├── banned_words.py  # 100+ flagged marketing words
│   │   ├── academic_tone.py # Tone analysis
│   │   ├── hedging.py       # Hedging language detection
│   │   ├── precision.py     # Precision checking
│   │   └── enforcer.py      # Auto-fix violations
│   ├── writers/             # Section-specific writers
│   │   ├── abstract.py
│   │   ├── introduction.py
│   │   ├── literature_review.py
│   │   ├── methods.py
│   │   ├── results.py
│   │   ├── discussion.py
│   │   └── conclusion.py
│   ├── memory/              # Checkpointer and store
│   ├── config/              # Settings from environment
│   └── server.py            # Flask intake form server
├── studio/
│   ├── graphs.py            # LangGraph workflow definition
│   └── langgraph.json       # Studio configuration
├── tests/
│   ├── unit/                # Unit tests (622 tests)
│   └── integration/         # Integration tests
├── evaluation/              # Quality evaluation suite
│   ├── metrics.py           # Quality metrics
│   ├── run_evaluation.py    # Evaluation runner
│   └── test_queries.json    # Test query dataset
├── public/
│   └── research_intake_form.html
├── docs/
│   ├── IMPLEMENTATION_PLAN.md
│   ├── langgraph_architecture_spec.md
│   └── writing_style_guide.md
└── sprints/                 # Sprint documentation
```

## Data Analysis Tools

The system includes 35+ tools organized into categories:

### Loading & Registry
- `load_data` - CSV, Excel, Parquet, Stata, SPSS, JSON, ZIP
- `list_datasets` - View registered datasets
- `query_data` - SQL queries via DuckDB

### Profiling
- `describe_column` - Column statistics
- `compute_distributions` - Histograms, frequency tables
- `detect_outliers` - IQR/Z-score methods

### Transformation
- `filter_data` - Row filtering with expressions
- `create_variable` - Computed columns (safe eval)
- `merge_datasets` - Join operations
- `aggregate_data` - Group-by summaries

### Analysis
- `run_regression` - OLS with diagnostics
- `compute_correlation` - Correlation matrices
- `run_hypothesis_test` - t-test, chi-squared, etc.

### Interpretation
- `interpret_regression` - Plain-English explanations
- `generate_insights` - Key finding summaries
- `suggest_analyses` - Next-step recommendations

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | ✅ |
| `LANGSMITH_API_KEY` | LangSmith API key for tracing | ✅ |
| `TAVILY_API_KEY` | Tavily API key for web search | ✅ |
| `LANGSMITH_TRACING` | Enable tracing (default: true) | ❌ |
| `LANGSMITH_PROJECT` | Project name in LangSmith | ❌ |
| `CACHE_ENABLED` | Enable LLM response caching | ❌ |
| `CACHE_PATH` | SQLite cache location | ❌ |
| `CACHE_TTL_LITERATURE` | Literature search cache TTL | ❌ |
| `CACHE_TTL_SYNTHESIS` | Synthesis nodes cache TTL | ❌ |
| `CACHE_TTL_WRITER` | Writer node cache TTL | ❌ |

### Model Configuration

| Task Type | Model | Use Case |
|-----------|-------|----------|
| Complex Reasoning | `claude-opus-4-5-20251101` | Research, scientific analysis |
| General/Coding | `claude-sonnet-4-5-20250929` | Default for most tasks |
| High-Volume | `claude-haiku-4-5-20251001` | Classification, extraction |

### Caching System

LLM responses are cached in SQLite to speed up development:

```bash
# Enable/disable caching
export CACHE_ENABLED=true  # default

# Configure TTLs (seconds)
export CACHE_TTL_LITERATURE=3600   # 1 hour
export CACHE_TTL_SYNTHESIS=1800    # 30 minutes
export CACHE_TTL_WRITER=600        # 10 minutes
```

## Testing

```bash
# Run all tests (622 tests)
uv run pytest tests/ -v

# Run unit tests only
uv run pytest tests/unit/ -v

# Run integration tests
uv run pytest tests/integration/ -v

# Run with coverage
uv run pytest --cov=src tests/

# Run specific test file
uv run pytest tests/unit/test_reviewer.py -v
```

### Evaluation Suite

```bash
# Run evaluation with test queries
python -m evaluation.run_evaluation

# Run specific query
python -m evaluation.run_evaluation --query-id crypto-001

# Use mock responses for testing
python -m evaluation.run_evaluation --mock

# Dry run (show what would be evaluated)
python -m evaluation.run_evaluation --dry-run
```

## Error Handling

The system implements comprehensive error handling with graceful degradation:

- **RetryPolicy** - Exponential backoff with jitter for API calls
- **Recovery Strategies** - RETRY, SKIP, FALLBACK, ABORT
- **Fallback Node** - Generates partial output on unrecoverable errors
- **Max 3 retries** before fallback activation

### Exception Hierarchy

```
GIAError (base)
├── WorkflowError
├── NodeExecutionError
├── ToolExecutionError
├── APIError
│   └── RateLimitError
├── ContextOverflowError
├── DataValidationError
├── SearchError
│   └── LiteratureSearchError
├── AnalysisError
├── WritingError
└── ReviewError
```

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run LangGraph Studio
cd studio && uv run langgraph dev

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

## Security

- API keys loaded from environment variables (never hardcoded)
- ZIP extraction protected against zip bombs and path traversal
- Safe expression evaluation (regex-based pattern blocking)
- CORS restricted to localhost in development
- Timezone-aware datetime handling (UTC)

## Documentation

- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) - Sprint-by-sprint development guide
- [Architecture Spec](docs/langgraph_architecture_spec.md) - Detailed architecture specification
- [Writing Style Guide](docs/writing_style_guide.md) - Academic writing standards

## Author

**Gia Tenica** (me@giatenica.com)

Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher.
For more information: https://giatenica.com

## License

MIT License - see [LICENSE](LICENSE) for details.
