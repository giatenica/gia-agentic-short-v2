# GIA Agentic Research System

LangGraph-based autonomous research system using **Anthropic Claude** for academic research automation. Features a multi-node workflow that takes a research question through literature review, gap analysis, methodology planning, data analysis, and paper writing.

## Overview

GIA (Gia Tenica) is an AI-powered research assistant that automates the academic research workflow:

1. **Intake** - Parse research question, validate inputs, process uploaded data files
2. **Data Exploration** - Analyze datasets with DuckDB backend (handles 46M+ rows)
3. **Literature Review** - Search Semantic Scholar, arXiv, and Tavily for papers
4. **Literature Synthesis** - Extract themes, identify gaps, generate contributions
5. **Gap Identification** - Analyze literature gaps with human approval checkpoints
6. **Research Planning** - Design methodology with human approval
7. **Data Analysis** - Execute statistical analysis using 35+ tools
8. **Paper Writing** - Generate academic sections with style enforcement

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

## Project Structure

```
gia-agentic-short-v2/
├── src/
│   ├── agents/              # Agent implementations
│   │   ├── base.py          # ReAct agent with tools
│   │   ├── research.py      # Research-focused agent
│   │   └── data_analyst.py  # Data analysis agent
│   ├── nodes/               # LangGraph workflow nodes
│   │   ├── intake.py        # Research intake processing
│   │   ├── data_explorer.py # Dataset analysis (DuckDB)
│   │   ├── literature_reviewer.py    # Academic search
│   │   ├── literature_synthesizer.py # Theme extraction
│   │   ├── gap_identifier.py         # Gap analysis
│   │   ├── planner.py                # Methodology planning
│   │   ├── data_analyst.py           # Statistical analysis
│   │   ├── conceptual_synthesizer.py # Theoretical research
│   │   └── writer.py                 # Paper composition
│   ├── tools/               # 17 tool modules with 35+ tools
│   │   ├── academic_search.py    # Semantic Scholar, arXiv, Tavily
│   │   ├── citation_analysis.py  # Citation metrics
│   │   ├── data_loading.py       # Load CSV, Parquet, Excel, etc.
│   │   ├── data_profiling.py     # Column statistics, distributions
│   │   ├── data_transformation.py # Filter, join, aggregate
│   │   ├── analysis.py           # Regression, correlation
│   │   ├── llm_interpretation.py # Insights, recommendations
│   │   ├── gap_analysis.py       # Literature gap detection
│   │   ├── methodology.py        # Research design
│   │   ├── contribution.py       # Contribution framing
│   │   └── synthesis.py          # Conceptual framework tools
│   ├── state/               # State management
│   │   ├── schema.py        # WorkflowState TypedDict
│   │   ├── models.py        # 50+ Pydantic models
│   │   └── enums.py         # Research status enums
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
│   └── unit/                # 281 unit tests
├── public/
│   └── research_intake_form.html
├── docs/                    # Architecture documentation
└── pyproject.toml
```

## Workflow Architecture

```
INTAKE → DATA_EXPLORER → LITERATURE_REVIEWER → LITERATURE_SYNTHESIZER
    ↓
GAP_IDENTIFIER (human approval) → PLANNER (human approval)
    ↓
    ├── DATA_ANALYST (empirical research)
    │       ↓
    └── CONCEPTUAL_SYNTHESIZER (theoretical)
            ↓
        WRITER → END
```

### Node Descriptions

| Node | Purpose |
|------|---------|
| `intake` | Parse form data, validate research question, process uploads |
| `data_explorer` | Parallel loading of datasets, schema detection, quality assessment |
| `literature_reviewer` | Generate search queries, execute multi-source academic search |
| `literature_synthesizer` | Extract themes, synthesize findings, identify gaps |
| `gap_identifier` | Analyze gaps, generate refined question (with interrupt) |
| `planner` | Design methodology, assess feasibility (with interrupt) |
| `data_analyst` | Execute regressions, correlations, hypothesis tests |
| `conceptual_synthesizer` | Build theoretical frameworks |
| `writer` | Generate paper sections with style enforcement |

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
- `create_variable` - Computed columns (safe eval with input validation)
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

## Caching System

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
# Run all 281 tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/unit/test_data_explorer.py -v

# Run with coverage
uv run pytest --cov=src tests/
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | ✅ |
| `LANGSMITH_API_KEY` | LangSmith API key for tracing | ✅ |
| `TAVILY_API_KEY` | Tavily API key for web search | ✅ |
| `LANGSMITH_TRACING` | Enable tracing (default: true) | ❌ |
| `LANGSMITH_PROJECT` | Project name in LangSmith | ❌ |
| `CACHE_ENABLED` | Enable LLM response caching | ❌ |
| `CACHE_PATH` | SQLite cache location | ❌ |
| `CACHE_TTL_DEFAULT` | Default cache TTL in seconds | ❌ |
| `CACHE_TTL_LITERATURE` | Literature search cache TTL | ❌ |
| `CACHE_TTL_SYNTHESIS` | Synthesis nodes cache TTL | ❌ |
| `CACHE_TTL_WRITER` | Writer node cache TTL | ❌ |

## Model Configuration

| Task Type | Model | Use Case |
|-----------|-------|----------|
| Complex Reasoning | `claude-opus-4-5-20251101` | Research, scientific analysis |
| General/Coding | `claude-sonnet-4-5-20250929` | Default for most tasks |
| High-Volume | `claude-haiku-4-5-20251001` | Classification, extraction |

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run specific tests
uv run pytest tests/unit/test_intake.py -v

# Run LangGraph Studio
cd studio && uv run langgraph dev
```

## Security

- API keys loaded from environment variables (never hardcoded)
- ZIP extraction protected against zip bombs and path traversal
- Safe expression evaluation (AST-based, no `eval()` on user input)
- Expression validation blocks code injection patterns
- CORS restricted to localhost in development
- Timezone-aware datetime handling (UTC)

## Author

**Gia Tenica** (me@giatenica.com)

Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher.
For more information: https://giatenica.com

## License

MIT License - see [LICENSE](LICENSE) for details.
