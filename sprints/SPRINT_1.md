# Sprint 1: Intake Processing and State Schema

## Overview

Sprint 1 establishes the foundation for the GIA Agentic research pipeline by implementing:

1. **State Management Layer** - Type-safe workflow state with Pydantic models
2. **INTAKE Node** - Processing and validation of research intake forms
3. **DATA_EXPLORER Node** - Automated data exploration and quality assessment
4. **Data Exploration Tools** - LangChain tools for CSV/Excel analysis

## Components

### State Management (`src/state/`)

#### Enums (`enums.py`)

Defines type-safe enumerations for workflow status and data types:

- `ResearchStatus` - 19 workflow states (INITIALIZED, INTAKE_PENDING, DATA_EXPLORED, etc.)
- `CritiqueSeverity` - Issue severity levels (CRITICAL, MAJOR, MINOR, SUGGESTION)
- `EvidenceStrength` - Evidence quality (STRONG, MODERATE, WEAK, INSUFFICIENT)
- `PaperType` - Publication types (FULL_PAPER, SHORT_PAPER, EXTENDED_ABSTRACT)
- `ResearchType` - Research methodologies (EMPIRICAL, THEORETICAL, MIXED_METHODS)
- `TargetJournal` - Target publication venues
- `DataQualityLevel` - Data quality assessment levels
- `ColumnType` - Data column types (INTEGER, FLOAT, STRING, DATETIME, etc.)

#### Models (`models.py`)

Pydantic v2 models with validation:

- `IntakeFormData` - Research submission form with validation rules
- `DataFile` - Uploaded file metadata
- `ColumnAnalysis` - Statistical analysis of data columns
- `QualityIssue` - Detected data quality problems
- `VariableMapping` - Maps user variables to data columns
- `DataExplorationResult` - Complete exploration results
- `WorkflowError` - Error tracking with context
- `ResearchPlan`, `SearchResult`, `Critique`, `EvidenceItem` - Future sprint models

#### Schema (`schema.py`)

TypedDict-based workflow state for LangGraph:

```python
class WorkflowState(TypedDict, total=False):
    thread_id: str
    status: ResearchStatus
    intake_form: IntakeFormData
    uploaded_data: list[DataFile]
    data_exploration: DataExplorationResult
    variable_mappings: list[VariableMapping]
    # ... additional fields
```

Functions:
- `create_initial_state()` - Create new workflow state
- `validate_state_for_node()` - Validate state requirements per node

### INTAKE Node (`src/nodes/intake.py`)

Processes research intake form submissions:

```python
def intake_node(state: WorkflowState) -> dict:
    """Process and validate intake form data."""
```

Features:
- Parses form data with type coercion
- Validates research question quality (length, punctuation)
- Checks hypothesis consistency
- Validates deadlines
- Processes file uploads
- Routes to DATA_EXPLORER or LITERATURE_REVIEWER

Routing Logic (`route_after_intake`):
- → `end` if validation fails
- → `data_explorer` if data files uploaded
- → `literature_reviewer` if no data files

### DATA_EXPLORER Node (`src/nodes/data_explorer.py`)

Analyzes uploaded research data:

```python
def data_explorer_node(state: WorkflowState) -> dict:
    """Explore and assess uploaded data files."""
```

Features:
- Analyzes CSV and Excel files
- Detects column types and statistics
- Assesses data quality (missing values, outliers, duplicates)
- Maps user-specified variables to data columns
- Determines research feasibility

Routing Logic (`route_after_data_explorer`):
- → `end` if quality issues prevent analysis
- → `literature_reviewer` otherwise

### Data Exploration Tools (`src/tools/data_exploration.py`)

LangChain `@tool` decorated functions:

| Tool | Description |
|------|-------------|
| `parse_csv_file` | Parse CSV and return structure info |
| `parse_excel_file` | Parse Excel with sheet selection |
| `detect_schema` | Detect column types and nullability |
| `generate_summary_stats` | Compute descriptive statistics |
| `detect_missing_values` | Find missing data patterns |
| `detect_outliers` | Identify outliers (IQR or z-score) |
| `assess_data_quality` | Overall quality assessment |

Helper Function:
- `analyze_file()` - Complete file analysis pipeline

## Testing

### Test Structure

```
tests/unit/
├── test_state.py        # 27 tests for state management
├── test_intake.py       # 27 tests for INTAKE node
└── test_data_explorer.py # 22 tests for DATA_EXPLORER
```

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific module
uv run pytest tests/unit/test_state.py -v

# With coverage
uv run pytest tests/ --cov=src
```

### Test Coverage

- State enums: All values and orderings
- Pydantic models: Creation, validation, computed properties
- Workflow state: Initialization, validation per node
- INTAKE node: Parsing, validation, routing
- DATA_EXPLORER: File analysis, variable mapping, routing

## Dependencies Added

```toml
[project.dependencies]
pandas = ">=2.0.0"
openpyxl = ">=3.1.0"
pydantic = ">=2.0.0"

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]
```

## Usage Example

```python
from src.state.schema import create_initial_state
from src.state.models import IntakeFormData
from src.state.enums import TargetJournal, PaperType, ResearchType
from src.nodes.intake import intake_node

# Create initial state
state = create_initial_state()

# Add intake form data
state["intake_form"] = IntakeFormData(
    title="Effect of AI on Academic Writing",
    research_question="How does AI assistance affect academic writing quality?",
    target_journal=TargetJournal.MIS_QUARTERLY,
    paper_type=PaperType.FULL_PAPER,
    research_type=ResearchType.EMPIRICAL,
    key_variables="writing_quality, ai_usage, word_count",
)

# Process intake
result = intake_node(state)
print(f"Status: {result['status']}")
```

## Related Documentation

- [Implementation Plan](../IMPLEMENTATION_PLAN.md) - Full project roadmap
- [Copilot Instructions](../../.github/copilot-instructions.md) - Development guidelines

## Next Steps (Sprint 2)

- LITERATURE_REVIEWER node with Tavily integration
- Search result parsing and citation extraction
- Source credibility assessment
