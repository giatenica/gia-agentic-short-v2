# API Reference

This document provides a comprehensive API reference for the GIA Agentic Research System v2.

## Table of Contents

- [Workflow Factory](#workflow-factory)
- [Nodes](#nodes)
- [State Schema](#state-schema)
- [Tools](#tools)
- [Error Handling](#error-handling)
- [Agents](#agents)

---

## Workflow Factory

### `src.graphs.research_workflow`

The main entry point for creating research workflows.

#### `create_research_workflow`

```python
def create_research_workflow(
    config: WorkflowConfig | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> StateGraph:
    """
    Create the main research workflow graph.
    
    Assembles all nodes into the complete academic research workflow:
    INTAKE → DATA_EXPLORER → LITERATURE_REVIEWER → LITERATURE_SYNTHESIZER 
        → GAP_IDENTIFIER → PLANNER
        → DATA_ANALYST or CONCEPTUAL_SYNTHESIZER 
        → WRITER → REVIEWER → OUTPUT
    
    Args:
        config: Workflow configuration (optional)
        checkpointer: Override checkpointer from config
        store: Override store from config
        
    Returns:
        Compiled StateGraph ready for execution
        
    Example:
        # Basic usage (LangGraph Studio)
        workflow = create_research_workflow()
        
        # With custom configuration
        config = WorkflowConfig(
            checkpointer=SqliteSaver.from_conn_string("sqlite:///research.db"),
            interrupt_before=["gap_identifier", "planner"],
        )
        workflow = create_research_workflow(config)
    """
```

#### `create_studio_workflow`

```python
def create_studio_workflow() -> StateGraph:
    """
    Create workflow for LangGraph Studio.
    
    Studio manages persistence automatically, so no checkpointer is configured.
    
    Returns:
        Compiled StateGraph for Studio
    """
```

#### `create_production_workflow`

```python
def create_production_workflow(db_path: str = "research.db") -> StateGraph:
    """
    Create workflow for production with SQLite persistence.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        Compiled StateGraph with persistent checkpointer
    """
```

#### `WorkflowConfig`

```python
@dataclass
class WorkflowConfig:
    """Configuration for research workflow compilation.
    
    Attributes:
        checkpointer: Checkpoint saver for persistence (optional)
        store: Memory store for long-term data (optional)
        cache: Cache backend for node caching (optional)
        interrupt_before: Nodes to pause before execution
        interrupt_after: Nodes to pause after execution
        enable_caching: Whether to enable node-level caching
        debug: Enable debug logging
    """
    checkpointer: BaseCheckpointSaver | None = None
    store: BaseStore | None = None
    cache: Any | None = None
    interrupt_before: list[str] = field(default_factory=lambda: ["gap_identifier", "planner"])
    interrupt_after: list[str] = field(default_factory=lambda: ["reviewer"])
    enable_caching: bool = True
    debug: bool = False
```

---

## Nodes

### `src.nodes.intake`

#### `intake_node`

```python
def intake_node(state: WorkflowState) -> dict:
    """
    Process intake form submission and validate inputs.
    
    Parses form data into structured state and validates required fields.
    
    Args:
        state: Current workflow state with form_data
        
    Returns:
        State update with parsed intake data and INTAKE_COMPLETE status
    """
```

### `src.nodes.data_explorer`

#### `data_explorer_node`

```python
async def data_explorer_node(state: WorkflowState) -> dict:
    """
    Analyze uploaded data before planning phase.
    
    Performs parallel loading of datasets using DuckDB backend,
    schema detection, and quality assessment.
    
    Args:
        state: Current workflow state with uploaded_data
        
    Returns:
        State update with data_exploration_results
    """
```

### `src.nodes.literature_reviewer`

#### `literature_reviewer_node`

```python
async def literature_reviewer_node(state: WorkflowState) -> dict:
    """
    Conduct systematic literature review.
    
    Generates search queries and executes multi-source academic search
    across Semantic Scholar, arXiv, and Tavily.
    
    Args:
        state: Current workflow state with original_query
        
    Returns:
        State update with literature_search_results
    """
```

### `src.nodes.literature_synthesizer`

#### `literature_synthesizer_node`

```python
async def literature_synthesizer_node(state: WorkflowState) -> dict:
    """
    Synthesize findings from literature searches.
    
    Extracts key themes, identifies seminal works, and surfaces
    methodological precedents and debates.
    
    Args:
        state: Current workflow state with literature_search_results
        
    Returns:
        State update with literature_synthesis
    """
```

### `src.nodes.gap_identifier`

#### `gap_identifier_node`

```python
async def gap_identifier_node(state: WorkflowState) -> dict:
    """
    Identify gaps and refine research question.
    
    Analyzes what literature covers vs. what user asked,
    generates refined research question, and requests human approval.
    
    Uses interrupt() for HITL checkpoint.
    
    Args:
        state: Current workflow state with literature_synthesis
        
    Returns:
        State update with gap_analysis and refined_query
    """
```

### `src.nodes.planner`

#### `planner_node`

```python
async def planner_node(state: WorkflowState) -> dict:
    """
    Design methodology based on gap analysis.
    
    Selects appropriate methodology, designs analysis approach,
    and requests human approval of the research plan.
    
    Uses interrupt() for HITL checkpoint.
    
    Args:
        state: Current workflow state with gap_analysis
        
    Returns:
        State update with research_plan
    """
```

### `src.nodes.data_analyst`

#### `data_analyst_node`

```python
async def data_analyst_node(state: WorkflowState) -> dict:
    """
    Execute data analysis per research plan.
    
    Performs statistical analysis using methodology from plan,
    generates findings with statistical backing.
    
    Args:
        state: Current workflow state with research_plan and data
        
    Returns:
        State update with analysis_results and findings
    """
```

### `src.nodes.conceptual_synthesizer`

#### `conceptual_synthesizer_node`

```python
async def conceptual_synthesizer_node(state: WorkflowState) -> dict:
    """
    Build theoretical framework from literature.
    
    Used for non-empirical research. Synthesizes concepts,
    generates propositions, and grounds in existing theory.
    
    Args:
        state: Current workflow state with literature_synthesis
        
    Returns:
        State update with conceptual_framework and propositions
    """
```

### `src.nodes.writer`

#### `writer_node`

```python
async def writer_node(state: WorkflowState) -> dict:
    """
    Orchestrate section writing based on paper type.
    
    Determines sections to write based on paper type and
    coordinates section writers with style enforcement.
    
    Args:
        state: Current workflow state with analysis/synthesis results
        
    Returns:
        State update with draft sections
    """
```

### `src.nodes.reviewer`

#### `reviewer_node`

```python
async def reviewer_node(state: WorkflowState) -> dict:
    """
    Critically evaluate draft against academic standards.
    
    Evaluates paper across 5 dimensions:
    - Contribution (25%)
    - Methodology (25%)
    - Evidence (20%)
    - Coherence (15%)
    - Writing (15%)
    
    Uses interrupt() for human approval of decision.
    
    Args:
        state: Current workflow state with draft
        
    Returns:
        State update with review_critique and review_decision
    """
```

### `src.nodes.fallback`

#### `fallback_node`

```python
def fallback_node(state: WorkflowState) -> dict:
    """
    Generate partial output on errors.
    
    Activates when workflow cannot complete normally.
    Collects available partial results and generates
    informative error summary.
    
    Args:
        state: Current workflow state with errors
        
    Returns:
        State update with partial output and error summary
    """
```

---

## State Schema

### `src.state.schema.WorkflowState`

```python
class WorkflowState(TypedDict):
    """Central state schema for research workflow.
    
    Categories:
    - Intake Data: form_data, project_title, original_query, etc.
    - Data Context: uploaded_data, data_exploration_results, etc.
    - Research Context: literature_synthesis, gap_analysis, etc.
    - Workflow State: messages, research_plan, draft, critique, etc.
    - Review State: review_critique, review_decision, revision_count, etc.
    - Error State: errors, error_count, recovery_strategy, etc.
    """
    
    # Intake form data
    form_data: dict | None
    project_title: str
    original_query: str
    target_journal: str | None
    paper_type: str | None
    research_type: str | None
    user_hypothesis: str | None
    
    # Data context
    uploaded_data: list[DataFile] | None
    data_context: str | None
    data_exploration_results: DataExplorationResult | None
    key_variables: list[str] | None
    
    # Research context
    proposed_methodology: str | None
    seed_literature: list[str] | None
    expected_contribution: str | None
    literature_synthesis: LiteratureSynthesis | None
    gap_analysis: GapAnalysis | None
    refined_query: str | None
    contribution_statement: str | None
    
    # Workflow state
    messages: Annotated[list[AnyMessage], add_messages]
    research_plan: ResearchPlan | None
    analysis_results: AnalysisResult | None
    conceptual_framework: ConceptualFramework | None
    draft: ResearchDraft | None
    status: ResearchStatus
    iteration_count: int
    
    # Review state
    review_critique: ReviewCritique | None
    review_decision: str | None
    reviewer_output: ReviewerOutput | None
    human_approved: bool
    human_feedback: str | None
    revision_count: int
    max_revisions: int
    
    # Error state
    errors: list[WorkflowError]
    error_count: int
    recovery_strategy: str | None
```

### `src.state.enums`

```python
class ResearchStatus(str, Enum):
    """Workflow status enumeration."""
    PENDING = "pending"
    INTAKE_COMPLETE = "intake_complete"
    EXPLORING_DATA = "exploring_data"
    LITERATURE_REVIEW = "literature_review"
    SYNTHESIZING = "synthesizing"
    GAP_IDENTIFIED = "gap_identified"
    PLANNING = "planning"
    ANALYZING = "analyzing"
    WRITING = "writing"
    REVIEWING = "reviewing"
    REVISING = "revising"
    COMPLETED = "completed"
    FAILED = "failed"

class ReviewDecision(str, Enum):
    """Review decision options."""
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"

class PaperType(str, Enum):
    """Types of academic papers."""
    FULL_PAPER = "full_paper"
    SHORT_PAPER = "short_paper"
    REVIEW = "review"
    PERSPECTIVE = "perspective"
    COMMENTARY = "commentary"

class ResearchType(str, Enum):
    """Types of research methodology."""
    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"
    MIXED_METHODS = "mixed_methods"
    META_ANALYSIS = "meta_analysis"
    CASE_STUDY = "case_study"
```

---

## Tools

### Academic Search Tools (`src.tools.academic_search`)

```python
@tool
def search_semantic_scholar(query: str, limit: int = 10) -> list[dict]:
    """Search Semantic Scholar for academic papers.
    
    Args:
        query: Search query
        limit: Maximum results to return
        
    Returns:
        List of paper metadata with citations
    """

@tool
def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    """Search arXiv for preprints.
    
    Args:
        query: Search query
        max_results: Maximum results to return
        
    Returns:
        List of preprint metadata
    """

@tool
def search_tavily(query: str) -> dict:
    """Search web using Tavily API.
    
    Args:
        query: Search query
        
    Returns:
        Search results with snippets
    """
```

### Data Loading Tools (`src.tools.data_loading`)

```python
@tool
def load_data(
    file_path: str,
    format: str | None = None,
    sheet_name: str | None = None,
) -> str:
    """Load data file into DuckDB registry.
    
    Supports: CSV, Excel, Parquet, Stata, SPSS, JSON, ZIP
    
    Args:
        file_path: Path to data file
        format: Optional format override
        sheet_name: For Excel files, which sheet to load
        
    Returns:
        Dataset ID for subsequent operations
    """

@tool
def list_datasets() -> list[dict]:
    """List all registered datasets.
    
    Returns:
        List of dataset metadata
    """

@tool
def query_data(dataset_id: str, sql: str) -> dict:
    """Execute SQL query on dataset.
    
    Args:
        dataset_id: ID of registered dataset
        sql: SQL query to execute
        
    Returns:
        Query results as dict
    """
```

### Analysis Tools (`src.tools.analysis`)

```python
@tool
def run_regression(
    dataset_id: str,
    dependent_var: str,
    independent_vars: list[str],
    robust_se: bool = True,
) -> dict:
    """Run OLS regression with diagnostics.
    
    Args:
        dataset_id: ID of dataset
        dependent_var: Name of dependent variable
        independent_vars: List of independent variable names
        robust_se: Whether to use robust standard errors
        
    Returns:
        Regression results with coefficients, p-values, diagnostics
    """

@tool
def compute_correlation(
    dataset_id: str,
    variables: list[str],
    method: str = "pearson",
) -> dict:
    """Compute correlation matrix.
    
    Args:
        dataset_id: ID of dataset
        variables: Variables to correlate
        method: pearson, spearman, or kendall
        
    Returns:
        Correlation matrix
    """

@tool
def run_hypothesis_test(
    dataset_id: str,
    test_type: str,
    **kwargs,
) -> dict:
    """Run statistical hypothesis test.
    
    Supported tests: t-test, chi-squared, ANOVA, Mann-Whitney
    
    Args:
        dataset_id: ID of dataset
        test_type: Type of test
        **kwargs: Test-specific parameters
        
    Returns:
        Test statistic, p-value, interpretation
    """
```

---

## Error Handling

### Exception Classes (`src.errors.exceptions`)

```python
class GIAError(Exception):
    """Base exception for GIA system.
    
    Attributes:
        message: Human-readable error message
        details: Additional error context
        recoverable: Whether error can be recovered from
    """

class WorkflowError(GIAError):
    """Workflow orchestration error."""

class NodeExecutionError(GIAError):
    """Node execution failure."""

class APIError(GIAError):
    """External API error."""

class RateLimitError(APIError):
    """Rate limit exceeded.
    
    Attributes:
        retry_after: Seconds to wait before retry
    """

class ContextOverflowError(GIAError):
    """Context window exceeded."""

class DataValidationError(GIAError):
    """Input validation error."""
```

### Recovery Strategies (`src.errors.recovery`)

```python
class RecoveryAction(str, Enum):
    """Available recovery actions."""
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    ABORT = "abort"
    REDUCE_CONTEXT = "reduce_context"

def determine_recovery_strategy(
    error: Exception,
    retry_count: int = 0,
    max_retries: int = 3,
) -> RecoveryStrategy:
    """Determine appropriate recovery action for error.
    
    Args:
        error: The exception that occurred
        retry_count: Current retry attempt
        max_retries: Maximum retry attempts
        
    Returns:
        RecoveryStrategy with action and parameters
    """

def execute_recovery(
    strategy: RecoveryStrategy,
    state: WorkflowState,
) -> dict:
    """Execute the recovery strategy.
    
    Args:
        strategy: Recovery strategy to execute
        state: Current workflow state
        
    Returns:
        State updates from recovery
    """
```

### Retry Policies (`src.errors.policies`)

```python
@dataclass
class RetryPolicy:
    """Configuration for retry behavior.
    
    Attributes:
        max_attempts: Maximum retry attempts
        initial_interval: Initial wait time (seconds)
        backoff_factor: Multiplier for each retry
        max_interval: Maximum wait time
        jitter: Whether to add random jitter
        retry_on: Exception types to retry on
    """
    max_attempts: int = 3
    initial_interval: float = 1.0
    backoff_factor: float = 2.0
    max_interval: float = 60.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] = (APIError, TimeoutError)

# Predefined policies
DEFAULT_RETRY_POLICY = RetryPolicy()
AGGRESSIVE_RETRY_POLICY = RetryPolicy(max_attempts=5, max_interval=120.0)
CONSERVATIVE_RETRY_POLICY = RetryPolicy(max_attempts=2, initial_interval=0.5)
```

---

## Agents

### `src.agents.base`

#### `create_react_agent`

```python
def create_react_agent(
    tools: list | None = None,
    model_name: str | None = None,
    system_prompt: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> StateGraph:
    """Create a ReAct agent with tool use.
    
    Args:
        tools: List of tools to bind
        model_name: Claude model to use (default: claude-sonnet-4-5-20250929)
        system_prompt: Custom system prompt
        checkpointer: For conversation persistence
        store: For long-term memory
        
    Returns:
        Compiled agent graph
    """
```

### `src.agents.research`

#### `create_research_agent`

```python
def create_research_agent(
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> StateGraph:
    """Create research-focused agent with academic tools.
    
    Includes: academic search, citation analysis, gap identification
    
    Args:
        checkpointer: For conversation persistence
        store: For long-term memory
        
    Returns:
        Compiled research agent graph
    """
```

### `src.agents.data_analyst`

#### `create_data_analyst_agent`

```python
def create_data_analyst_agent(
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> StateGraph:
    """Create data analysis agent with statistical tools.
    
    Includes: data loading, profiling, transformation, analysis
    
    Args:
        checkpointer: For conversation persistence
        store: For long-term memory
        
    Returns:
        Compiled data analyst agent graph
    """
```

---

## Usage Examples

### Basic Workflow Execution

```python
from src.graphs import create_research_workflow, WorkflowConfig
from langgraph.checkpoint.memory import MemorySaver

# Create workflow with memory
config = WorkflowConfig(checkpointer=MemorySaver())
workflow = create_research_workflow(config)

# Execute with initial state
initial_state = {
    "form_data": {
        "research_question": "What factors drive cryptocurrency adoption?",
        "paper_type": "full_paper",
        "research_type": "empirical",
    }
}

# Run workflow
config = {"configurable": {"thread_id": "research-001"}}
result = workflow.invoke(initial_state, config)
```

### Streaming Progress

```python
from src.graphs import create_research_workflow
from src.graphs.streaming import stream_workflow

workflow = create_research_workflow()

async for event in stream_workflow(workflow, initial_state, config):
    if event.type == "status":
        print(f"Status: {event.data['status']}")
    elif event.type == "progress":
        print(f"Progress: {event.data['message']}")
```

### Handling HITL Interrupts

```python
from langgraph.types import Command

# Workflow pauses at gap_identifier
result = workflow.invoke(initial_state, config)

# User reviews and approves
workflow.invoke(
    Command(resume={"approved": True, "feedback": "Looks good"}),
    config
)
```

---

*API Reference maintained by Gia Tenica. Last updated: January 2026.*
