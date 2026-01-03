# Sprint 9: Error Handling and Fallbacks

## Overview

Sprint 9 implements comprehensive error handling and graceful degradation for the GIA research workflow. This includes a custom exception hierarchy, retry policies, error handlers, recovery strategies, and a fallback node for partial output generation.

## Objectives

- ✅ Create custom exception hierarchy for categorized error handling
- ✅ Implement RetryPolicy configurations for different scenarios
- ✅ Build error handlers for tools, nodes, and API calls
- ✅ Develop recovery strategies for workflow failures
- ✅ Implement fallback node for graceful degradation
- ✅ Integrate error handling into workflow routing
- ✅ Write comprehensive tests

## Implementation

### 1. Exception Hierarchy (`src/errors/exceptions.py`)

Custom exception classes for categorized error handling:

```
GIAError (base)
├── WorkflowError - Workflow orchestration issues
├── NodeExecutionError - Node execution failures
├── ToolExecutionError - Tool execution failures
├── APIError - External API errors
│   ├── RateLimitError - Rate limit exceeded
│   └── ContextOverflowError - Context window exceeded
├── DataValidationError - Input validation errors
├── SearchError - Search operation failures
│   └── LiteratureSearchError - Academic search failures
├── AnalysisError - Data analysis failures
├── WritingError - Content writing failures
└── ReviewError - Review process failures
```

### 2. Retry Policies (`src/errors/policies.py`)

Configurable retry behavior with exponential backoff:

```python
@dataclass
class RetryPolicy:
    max_attempts: int = 3
    initial_interval: float = 1.0
    backoff_factor: float = 2.0
    max_interval: float = 60.0
    jitter: bool = True
    retry_on: tuple[Type[Exception], ...] = (Exception,)
```

Pre-configured policies:
- `DEFAULT_RETRY_POLICY` - Standard 3 retries
- `AGGRESSIVE_RETRY_POLICY` - 5 retries with longer delays
- `CONSERVATIVE_RETRY_POLICY` - 2 retries, quick fail
- `create_api_retry_policy()` - API-specific (rate limits, timeouts)
- `create_search_retry_policy()` - Search-specific (connection errors)
- `create_analysis_retry_policy()` - Analysis-specific

### 3. Error Handlers (`src/errors/handlers.py`)

Functions for graceful error management:

- `create_error_response()` - Standardized error response dictionary
- `create_workflow_error_model()` - Create WorkflowError for state tracking
- `log_error_with_context()` - Log errors with full context
- `handle_tool_error()` - Handle tool errors (returns user-friendly string)
- `handle_node_error()` - Handle node errors (returns state updates)
- `handle_api_error()` - Handle API errors (returns message)

```python
class ErrorHandler:
    """Centralized error handler for workflow operations."""
    
    def __init__(self, node: str, max_errors: int = 3):
        ...
    
    def handle(self, error: Exception, state: dict) -> dict:
        """Handle error and return state updates."""
        ...
    
    def should_fail(self, error: Exception, state: dict) -> bool:
        """Determine if workflow should fail/fallback."""
        ...
```

### 4. Recovery Strategies (`src/errors/recovery.py`)

Recovery actions and strategies:

```python
class RecoveryAction(str, Enum):
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    REDUCE_CONTENT = "reduce_content"
    WAIT_AND_RETRY = "wait_and_retry"
    ABORT = "abort"
    PARTIAL_OUTPUT = "partial_output"

@dataclass
class RecoveryStrategy:
    action: RecoveryAction
    reason: str
    params: dict[str, Any] = field(default_factory=dict)
    fallback_actions: list[RecoveryAction] = field(default_factory=list)
```

Key functions:
- `determine_recovery_strategy()` - Select appropriate recovery action
- `execute_recovery()` - Execute the recovery strategy
- `can_continue_workflow()` - Check if workflow can proceed
- `get_partial_output()` - Collect available output from state
- `create_fallback_content()` - Generate fallback section content

### 5. Fallback Node (`src/nodes/fallback.py`)

Graceful degradation when workflow cannot complete:

```python
def fallback_node(state: WorkflowState) -> dict[str, Any]:
    """Generate partial output when workflow encounters too many errors."""
    ...

def should_fallback(state: WorkflowState) -> bool:
    """Determine if workflow should route to fallback."""
    # Triggers on:
    # - Explicit _should_fallback flag
    # - 3+ errors accumulated
    # - Any unrecoverable error
    # - FAILED status
    ...
```

Fallback output includes:
- Error summary with categories and affected nodes
- Incomplete stage list
- Recovery suggestions
- Available partial output
- Fallback paper sections
- Final assembled paper with completion notes

### 6. Workflow Integration

All routers updated with fallback routing:

```python
def route_after_intake(state) -> Literal["data_explorer", "literature_reviewer", "fallback", "__end__"]:
    if _should_fallback(state):
        return "fallback"
    ...
```

Workflow edges updated to include fallback:
```python
workflow.add_conditional_edges(
    "intake",
    route_after_intake,
    ["data_explorer", "literature_reviewer", "fallback", END]
)
```

## File Changes

### New Files
- `src/errors/__init__.py` - Module exports
- `src/errors/exceptions.py` - Exception hierarchy
- `src/errors/policies.py` - RetryPolicy configurations
- `src/errors/handlers.py` - Error handlers
- `src/errors/recovery.py` - Recovery strategies
- `src/nodes/fallback.py` - Fallback node
- `tests/unit/test_errors.py` - 82 tests for error handling

### Modified Files
- `src/graphs/research_workflow.py` - Added fallback node, updated edges
- `src/graphs/routers.py` - Added fallback routing to all routers
- `src/graphs/__init__.py` - Added WORKFLOW_NODES export
- `src/graphs/streaming.py` - Added fallback progress message
- `src/nodes/__init__.py` - Added fallback node exports
- `tests/unit/test_graphs.py` - Updated tests for fallback routing

## Error Handling Flow

```
Error occurs in node
        ↓
log_error_with_context() - Log with full details
        ↓
handle_node_error() - Create state updates
        ↓
_should_fallback() check in router
        ↓
    ┌───┴───┐
    ↓       ↓
continue  fallback
    ↓       ↓
next node  fallback_node()
            ↓
        Generate partial output
            ↓
          END
```

## Fallback Triggers

1. **Explicit Flag**: `state["_should_fallback"] = True`
2. **Error Count**: 3+ errors accumulated
3. **Unrecoverable Error**: Any error with `recoverable=False`
4. **Failed Status**: `state["status"] == ResearchStatus.FAILED`

## Testing

```bash
# Run error handling tests
uv run pytest tests/unit/test_errors.py -v

# Run all tests (482 total)
uv run pytest tests/ -v
```

## Usage Examples

### Creating Custom Errors

```python
from src.errors import (
    GIAError,
    RateLimitError,
    AnalysisError,
)

# Basic error
raise GIAError("Something went wrong", recoverable=True)

# Rate limit with retry info
raise RateLimitError(
    "API rate limited",
    service="semantic_scholar",
    retry_after=60,
)

# Analysis error
raise AnalysisError(
    "Regression failed",
    analysis_type="linear_regression",
    dataset="survey_data.csv",
)
```

### Using ErrorHandler in Nodes

```python
from src.errors.handlers import ErrorHandler

def my_node(state: WorkflowState) -> dict:
    handler = ErrorHandler(node="my_node", max_errors=3)
    
    try:
        # Node logic...
        result = do_something()
        return {"output": result}
    except Exception as e:
        return handler.handle(e, state)
```

### Using @with_error_handling Decorator

```python
from src.errors.handlers import with_error_handling

@with_error_handling("literature_reviewer", max_errors=3)
def literature_reviewer_node(state: WorkflowState) -> dict:
    # Node logic - errors automatically handled
    ...
```

## Future Enhancements

- Implement retry execution with backoff
- Add circuit breaker pattern for external APIs
- Integrate with monitoring/alerting system
- Add error analytics and reporting

## Related Issues

- Issue #20: Sprint 9 - Error Handling and Fallbacks
