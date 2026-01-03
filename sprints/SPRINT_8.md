# Sprint 8: Graph Assembly and Full Workflow Integration

**Status**: ✅ Completed
**Duration**: 1 day
**Issue**: #18
**PR**: TBD

## Overview

Sprint 8 refactors the research workflow graph into a modular, maintainable architecture with proper separation of concerns. This sprint introduces a dedicated `src/graphs/` module containing workflow factories, routing logic, streaming utilities, and debugging capabilities.

## Objectives

1. ✅ Extract routing functions from studio/graphs.py to dedicated module
2. ✅ Create workflow factory with configurable options
3. ✅ Implement streaming utilities for real-time UI updates
4. ✅ Add time travel and debugging capabilities
5. ✅ Support modular subgraph compositions
6. ✅ Maintain backward compatibility with existing code

## Implementation

### New Module Structure

```
src/graphs/
├── __init__.py              # Module exports (20+ functions/classes)
├── routers.py               # All routing functions (~310 lines)
├── research_workflow.py     # Main workflow factory (~370 lines)
├── streaming.py             # Streaming utilities (~230 lines)
├── debug.py                 # Time travel/debugging (~340 lines)
└── subgraphs.py             # Modular subgraphs (~300 lines)
```

### Key Components

#### 1. Routing Functions (`routers.py`)

All conditional edge routing logic consolidated in one place:

- `route_after_intake()` - Routes to data_explorer or literature_reviewer
- `route_after_data_explorer()` - Routes to literature_reviewer
- `route_after_literature_reviewer()` - Routes to literature_synthesizer
- `route_after_synthesizer()` - Routes to gap_identifier
- `route_after_gap_identifier()` - Routes to planner or end
- `route_after_planner()` - Routes by research type
- `route_by_research_type()` - Routes to data_analyst or conceptual_synthesizer
- `route_after_analysis()` - Routes to writer
- `route_after_writer()` - Routes to reviewer
- `route_after_reviewer()` - Routes to output, writer (revise), or end

#### 2. Workflow Factory (`research_workflow.py`)

```python
@dataclass
class WorkflowConfig:
    """Configuration for research workflow compilation."""
    checkpointer: BaseCheckpointSaver | None = None
    store: BaseStore | None = None
    cache: Any | None = None
    interrupt_before: list[str] = field(default_factory=lambda: INTERRUPT_BEFORE_NODES)
    interrupt_after: list[str] = field(default_factory=lambda: INTERRUPT_AFTER_NODES)
    enable_caching: bool = True
    debug: bool = False
```

Factory functions:
- `create_research_workflow(config)` - Main factory with full configuration
- `create_studio_workflow()` - LangGraph Studio configuration
- `create_production_workflow(db_path)` - Production with SQLite persistence

#### 3. Streaming Utilities (`streaming.py`)

```python
class StreamMode(Enum):
    """Streaming mode options."""
    VALUES = "values"
    UPDATES = "updates"
    MESSAGES = "messages"
    DEBUG = "debug"

@dataclass
class StreamEvent:
    """Structured streaming event."""
    mode: StreamMode
    node: str
    data: Any
    metadata: dict | None = None
```

Functions:
- `stream_research_workflow()` - Async generator for streaming updates
- `stream_with_progress()` - Streaming with progress callbacks
- `format_for_sse()` - Format events for Server-Sent Events
- `format_for_websocket()` - Format events for WebSocket

#### 4. Debug Utilities (`debug.py`)

```python
@dataclass
class StateSnapshot:
    """Snapshot of workflow state at a point in time."""
    checkpoint_id: str
    thread_id: str
    created_at: datetime
    node: str
    next_nodes: list[str]
    values: dict
    metadata: dict

class WorkflowInspector:
    """High-level interface for workflow inspection and debugging."""
    
    @property
    def status(self) -> WorkflowStatus: ...
    
    @property
    def history(self) -> list[StateSnapshot]: ...
    
    def get_state_at(self, node: str) -> StateSnapshot | None: ...
    
    def replay_from(self, checkpoint_id: str, new_input: dict | None = None): ...
    
    def fork(self, checkpoint_id: str, new_values: dict) -> str: ...
```

Functions:
- `inspect_workflow_state()` - Get current workflow status
- `get_state_history()` - Get full checkpoint history
- `get_state_at_node()` - Get state at specific node
- `replay_from_checkpoint()` - Time travel to checkpoint
- `fork_from_state()` - Fork workflow with modifications

#### 5. Subgraphs (`subgraphs.py`)

Modular subgraph factories for pipeline composition:

- `create_literature_review_subgraph()` - Literature review pipeline
- `create_analysis_subgraph()` - Data analysis pipeline
- `create_writing_subgraph()` - Writing and review pipeline
- `create_research_pipeline_subgraph()` - Full research pipeline

### Updated Files

- **studio/graphs.py** - Simplified to import from src/graphs
  - Maintains backward compatibility through re-exports
  - Uses `create_studio_workflow()` factory

## Testing

71 new tests added in `tests/unit/test_graphs.py`:

- `TestWorkflowConfig` - Configuration dataclass tests
- `TestWorkflowFactory` - Factory function tests
- `TestOutputNode` - Output node tests
- `TestRouteAfter*` - All routing function tests
- `TestStreamEvent` - Streaming event tests
- `TestProgressMessages` - Progress message tests
- `TestStreamFormatters` - SSE/WebSocket formatters
- `TestStateSnapshot` - State snapshot tests
- `TestWorkflowStatus` - Workflow status tests
- `TestInspectWorkflowState` - Inspection function tests
- `TestWorkflowInspector` - Inspector class tests
- `TestSubgraphs` - Subgraph factory tests
- `TestConstants` - Module constants tests
- `TestWorkflowIntegration` - Integration tests

**Test Results**: 400 tests passing (329 original + 71 new)

## Constants

### Workflow Nodes (in order)
```python
WORKFLOW_NODES = [
    "intake", "data_explorer", "literature_reviewer",
    "literature_synthesizer", "gap_identifier", "planner",
    "data_analyst", "conceptual_synthesizer", "writer",
    "reviewer", "output"
]
```

### Interrupt Points
```python
INTERRUPT_BEFORE_NODES = ["gap_identifier", "planner", "reviewer"]
INTERRUPT_AFTER_NODES = ["reviewer"]
```

### Theoretical Methodologies
```python
THEORETICAL_METHODOLOGIES = {
    "systematic_review", "meta_analysis", "conceptual_framework",
    "theoretical_synthesis", "critical_analysis", "literature_review"
}
```

## Usage Examples

### Basic Usage
```python
from src.graphs import create_research_workflow

# Create default workflow
workflow = create_research_workflow()

# Run workflow
result = workflow.invoke(initial_state)
```

### With Configuration
```python
from src.graphs import create_research_workflow, WorkflowConfig
from langgraph.checkpoint.sqlite import SqliteSaver

config = WorkflowConfig(
    checkpointer=SqliteSaver.from_conn_string("sqlite:///workflow.db"),
    interrupt_before=["planner", "reviewer"],
    enable_caching=True,
    debug=True,
)
workflow = create_research_workflow(config)
```

### Streaming
```python
from src.graphs import stream_research_workflow, StreamMode

async for event in stream_research_workflow(
    workflow, 
    initial_state, 
    thread_id="session-123",
    mode=StreamMode.UPDATES
):
    print(f"Node: {event.node}, Data: {event.data}")
```

### Debugging
```python
from src.graphs import WorkflowInspector

inspector = WorkflowInspector(workflow, "session-123")

# Check status
print(inspector.status)

# View history
for snapshot in inspector.history:
    print(f"{snapshot.node}: {snapshot.checkpoint_id}")

# Time travel
result = inspector.replay_from(checkpoint_id)

# Fork with modifications
new_thread = inspector.fork(checkpoint_id, {"custom_field": "value"})
```

## Dependencies

No new dependencies added. Uses existing:
- `langgraph` - StateGraph, checkpointers
- `langchain-core` - Base types
- Standard library only for new utilities

## Backward Compatibility

Full backward compatibility maintained:
- `studio/graphs.py` re-exports all routing functions
- Existing imports continue to work
- All 329 original tests pass unchanged

## Next Steps (Sprint 9+)

1. Add WebSocket server for real-time streaming
2. Build visualization dashboard for workflow state
3. Implement replay UI in LangGraph Studio
4. Add metrics collection for performance monitoring
5. Create CI/CD pipeline with graph validation

## Files Changed

### New Files
- `src/graphs/__init__.py`
- `src/graphs/routers.py`
- `src/graphs/research_workflow.py`
- `src/graphs/streaming.py`
- `src/graphs/debug.py`
- `src/graphs/subgraphs.py`
- `tests/unit/test_graphs.py`
- `sprints/SPRINT_8.md`

### Modified Files
- `studio/graphs.py`

## Checklist

- [x] Create src/graphs/ module structure
- [x] Extract routing functions to routers.py
- [x] Implement WorkflowConfig dataclass
- [x] Create workflow factory functions
- [x] Implement streaming utilities
- [x] Add time travel/debugging capabilities
- [x] Create subgraph factories
- [x] Update studio/graphs.py for compatibility
- [x] Write comprehensive tests (71 tests)
- [x] All tests pass (400 total)
- [x] Document Sprint 8 in SPRINT_8.md
- [ ] Create and merge PR
