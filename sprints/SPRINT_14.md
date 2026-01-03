# Sprint 14: Data Acquisition Node & Code Execution

**Status:** ✅ Complete  
**Branch:** `feature/sprint-14-data-acquisition-node`  
**Issue:** #30  
**Duration:** 1 day  
**PR:** #31  

## Overview

Sprint 14 introduces the intelligent data acquisition node, which autonomously fetches external data from sources like yfinance, FRED, and CoinGecko. It also includes a secure Python code execution sandbox for generating custom data acquisition scripts when built-in tools are insufficient.

## Goals

1. ✅ Create intelligent data acquisition node that:
   - Parses data requirements from research plans
   - Checks uploaded datasets against requirements
   - Automatically fetches missing data from external APIs
   - Falls back to custom code generation when needed

2. ✅ Implement secure code execution sandbox:
   - Pattern-based and AST validation
   - Restricted globals/builtins
   - Timeout enforcement
   - Safe import system

3. ✅ Integrate into workflow between planner and data_analyst nodes

## Implementation Details

### New Files

| File | Purpose | Lines |
|------|---------|-------|
| `src/nodes/data_acquisition.py` | Intelligent data acquisition node | ~400 |
| `src/tools/code_execution.py` | Safe Python code sandbox | ~540 |
| `tests/unit/test_code_execution.py` | Code execution tests | ~290 |
| `tests/unit/test_data_acquisition.py` | Data acquisition tests | ~280 |

### Modified Files

| File | Changes |
|------|---------|
| `src/state/enums.py` | Added `DataRequirementPriority`, `AcquisitionStatus`, `CodeExecutionStatus` |
| `src/state/models.py` | Added 7 new models for data acquisition |
| `src/state/schema.py` | Added 4 new state fields |
| `src/nodes/__init__.py` | Export data_acquisition_node |
| `src/graphs/research_workflow.py` | Added data_acquisition to workflow |
| `src/graphs/routers.py` | Added route_after_data_acquisition |
| `studio/graphs.py` | Updated router imports |

### State Models Added

```python
# src/state/enums.py
class DataRequirementPriority(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    OPTIONAL = "optional"

class AcquisitionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"

class CodeExecutionStatus(str, Enum):
    NOT_EXECUTED = "not_executed"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    VALIDATION_FAILED = "validation_failed"

# src/state/models.py
TimeRange          # start_date, end_date, relative
DataRequirement    # variable_name, data_type, required_fields, entities, priority
DataAcquisitionTask # requirement, source, tool_to_use, params, status
DataAcquisitionPlan # requirements, available_in_upload, to_acquire
AcquisitionFailure  # requirement, attempted_sources, error_messages
CodeSnippet        # code, description, status, execution_result
AcquiredDataset    # dataset_name, source, requirement_id, row_count

# src/state/schema.py additions
data_acquisition_plan: Optional[Dict[str, Any]]
acquired_datasets: Optional[List[Dict[str, Any]]]
acquisition_failures: Optional[List[Dict[str, Any]]]
generated_code_snippets: Optional[List[Dict[str, Any]]]
```

### Code Execution Safety Features

1. **Forbidden Patterns (30+)**
   - System access: `os`, `sys`, `subprocess`, `shutil`
   - Code execution: `eval`, `exec`, `compile`, `__import__`
   - File operations: `open`, `file`, `input`
   - Dangerous modules: `socket`, `ctypes`, `pickle`

2. **Safe Builtins Whitelist**
   - Types: `bool`, `int`, `float`, `str`, `list`, `dict`, `set`, `tuple`
   - Functions: `len`, `range`, `enumerate`, `zip`, `map`, `filter`, `sorted`, `max`, `min`
   - Exceptions: `Exception`, `ValueError`, `TypeError`, etc.

3. **Restricted Import System**
   - Allowed: `pandas`, `numpy`, `json`, `math`, `statistics`, `datetime`, `requests`
   - All other imports blocked

4. **Execution Environment**
   - Timeout enforcement (Unix via signal, graceful on Windows)
   - Stdout/stderr capture
   - Local variable extraction

### Workflow Integration

```
PLANNER → DATA_ACQUISITION → DATA_ANALYST
             ↓
    (if theoretical)
             ↓
    CONCEPTUAL_SYNTHESIZER
```

The data acquisition node:
1. Parses `research_plan.data_requirements` 
2. Checks `loaded_datasets` for matches
3. For unmet requirements, finds appropriate external source
4. Executes acquisition tools (acquire_stock_data, etc.)
5. On failure, generates custom code as fallback
6. Routes to data_analyst or human_interrupt based on results

### Routing Logic

```python
def route_after_planner(state):
    # Theoretical research skips data acquisition
    if methodology in THEORETICAL_METHODOLOGIES:
        return "conceptual_synthesizer"
    return "data_acquisition"

def route_after_data_acquisition(state):
    failures = state.get("acquisition_failures", [])
    if any required data missing:
        return "human_interrupt"
    return route_by_research_type(state)  # data_analyst or conceptual_synthesizer
```

## Test Results

```
tests/unit/test_code_execution.py    - 28 passed ✅
tests/unit/test_data_acquisition.py  - 22 passed ✅
Total: 50 tests                      - All passed ✅
```

### Test Coverage

**Code Execution Tests:**
- Code validation (safe code, forbidden patterns, syntax errors)
- Execution safety (blocked imports, timeout handling)
- Module support (pandas, numpy, statistics, json)
- Output capture (stdout, variables, errors)

**Data Acquisition Tests:**
- Requirement parsing (explicit, inferred, empty)
- Requirement matching against uploads
- Source finding (stock, economic, crypto)
- Node execution flow
- Routing logic

## Dependencies

- Sprint 13 (Data Source Registry) - Complete ✅
- Sprint 12 (Data Explorer) - Complete ✅

## Usage Example

```python
from src.nodes.data_acquisition import data_acquisition_node

state = {
    "research_plan": {
        "methodology": "regression analysis",
        "data_requirements": [
            {
                "variable_name": "stock_returns",
                "data_type": "stock_prices",
                "entities": ["AAPL", "MSFT"],
                "priority": "required",
            }
        ]
    },
    "loaded_datasets": [],
}

result = data_acquisition_node(state)
# Result includes:
# - data_acquisition_plan: plan with tasks
# - acquired_datasets: successfully fetched data
# - acquisition_failures: any failed requirements
# - loaded_datasets: updated with new data
```

## Lessons Learned

1. **Safe Import System**: Python's `import` statement requires `__import__` in builtins. Created a restricted `_safe_import` function that only allows whitelisted modules.

2. **Signal-based Timeout**: Unix-only via `signal.alarm()`. Windows requires alternative approaches (threading/multiprocessing).

3. **Pattern + AST Validation**: Dual validation catches both string patterns and semantic AST issues.

## Next Steps

Sprint 15: Gap Resolution Agent
- Human-assisted data acquisition
- Manual upload integration
- Data quality validation
- Requirement relaxation logic
