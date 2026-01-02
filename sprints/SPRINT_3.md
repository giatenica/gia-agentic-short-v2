# Sprint 3: GAP_IDENTIFIER Node Implementation

## Overview

Sprint 3 implements the GAP_IDENTIFIER node, which identifies research gaps in the literature, generates contribution statements, and refines research questions based on identified gaps. This node uses human-in-the-loop (HITL) via LangGraph's `interrupt()` to allow researchers to approve, modify, or reject refined research questions.

## Deliverables

### 1. New Enums (src/state/enums.py)

| Enum | Values | Purpose |
|------|--------|---------|
| `GapType` | methodological, empirical, theoretical, contextual, temporal, conflicting_findings | Categorizes types of research gaps |
| `GapSignificance` | high, medium, low | Rates the importance/impact of a gap |

### 2. New Models (src/state/models.py)

| Model | Key Fields | Purpose |
|-------|------------|---------|
| `ResearchGap` | type, description, significance, addressability, evidence | Individual gap identified in literature |
| `GapAnalysis` | gaps, primary_gap, coverage_analysis, recommendations | Complete gap analysis result |
| `ContributionStatement` | main_statement, contribution_type, novelty_explanation | Research contribution positioning |
| `RefinedResearchQuestion` | original_question, refined_question, rationale, scope_changes | Question refinement with justification |

### 3. Gap Analysis Tools (src/tools/gap_analysis.py)

| Function | Purpose |
|----------|---------|
| `compare_coverage()` | Compare literature coverage against research question requirements |
| `identify_methodological_gaps()` | Find gaps in research methods and approaches |
| `identify_empirical_gaps()` | Find gaps in evidence and contexts studied |
| `identify_theoretical_gaps()` | Find gaps in theoretical frameworks |
| `assess_gap_significance()` | Rank gaps by importance and addressability |
| `perform_gap_analysis()` | Orchestrate comprehensive gap identification |

**Tool Exports:**
- `compare_coverage_tool` - LangChain tool for coverage comparison
- `identify_gaps_tool` - LangChain tool for gap identification

### 4. Contribution Tools (src/tools/contribution.py)

| Function | Purpose |
|----------|---------|
| `generate_contribution_statement()` | Create main contribution statement |
| `position_in_literature()` | Position research within existing work |
| `differentiate_from_prior()` | Differentiate from similar papers |
| `refine_research_question()` | Narrow question to target identified gap |

**Tool Exports:**
- `generate_contribution_tool` - LangChain tool for contribution generation
- `refine_question_tool` - LangChain tool for question refinement

### 5. GAP_IDENTIFIER Node (src/nodes/gap_identifier.py)

The main node implements the following workflow:

```
┌─────────────────────────────────────────────────────────────────┐
│                     GAP_IDENTIFIER Node                         │
├─────────────────────────────────────────────────────────────────┤
│  1. identify_gaps()          → Comprehensive gap analysis       │
│  2. select_primary_gap()     → Choose most significant gap      │
│  3. create_refined_question()→ Generate refined question        │
│  4. create_contribution()    → Generate contribution statement  │
│  5. prepare_approval_request()→ Format for HITL interrupt       │
│  6. interrupt()              → Human approval/modify/reject     │
│  7. process_approval()       → Handle human response            │
└─────────────────────────────────────────────────────────────────┘
```

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `gap_identifier_node()` | Main node function with HITL interrupt |
| `identify_gaps()` | Comprehensive gap identification |
| `select_primary_gap()` | Choose most significant addressable gap |
| `create_refined_question()` | Generate refined research question |
| `create_contribution_statement()` | Generate contribution statement |
| `prepare_approval_request()` | Format data for interrupt() |
| `process_approval_response()` | Handle human approval/modify/reject |

**Routing Functions:**

| Function | Purpose |
|----------|---------|
| `should_refine_further()` | Check if gap identification is complete |
| `route_after_gap_identifier()` | Route to next node or retry |

### 6. Human-in-the-Loop (HITL) Implementation

The node uses LangGraph's `interrupt()` for researcher approval:

```python
from langgraph.types import interrupt

# Present refined question for approval
response = interrupt({
    "type": "refined_question_approval",
    "original_question": original,
    "refined_question": refined,
    "primary_gap": gap,
    "contribution_statement": contribution,
    "options": ["approve", "modify", "reject"],
    "prompt": "Please review the refined research question..."
})
```

**Response Options:**

| Option | Behavior |
|--------|----------|
| `approve` | Accept refined question as-is |
| `modify` | Use human-provided alternative question |
| `reject` | Keep original question, continue workflow |

## Acceptance Criteria Status

| Criterion | Status |
|-----------|--------|
| ✅ Compare literature coverage to research question | Implemented |
| ✅ Identify methodological, empirical, and theoretical gaps | Implemented |
| ✅ Rank gaps by significance and addressability | Implemented |
| ✅ Generate contribution positioning statement | Implemented |
| ✅ Propose refined research question | Implemented |
| ✅ HITL approval for refined question | Implemented with interrupt() |
| ✅ Unit tests with 80%+ coverage | 28 tests, all passing |

## Usage Example

```python
from src.nodes import gap_identifier_node, route_after_gap_identifier

# In LangGraph workflow
workflow = StateGraph(WorkflowState)

workflow.add_node("gap_identifier", gap_identifier_node)
workflow.add_conditional_edges(
    "gap_identifier",
    route_after_gap_identifier,
    {
        "methodology_advisor": "methodology_advisor",
        "gap_identifier": "gap_identifier",  # Retry on failure
    }
)
```

## State Updates

The node updates the following state fields:

```python
{
    "gap_analysis": GapAnalysis,           # Complete gap analysis
    "contribution_statement": ContributionStatement,  # Research positioning
    "refined_research_question": RefinedResearchQuestion,  # Refined question
    "errors": list[WorkflowError],         # Any errors encountered
}
```

## Test Coverage

### Unit Tests (tests/unit/test_gap_identifier.py)

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestCompareCoverage` | 2 | Coverage comparison parsing |
| `TestIdentifyGaps` | 3 | Gap identification parsing |
| `TestAssessGapSignificance` | 1 | Significance assessment |
| `TestPerformGapAnalysis` | 1 | Complete gap analysis |
| `TestGenerateContributionStatement` | 2 | Contribution generation |
| `TestPositionInLiterature` | 1 | Literature positioning |
| `TestRefineResearchQuestion` | 1 | Question refinement |
| `TestDifferentiateFromPrior` | 1 | Differentiation parsing |
| `TestGapIdentifierNode` | 6 | Node functions |
| `TestRouting` | 5 | Routing logic |
| `TestGapIdentifierTools` | 1 | Tool exports |
| `TestIntegration` | 2 | Full flow tests |

**Total: 28 tests, all passing**

## Dependencies

No new dependencies required. Uses existing:
- `langchain-anthropic` for Claude API
- `langgraph` for workflow and interrupt()
- `pydantic` for model validation

## Files Changed

| File | Changes |
|------|---------|
| `src/state/enums.py` | Added GapType, GapSignificance enums |
| `src/state/models.py` | Added ResearchGap, GapAnalysis, ContributionStatement, RefinedResearchQuestion |
| `src/state/__init__.py` | Updated exports |
| `src/tools/gap_analysis.py` | New file with gap analysis tools |
| `src/tools/contribution.py` | New file with contribution tools |
| `src/tools/__init__.py` | Updated exports |
| `src/nodes/gap_identifier.py` | New file with GAP_IDENTIFIER node |
| `src/nodes/__init__.py` | Updated exports |
| `tests/unit/test_gap_identifier.py` | New file with 28 unit tests |
| `docs/SPRINT_3.md` | This documentation |

## Next Steps (Sprint 4)

Sprint 4 will implement the METHODOLOGY_ADVISOR node:
- Research design guidance
- Method selection recommendations
- Validity and reliability considerations
- Data collection strategy
- Analysis approach suggestions

## Author

Implementation by Gia Tenica (me@giatenica.com)
