# Sprint 7: REVIEWER Node and Revision Loop

## Overview

Sprint 7 implements the critical quality assurance layer for academic paper generation. The REVIEWER node evaluates generated papers across five dimensions, providing structured feedback and routing decisions for iterative improvement.

## Objectives

1. ✅ Create paper quality evaluation criteria module
2. ✅ Implement REVIEWER node with Human-in-the-Loop (HITL) interrupt
3. ✅ Build revision routing system with iteration limits
4. ✅ Integrate revision loop into workflow graph
5. ✅ Comprehensive test coverage

## Implementation Details

### 1. Review State Models (`src/state/models.py`, `src/state/enums.py`)

**New Enums:**
- `ReviewDecision`: APPROVE, REVISE, REJECT
- `ReviewDimension`: CONTRIBUTION, METHODOLOGY, EVIDENCE, COHERENCE, WRITING
- `RevisionPriority`: CRITICAL, HIGH, MEDIUM, LOW

**New Models:**
- `QualityScore`: Individual dimension score (0-10) with justification
- `ReviewCritiqueItem`: Specific issue with location, description, and priority
- `ReviewCritique`: Complete critique with all dimension scores and items
- `RevisionRequest`: Prioritized revision instructions for writer node
- `ReviewerOutput`: Final output including decision and human approval status

**Scoring Functions:**
- `calculate_overall_score()`: Weighted average (contribution 25%, methodology 25%, evidence 20%, coherence 15%, writing 15%)
- `determine_review_decision()`: Approve (≥7.0), Revise (4.0-6.9), Reject (<4.0)

### 2. Review Criteria Module (`src/review/criteria.py`)

Five evaluation functions with detailed criteria:

| Function | Dimension | Weight | Key Criteria |
|----------|-----------|--------|--------------|
| `evaluate_contribution()` | Contribution | 25% | Novelty, significance, gap addressing, advancement |
| `evaluate_methodology()` | Methodology | 25% | Rigor, data quality, validity, reproducibility |
| `evaluate_evidence()` | Evidence | 20% | Robustness, interpretation accuracy, limitations |
| `evaluate_coherence()` | Coherence | 15% | Logic flow, section integration, argument consistency |
| `evaluate_writing()` | Writing | 15% | Academic tone, banned words, citation quality |

**Main Entry Point:**
```python
async def evaluate_paper(state: WorkflowState) -> ReviewCritique
```

### 3. REVIEWER Node (`src/nodes/reviewer.py`)

**Core Functions:**

1. **`reviewer_node(state: WorkflowState) -> dict`**
   - Evaluates paper using all criteria
   - Calculates overall score and decision
   - Creates revision request for "revise" decisions
   - Implements HITL interrupt for human approval
   - Tracks revision iterations

2. **`route_after_reviewer(state: WorkflowState) -> str`**
   - Routes to "output" for approved papers
   - Routes to "writer" for revisions (with max iterations)
   - Handles rejection cases

3. **`_generate_paper_markdown(state: WorkflowState) -> str`**
   - Combines all paper sections into formatted markdown
   - Used for human review presentation

**Human-in-the-Loop Integration:**
```python
from langgraph.types import interrupt

# Interrupt for human decision
human_response = interrupt({
    "paper_markdown": paper_markdown,
    "critique": critique,
    "overall_score": overall_score,
    "ai_decision": decision.value,
    "revision_count": revision_count,
    "message": "Please review the paper and decide..."
})
```

### 4. Workflow Graph Updates (`studio/graphs.py`)

**New Nodes:**
- `reviewer_node`: Paper evaluation and HITL interrupt
- `output_node`: Final output preparation

**New Routing:**
- `route_after_writer()`: Routes WRITER → REVIEWER
- `_route_after_reviewer()`: Routes based on review decision

**Revision Loop:**
```
WRITER → REVIEWER → (decision) → OUTPUT (approve)
                  → (decision) → WRITER (revise, max 3)
                  → (decision) → OUTPUT (reject or max reached)
```

### 5. Test Coverage (`tests/unit/test_reviewer.py`)

**48 tests across 14 test classes:**

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestEvaluateContribution` | 4 | Contribution scoring |
| `TestEvaluateMethodology` | 4 | Methodology scoring |
| `TestEvaluateEvidence` | 4 | Evidence scoring |
| `TestEvaluateCoherence` | 4 | Coherence scoring |
| `TestEvaluateWriting` | 4 | Writing quality scoring |
| `TestCalculateOverallScore` | 3 | Weighted score calculation |
| `TestDetermineReviewDecision` | 3 | Decision thresholds |
| `TestRouteAfterReviewer` | 3 | Routing logic |
| `TestCreateRevisionRequest` | 2 | Revision request generation |
| `TestGeneratePaperMarkdown` | 3 | Markdown generation |
| `TestReviewerOutputModel` | 2 | Output model validation |
| `TestRevisionCountTracking` | 3 | Iteration tracking |
| `TestReviewerNodeBasic` | 5 | Basic node functionality |
| `TestReviewerIntegration` | 4 | End-to-end integration |

## Quality Dimensions

### Contribution (25%)
- **Novelty**: Does the work present new ideas?
- **Significance**: How important is the contribution?
- **Gap Addressing**: Does it fill an identified gap?
- **Field Advancement**: Does it advance the field?

### Methodology (25%)
- **Rigor**: Is the approach scientifically sound?
- **Data Quality**: Is the data appropriate and sufficient?
- **Validity**: Are methods valid for the claims?
- **Reproducibility**: Can the work be replicated?

### Evidence (20%)
- **Robustness**: Are results well-supported?
- **Interpretation**: Are conclusions appropriate?
- **Limitations**: Are limitations acknowledged?
- **Alternative Explanations**: Are alternatives considered?

### Coherence (15%)
- **Logic Flow**: Does the argument flow logically?
- **Integration**: Do sections connect well?
- **Consistency**: Are claims consistent throughout?
- **Structure**: Is the paper well-organized?

### Writing Quality (15%)
- **Academic Tone**: Is the tone appropriate?
- **Banned Words**: Are AI-flagged words avoided?
- **Citations**: Are citations proper and complete?
- **Grammar**: Is the writing clear and correct?

## Decision Thresholds

| Score Range | Decision | Action |
|-------------|----------|--------|
| ≥ 7.0 | APPROVE | Route to OUTPUT node |
| 4.0 - 6.9 | REVISE | Route to WRITER node (max 3 times) |
| < 4.0 | REJECT | Route to OUTPUT with rejection |

## Revision Loop Behavior

1. **First Pass**: WRITER generates paper → REVIEWER evaluates
2. **Revisions**: If score < 7.0, REVIEWER creates `RevisionRequest`
3. **Writer Revision**: WRITER receives prioritized feedback
4. **Iteration**: Loop continues until approve or max iterations (3)
5. **Human Escalation**: After 3 iterations, human must decide

## Files Changed

### New Files
- `src/review/__init__.py` - Module exports
- `src/review/criteria.py` - Evaluation functions (~850 lines)
- `src/nodes/reviewer.py` - REVIEWER node (~350 lines)
- `tests/unit/test_reviewer.py` - Test suite (48 tests)

### Modified Files
- `src/state/enums.py` - Added 3 enums
- `src/state/models.py` - Added 5 models, 2 functions
- `src/state/schema.py` - Added 8 review fields
- `src/nodes/__init__.py` - Added reviewer exports
- `studio/graphs.py` - Added nodes and routing

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run reviewer tests only
uv run pytest tests/unit/test_reviewer.py -v

# Run with coverage
uv run pytest tests/unit/test_reviewer.py -v --cov=src/review --cov=src/nodes/reviewer
```

**Results**: All 329 tests pass (48 new + 281 existing)

## Usage Example

```python
from src.nodes.reviewer import reviewer_node, route_after_reviewer

# In workflow graph
graph.add_node("reviewer", reviewer_node)
graph.add_node("output", output_node)

graph.add_conditional_edges(
    "reviewer",
    route_after_reviewer,
    {"output": "output", "writer": "writer"}
)
```

## Future Improvements

1. **Multi-Reviewer**: Simulate multiple reviewer perspectives
2. **Revision Tracking**: Detailed diff between revisions
3. **Learning**: Improve criteria weights based on outcomes
4. **External API**: Optional external review service integration

## Related Issues

- GitHub Issue #16: Sprint 7 - REVIEWER Node and Revision Loop

## Sprint Status

✅ **COMPLETE** - All objectives achieved, tests passing, ready for merge
