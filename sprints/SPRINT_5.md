# Sprint 5: DATA_ANALYST and CONCEPTUAL_SYNTHESIZER Nodes

## Overview

Sprint 5 implements the analysis phase of the GIA Agentic research pipeline. After planning (Sprint 4), the workflow routes to one of two nodes based on research type:

- **DATA_ANALYST** for empirical research with quantitative data
- **CONCEPTUAL_SYNTHESIZER** for theoretical/conceptual research

This sprint completes the research execution phase, preparing findings for the WRITER node (Sprint 6).

## Components

### Research Type Router

Routes from PLANNER to the appropriate analysis node based on:
- Research type (empirical, theoretical, mixed)
- Availability of user data
- Methodology type from the research plan

```
PLANNER
   │
   ▼
[Research Type Router]
   │
   ├──► DATA_ANALYST (empirical + has data)
   │         │
   │         ▼
   │    [Quantitative Analysis]
   │         │
   └──► CONCEPTUAL_SYNTHESIZER (theoretical or no data)
             │
             ▼
        [Framework Building]
             │
             ▼
         WRITER (Sprint 6)
```

### DATA_ANALYST Node (`src/nodes/data_analyst.py`)

Executes data analysis per the research plan methodology:

| Task | Description |
|------|-------------|
| Execute Methodology | Run analysis matching research plan |
| Generate Statistics | Descriptive and inferential statistics |
| Test Hypotheses | Statistical hypothesis testing |
| Generate Findings | Structured findings with evidence |
| Assess Gap Coverage | Check if findings address the gap |

Features:
- Follows methodology from PLANNER node
- Uses variables from DATA_EXPLORER mapping
- Produces structured findings with statistical backing
- Links findings to research question
- Assesses whether findings address the identified gap

### CONCEPTUAL_SYNTHESIZER Node (`src/nodes/conceptual_synthesizer.py`)

Builds theoretical framework from literature:

| Task | Description |
|------|-------------|
| Extract Concepts | Key concepts from literature synthesis |
| Build Framework | Construct conceptual relationships |
| Generate Propositions | Testable theoretical propositions |
| Ground in Theory | Link to existing theoretical frameworks |
| Assess Contribution | Evaluate theoretical contribution |

Features:
- Synthesizes concepts from literature review
- Constructs conceptual framework
- Generates testable propositions
- Links to seminal works and existing theory
- Assesses theoretical contribution

### Analysis Tools (`src/tools/analysis.py`)

Statistical analysis tools for DATA_ANALYST:

| Tool | Description |
|------|-------------|
| `execute_descriptive_stats` | Summary statistics (mean, std, quartiles) |
| `execute_correlation_analysis` | Correlation matrices and tests |
| `execute_regression_analysis` | OLS, fixed effects, panel data |
| `execute_hypothesis_test` | t-tests, chi-square, ANOVA |
| `execute_event_study` | Event study methodology |
| `generate_findings` | Structure results into findings |

### Synthesis Tools (`src/tools/synthesis.py`)

Conceptual analysis tools for CONCEPTUAL_SYNTHESIZER:

| Tool | Description |
|------|-------------|
| `extract_key_concepts` | Extract concepts from literature |
| `build_conceptual_framework` | Construct theoretical framework |
| `generate_propositions` | Generate testable propositions |
| `map_concept_relationships` | Map concept interconnections |
| `ground_in_theory` | Link to existing theoretical frameworks |
| `assess_theoretical_contribution` | Evaluate contribution strength |

### State Schema Updates

New models in `src/state/models.py`:

```python
class Finding(BaseModel):
    """A research finding with evidence."""
    finding_id: str
    finding_type: FindingType
    statement: str
    evidence: list[EvidenceItem]
    statistical_support: StatisticalResult | None
    confidence_level: float
    addresses_gap: bool

class StatisticalResult(BaseModel):
    """Statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    confidence_interval: tuple[float, float] | None
    degrees_of_freedom: int | None
    interpretation: str

class ConceptualFramework(BaseModel):
    """Theoretical framework from conceptual analysis."""
    framework_id: str
    title: str
    description: str
    key_concepts: list[Concept]
    relationships: list[ConceptRelationship]
    propositions: list[Proposition]
    theoretical_grounding: list[str]

class Proposition(BaseModel):
    """Testable theoretical proposition."""
    proposition_id: str
    statement: str
    derived_from: list[str]  # Concept IDs
    testable: bool
    empirical_support: EvidenceStrength
```

New enums in `src/state/enums.py`:

```python
class FindingType(str, Enum):
    """Types of research findings."""
    MAIN_RESULT = "main_result"
    SUPPORTING = "supporting"
    UNEXPECTED = "unexpected"
    NULL_RESULT = "null_result"
    ROBUSTNESS = "robustness"

class AnalysisStatus(str, Enum):
    """Status of analysis execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    PARTIAL = "partial"
```

## Testing

### Test Structure

```
tests/unit/
├── test_data_analyst.py      # DATA_ANALYST node tests
├── test_conceptual_synthesizer.py  # CONCEPTUAL_SYNTHESIZER tests
├── test_analysis_tools.py    # Analysis tools tests
├── test_synthesis_tools.py   # Synthesis tools tests
```

### Test Coverage

- DATA_ANALYST: Analysis execution, finding generation, gap coverage
- CONCEPTUAL_SYNTHESIZER: Framework building, proposition generation
- Analysis Tools: Statistical functions, result interpretation
- Synthesis Tools: Concept extraction, framework construction
- Routing: Research type routing after PLANNER

### Running Tests

```bash
# All Sprint 5 tests
uv run pytest tests/unit/test_data_analyst.py tests/unit/test_conceptual_synthesizer.py -v

# With coverage
uv run pytest tests/unit/ --cov=src/nodes --cov=src/tools
```

## LangGraph Capabilities Used

| Capability | Usage |
|------------|-------|
| **Send API** | Parallel analysis tasks (if needed) |
| **InjectedState** | Tools access data and literature |
| **InjectedStore** | Persisting findings |
| **RetryPolicy** | Robust computation |
| **CachePolicy** | Cache expensive analyses |
| **Conditional Edges** | Research type routing |

## Acceptance Criteria

- [ ] Research type routing works correctly (empirical vs theoretical)
- [ ] DATA_ANALYST executes analysis per methodology from plan
- [ ] DATA_ANALYST generates findings with statistical backing
- [ ] DATA_ANALYST links findings to research question
- [ ] DATA_ANALYST assesses gap coverage
- [ ] CONCEPTUAL_SYNTHESIZER extracts concepts from literature
- [ ] CONCEPTUAL_SYNTHESIZER builds coherent framework
- [ ] CONCEPTUAL_SYNTHESIZER generates testable propositions
- [ ] CONCEPTUAL_SYNTHESIZER grounds framework in theory
- [ ] Both nodes produce output compatible with WRITER node
- [ ] Unit tests pass with good coverage

## Dependencies

```toml
[project.dependencies]
scipy = ">=1.10.0"  # For statistical tests
statsmodels = ">=0.14.0"  # For regression analysis
```

## Related Documentation

- [Implementation Plan](../docs/IMPLEMENTATION_PLAN.md) - Full project roadmap
- [Sprint 4 (PLANNER)](./SPRINT_4.md) - Prior sprint
- [Copilot Instructions](../.github/copilot-instructions.md) - Development guidelines

## Next Steps (Sprint 6)

- WRITER node for draft construction
- Section-by-section writing
- Citation integration
- Style enforcement per writing guide

---

**Author:** Gia Tenica*  
**Date:** 2 January 2026

*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher.
