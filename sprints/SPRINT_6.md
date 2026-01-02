# Sprint 6: WRITER Node

**Duration:** 5-6 days  
**Goal:** Construct the academic argument through section-by-section writing

---

## Overview

Writing in academic research is not mere reporting; it is argument construction. Each section serves a rhetorical purpose:

- **Introduction**: Establish the gap and promise the contribution
- **Literature Review**: Position within existing scholarship
- **Methods**: Justify analytical approach
- **Results/Findings**: Present evidence
- **Discussion**: Interpret and connect to literature
- **Conclusion**: Restate contribution and implications
- **Abstract**: Written LAST; summarizes the paper

---

## Architecture

### Input State

The WRITER node receives state from either DATA_ANALYST (empirical research) or CONCEPTUAL_SYNTHESIZER (theoretical research):

```python
# From DATA_ANALYST
analysis: AnalysisResult  # Contains DataAnalysisResult with findings
data_exploration_results: DataExplorationResult

# From CONCEPTUAL_SYNTHESIZER  
analysis: AnalysisResult  # Contains ConceptualSynthesisResult with framework

# Common inputs from prior nodes
research_plan: ResearchPlan
gap_analysis: GapAnalysis
literature_synthesis: dict
refined_query: str
contribution: ContributionStatement
```

### Output State

```python
draft: ResearchDraft  # Complete paper draft
sections: list[Section]  # Individual sections with metadata
style_violations: list[StyleViolation]  # Any style guide violations
status: ResearchStatus.WRITING_COMPLETE
```

---

## LangGraph Capabilities Used

- **Send API** for parallel section writing (where independent)
- **Subgraphs** for section writer encapsulation
- `stream_mode="messages"` for real-time token streaming
- `StreamWriter` for progress events

---

## File Structure

```
src/
├── nodes/
│   └── writer.py           # WRITER node orchestrator
├── writers/
│   ├── __init__.py
│   ├── base.py             # Base section writer
│   ├── introduction.py     # Introduction writer
│   ├── literature_review.py # Literature review writer
│   ├── methods.py          # Methods/methodology writer
│   ├── results.py          # Results/findings writer
│   ├── discussion.py       # Discussion writer
│   ├── conclusion.py       # Conclusion writer
│   ├── abstract.py         # Abstract writer (last)
│   └── argument.py         # Argument structure manager
├── citations/
│   ├── __init__.py
│   ├── formatter.py        # Chicago Author-Date formatting
│   ├── manager.py          # Citation tracking and verification
│   └── reference_list.py   # Reference list generation
├── style/
│   ├── __init__.py
│   ├── enforcer.py         # Main style enforcement
│   ├── banned_words.py     # BannedWordsFilter
│   ├── academic_tone.py    # AcademicToneChecker
│   ├── journal_style.py    # JournalStyleMatcher
│   ├── hedging.py          # HedgingLanguageChecker
│   └── precision.py        # PrecisionChecker
└── state/
    ├── enums.py            # Add WritingStatus, SectionType, etc.
    └── models.py           # Add Section, StyleViolation, etc.
```

---

## Implementation Tasks

### Task 1: Add State Models

**File: `src/state/enums.py`**

Add new enums:

```python
class SectionType(str, Enum):
    """Types of paper sections."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    LITERATURE_REVIEW = "literature_review"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    APPENDIX = "appendix"

class WritingStatus(str, Enum):
    """Status of section writing."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DRAFT_COMPLETE = "draft_complete"
    STYLE_CHECKED = "style_checked"
    CITATIONS_VERIFIED = "citations_verified"
    FINALIZED = "finalized"

class StyleViolationType(str, Enum):
    """Types of style guide violations."""
    BANNED_WORD = "banned_word"
    INFORMAL_TONE = "informal_tone"
    MISSING_HEDGE = "missing_hedge"
    VAGUE_TERM = "vague_term"
    CITATION_FORMAT = "citation_format"
    OVERCLAIM = "overclaim"
    JOURNAL_MISMATCH = "journal_mismatch"
```

**File: `src/state/models.py`**

Add new models:

```python
class Section(BaseModel):
    """A paper section."""
    section_type: SectionType
    title: str
    content: str
    word_count: int
    citations: list[str]
    status: WritingStatus = WritingStatus.PENDING
    style_violations: list["StyleViolation"] = []

class StyleViolation(BaseModel):
    """A style guide violation."""
    violation_type: StyleViolationType
    severity: CritiqueSeverity
    location: str  # e.g., "paragraph 2, sentence 3"
    original_text: str
    suggestion: str
    rule_reference: str  # Reference to style guide section

class CitationEntry(BaseModel):
    """A citation entry."""
    key: str  # e.g., "Fama1970"
    authors: list[str]
    year: int
    title: str
    journal: str | None = None
    volume: str | None = None
    pages: str | None = None
    doi: str | None = None
    
class ReferenceList(BaseModel):
    """Complete reference list."""
    entries: list[CitationEntry]
    format: str = "chicago_author_date"

class WriterOutput(BaseModel):
    """Output from the WRITER node."""
    sections: list[Section]
    reference_list: ReferenceList
    total_word_count: int
    style_violations: list[StyleViolation]
    argument_coherence_score: float
    contribution_delivered: bool
```

### Task 2: Create Citation Module

**File: `src/citations/formatter.py`**

```python
def format_inline_citation(authors: list[str], year: int) -> str:
    """Format inline citation in Chicago Author-Date style."""
    if len(authors) == 1:
        return f"({authors[0]} {year})"
    elif len(authors) == 2:
        return f"({authors[0]} and {authors[1]} {year})"
    else:
        return f"({authors[0]} et al. {year})"
```

**File: `src/citations/reference_list.py`**

Generate Chicago-style reference list from CitationEntry list.

### Task 3: Create Style Module

**File: `src/style/banned_words.py`**

Load banned words from `docs/writing_style_guide.md` and check text.

**File: `src/style/enforcer.py`**

```python
class StyleEnforcer:
    """Enforce academic writing style from style guide."""
    
    @classmethod
    def from_guide(cls, guide_path: str) -> "StyleEnforcer":
        """Load style rules from markdown guide."""
        ...
    
    def check(self, text: str) -> list[StyleViolation]:
        """Check text against all style rules."""
        ...
```

### Task 4: Create Section Writers

Each writer follows the same pattern:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from src.state.schema import WorkflowState
from src.state.models import Section
from src.style import StyleEnforcer

def write_introduction(state: WorkflowState) -> Section:
    """Write the introduction section."""
    # Components: hook, research question, preview of findings,
    # contribution, literature positioning, roadmap
    ...
```

### Task 5: Create WRITER Node

**File: `src/nodes/writer.py`**

```python
from langgraph.types import Send
from src.state.schema import WorkflowState
from src.state.enums import ResearchStatus, SectionType

def writer_node(state: WorkflowState) -> dict:
    """Orchestrate section writing based on paper type."""
    
    sections = determine_sections(
        paper_type=state.get("paper_type"),
        research_type=state.get("research_type"),
        methodology=state["research_plan"].methodology
    )
    
    return {
        "sections_to_write": sections,
        "status": ResearchStatus.WRITING
    }

def route_to_section_writers(state: WorkflowState):
    """Fan out to parallel section writers where possible."""
    parallel_sections = ["methods", "results", "related_work"]
    return [
        Send(f"write_{s}", state) 
        for s in state["sections_to_write"] 
        if s in parallel_sections
    ]
```

### Task 6: Create Argument Manager

**File: `src/writers/argument.py`**

```python
def build_argument_thread(sections: list[Section]) -> ArgumentThread:
    """Ensure logical flow across sections."""
    ...

def verify_claim_support(draft: ResearchDraft) -> list[UnsupportedClaim]:
    """Every claim backed by evidence."""
    ...

def check_contribution_delivery(
    promised: ContributionStatement,
    delivered: Section  # conclusion
) -> bool:
    """Does paper deliver promised contribution?"""
    ...
```

### Task 7: Update Graph Routing

**File: `studio/graphs.py`**

```python
# Change from:
workflow.add_edge("data_analyst", END)
workflow.add_edge("conceptual_synthesizer", END)

# To:
workflow.add_edge("data_analyst", "writer")
workflow.add_edge("conceptual_synthesizer", "writer")
workflow.add_edge("writer", END)  # Sprint 7 will add REVIEWER
```

---

## Section Writing Guidelines

### Introduction (1-2 pages for short papers)

Must include:
1. **Opening hook** (1-2 sentences): Why this matters
2. **Research question** (1 paragraph): What exactly you study
3. **Preview of findings** (1 paragraph): What you find
4. **Contribution** (1 paragraph): How this advances knowledge
5. **Literature positioning** (1-2 paragraphs): Where this fits
6. **Paper roadmap** (1-2 sentences): Structure of paper

### Literature Review

For short papers, integrated into introduction. Must:
- Thematic synthesis, not annotated bibliography
- Position work within existing scholarship
- Identify gaps (from GAP_IDENTIFIER)

### Methods

Must include:
- Data sources with citations
- Sample period and construction
- Variable definitions
- Econometric specification
- Identification strategy
- Limitations

### Results

Must include:
- Main findings with economic interpretation
- Statistical significance AND economic magnitude
- Robustness checks (abbreviated for short papers)
- Findings constrained by data (no interpretation yet)

### Discussion

Must include:
- Interpret findings
- Connect to literature
- Address limitations
- Alternative explanations

### Conclusion (0.5-1 page)

Must include:
- Summary of findings (2-3 sentences)
- Economic implications
- Limitations
- Future research directions

Do NOT:
- Introduce new findings
- Repeat detailed statistics
- Over-claim implications

### Abstract (50-75 words for short papers)

Written LAST. Must:
- State the research question
- Describe methodology briefly
- Present key findings
- Indicate the contribution

---

## Style Enforcement Rules

### Banned Words (from `docs/writing_style_guide.md`)

```python
BANNED_WORDS = {
    "delve", "realm", "harness", "unlock", "tapestry", "paradigm",
    "cutting-edge", "revolutionize", "landscape", "potential", "findings",
    "intricate", "showcasing", "crucial", "pivotal", "surpass",
    "meticulously", "vibrant", "unparalleled", "underscore", "leverage",
    "synergy", "innovative", "game-changer", "testament", "commendable",
    "meticulous", "highlight", "emphasize", "boast", "groundbreaking",
    "align", "foster", "showcase", "enhance", "holistic", "garner",
    "accentuate", "pioneering", "trailblazing", "unleash", "versatile",
    "transformative", "redefine", "seamless", "optimize", "scalable",
    "robust",  # non-statistical usage
    "breakthrough", "empower", "streamline", "intelligent", "smart",
    "next-gen", "frictionless", "elevate", "adaptive", "effortless",
    "data-driven", "insightful", "proactive", "mission-critical",
    "visionary", "disruptive", "reimagine", "agile", "customizable",
    "personalized", "unprecedented", "intuitive", "leading-edge",
    "synergize", "democratize", "automate", "accelerate",
    "state-of-the-art", "dynamic",  # non-technical usage
    "reliable", "efficient", "cloud-native", "immersive", "predictive",
    "transparent", "proprietary", "integrated", "plug-and-play",
    "turnkey", "future-proof", "open-ended", "AI-powered",
    "next-generation", "always-on", "hyper-personalized",
    "results-driven", "machine-first", "paradigm-shifting", "novel",
    "unique", "utilize", "impactful"
}
```

### Replacements

- "examine" instead of "delve"
- "use" instead of "utilize"  
- "new" instead of "novel"
- "important" instead of "crucial"
- "shows" instead of "showcases"

### Academic Tone Rules

- Active voice preferred
- First person plural acceptable ("We find...")
- No contractions
- No colloquialisms
- Appropriate hedging language

### Hedging Language

Required expressions:
- "The results suggest..."
- "These findings are consistent with..."
- "One interpretation is..."
- "The evidence points toward..."

### Precision Rules

- Use specific numbers: "a 15% increase" not "a large increase"
- Define all variables explicitly
- Avoid vague terms: "many", "various", "some"
- Quantify economic magnitude

---

## Citation Format

### Chicago Author-Date (In-Text)

| Type | Format | Example |
|------|--------|---------|
| Single author | (Author Year) | (Fama 1970) |
| Two authors | (Author and Author Year) | (Fama and French 1993) |
| Three+ authors | (Author et al. Year) | (Barberis et al. 2001) |
| With page | (Author Year, page) | (Jensen 1986, 323) |
| Multiple sources | (Author Year; Author Year) | (Fama 1970; Jensen 1986) |
| Narrative | Author (Year) | Fama (1970) shows that... |

### Reference List Format

```
Fama, Eugene F., and Kenneth R. French. 1993. "Common Risk Factors in the 
    Returns on Stocks and Bonds." Journal of Financial Economics 33 (1): 3-56.
```

---

## Acceptance Criteria

- [ ] Each section serves its rhetorical purpose
- [ ] Claims are backed by evidence or citations
- [ ] Argument flows logically across sections
- [ ] Contribution promised in intro is delivered in conclusion
- [ ] Banned words are filtered per `docs/writing_style_guide.md`
- [ ] Style matches target journal (RFS, JFE, JF, JFQA conventions)
- [ ] Citations follow Chicago Author-Date format
- [ ] Hedging language used appropriately for claims
- [ ] Word counts within target ranges for paper type
- [ ] Abstract written last and summarizes paper accurately

---

## Dependencies

### Python Packages (already in pyproject.toml)

- `langchain-anthropic`: LLM for writing
- `pydantic`: Data validation

### Internal Dependencies

- `src/state/schema.py`: WorkflowState
- `src/state/models.py`: ResearchPlan, GapAnalysis, etc.
- `src/state/enums.py`: ResearchStatus, etc.
- `docs/writing_style_guide.md`: Style rules

---

## Testing Strategy

### Unit Tests

1. **test_writer.py**: WRITER node orchestration
2. **test_section_writers.py**: Individual section writers
3. **test_style.py**: Style enforcement rules
4. **test_citations.py**: Citation formatting

### Integration Tests

1. Writer node with mock state from DATA_ANALYST
2. Writer node with mock state from CONCEPTUAL_SYNTHESIZER
3. Full pipeline from PLANNER to WRITER

### Test Cases

- Short paper section determination
- Full paper section determination
- Banned word detection and replacement
- Citation formatting (all variants)
- Reference list generation
- Argument coherence checking
- Contribution delivery verification

---

## Notes

- Abstract is always written last (after all other sections)
- For short papers, literature review is integrated into introduction
- Methods section varies by research type (empirical vs theoretical)
- Discussion section interprets; Results section presents
- Style violations should be warnings, not blockers (REVIEWER handles final check)

---

## Completion Status

### Acceptance Criteria Checklist

- [x] Each section serves its rhetorical purpose
- [x] Claims are backed by evidence or citations
- [x] Argument flows logically across sections
- [x] Contribution promised in intro is delivered in conclusion
- [x] Banned words are filtered per `docs/writing_style_guide.md`
- [x] Style matches target journal (RFS, JFE, JF, JFQA conventions)
- [x] Citations follow Chicago Author-Date format
- [x] Hedging language used appropriately for claims

### Test Results

- **Unit Tests**: 52 new tests passing
- **Total Tests**: 257 tests passing
- **All existing tests continue to pass**

### Files Created/Modified

**New Files:**
- `src/nodes/writer.py` - WRITER node orchestrator
- `src/writers/__init__.py` - Module exports
- `src/writers/base.py` - BaseSectionWriter ABC
- `src/writers/introduction.py` - IntroductionWriter
- `src/writers/literature_review.py` - LiteratureReviewWriter
- `src/writers/methods.py` - MethodsWriter
- `src/writers/results.py` - ResultsWriter
- `src/writers/discussion.py` - DiscussionWriter
- `src/writers/conclusion.py` - ConclusionWriter
- `src/writers/abstract.py` - AbstractWriter
- `src/writers/argument.py` - ArgumentManager
- `src/citations/__init__.py` - Module exports
- `src/citations/formatter.py` - Chicago Author-Date formatting
- `src/citations/manager.py` - Citation tracking
- `src/citations/reference_list.py` - Reference list generation
- `src/style/__init__.py` - Module exports
- `src/style/enforcer.py` - StyleEnforcer
- `src/style/banned_words.py` - BannedWordsFilter
- `src/style/academic_tone.py` - AcademicToneChecker
- `src/style/hedging.py` - HedgingLanguageChecker
- `src/style/precision.py` - PrecisionChecker
- `src/style/journal_style.py` - JournalStyleMatcher
- `tests/unit/test_writer.py` - 52 unit tests

**Modified Files:**
- `src/state/enums.py` - Added SectionType, WritingStatus, StyleViolationType, CitationStyle, JournalTarget
- `src/state/models.py` - Added StyleViolation, CitationEntry, ReferenceList, PaperSection, ArgumentThread, WriterOutput, SectionWritingContext, SECTION_WORD_COUNTS
- `src/state/schema.py` - Added writer_output, completed_sections, reference_list, style_violations fields
- `src/state/__init__.py` - Updated exports
- `src/nodes/__init__.py` - Added writer_node exports
- `studio/graphs.py` - Added writer node and route_after_analysis routing
- `docs/IMPLEMENTATION_PLAN.md` - Marked Sprint 6 acceptance criteria complete

---

*Sprint 6 Completed: January 2026*
