# Changelog

All notable changes to the GIA Agentic Research System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Sprint 11: Documentation and Polish (in progress)

---

## [2.0.0] - 2026-01-03

### Added

#### Sprint 10: Testing and Evaluation
- Integration test framework with MockChatModel
- End-to-end workflow tests (23 tests)
- State persistence tests (13 tests)
- Human-in-the-loop tests (15 tests)
- Evaluation suite with quality metrics
  - Completeness evaluation
  - Theme coverage evaluation
  - Citation quality evaluation
  - Methodology quality evaluation
  - Writing quality evaluation
  - Coherence evaluation
- Evaluation runner CLI
- Test queries dataset (10 finance domain queries)
- New unit tests: test_basic_tools.py, test_formatter.py, test_abstract_writer.py, test_evaluation_metrics.py

#### Sprint 9: Error Handling and Fallbacks
- Custom exception hierarchy (GIAError, WorkflowError, NodeExecutionError, etc.)
- RetryPolicy configurations with exponential backoff
- Error handler functions and decorators
- Recovery strategies (RETRY, SKIP, FALLBACK, ABORT)
- Fallback node for graceful degradation
- Error routing in all workflow routers
- 82 new tests for error handling

#### Sprint 8: Graph Assembly and Full Workflow Integration
- `src/graphs/` module with modular workflow components
- WorkflowConfig dataclass for flexible configuration
- Factory functions: create_research_workflow, create_studio_workflow, create_production_workflow
- Streaming utilities with StreamMode and StreamEvent
- WorkflowInspector for time travel debugging
- Subgraph compositions for modular workflows
- 71 new tests for Sprint 8 functionality

#### Sprint 7: REVIEWER Node and Revision Loop
- REVIEWER node with 5-dimension evaluation
  - Contribution (25%)
  - Methodology (25%)
  - Evidence (20%)
  - Coherence (15%)
  - Writing (15%)
- Review decision thresholds (APPROVE ≥7.0, REVISE 4.0-6.9, REJECT <4.0)
- Revision loop with max 3 iterations
- Human approval via interrupt()
- ReviewDecision, ReviewDimension, RevisionPriority enums
- QualityScore, ReviewCritiqueItem, ReviewCritique models
- 48 new tests for reviewer functionality

#### Sprint 6: WRITER Node
- WRITER node orchestrator
- Section writers: abstract, introduction, literature_review, methods, results, discussion, conclusion
- Argument structure manager
- Citation integration (APA format)
- Style enforcement module
  - BannedWordsFilter (100+ words)
  - AcademicToneChecker
  - HedgingLanguageChecker
  - PrecisionChecker
  - JournalStyleMatcher (RFS, JFE, JF, JFQA)

#### Sprint 5: DATA_ANALYST and CONCEPTUAL_SYNTHESIZER
- DATA_ANALYST node for empirical research
- CONCEPTUAL_SYNTHESIZER node for theoretical research
- Research type routing
- 35+ analysis tools
  - Data loading (CSV, Excel, Parquet, Stata, SPSS, ZIP)
  - Data profiling (statistics, distributions, outliers)
  - Data transformation (filter, join, aggregate)
  - Analysis (regression, correlation, hypothesis tests)
  - Interpretation (insights, recommendations)
- DuckDB backend for large datasets (46M+ rows)

#### Sprint 4: PLANNER Node
- PLANNER node with methodology selection
- Analysis design tools
- HITL approval via interrupt()
- MethodologyType enum (25+ methodologies)
- AnalysisApproach enum (15+ approaches)
- Research plan validation and feasibility assessment
- 34 new tests

#### Sprint 3: GAP_IDENTIFIER Node
- GAP_IDENTIFIER node
- Gap analysis tools (methodological, empirical, theoretical)
- Contribution statement generation
- Research question refinement
- HITL checkpoint for approval

#### Sprint 2: LITERATURE_REVIEWER Node
- LITERATURE_REVIEWER node
- Multi-source academic search
  - Semantic Scholar API
  - arXiv API
  - Tavily web search
- Literature synthesis
- Citation graph analysis
- Theme extraction

#### Sprint 1: Intake Processing and State Schema
- INTAKE node for form processing
- DATA_EXPLORER node for uploaded data
- WorkflowState TypedDict (30+ fields)
- 50+ Pydantic models
- Research status enums
- Data quality assessment

#### Sprint 0: Foundation
- Project structure with uv
- Basic agent framework (create_react_agent, create_research_agent)
- Memory system (checkpointer + store)
- Tool definitions
- LangGraph Studio integration
- Environment configuration

### Changed
- Migrated from custom agent framework to LangGraph
- Replaced 25 discrete agents with 10 core workflow nodes
- Centralized state management via WorkflowState
- Automatic LangSmith tracing

### Fixed
- Timezone-aware datetime handling (UTC)
- Safe expression evaluation (regex-based pattern blocking)
- ZIP extraction protection against zip bombs and path traversal

---

## [1.0.0] - 2025-12-01

### Added
- Initial release of GIA Agentic v1
- Custom agent framework with 25 discrete agents
- Manual workflow orchestration
- Basic research capabilities

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 2.0.0 | 2026-01-03 | LangGraph migration, complete workflow |
| 1.0.0 | 2025-12-01 | Initial release |

---

*Changelog maintained by Gia Tenica.*
