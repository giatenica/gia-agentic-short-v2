"""WorkflowState schema for GIA Agentic v2.

This module defines the central state object that flows through all nodes
in the LangGraph workflow. It uses TypedDict with annotations for proper
state management and message accumulation.
"""

from datetime import date, datetime, timezone
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.state.models import (
    DataExplorationResult,
    DataExplorationSummary,
    DataFile,
    ResearchPlan,
    SearchResult,
    AnalysisResult,
    ResearchDraft,
    Critique,
    EvidenceItem,
    WorkflowError,
    GapAnalysis,
    ContributionStatement,
    RefinedResearchQuestion,
    # Sprint 6 models
    WriterOutput,
    PaperSection,
    ReferenceList,
    StyleViolation,
    # Sprint 7 models
    ReviewCritique,
    RevisionRequest,
    ReviewerOutput,
    # Sprint 14 models
    DataAcquisitionPlan,
    AcquiredDataset,
    AcquisitionFailure,
    CodeSnippet,
    # Sprint 15 models
    TableArtifact,
    FigureArtifact,
)
from src.state.enums import ResearchStatus


class WorkflowState(TypedDict, total=False):
    """
    Central state schema for the GIA Agentic v2 research workflow.
    
    This state object persists across all graph nodes and maintains the
    complete context of the research workflow. Uses TypedDict for LangGraph
    compatibility with optional fields (total=False).
    
    The state is structured in logical groups:
    1. Intake form data - User-provided research context
    2. Data context - Uploaded data and exploration results
    3. Research context - Planning and methodology
    4. Literature context - Literature review results
    5. Analysis context - Findings and synthesis
    6. Draft context - Written output and review
    7. Workflow metadata - Status, messages, errors
    
    Usage with LangGraph:
        ```python
        from langgraph.graph import StateGraph
        from src.state import WorkflowState
        
        graph = StateGraph(WorkflowState)
        graph.add_node("intake", intake_node)
        # ... add more nodes
        ```
    """
    
    # =========================================================================
    # Intake Form Data
    # =========================================================================
    
    # Raw form submission (before parsing)
    form_data: dict[str, Any]
    
    # Basic information (from intake form)
    project_title: str
    original_query: str  # research_question from form
    target_journal: str
    paper_type: str
    research_type: str
    
    # Hypothesis (optional from form)
    user_hypothesis: str | None
    
    # =========================================================================
    # Data Context (from intake and exploration)
    # =========================================================================
    
    # Uploaded data files
    uploaded_data: list[DataFile]
    
    # Dataset names loaded into DataRegistry (from data_explorer)
    loaded_datasets: list[str]
    
    # User's description of their data
    data_context: str | None
    
    # Results from DATA_EXPLORER node
    data_exploration_results: DataExplorationResult | None

    # Additional exploration artifacts (non-canonical but used by nodes)
    all_exploration_results: list[DataExplorationResult] | None
    variable_mappings: list[dict[str, Any]] | None
    
    # Sprint 12: Enhanced data exploration summary with LLM-generated prose
    data_exploration_summary: DataExplorationSummary | None
    
    # User-specified key variables
    key_variables: list[str]
    
    # =========================================================================
    # Research Context (from intake and planning)
    # =========================================================================
    
    # User's proposed methodology
    proposed_methodology: str | None
    
    # Seed literature from user (starting point for lit review)
    seed_literature: list[str]
    
    # User's expected contribution
    expected_contribution: str | None
    
    # Constraints
    deadline: date | None
    constraints: str | None
    
    # Research plan from PLANNER node
    research_plan: ResearchPlan | None
    
    # =========================================================================
    # Literature Context (from LITERATURE_REVIEWER and GAP_IDENTIFIER)
    # =========================================================================
    
    # Literature synthesis (structured summary of literature)
    literature_synthesis: dict[str, Any] | None
    
    # Key themes from literature
    literature_themes: list[str]
    
    # Methodology precedents from literature
    methodology_precedents: list[str]
    
    # Identified gaps from GAP_IDENTIFIER
    identified_gaps: list[str]
    
    # Gap analysis result (structured from GAP_IDENTIFIER)
    gap_analysis: GapAnalysis | dict[str, Any] | None
    
    # Refined research question (after gap analysis)
    refined_query: str | None
    
    # Refined research question object
    refined_research_question: RefinedResearchQuestion | dict[str, Any] | None
    
    # Contribution statement (from gap analysis)
    contribution_statement: str | None
    
    # Contribution statement object
    contribution: ContributionStatement | dict[str, Any] | None
    
    # =========================================================================
    # Search and Analysis Context
    # =========================================================================
    
    # Search results from SEARCHER node
    search_results: list[SearchResult]
    
    # Analysis results from DATA_ANALYST or CONCEPTUAL_SYNTHESIZER
    analysis: AnalysisResult | None

    # Legacy analysis keys used for routing/backward compatibility
    data_analyst_output: dict[str, Any] | None
    conceptual_synthesis_output: dict[str, Any] | None
    
    # Evidence registry for claim tracking
    evidence_items: list[EvidenceItem]
    
    # =========================================================================
    # Draft Context (from WRITER and REVIEWER)
    # =========================================================================
    
    # Current draft from WRITER node
    draft: ResearchDraft | None
    
    # Writer output from Sprint 6 WRITER node
    writer_output: WriterOutput | dict[str, Any] | None
    
    # Sections to write (determined by writer node)
    sections_to_write: list[str]
    
    # Completed sections from section writers
    completed_sections: list[PaperSection]
    
    # Reference list
    reference_list: ReferenceList | dict[str, Any] | None
    
    # Style violations collected during writing
    style_violations: list[StyleViolation]
    
    # Critique from REVIEWER node (legacy)
    critique: Critique | None
    
    # =========================================================================
    # Sprint 7: Review Context (from REVIEWER)
    # =========================================================================
    
    # Review critique from REVIEWER node
    review_critique: ReviewCritique | dict[str, Any] | None
    
    # Review decision (approve, revise, reject)
    review_decision: str | None
    
    # Revision request (if revision needed)
    revision_request: RevisionRequest | dict[str, Any] | None
    
    # Reviewer output
    reviewer_output: ReviewerOutput | dict[str, Any] | None
    
    # Number of revision cycles completed
    revision_count: int
    
    # Maximum revision cycles allowed
    max_revisions: int
    
    # Human approval status for final output
    human_approved: bool
    
    # Human feedback on the review
    human_feedback: str | None
    
    # =========================================================================
    # Sprint 14: Data Acquisition Context
    # =========================================================================
    
    # Plan for acquiring external data
    data_acquisition_plan: DataAcquisitionPlan | dict[str, Any] | None
    
    # Successfully acquired datasets
    acquired_datasets: list[AcquiredDataset]
    
    # Failed acquisition attempts
    acquisition_failures: list[AcquisitionFailure]
    
    # Generated code snippets for custom acquisition
    generated_code_snippets: list[CodeSnippet]
    
    # =========================================================================
    # Sprint 15: Visualization & Tables Context
    # =========================================================================
    
    # Generated table artifacts (summary stats, regressions, correlations)
    tables: list[TableArtifact]
    
    # Generated figure artifacts (time series, scatter, distributions)
    figures: list[FigureArtifact]

    # =========================================================================
    # Fallback Output (graceful degradation)
    # =========================================================================

    fallback_report: dict[str, Any] | None
    final_paper: str | None
    _fallback_activated: bool
    
    # =========================================================================
    # Workflow Metadata
    # =========================================================================
    
    # Message history with add_messages reducer for accumulation
    messages: Annotated[list[AnyMessage], add_messages]
    
    # Current workflow status
    status: ResearchStatus
    
    # Number of revision iterations
    iteration_count: int
    
    # Maximum allowed iterations
    max_iterations: int
    
    # Workflow errors
    errors: list[WorkflowError]
    
    # Checkpoint log
    checkpoints: list[str]
    
    # Workflow timestamps
    created_at: datetime
    updated_at: datetime
    
    # Configuration overrides
    config: dict[str, Any]


def create_initial_state(
    form_data: dict[str, Any] | None = None,
    **kwargs
) -> WorkflowState:
    """
    Create an initial WorkflowState with sensible defaults.
    
    Args:
        form_data: Raw form submission data (optional).
        **kwargs: Additional state fields to set.
        
    Returns:
        Initialized WorkflowState.
        
    Example:
        ```python
        state = create_initial_state(
            form_data={"title": "My Research", ...},
            project_title="My Research"
        )
        ```
    """
    now = datetime.now(timezone.utc)
    
    defaults: WorkflowState = {
        # Intake
        "form_data": form_data or {},
        "project_title": "",
        "original_query": "",
        "target_journal": "",
        "paper_type": "",
        "research_type": "",
        "user_hypothesis": None,
        
        # Data
        "uploaded_data": [],
        "loaded_datasets": [],
        "data_context": None,
        "data_exploration_results": None,
        "data_exploration_summary": None,  # Sprint 12: LLM-generated data summary
        "all_exploration_results": None,
        "variable_mappings": None,
        "key_variables": [],
        
        # Research
        "proposed_methodology": None,
        "seed_literature": [],
        "expected_contribution": None,
        "deadline": None,
        "constraints": None,
        "research_plan": None,
        
        # Literature
        "literature_synthesis": None,
        "literature_themes": [],
        "methodology_precedents": [],
        "identified_gaps": [],
        "gap_analysis": None,
        "refined_query": None,
        "refined_research_question": None,
        "contribution_statement": None,
        "contribution": None,
        
        # Search/Analysis
        "search_results": [],
        "analysis": None,
        "data_analyst_output": None,
        "conceptual_synthesis_output": None,
        "evidence_items": [],
        
        # Draft
        "draft": None,
        "writer_output": None,
        "sections_to_write": [],
        "completed_sections": [],
        "reference_list": None,
        "style_violations": [],
        "critique": None,
        
        # Review (Sprint 7)
        "review_critique": None,
        "review_decision": None,
        "revision_request": None,
        "reviewer_output": None,
        "revision_count": 0,
        "max_revisions": 3,
        "human_approved": False,
        "human_feedback": None,

        # Sprint 14: Data acquisition
        "data_acquisition_plan": None,
        "acquired_datasets": [],
        "acquisition_failures": [],
        "generated_code_snippets": [],

        # Sprint 15: Tables/Figures
        "tables": [],
        "figures": [],

        # Fallback
        "fallback_report": None,
        "final_paper": None,
        "_fallback_activated": False,
        
        # Metadata
        "messages": [],
        "status": ResearchStatus.INTAKE_PENDING,
        "iteration_count": 0,
        "max_iterations": 3,
        "errors": [],
        "checkpoints": [],
        "created_at": now,
        "updated_at": now,
        "config": {},
    }
    
    # Override with provided kwargs
    defaults.update(kwargs)
    
    return defaults


def validate_state_for_node(state: WorkflowState, node_name: str) -> tuple[bool, list[str]]:
    """
    Validate that the state has required fields for a specific node.
    
    Args:
        state: Current workflow state.
        node_name: Name of the node to validate for.
        
    Returns:
        Tuple of (is_valid, list_of_missing_fields).
    """
    required_fields: dict[str, list[str]] = {
        "intake": ["form_data"],
        "data_explorer": ["uploaded_data"],
        "literature_reviewer": ["original_query"],
        # GAP_IDENTIFIER can synthesize minimally from search_results
        "gap_identifier": ["original_query"],
        # PLANNER can proceed as long as it has a research question
        "planner": ["original_query"],
        "data_analyst": ["research_plan", "data_exploration_results"],
        "conceptual_synthesizer": ["literature_synthesis"],
        # WRITER can proceed with canonical analysis or legacy data_analyst_output
        "writer": ["writer_input"],
        "reviewer": ["writer_output"],
        "output": ["reviewer_output", "human_approved"],
    }
    
    node_requirements = required_fields.get(node_name, [])
    missing = []
    
    for field in node_requirements:
        if field == "writer_input":
            has_analysis = bool(state.get("analysis"))
            has_legacy = bool(state.get("data_analyst_output")) or bool(state.get("conceptual_synthesis_output"))
            if not (has_analysis or has_legacy):
                missing.append("analysis")
            continue

        value = state.get(field)
        if value is None or (isinstance(value, (list, dict, str)) and not value):
            missing.append(field)
    
    return len(missing) == 0, missing
