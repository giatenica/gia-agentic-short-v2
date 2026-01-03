"""Graph definitions for LangGraph Studio.

Note: LangGraph Studio/API handles persistence automatically.
Don't pass checkpointer/store here - the platform provides these.

Node-level caching is enabled by default to speed up development and testing.
Configure via environment variables:
    CACHE_ENABLED=true/false
    CACHE_TTL_DEFAULT=1800 (seconds)
See src/cache/__init__.py for full configuration options.
"""

import os
import sys
from typing import Literal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, START, END

from src.agents import create_react_agent, create_research_agent, create_data_analyst_agent
from src.state.schema import WorkflowState
from src.state.enums import ResearchStatus
from src.nodes import (
    intake_node,
    data_explorer_node,
    literature_reviewer_node,
    literature_synthesizer_node,
    gap_identifier_node,
    route_after_gap_identifier,
    planner_node,
    route_after_planner,
    data_analyst_node,
    route_after_data_analyst,
    conceptual_synthesizer_node,
    route_after_conceptual_synthesizer,
    writer_node,
    reviewer_node,
    route_after_reviewer,
)
from src.cache import get_cache, get_cache_policy
from src.config import settings

# Create agent instances for Studio
# Note: Don't pass checkpointer/store - LangGraph API handles persistence
react_agent = create_react_agent()
research_agent = create_research_agent()
data_analyst_agent = create_data_analyst_agent()


# =============================================================================
# Research Workflow Graph (Sprints 1-4)
# =============================================================================


def route_after_intake(state: WorkflowState) -> Literal["data_explorer", "literature_reviewer", "__end__"]:
    """Route after intake node.
    
    Routes to data_explorer if uploaded data files exist,
    otherwise directly to literature_reviewer.
    """
    if state.get("errors"):
        return END
    if state.get("status") == ResearchStatus.INTAKE_COMPLETE:
        # Check if there are uploaded data files to explore
        uploaded_data = state.get("uploaded_data", [])
        if uploaded_data:
            return "data_explorer"
        return "literature_reviewer"
    return END


def route_after_data_explorer(state: WorkflowState) -> Literal["literature_reviewer", "__end__"]:
    """Route after data explorer node."""
    if state.get("errors"):
        # Check if errors are recoverable (data quality issues)
        errors = state.get("errors", [])
        if all(getattr(e, "recoverable", True) for e in errors):
            # Continue with warnings
            return "literature_reviewer"
        return END
    return "literature_reviewer"


def route_after_literature_reviewer(state: WorkflowState) -> Literal["literature_synthesizer", "__end__"]:
    """Route after literature reviewer node.
    
    Continue workflow even without search results - the research can proceed
    with the data analysis path. Only stop on fatal errors.
    """
    # Check for fatal (non-recoverable) errors only
    errors = state.get("errors", [])
    fatal_errors = [e for e in errors if hasattr(e, 'recoverable') and not e.recoverable]
    if fatal_errors:
        return END
    
    # Always proceed to synthesizer - even with empty results, we need to 
    # acknowledge the literature gap and continue with data-driven research
    return "literature_synthesizer"


def route_after_synthesizer(state: WorkflowState) -> Literal["gap_identifier", "__end__"]:
    """Route after literature synthesizer node.
    
    Continue to gap identification even without full synthesis - the gap
    identifier can work with partial information or generate gaps from
    the research question alone.
    """
    # Only stop on fatal (non-recoverable) errors
    errors = state.get("errors", [])
    fatal_errors = [e for e in errors if hasattr(e, 'recoverable') and not e.recoverable]
    if fatal_errors:
        return END
    
    # Always proceed to gap identifier - it can work with whatever we have
    return "gap_identifier"


def _route_after_gap_identifier(state: WorkflowState) -> Literal["planner", "__end__"]:
    """Route after gap identifier to PLANNER node."""
    if state.get("errors"):
        return END
    # Check if gap analysis is complete
    if state.get("gap_analysis") or state.get("refined_query"):
        return "planner"
    return END


# =============================================================================
# Research Type Router (Sprint 5)
# =============================================================================


# Methodology types that indicate theoretical/conceptual research
THEORETICAL_METHODOLOGIES = {
    "analytical_model",
    "simulation",
    "conceptual_framework",
    "systematic_review",
    "meta_analysis",
    "narrative_review",
}


def route_by_research_type(state: WorkflowState) -> Literal["data_analyst", "conceptual_synthesizer"]:
    """
    Route to appropriate analysis node based on research type.
    
    Routes to DATA_ANALYST if:
    - Research type is empirical or mixed methods AND has data
    - Methodology is quantitative
    
    Routes to CONCEPTUAL_SYNTHESIZER if:
    - Research type is theoretical
    - No data available
    - Methodology is theoretical/conceptual
    """
    # Check if we have data from exploration
    has_data = state.get("data_exploration_results") is not None
    
    # Get research type from state
    research_type = state.get("research_type", "").lower()
    
    # Get methodology type from research plan
    plan = state.get("research_plan")
    methodology_type = None
    if plan:
        if isinstance(plan, dict):
            methodology_type = plan.get("methodology_type", "")
        elif hasattr(plan, "methodology_type"):
            methodology_type = plan.methodology_type
            if hasattr(methodology_type, "value"):
                methodology_type = methodology_type.value
    
    # Route to conceptual synthesizer for theoretical work
    if research_type in ["theoretical", "literature_review"]:
        return "conceptual_synthesizer"
    
    # Route to conceptual synthesizer for theoretical methodologies
    if methodology_type and str(methodology_type).lower() in THEORETICAL_METHODOLOGIES:
        return "conceptual_synthesizer"
    
    # Route to data analyst if we have data and empirical research
    if has_data and research_type in ["empirical", "mixed", "experimental", "case_study"]:
        return "data_analyst"
    
    # Default: if no data, use conceptual synthesizer
    if not has_data:
        return "conceptual_synthesizer"
    
    # Default with data: use data analyst
    return "data_analyst"


def _route_after_planner(state: WorkflowState) -> Literal["data_analyst", "conceptual_synthesizer", "__end__"]:
    """Route after planner to analysis nodes based on research type."""
    if state.get("errors"):
        return END
    
    # Check if plan is approved
    plan = state.get("research_plan")
    if not plan:
        return END
    
    # Check approval status
    approval_status = None
    if isinstance(plan, dict):
        approval_status = plan.get("approval_status", "pending")
    elif hasattr(plan, "approval_status"):
        approval_status = plan.approval_status
        if hasattr(approval_status, "value"):
            approval_status = approval_status.value
    
    # Only proceed if plan is approved
    if approval_status == "rejected":
        return END
    
    # Route based on research type
    return route_by_research_type(state)


# =============================================================================
# Analysis to Writer Routing (Sprint 6)
# =============================================================================


def route_after_analysis(state: WorkflowState) -> Literal["writer", "__end__"]:
    """
    Route from analysis nodes to writer.
    
    Proceeds to writer if:
    - No errors in state
    - Analysis output exists (data_analyst_output or conceptual_synthesis_output)
    """
    if state.get("errors"):
        return END
    
    # Check for analysis completion
    has_data_analysis = state.get("data_analyst_output") is not None
    has_conceptual_synthesis = state.get("conceptual_synthesis_output") is not None
    
    if has_data_analysis or has_conceptual_synthesis:
        return "writer"
    
    return END


# =============================================================================
# Writer to Reviewer Routing (Sprint 7)
# =============================================================================


def route_after_writer(state: WorkflowState) -> Literal["reviewer", "__end__"]:
    """
    Route from writer to reviewer.
    
    Proceeds to reviewer if:
    - No errors in state
    - Writer output exists
    """
    if state.get("errors"):
        return END
    
    # Check for writer output
    if state.get("writer_output"):
        return "reviewer"
    
    return END


def _route_after_reviewer(state: WorkflowState) -> Literal["writer", "output", "__end__"]:
    """
    Route after reviewer based on review decision.
    
    Routes to:
    - "output" if approved by human
    - "writer" if revision needed (loops back)
    - END if rejected or error
    """
    decision = state.get("review_decision")
    human_approved = state.get("human_approved", False)
    
    if decision == "approve" and human_approved:
        return "output"
    elif decision == "revise":
        return "writer"
    else:
        return END


def output_node(state: WorkflowState) -> dict:
    """
    Output node - final node that prepares the completed paper.
    
    This node:
    1. Extracts the final paper from reviewer output
    2. Logs completion status
    3. Returns final state
    """
    import logging
    from src.state.enums import ResearchStatus
    
    logger = logging.getLogger(__name__)
    logger.info("OUTPUT: Preparing final paper output")
    
    reviewer_output = state.get("reviewer_output")
    final_paper = None
    
    if reviewer_output:
        if isinstance(reviewer_output, dict):
            final_paper = reviewer_output.get("final_paper")
        elif hasattr(reviewer_output, "final_paper"):
            final_paper = reviewer_output.final_paper
    
    if final_paper:
        logger.info(f"OUTPUT: Final paper ready ({len(final_paper)} characters)")
    else:
        logger.warning("OUTPUT: No final paper content available")
    
    return {
        "status": ResearchStatus.COMPLETED,
    }


def create_research_workflow() -> StateGraph:
    """
    Create the main research workflow graph.
    
    Current implementation (Sprints 1-7):
    INTAKE -> [if data] DATA_EXPLORER -> LITERATURE_REVIEWER -> LITERATURE_SYNTHESIZER 
        -> GAP_IDENTIFIER -> PLANNER
        -> [route by research type] -> DATA_ANALYST or CONCEPTUAL_SYNTHESIZER 
        -> WRITER -> REVIEWER -> [approve] -> OUTPUT
                         ↓
                    [revise] -> WRITER (revision loop)
    
    The DATA_EXPLORER node analyzes uploaded data files:
    - Schema detection and summary statistics
    - Data quality assessment
    - Variable mapping to research questions
    - Skipped if no data files are uploaded
    
    The GAP_IDENTIFIER node includes an interrupt() for human approval
    of the refined research question.
    
    The PLANNER node includes an interrupt() for human approval
    of the research methodology and plan.
    
    Sprint 5 adds:
    - Research type routing after PLANNER
    - DATA_ANALYST node for empirical research
    - CONCEPTUAL_SYNTHESIZER node for theoretical research
    
    Sprint 6 adds:
    - WRITER node for paper composition
    - Section writers for each paper component
    - Style enforcement and citation management
    
    Node-level caching:
    - Enabled by default (CACHE_ENABLED=true)
    - Caches LLM responses to avoid redundant computation
    - TTLs configured per node type for optimal freshness
    - Nodes with interrupt() (planner) are NOT cached
    - DATA_EXPLORER is not cached (always analyze fresh data)
    """
    workflow = StateGraph(WorkflowState)
    
    # Get cache policies for each node type
    # Nodes that should NOT be cached: intake (always fresh), planner (has interrupt)
    literature_policy = get_cache_policy(ttl=settings.cache_ttl_literature)
    synthesis_policy = get_cache_policy(ttl=settings.cache_ttl_synthesis)
    gap_policy = get_cache_policy(ttl=settings.cache_ttl_gap_analysis)
    writer_policy = get_cache_policy(ttl=settings.cache_ttl_writer)
    
    # Add nodes (Sprints 1-4)
    # intake: No caching - always process fresh user input
    workflow.add_node("intake", intake_node)
    
    # data_explorer: No caching - always analyze fresh data
    workflow.add_node("data_explorer", data_explorer_node)
    
    # literature_reviewer: Cache for 1 hour (API calls are expensive)
    workflow.add_node(
        "literature_reviewer", 
        literature_reviewer_node,
        cache_policy=literature_policy
    )
    
    # literature_synthesizer: Cache for 30 minutes
    workflow.add_node(
        "literature_synthesizer", 
        literature_synthesizer_node,
        cache_policy=synthesis_policy
    )
    
    # gap_identifier: Cache for 30 minutes
    workflow.add_node(
        "gap_identifier", 
        gap_identifier_node,
        cache_policy=gap_policy
    )
    
    # planner: No caching - has interrupt() for human approval
    workflow.add_node("planner", planner_node)
    
    # Add Sprint 5 nodes
    # data_analyst: Cache for 30 minutes (analysis is expensive)
    workflow.add_node(
        "data_analyst", 
        data_analyst_node,
        cache_policy=synthesis_policy
    )
    
    # conceptual_synthesizer: Cache for 30 minutes
    workflow.add_node(
        "conceptual_synthesizer", 
        conceptual_synthesizer_node,
        cache_policy=synthesis_policy
    )
    
    # Add Sprint 6 nodes
    # writer: Cache for 10 minutes (shorter TTL as writing may need iteration)
    workflow.add_node(
        "writer", 
        writer_node,
        cache_policy=writer_policy
    )
    
    # Add Sprint 7 nodes
    # reviewer: No caching - has interrupt() for human approval
    workflow.add_node("reviewer", reviewer_node)
    
    # output: No caching - final node
    workflow.add_node("output", output_node)
    
    # Add edges (Sprints 1-4)
    workflow.add_edge(START, "intake")
    workflow.add_conditional_edges(
        "intake",
        route_after_intake,
        ["data_explorer", "literature_reviewer", END]
    )
    workflow.add_conditional_edges(
        "data_explorer",
        route_after_data_explorer,
        ["literature_reviewer", END]
    )
    workflow.add_conditional_edges(
        "literature_reviewer",
        route_after_literature_reviewer,
        ["literature_synthesizer", END]
    )
    workflow.add_conditional_edges(
        "literature_synthesizer",
        route_after_synthesizer,
        ["gap_identifier", END]
    )
    workflow.add_conditional_edges(
        "gap_identifier",
        _route_after_gap_identifier,
        ["planner", END]
    )
    
    # Sprint 5: Research type routing after PLANNER
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        ["data_analyst", "conceptual_synthesizer", END]
    )
    
    # Sprint 6: Analysis nodes route to WRITER
    workflow.add_conditional_edges(
        "data_analyst",
        route_after_analysis,
        ["writer", END]
    )
    workflow.add_conditional_edges(
        "conceptual_synthesizer",
        route_after_analysis,
        ["writer", END]
    )
    
    # Sprint 7: WRITER routes to REVIEWER
    workflow.add_conditional_edges(
        "writer",
        route_after_writer,
        ["reviewer", END]
    )
    
    # Sprint 7: REVIEWER routes to OUTPUT or back to WRITER (revision loop)
    workflow.add_conditional_edges(
        "reviewer",
        _route_after_reviewer,
        ["writer", "output", END]
    )
    
    # Sprint 7: OUTPUT leads to END
    workflow.add_edge("output", END)
    
    # Compile with cache backend (None if caching disabled)
    cache = get_cache()
    return workflow.compile(cache=cache)


# Create the workflow instance for Studio
research_workflow = create_research_workflow()
