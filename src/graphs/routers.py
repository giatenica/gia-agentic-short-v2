"""Routing functions for the research workflow graph.

This module contains all routing logic for conditional edges in the workflow,
extracted from the graph definition for modularity and testability.
"""

import logging
from typing import Literal

from src.state.schema import WorkflowState
from src.state.enums import ResearchStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
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

# Maximum number of errors before fallback
MAX_ERRORS_BEFORE_FALLBACK = 3


# =============================================================================
# Fallback Check Helper
# =============================================================================


def _should_fallback(state: WorkflowState) -> bool:
    """Check if workflow should route to fallback node.
    
    Args:
        state: Current workflow state
        
    Returns:
        True if should route to fallback
    """
    # Check explicit fallback flag
    if state.get("_should_fallback"):
        return True
    
    # Check error count
    errors = state.get("errors", [])
    if len(errors) >= MAX_ERRORS_BEFORE_FALLBACK:
        return True
    
    # Check for unrecoverable errors
    unrecoverable_count = sum(
        1 for e in errors
        if not getattr(e, 'recoverable', True)
    )
    if unrecoverable_count >= 1:
        return True
    
    # Check failed status
    if state.get("status") == ResearchStatus.FAILED:
        return True
    
    return False


# =============================================================================
# Intake and Data Explorer Routing
# =============================================================================


def route_after_intake(state: WorkflowState) -> Literal["data_explorer", "literature_reviewer", "fallback", "__end__"]:
    """
    Route after intake node.
    
    Routes to data_explorer if uploaded data files exist,
    otherwise directly to literature_reviewer.
    Routes to fallback if too many errors accumulated.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name, "fallback", or "__end__"
    """
    # Check for fallback condition
    if _should_fallback(state):
        logger.warning("Routing to fallback from intake due to errors")
        return "fallback"
    
    if state.get("errors"):
        return "__end__"
    
    if state.get("status") == ResearchStatus.INTAKE_COMPLETE:
        # Check if there are uploaded data files to explore
        uploaded_data = state.get("uploaded_data", [])
        if uploaded_data:
            logger.info("Routing to data_explorer: uploaded data found")
            return "data_explorer"
        logger.info("Routing to literature_reviewer: no uploaded data")
        return "literature_reviewer"
    
    return "__end__"


def route_after_data_explorer(state: WorkflowState) -> Literal["literature_reviewer", "fallback", "__end__"]:
    """
    Route after data explorer node.
    
    Continues to literature reviewer unless there are non-recoverable errors.
    Routes to fallback if too many errors accumulated.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name, "fallback", or "__end__"
    """
    # Check for fallback condition
    if _should_fallback(state):
        logger.warning("Routing to fallback from data_explorer due to errors")
        return "fallback"
    
    if state.get("errors"):
        # Check if errors are recoverable (data quality issues)
        errors = state.get("errors", [])
        if all(getattr(e, "recoverable", True) for e in errors):
            # Continue with warnings
            logger.warning("Continuing with recoverable data explorer errors")
            return "literature_reviewer"
        return "__end__"
    return "literature_reviewer"


# =============================================================================
# Literature Review Routing
# =============================================================================


def route_after_literature_reviewer(state: WorkflowState) -> Literal["literature_synthesizer", "fallback", "__end__"]:
    """
    Route after literature reviewer node.
    
    Continue workflow even without search results - the research can proceed
    with the data analysis path. Only stop on fatal errors.
    Routes to fallback if too many errors accumulated.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name, "fallback", or "__end__"
    """
    # Check for fallback condition
    if _should_fallback(state):
        logger.warning("Routing to fallback from literature_reviewer due to errors")
        return "fallback"
    
    # Check for fatal (non-recoverable) errors only
    errors = state.get("errors", [])
    fatal_errors = [e for e in errors if hasattr(e, 'recoverable') and not e.recoverable]
    if fatal_errors:
        logger.error(f"Fatal errors in literature reviewer: {fatal_errors}")
        return "__end__"
    
    # Always proceed to synthesizer - even with empty results, we need to 
    # acknowledge the literature gap and continue with data-driven research
    return "literature_synthesizer"


def route_after_synthesizer(state: WorkflowState) -> Literal["gap_identifier", "fallback", "__end__"]:
    """
    Route after literature synthesizer node.
    
    Continue to gap identification even without full synthesis - the gap
    identifier can work with partial information or generate gaps from
    the research question alone.
    Routes to fallback if too many errors accumulated.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name, "fallback", or "__end__"
    """
    # Check for fallback condition
    if _should_fallback(state):
        logger.warning("Routing to fallback from synthesizer due to errors")
        return "fallback"
    
    # Only stop on fatal (non-recoverable) errors
    errors = state.get("errors", [])
    fatal_errors = [e for e in errors if hasattr(e, 'recoverable') and not e.recoverable]
    if fatal_errors:
        logger.error(f"Fatal errors in synthesizer: {fatal_errors}")
        return "__end__"
    
    # Always proceed to gap identifier - it can work with whatever we have
    return "gap_identifier"


# =============================================================================
# Gap Identifier and Planner Routing
# =============================================================================


def route_after_gap_identifier(state: WorkflowState) -> Literal["planner", "fallback", "__end__"]:
    """
    Route after gap identifier to PLANNER node.
    Routes to fallback if too many errors accumulated.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name, "fallback", or "__end__"
    """
    # Check for fallback condition
    if _should_fallback(state):
        logger.warning("Routing to fallback from gap_identifier due to errors")
        return "fallback"
    
    if state.get("errors"):
        return "__end__"
    
    # Check if gap analysis is complete
    if state.get("gap_analysis") or state.get("refined_query"):
        return "planner"
    
    return "__end__"


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
    
    Args:
        state: Current workflow state
        
    Returns:
        Either "data_analyst" or "conceptual_synthesizer"
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
        logger.info(f"Routing to conceptual_synthesizer: research_type={research_type}")
        return "conceptual_synthesizer"
    
    # Route to conceptual synthesizer for theoretical methodologies
    if methodology_type and str(methodology_type).lower() in THEORETICAL_METHODOLOGIES:
        logger.info(f"Routing to conceptual_synthesizer: methodology={methodology_type}")
        return "conceptual_synthesizer"
    
    # Route to data analyst if we have data and empirical research
    if has_data and research_type in ["empirical", "mixed", "experimental", "case_study"]:
        logger.info(f"Routing to data_analyst: research_type={research_type}, has_data=True")
        return "data_analyst"
    
    # Default: if no data, use conceptual synthesizer
    if not has_data:
        logger.info("Routing to conceptual_synthesizer: no data available")
        return "conceptual_synthesizer"
    
    # Default with data: use data analyst
    logger.info("Routing to data_analyst: default with data")
    return "data_analyst"


def route_after_planner(state: WorkflowState) -> Literal["data_analyst", "conceptual_synthesizer", "fallback", "__end__"]:
    """
    Route after planner to analysis nodes based on research type.
    Routes to fallback if too many errors accumulated.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name, "fallback", or "__end__"
    """
    # Check for fallback condition
    if _should_fallback(state):
        logger.warning("Routing to fallback from planner due to errors")
        return "fallback"
    
    if state.get("errors"):
        return "__end__"
    
    # Check if plan is approved
    plan = state.get("research_plan")
    if not plan:
        return "__end__"
    
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
        logger.info("Plan rejected, ending workflow")
        return "__end__"
    
    # Route based on research type
    return route_by_research_type(state)


# =============================================================================
# Analysis and Writer Routing
# =============================================================================


def route_after_analysis(state: WorkflowState) -> Literal["writer", "fallback", "__end__"]:
    """
    Route from analysis nodes to writer.
    
    Proceeds to writer if:
    - No errors in state
    - Analysis output exists (data_analyst_output or conceptual_synthesis_output)
    
    Routes to fallback if too many errors accumulated.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name, "fallback", or "__end__"
    """
    # Check for fallback condition
    if _should_fallback(state):
        logger.warning("Routing to fallback from analysis due to errors")
        return "fallback"
    
    if state.get("errors"):
        return "__end__"
    
    # Check for analysis completion
    has_data_analysis = state.get("data_analyst_output") is not None
    has_conceptual_synthesis = state.get("conceptual_synthesis_output") is not None
    
    if has_data_analysis or has_conceptual_synthesis:
        return "writer"
    
    return "__end__"


def route_after_writer(state: WorkflowState) -> Literal["reviewer", "fallback", "__end__"]:
    """
    Route from writer to reviewer.
    
    Proceeds to reviewer if:
    - No errors in state
    - Writer output exists
    
    Routes to fallback if too many errors accumulated.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name, "fallback", or "__end__"
    """
    # Check for fallback condition
    if _should_fallback(state):
        logger.warning("Routing to fallback from writer due to errors")
        return "fallback"
    
    if state.get("errors"):
        return "__end__"
    
    # Check for writer output
    if state.get("writer_output"):
        return "reviewer"
    
    return "__end__"


# =============================================================================
# Reviewer Routing
# =============================================================================


def route_after_reviewer(state: WorkflowState) -> Literal["writer", "output", "fallback", "__end__"]:
    """
    Route after reviewer based on review decision.
    
    Routes to:
    - "output" if approved by human or escalated
    - "writer" if revision needed (loops back)
    - "fallback" if too many errors accumulated
    - "__end__" if rejected or error
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name, "fallback", or "__end__"
    """
    # Check for fallback condition
    if _should_fallback(state):
        logger.warning("Routing to fallback from reviewer due to errors")
        return "fallback"
    
    decision = state.get("review_decision")
    human_approved = state.get("human_approved", False)
    
    logger.info(f"Reviewer routing: decision={decision}, human_approved={human_approved}")
    
    if decision == "approve" and human_approved:
        return "output"
    elif decision == "escalate":
        # Escalated to human - route to output where human makes final decision
        return "output"
    elif decision == "revise":
        return "writer"
    else:
        # reject or error
        return "__end__"
