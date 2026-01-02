"""Graph definitions for LangGraph Studio.

Note: LangGraph Studio/API handles persistence automatically.
Don't pass checkpointer/store here - the platform provides these.
"""

import os
import sys
from typing import Literal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, START, END

from src.agents import create_react_agent, create_research_agent
from src.state.schema import WorkflowState
from src.state.enums import ResearchStatus
from src.nodes import (
    intake_node,
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
)

# Create agent instances for Studio
# Note: Don't pass checkpointer/store - LangGraph API handles persistence
react_agent = create_react_agent()
research_agent = create_research_agent()


# =============================================================================
# Research Workflow Graph (Sprints 1-4)
# =============================================================================


def route_after_intake(state: WorkflowState) -> Literal["literature_reviewer", "__end__"]:
    """Route after intake node."""
    if state.get("errors"):
        return END
    if state.get("status") == ResearchStatus.INTAKE_COMPLETE:
        return "literature_reviewer"
    return END


def route_after_literature_reviewer(state: WorkflowState) -> Literal["literature_synthesizer", "__end__"]:
    """Route after literature reviewer node."""
    if state.get("errors"):
        return END
    if state.get("search_results"):
        return "literature_synthesizer"
    return END


def route_after_synthesizer(state: WorkflowState) -> Literal["gap_identifier", "__end__"]:
    """Route after literature synthesizer node."""
    if state.get("errors"):
        return END
    if state.get("literature_synthesis"):
        return "gap_identifier"
    return END


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


def create_research_workflow() -> StateGraph:
    """
    Create the main research workflow graph.
    
    Current implementation (Sprints 1-6):
    INTAKE -> LITERATURE_REVIEWER -> LITERATURE_SYNTHESIZER -> GAP_IDENTIFIER -> PLANNER
        -> [route by research type] -> DATA_ANALYST or CONCEPTUAL_SYNTHESIZER 
        -> WRITER -> END
    
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
    """
    workflow = StateGraph(WorkflowState)
    
    # Add nodes (Sprints 1-4)
    workflow.add_node("intake", intake_node)
    workflow.add_node("literature_reviewer", literature_reviewer_node)
    workflow.add_node("literature_synthesizer", literature_synthesizer_node)
    workflow.add_node("gap_identifier", gap_identifier_node)
    workflow.add_node("planner", planner_node)
    
    # Add Sprint 5 nodes
    workflow.add_node("data_analyst", data_analyst_node)
    workflow.add_node("conceptual_synthesizer", conceptual_synthesizer_node)
    
    # Add Sprint 6 nodes
    workflow.add_node("writer", writer_node)
    
    # Add edges (Sprints 1-4)
    workflow.add_edge(START, "intake")
    workflow.add_conditional_edges(
        "intake",
        route_after_intake,
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
    
    # Sprint 6: WRITER leads to END
    workflow.add_edge("writer", END)
    
    return workflow.compile()


# Create the workflow instance for Studio
research_workflow = create_research_workflow()
