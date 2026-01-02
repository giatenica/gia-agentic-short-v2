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
from src.state.enums import ResearchStatus, PlanApprovalStatus
from src.nodes import (
    intake_node,
    literature_reviewer_node,
    literature_synthesizer_node,
    gap_identifier_node,
    route_after_gap_identifier,
    planner_node,
    route_after_planner,
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


def _route_after_planner(state: WorkflowState) -> Literal["__end__"]:
    """Route after planner - currently end (Sprint 5 will add DATA_ANALYST/CONCEPTUAL_SYNTHESIZER)."""
    # For now, always end after planner
    # Sprint 5 will route to data_analyst or conceptual_synthesizer
    return END


def create_research_workflow() -> StateGraph:
    """
    Create the main research workflow graph.
    
    Current implementation (Sprints 1-4):
    INTAKE -> LITERATURE_REVIEWER -> LITERATURE_SYNTHESIZER -> GAP_IDENTIFIER -> PLANNER -> END
    
    The GAP_IDENTIFIER node includes an interrupt() for human approval
    of the refined research question.
    
    The PLANNER node includes an interrupt() for human approval
    of the research methodology and plan.
    """
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("intake", intake_node)
    workflow.add_node("literature_reviewer", literature_reviewer_node)
    workflow.add_node("literature_synthesizer", literature_synthesizer_node)
    workflow.add_node("gap_identifier", gap_identifier_node)
    workflow.add_node("planner", planner_node)
    
    # Add edges
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
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        [END]
    )
    
    return workflow.compile()


# Create the workflow instance for Studio
research_workflow = create_research_workflow()
