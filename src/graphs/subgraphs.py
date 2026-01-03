"""Subgraph compositions for modular workflow design.

This module provides subgraph factories for composing the research workflow
from smaller, reusable components.
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END

from src.state.schema import WorkflowState
from src.state.enums import ResearchStatus
from src.nodes import (
    literature_reviewer_node,
    literature_synthesizer_node,
    data_analyst_node,
    conceptual_synthesizer_node,
)
from src.graphs.routers import (
    route_after_literature_reviewer,
    route_by_research_type,
    route_after_analysis,
)
from src.cache import get_cache_policy
from src.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Literature Review Subgraph
# =============================================================================


def create_literature_review_subgraph(
    enable_caching: bool = True,
) -> StateGraph:
    """
    Create a subgraph for literature review operations.
    
    This subgraph encapsulates:
    - LITERATURE_REVIEWER node (academic search)
    - LITERATURE_SYNTHESIZER node (theme extraction, synthesis)
    
    Args:
        enable_caching: Whether to enable node-level caching
        
    Returns:
        Compiled subgraph for literature review
        
    Example:
        lit_review = create_literature_review_subgraph()
        result = lit_review.invoke({"original_query": "..."})
    """
    subgraph = StateGraph(WorkflowState)
    
    # Get cache policies
    literature_policy = None
    synthesis_policy = None
    if enable_caching and settings.cache_enabled:
        literature_policy = get_cache_policy(ttl=settings.cache_ttl_literature)
        synthesis_policy = get_cache_policy(ttl=settings.cache_ttl_synthesis)
    
    # Add nodes
    if literature_policy:
        subgraph.add_node("literature_reviewer", literature_reviewer_node, cache_policy=literature_policy)
    else:
        subgraph.add_node("literature_reviewer", literature_reviewer_node)
    
    if synthesis_policy:
        subgraph.add_node("literature_synthesizer", literature_synthesizer_node, cache_policy=synthesis_policy)
    else:
        subgraph.add_node("literature_synthesizer", literature_synthesizer_node)
    
    # Add edges
    subgraph.add_edge(START, "literature_reviewer")
    
    def route_to_synthesizer(state: WorkflowState) -> Literal["literature_synthesizer", "__end__"]:
        """Route from reviewer to synthesizer."""
        result = route_after_literature_reviewer(state)
        return "literature_synthesizer" if result == "literature_synthesizer" else "__end__"
    
    subgraph.add_conditional_edges(
        "literature_reviewer",
        route_to_synthesizer,
        ["literature_synthesizer", END]
    )
    
    subgraph.add_edge("literature_synthesizer", END)
    
    return subgraph.compile()


# =============================================================================
# Analysis Subgraph
# =============================================================================


def create_analysis_subgraph(
    enable_caching: bool = True,
) -> StateGraph:
    """
    Create a subgraph for analysis operations.
    
    This subgraph encapsulates:
    - Research type routing (empirical vs theoretical)
    - DATA_ANALYST node (quantitative analysis)
    - CONCEPTUAL_SYNTHESIZER node (theoretical framework)
    
    The routing is based on:
    - Research type from state (empirical, theoretical, mixed)
    - Available data from data exploration
    - Methodology type from research plan
    
    Args:
        enable_caching: Whether to enable node-level caching
        
    Returns:
        Compiled subgraph for analysis
        
    Example:
        analysis = create_analysis_subgraph()
        result = analysis.invoke({
            "research_type": "empirical",
            "data_exploration_results": {...}
        })
    """
    subgraph = StateGraph(WorkflowState)
    
    # Get cache policy
    analysis_policy = None
    if enable_caching and settings.cache_enabled:
        analysis_policy = get_cache_policy(ttl=settings.cache_ttl_synthesis)
    
    # Entry node that routes to appropriate analysis
    def analysis_router_node(state: WorkflowState) -> dict:
        """Entry node that records the routing decision."""
        route = route_by_research_type(state)
        logger.info(f"Analysis router: routing to {route}")
        return {"_analysis_route": route}
    
    subgraph.add_node("router", analysis_router_node)
    
    # Add analysis nodes
    if analysis_policy:
        subgraph.add_node("data_analyst", data_analyst_node, cache_policy=analysis_policy)
        subgraph.add_node("conceptual_synthesizer", conceptual_synthesizer_node, cache_policy=analysis_policy)
    else:
        subgraph.add_node("data_analyst", data_analyst_node)
        subgraph.add_node("conceptual_synthesizer", conceptual_synthesizer_node)
    
    # Add edges
    subgraph.add_edge(START, "router")
    
    # Route from router to appropriate analysis node
    subgraph.add_conditional_edges(
        "router",
        route_by_research_type,
        ["data_analyst", "conceptual_synthesizer"]
    )
    
    # Both analysis nodes lead to END
    subgraph.add_edge("data_analyst", END)
    subgraph.add_edge("conceptual_synthesizer", END)
    
    return subgraph.compile()


# =============================================================================
# Writing Subgraph
# =============================================================================


def create_writing_subgraph(
    max_revisions: int = 3,
    enable_caching: bool = True,
) -> StateGraph:
    """
    Create a subgraph for writing and review operations.
    
    This subgraph encapsulates:
    - WRITER node (paper composition)
    - REVIEWER node (quality assessment)
    - Revision loop with max iterations
    
    Args:
        max_revisions: Maximum revision iterations
        enable_caching: Whether to enable node-level caching
        
    Returns:
        Compiled subgraph for writing/review
    """
    from src.nodes import writer_node, reviewer_node
    from src.graphs.routers import route_after_writer, route_after_reviewer
    
    subgraph = StateGraph(WorkflowState)
    
    # Get cache policy for writer (reviewer shouldn't be cached)
    writer_policy = None
    if enable_caching and settings.cache_enabled:
        writer_policy = get_cache_policy(ttl=settings.cache_ttl_writer)
    
    # Add nodes
    if writer_policy:
        subgraph.add_node("writer", writer_node, cache_policy=writer_policy)
    else:
        subgraph.add_node("writer", writer_node)
    
    subgraph.add_node("reviewer", reviewer_node)
    
    # Output node for this subgraph
    def subgraph_output(state: WorkflowState) -> dict:
        """Mark subgraph as complete."""
        return {"_writing_complete": True}
    
    subgraph.add_node("output", subgraph_output)
    
    # Add edges
    subgraph.add_edge(START, "writer")
    
    # Writer -> Reviewer
    def route_writer_to_reviewer(state: WorkflowState) -> Literal["reviewer", "__end__"]:
        result = route_after_writer(state)
        return "reviewer" if result == "reviewer" else "__end__"
    
    subgraph.add_conditional_edges(
        "writer",
        route_writer_to_reviewer,
        ["reviewer", END]
    )
    
    # Reviewer -> Writer (revision) or Output (approval)
    def route_review(state: WorkflowState) -> Literal["writer", "output", "__end__"]:
        # Check revision count
        revision_count = state.get("revision_count", 0)
        if revision_count >= max_revisions:
            logger.info(f"Max revisions ({max_revisions}) reached")
            return "output"
        
        result = route_after_reviewer(state)
        if result == "writer":
            return "writer"
        elif result == "output":
            return "output"
        return "__end__"
    
    subgraph.add_conditional_edges(
        "reviewer",
        route_review,
        ["writer", "output", END]
    )
    
    subgraph.add_edge("output", END)
    
    return subgraph.compile()


# =============================================================================
# Full Pipeline Subgraph (Alternative to main workflow)
# =============================================================================


def create_research_pipeline_subgraph() -> StateGraph:
    """
    Create a simplified research pipeline using subgraph composition.
    
    This demonstrates how subgraphs can be composed into a larger workflow.
    Note: This is an alternative approach to the main create_research_workflow()
    function and is primarily for demonstration.
    
    Returns:
        Compiled pipeline using composed subgraphs
    """
    from src.nodes import (
        intake_node,
        data_explorer_node,
        gap_identifier_node,
        planner_node,
    )
    from src.graphs.routers import (
        route_after_intake,
        route_after_data_explorer,
        route_after_synthesizer,
        route_after_gap_identifier,
        route_after_planner,
    )
    
    pipeline = StateGraph(WorkflowState)
    
    # Add individual nodes
    pipeline.add_node("intake", intake_node)
    pipeline.add_node("data_explorer", data_explorer_node)
    pipeline.add_node("gap_identifier", gap_identifier_node)
    pipeline.add_node("planner", planner_node)
    
    # Add subgraphs as nodes
    lit_review_subgraph = create_literature_review_subgraph()
    analysis_subgraph = create_analysis_subgraph()
    writing_subgraph = create_writing_subgraph()
    
    pipeline.add_node("literature_review", lit_review_subgraph)
    pipeline.add_node("analysis", analysis_subgraph)
    pipeline.add_node("writing", writing_subgraph)
    
    # Wire up the pipeline
    pipeline.add_edge(START, "intake")
    
    # Intake routing
    def route_intake(state: WorkflowState) -> Literal["data_explorer", "literature_review", "__end__"]:
        result = route_after_intake(state)
        if result == "literature_reviewer":
            return "literature_review"
        elif result == "data_explorer":
            return "data_explorer"
        return "__end__"
    
    pipeline.add_conditional_edges(
        "intake",
        route_intake,
        ["data_explorer", "literature_review", END]
    )
    
    # Data explorer -> Literature review
    def route_data_explorer(state: WorkflowState) -> Literal["literature_review", "__end__"]:
        result = route_after_data_explorer(state)
        return "literature_review" if result == "literature_reviewer" else "__end__"
    
    pipeline.add_conditional_edges(
        "data_explorer",
        route_data_explorer,
        ["literature_review", END]
    )
    
    # Literature review -> Gap identifier
    def route_lit_review(state: WorkflowState) -> Literal["gap_identifier", "__end__"]:
        result = route_after_synthesizer(state)
        return "gap_identifier" if result == "gap_identifier" else "__end__"
    
    pipeline.add_conditional_edges(
        "literature_review",
        route_lit_review,
        ["gap_identifier", END]
    )
    
    # Gap identifier -> Planner
    def route_gap(state: WorkflowState) -> Literal["planner", "__end__"]:
        result = route_after_gap_identifier(state)
        return "planner" if result == "planner" else "__end__"
    
    pipeline.add_conditional_edges(
        "gap_identifier",
        route_gap,
        ["planner", END]
    )
    
    # Planner -> Analysis subgraph
    def route_planner(state: WorkflowState) -> Literal["analysis", "__end__"]:
        result = route_after_planner(state)
        if result in ["data_analyst", "conceptual_synthesizer"]:
            return "analysis"
        return "__end__"
    
    pipeline.add_conditional_edges(
        "planner",
        route_planner,
        ["analysis", END]
    )
    
    # Analysis -> Writing subgraph
    pipeline.add_edge("analysis", "writing")
    
    # Writing -> END
    pipeline.add_edge("writing", END)
    
    return pipeline.compile()
