"""Main research workflow graph assembly.

This module provides the factory function for creating the complete
academic research workflow graph with all nodes properly wired.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from src.state.schema import WorkflowState
from src.state.enums import ResearchStatus
from src.nodes import (
    intake_node,
    data_explorer_node,
    literature_reviewer_node,
    literature_synthesizer_node,
    gap_identifier_node,
    planner_node,
    data_analyst_node,
    conceptual_synthesizer_node,
    writer_node,
    reviewer_node,
)
from src.graphs.routers import (
    route_after_intake,
    route_after_data_explorer,
    route_after_literature_reviewer,
    route_after_synthesizer,
    route_after_gap_identifier,
    route_after_planner,
    route_after_analysis,
    route_after_writer,
    route_after_reviewer,
)
from src.cache import get_cache, get_cache_policy
from src.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Nodes that should have interrupt_before (human reviews before execution)
INTERRUPT_BEFORE_NODES = ["gap_identifier", "planner"]

# Nodes that should have interrupt_after (human reviews after execution)
INTERRUPT_AFTER_NODES = ["reviewer"]

# All nodes in workflow order
WORKFLOW_NODES = [
    "intake",
    "data_explorer",
    "literature_reviewer",
    "literature_synthesizer",
    "gap_identifier",
    "planner",
    "data_analyst",
    "conceptual_synthesizer",
    "writer",
    "reviewer",
    "output",
]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class WorkflowConfig:
    """Configuration for research workflow compilation.
    
    Attributes:
        checkpointer: Checkpoint saver for persistence (optional)
        store: Memory store for long-term data (optional)
        cache: Cache backend for node caching (optional)
        interrupt_before: Nodes to pause before execution
        interrupt_after: Nodes to pause after execution
        enable_caching: Whether to enable node-level caching
        debug: Enable debug logging
    """
    checkpointer: BaseCheckpointSaver | None = None
    store: BaseStore | None = None
    cache: Any | None = None
    interrupt_before: list[str] = field(default_factory=lambda: INTERRUPT_BEFORE_NODES.copy())
    interrupt_after: list[str] = field(default_factory=lambda: INTERRUPT_AFTER_NODES.copy())
    enable_caching: bool = True
    debug: bool = False


# =============================================================================
# Output Node
# =============================================================================


def output_node(state: WorkflowState) -> dict:
    """
    Output node - final node that prepares the completed paper.
    
    This node:
    1. Extracts the final paper from reviewer output
    2. Logs completion status
    3. Returns final state with COMPLETED status
    
    Args:
        state: Current workflow state
        
    Returns:
        State update with completion status
    """
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


# =============================================================================
# Workflow Factory
# =============================================================================


def create_research_workflow(
    config: WorkflowConfig | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> StateGraph:
    """
    Create the main research workflow graph.
    
    This function assembles all nodes into the complete academic research
    workflow following the proper academic sequence:
    
    INTAKE → [if data] DATA_EXPLORER → LITERATURE_REVIEWER → LITERATURE_SYNTHESIZER 
        → GAP_IDENTIFIER → PLANNER
        → [route by research type] → DATA_ANALYST or CONCEPTUAL_SYNTHESIZER 
        → WRITER → REVIEWER → [approve] → OUTPUT
                         ↓
                    [revise] → WRITER (revision loop)
    
    Args:
        config: Workflow configuration (optional, uses defaults if not provided)
        checkpointer: Override checkpointer from config
        store: Override store from config
        
    Returns:
        Compiled StateGraph ready for execution
        
    Example:
        # Basic usage with defaults (LangGraph Studio)
        workflow = create_research_workflow()
        
        # With custom configuration
        from langgraph.checkpoint.sqlite import SqliteSaver
        config = WorkflowConfig(
            checkpointer=SqliteSaver.from_conn_string("sqlite:///research.db"),
            interrupt_before=["gap_identifier", "planner"],
        )
        workflow = create_research_workflow(config)
        
        # Direct checkpointer/store override
        workflow = create_research_workflow(checkpointer=my_checkpointer)
    """
    # Use provided config or create default
    if config is None:
        config = WorkflowConfig()
    
    # Allow direct override of checkpointer/store
    if checkpointer is not None:
        config.checkpointer = checkpointer
    if store is not None:
        config.store = store
    
    if config.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Creating research workflow with debug enabled")
    
    # Create the graph
    workflow = StateGraph(WorkflowState)
    
    # ==========================================================================
    # Get cache policies (if caching enabled)
    # ==========================================================================
    
    literature_policy = None
    synthesis_policy = None
    gap_policy = None
    writer_policy = None
    
    if config.enable_caching and settings.cache_enabled:
        literature_policy = get_cache_policy(ttl=settings.cache_ttl_literature)
        synthesis_policy = get_cache_policy(ttl=settings.cache_ttl_synthesis)
        gap_policy = get_cache_policy(ttl=settings.cache_ttl_gap_analysis)
        writer_policy = get_cache_policy(ttl=settings.cache_ttl_writer)
    
    # ==========================================================================
    # Add nodes (Sprint 1-7)
    # ==========================================================================
    
    # Sprint 1: Intake (no caching - always process fresh user input)
    workflow.add_node("intake", intake_node)
    
    # Sprint 1: Data explorer (no caching - always analyze fresh data)
    workflow.add_node("data_explorer", data_explorer_node)
    
    # Sprint 2: Literature reviewer (cache for 1 hour - API calls expensive)
    if literature_policy:
        workflow.add_node("literature_reviewer", literature_reviewer_node, cache_policy=literature_policy)
    else:
        workflow.add_node("literature_reviewer", literature_reviewer_node)
    
    # Sprint 2: Literature synthesizer (cache for 30 minutes)
    if synthesis_policy:
        workflow.add_node("literature_synthesizer", literature_synthesizer_node, cache_policy=synthesis_policy)
    else:
        workflow.add_node("literature_synthesizer", literature_synthesizer_node)
    
    # Sprint 3: Gap identifier (cache for 30 minutes)
    if gap_policy:
        workflow.add_node("gap_identifier", gap_identifier_node, cache_policy=gap_policy)
    else:
        workflow.add_node("gap_identifier", gap_identifier_node)
    
    # Sprint 4: Planner (no caching - has interrupt for human approval)
    workflow.add_node("planner", planner_node)
    
    # Sprint 5: Data analyst (cache for 30 minutes - analysis expensive)
    if synthesis_policy:
        workflow.add_node("data_analyst", data_analyst_node, cache_policy=synthesis_policy)
    else:
        workflow.add_node("data_analyst", data_analyst_node)
    
    # Sprint 5: Conceptual synthesizer (cache for 30 minutes)
    if synthesis_policy:
        workflow.add_node("conceptual_synthesizer", conceptual_synthesizer_node, cache_policy=synthesis_policy)
    else:
        workflow.add_node("conceptual_synthesizer", conceptual_synthesizer_node)
    
    # Sprint 6: Writer (cache for 10 minutes - may need iteration)
    if writer_policy:
        workflow.add_node("writer", writer_node, cache_policy=writer_policy)
    else:
        workflow.add_node("writer", writer_node)
    
    # Sprint 7: Reviewer (no caching - has interrupt for human approval)
    workflow.add_node("reviewer", reviewer_node)
    
    # Sprint 7: Output (no caching - final node)
    workflow.add_node("output", output_node)
    
    # ==========================================================================
    # Add edges
    # ==========================================================================
    
    # Start -> Intake
    workflow.add_edge(START, "intake")
    
    # Intake -> Data Explorer (if data) or Literature Reviewer
    workflow.add_conditional_edges(
        "intake",
        route_after_intake,
        ["data_explorer", "literature_reviewer", END]
    )
    
    # Data Explorer -> Literature Reviewer
    workflow.add_conditional_edges(
        "data_explorer",
        route_after_data_explorer,
        ["literature_reviewer", END]
    )
    
    # Literature Reviewer -> Literature Synthesizer
    workflow.add_conditional_edges(
        "literature_reviewer",
        route_after_literature_reviewer,
        ["literature_synthesizer", END]
    )
    
    # Literature Synthesizer -> Gap Identifier
    workflow.add_conditional_edges(
        "literature_synthesizer",
        route_after_synthesizer,
        ["gap_identifier", END]
    )
    
    # Gap Identifier -> Planner
    workflow.add_conditional_edges(
        "gap_identifier",
        route_after_gap_identifier,
        ["planner", END]
    )
    
    # Planner -> Data Analyst or Conceptual Synthesizer (Sprint 5)
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        ["data_analyst", "conceptual_synthesizer", END]
    )
    
    # Data Analyst -> Writer (Sprint 6)
    workflow.add_conditional_edges(
        "data_analyst",
        route_after_analysis,
        ["writer", END]
    )
    
    # Conceptual Synthesizer -> Writer (Sprint 6)
    workflow.add_conditional_edges(
        "conceptual_synthesizer",
        route_after_analysis,
        ["writer", END]
    )
    
    # Writer -> Reviewer (Sprint 7)
    workflow.add_conditional_edges(
        "writer",
        route_after_writer,
        ["reviewer", END]
    )
    
    # Reviewer -> Writer (revision) or Output (approval) (Sprint 7)
    workflow.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        ["writer", "output", END]
    )
    
    # Output -> END
    workflow.add_edge("output", END)
    
    # ==========================================================================
    # Compile with configuration
    # ==========================================================================
    
    # Build compile kwargs
    compile_kwargs = {}
    
    if config.checkpointer:
        compile_kwargs["checkpointer"] = config.checkpointer
    
    if config.store:
        compile_kwargs["store"] = config.store
    
    if config.interrupt_before:
        compile_kwargs["interrupt_before"] = config.interrupt_before
    
    if config.interrupt_after:
        compile_kwargs["interrupt_after"] = config.interrupt_after
    
    # Add cache if enabled and configured
    if config.enable_caching:
        cache = config.cache or get_cache()
        if cache:
            compile_kwargs["cache"] = cache
    
    logger.info(f"Compiling workflow with config: {list(compile_kwargs.keys())}")
    
    return workflow.compile(**compile_kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_studio_workflow() -> StateGraph:
    """
    Create workflow for LangGraph Studio.
    
    Studio manages its own persistence, so we don't pass checkpointer/store.
    Caching is enabled by default for faster development.
    
    Returns:
        Compiled StateGraph for Studio
    """
    config = WorkflowConfig(
        checkpointer=None,
        store=None,
        interrupt_before=[],  # Studio handles interrupts differently
        interrupt_after=[],
        enable_caching=settings.cache_enabled,
    )
    return create_research_workflow(config)


def create_production_workflow(
    db_path: str = "sqlite:///research.db",
) -> StateGraph:
    """
    Create workflow for production use.
    
    Uses SQLite checkpointer for persistence and enables all HITL gates.
    
    Args:
        db_path: SQLite database path for checkpointing
        
    Returns:
        Compiled StateGraph for production
    """
    from langgraph.checkpoint.sqlite import SqliteSaver
    
    checkpointer = SqliteSaver.from_conn_string(db_path)
    
    config = WorkflowConfig(
        checkpointer=checkpointer,
        interrupt_before=INTERRUPT_BEFORE_NODES,
        interrupt_after=INTERRUPT_AFTER_NODES,
        enable_caching=True,
    )
    return create_research_workflow(config)
