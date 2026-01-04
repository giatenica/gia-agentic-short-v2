"""Main research workflow graph assembly.

This module provides the factory function for creating the complete
academic research workflow graph with all nodes properly wired.
"""

import logging
from pathlib import Path
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
    data_acquisition_node,
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
    route_after_data_acquisition,
    route_after_analysis,
    route_after_writer,
    route_after_reviewer,
)
from src.nodes.fallback import fallback_node
from src.cache import get_cache, get_cache_policy
from src.config import settings
from src.output.latex import build_and_compile
from src.tools.visualization import export_all_artifacts
from src.state.models import WorkflowError

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
    "data_acquisition",  # Sprint 14: External data fetching
    "data_analyst",
    "conceptual_synthesizer",
    "writer",
    "reviewer",
    "output",
    "fallback",  # Error recovery node
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
    writer_output = state.get("writer_output")
    completed_sections = state.get("completed_sections") or []

    paper_title = None
    run_id = None
    if isinstance(writer_output, dict):
        paper_title = (writer_output.get("title") or "").strip() or None
        arg_thread = writer_output.get("argument_thread")
        if isinstance(arg_thread, dict):
            run_id = (arg_thread.get("thread_id") or "").strip() or None

    if not paper_title:
        paper_title = "Untitled Paper"
    if not run_id:
        run_id = "run"

    tables = state.get("tables") or []
    figures = state.get("figures") or []

    output_base = Path(settings.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_base / "_tmp_artifacts" / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    exported_files: list[str] = []
    try:
        export_result = export_all_artifacts.invoke(
            {
                "output_dir": str(artifacts_dir),
                "tables": tables,
                "figures": figures,
            }
        )
        if isinstance(export_result, dict) and export_result.get("status") == "success":
            exported_files = export_result.get("exported_files", []) or []
    except Exception as e:
        logger.warning("OUTPUT: Failed to export artifacts: %s", e)

    # Prefer writer sections; reviewer_output.final_paper is treated as a narrative artifact.
    if not isinstance(completed_sections, list) or not completed_sections:
        if isinstance(writer_output, dict):
            sections = writer_output.get("sections") or []
            if isinstance(sections, list):
                completed_sections = sections

    build_result = build_and_compile(
        base_output_dir=str(output_base),
        run_id=run_id,
        title=paper_title,
        author="Gia Tenica",
        sections=[s if isinstance(s, dict) else getattr(s, "model_dump", lambda: {})() for s in completed_sections],
        tables=[t if isinstance(t, dict) else getattr(t, "model_dump", lambda: {})() for t in tables],
        figures=[f if isinstance(f, dict) else getattr(f, "model_dump", lambda: {})() for f in figures],
        exported_files=exported_files,
    )

    # Move exported artifacts into the build directory so LaTeX \input / \includegraphics works.
    final_artifacts_dir = Path(build_result.output_dir) / "artifacts"
    final_artifacts_dir.mkdir(parents=True, exist_ok=True)
    moved_exports: list[str] = []
    for p in exported_files:
        try:
            src = Path(p)
            if src.exists():
                dest = final_artifacts_dir / src.name
                dest.write_bytes(src.read_bytes())
                moved_exports.append(str(dest))
        except Exception:
            continue

    errors = state.get("errors") or []
    if not build_result.engine:
        errors = list(errors) + [
            WorkflowError(
                node="output",
                category="dependency",
                message="No LaTeX engine found. Install 'tectonic' or 'latexmk' to compile PDF.",
                recoverable=True,
                details={"output_dir": build_result.output_dir, "tex_path": build_result.tex_path},
            )
        ]
    elif not build_result.pdf_path:
        errors = list(errors) + [
            WorkflowError(
                node="output",
                category="latex",
                message="LaTeX compilation failed. See compilation logs in state.",
                recoverable=True,
                details={
                    "engine": build_result.engine,
                    "output_dir": build_result.output_dir,
                    "tex_path": build_result.tex_path,
                },
            )
        ]

    if build_result.pdf_path:
        logger.info("OUTPUT: PDF generated at %s", build_result.pdf_path)
    else:
        logger.warning("OUTPUT: PDF not generated; LaTeX source saved at %s", build_result.tex_path)

    final_paper = None
    if reviewer_output:
        if isinstance(reviewer_output, dict):
            final_paper = reviewer_output.get("final_paper")
        elif hasattr(reviewer_output, "final_paper"):
            final_paper = reviewer_output.final_paper

    return {
        "status": ResearchStatus.COMPLETED,
        "errors": errors,
        "final_paper": final_paper,
        "output_dir": build_result.output_dir,
        "latex_tex_path": build_result.tex_path,
        "latex_pdf_path": build_result.pdf_path,
        "exported_artifacts": moved_exports,
        "latex_engine": build_result.engine,
        "latex_compile_stdout": build_result.compilation_stdout,
        "latex_compile_stderr": build_result.compilation_stderr,
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
    
    # Sprint 14: Data acquisition (no caching - fetches fresh external data)
    workflow.add_node("data_acquisition", data_acquisition_node)
    
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
    
    # Sprint 9: Fallback node for graceful degradation (no caching)
    workflow.add_node("fallback", fallback_node)
    
    # ==========================================================================
    # Add edges
    # ==========================================================================
    
    # Start -> Intake
    workflow.add_edge(START, "intake")
    
    # Intake -> Data Explorer (if data) or Literature Reviewer or Fallback
    workflow.add_conditional_edges(
        "intake",
        route_after_intake,
        ["data_explorer", "literature_reviewer", "fallback", END]
    )
    
    # Data Explorer -> Literature Reviewer or Fallback
    workflow.add_conditional_edges(
        "data_explorer",
        route_after_data_explorer,
        ["literature_reviewer", "fallback", END]
    )
    
    # Literature Reviewer -> Literature Synthesizer or Fallback
    workflow.add_conditional_edges(
        "literature_reviewer",
        route_after_literature_reviewer,
        ["literature_synthesizer", "fallback", END]
    )
    
    # Literature Synthesizer -> Gap Identifier or Fallback
    workflow.add_conditional_edges(
        "literature_synthesizer",
        route_after_synthesizer,
        ["gap_identifier", "fallback", END]
    )
    
    # Gap Identifier -> Planner or Fallback
    workflow.add_conditional_edges(
        "gap_identifier",
        route_after_gap_identifier,
        ["planner", "fallback", END]
    )
    
    # Planner -> Data Acquisition or Conceptual Synthesizer or Fallback (Sprint 14)
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        ["data_acquisition", "conceptual_synthesizer", "fallback", END]
    )
    
    # Data Acquisition -> Data Analyst or Conceptual Synthesizer or Fallback (Sprint 14)
    workflow.add_conditional_edges(
        "data_acquisition",
        route_after_data_acquisition,
        ["data_analyst", "conceptual_synthesizer", "fallback", END]
    )
    
    # Data Analyst -> Writer or Fallback (Sprint 6)
    workflow.add_conditional_edges(
        "data_analyst",
        route_after_analysis,
        ["writer", "fallback", END]
    )
    
    # Conceptual Synthesizer -> Writer or Fallback (Sprint 6)
    workflow.add_conditional_edges(
        "conceptual_synthesizer",
        route_after_analysis,
        ["writer", "fallback", END]
    )
    
    # Writer -> Reviewer or Fallback (Sprint 7)
    workflow.add_conditional_edges(
        "writer",
        route_after_writer,
        ["reviewer", "fallback", END]
    )
    
    # Reviewer -> Writer (revision) or Output (approval) or Fallback (Sprint 7)
    workflow.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        ["writer", "output", "fallback", END]
    )
    
    # Output -> END
    workflow.add_edge("output", END)
    
    # Fallback -> END (Sprint 9)
    workflow.add_edge("fallback", END)
    
    # ==========================================================================
    # Compile with configuration
    # ==========================================================================
    
    # Build compile kwargs
    compile_kwargs = {}
    
    if config.checkpointer:
        compile_kwargs["checkpointer"] = config.checkpointer
    
    if config.store:
        compile_kwargs["store"] = config.store
    
    # Explicitly set interrupts (empty list = no interrupts = auto-approve)
    compile_kwargs["interrupt_before"] = config.interrupt_before or []
    compile_kwargs["interrupt_after"] = config.interrupt_after or []
    
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
