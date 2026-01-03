"""Graph definitions and workflow assembly for GIA Agentic v2.

This module provides:
- Main research workflow graph factory
- Streaming utilities for real-time UI updates
- Time travel and debugging capabilities
- Subgraph compositions for modular design
"""

from src.graphs.research_workflow import (
    create_research_workflow,
    create_studio_workflow,
    create_production_workflow,
    WorkflowConfig,
    INTERRUPT_BEFORE_NODES,
    INTERRUPT_AFTER_NODES,
)
from src.graphs.streaming import (
    stream_research_workflow,
    StreamEvent,
    StreamMode,
)
from src.graphs.debug import (
    inspect_workflow_state,
    get_state_history,
    replay_from_checkpoint,
    WorkflowInspector,
)
from src.graphs.subgraphs import (
    create_literature_review_subgraph,
    create_analysis_subgraph,
)
from src.graphs.routers import (
    route_after_intake,
    route_after_data_explorer,
    route_after_literature_reviewer,
    route_after_synthesizer,
    route_after_gap_identifier,
    route_after_planner,
    route_by_research_type,
    route_after_analysis,
    route_after_writer,
    route_after_reviewer,
)

__all__ = [
    # Main workflow
    "create_research_workflow",
    "create_studio_workflow",
    "create_production_workflow",
    "WorkflowConfig",
    "INTERRUPT_BEFORE_NODES",
    "INTERRUPT_AFTER_NODES",
    # Streaming
    "stream_research_workflow",
    "StreamEvent",
    "StreamMode",
    # Debug
    "inspect_workflow_state",
    "get_state_history",
    "replay_from_checkpoint",
    "WorkflowInspector",
    # Subgraphs
    "create_literature_review_subgraph",
    "create_analysis_subgraph",
    # Routers
    "route_after_intake",
    "route_after_data_explorer",
    "route_after_literature_reviewer",
    "route_after_synthesizer",
    "route_after_gap_identifier",
    "route_after_planner",
    "route_by_research_type",
    "route_after_analysis",
    "route_after_writer",
    "route_after_reviewer",
]
