"""Graph definitions for LangGraph Studio.

Note: LangGraph Studio/API handles persistence automatically.
Don't pass checkpointer/store here - the platform provides these.

This module now imports the workflow from src/graphs for modularity.
The workflow definition has been refactored into:
- src/graphs/research_workflow.py - Main workflow factory
- src/graphs/routers.py - Routing functions
- src/graphs/streaming.py - Streaming utilities
- src/graphs/debug.py - Time travel and debugging
- src/graphs/subgraphs.py - Modular subgraph compositions
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import create_react_agent, create_research_agent, create_data_analyst_agent
from src.graphs.research_workflow import (
    create_research_workflow,
    create_studio_workflow,
    WorkflowConfig,
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
from src.cache import get_cache, get_cache_policy
from src.config import settings


# =============================================================================
# Agent Instances for Studio
# =============================================================================

# Create agent instances for Studio
# Note: Don't pass checkpointer/store - LangGraph API handles persistence
react_agent = create_react_agent()
research_agent = create_research_agent()
data_analyst_agent = create_data_analyst_agent()


# =============================================================================
# Research Workflow for Studio
# =============================================================================

# Create the workflow instance for Studio using the factory
# Studio manages its own persistence, caching is enabled by default
research_workflow = create_studio_workflow()


# =============================================================================
# Alternative: Custom Configuration Example
# =============================================================================

def create_custom_workflow():
    """
    Example of creating a workflow with custom configuration.
    
    This demonstrates how to use WorkflowConfig for different scenarios.
    """
    from langgraph.checkpoint.memory import MemorySaver
    
    config = WorkflowConfig(
        checkpointer=MemorySaver(),
        interrupt_before=["gap_identifier", "planner"],
        interrupt_after=["reviewer"],
        enable_caching=True,
        debug=True,
    )
    return create_research_workflow(config)


# For debugging - expose the config
__workflow_config__ = WorkflowConfig(
    checkpointer=None,
    store=None,
    interrupt_before=[],
    interrupt_after=[],
    enable_caching=settings.cache_enabled,
)
