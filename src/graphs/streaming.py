"""Streaming utilities for the research workflow.

This module provides async streaming capabilities for real-time UI updates
during workflow execution.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Callable

from langgraph.graph import StateGraph

from src.state.schema import WorkflowState

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================


class StreamMode(str, Enum):
    """Stream modes for workflow execution."""
    UPDATES = "updates"      # Node state updates
    MESSAGES = "messages"    # LLM token streaming
    CUSTOM = "custom"        # Custom events from nodes
    VALUES = "values"        # Full state values
    DEBUG = "debug"          # Debug information


@dataclass
class StreamEvent:
    """Represents a streaming event from the workflow.
    
    Attributes:
        mode: Type of event (updates, messages, custom, etc.)
        node: Name of the node that produced this event
        data: Event payload
        metadata: Additional event metadata
    """
    mode: StreamMode
    node: str | None
    data: Any
    metadata: dict | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mode": self.mode.value,
            "node": self.node,
            "data": self.data,
            "metadata": self.metadata or {},
        }


# =============================================================================
# Progress Tracking
# =============================================================================

# Node to human-readable progress messages
NODE_PROGRESS_MESSAGES = {
    "intake": "Processing research intake form...",
    "data_explorer": "Analyzing uploaded data files...",
    "literature_reviewer": "Searching academic literature...",
    "literature_synthesizer": "Synthesizing literature findings...",
    "gap_identifier": "Identifying research gaps...",
    "planner": "Creating research plan...",
    "data_analyst": "Performing data analysis...",
    "conceptual_synthesizer": "Building conceptual framework...",
    "writer": "Writing paper sections...",
    "reviewer": "Reviewing paper quality...",
    "output": "Preparing final output...",
    "fallback": "Generating partial output (error recovery)...",
}


def get_progress_message(node: str) -> str:
    """Get human-readable progress message for a node."""
    return NODE_PROGRESS_MESSAGES.get(node, f"Processing {node}...")


# =============================================================================
# Streaming Functions
# =============================================================================


async def stream_research_workflow(
    workflow: StateGraph,
    initial_state: dict | WorkflowState,
    thread_id: str,
    stream_modes: list[StreamMode] | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Stream workflow execution with real-time updates.
    
    This function provides an async generator that yields events as the
    workflow executes. It's designed for use with real-time UIs.
    
    Args:
        workflow: Compiled workflow graph
        initial_state: Initial state or form data
        thread_id: Unique identifier for this execution thread
        stream_modes: Which stream modes to enable (default: updates + messages)
        
    Yields:
        StreamEvent objects containing workflow progress
        
    Example:
        async for event in stream_research_workflow(workflow, data, "thread-123"):
            if event.mode == StreamMode.UPDATES:
                update_ui_state(event.node, event.data)
            elif event.mode == StreamMode.MESSAGES:
                append_token(event.data)
    """
    if stream_modes is None:
        stream_modes = [StreamMode.UPDATES, StreamMode.MESSAGES]
    
    # Convert to LangGraph stream mode strings
    mode_strings = [mode.value for mode in stream_modes]
    
    config = {"configurable": {"thread_id": thread_id}}
    
    logger.info(f"Starting workflow stream for thread {thread_id}")
    
    current_node = None
    
    try:
        async for event in workflow.astream(
            initial_state,
            config=config,
            stream_mode=mode_strings,
        ):
            # Parse event based on structure
            if isinstance(event, tuple) and len(event) == 2:
                mode_str, data = event
                mode = StreamMode(mode_str) if mode_str in [m.value for m in StreamMode] else StreamMode.CUSTOM
                
                # Extract node name from data if available
                node = None
                if isinstance(data, dict):
                    node = data.get("node") or data.get("langgraph_node")
                    if node and node != current_node:
                        current_node = node
                        # Emit progress message for new node
                        yield StreamEvent(
                            mode=StreamMode.CUSTOM,
                            node=node,
                            data={
                                "type": "progress",
                                "message": get_progress_message(node),
                            }
                        )
                
                yield StreamEvent(
                    mode=mode,
                    node=node or current_node,
                    data=data,
                )
            else:
                # Handle other event formats
                yield StreamEvent(
                    mode=StreamMode.UPDATES,
                    node=current_node,
                    data=event,
                )
                
    except Exception as e:
        logger.error(f"Workflow stream error: {e}")
        yield StreamEvent(
            mode=StreamMode.CUSTOM,
            node=current_node,
            data={
                "type": "error",
                "message": str(e),
            },
            metadata={"error_type": type(e).__name__}
        )
        raise


async def stream_with_progress(
    workflow: StateGraph,
    initial_state: dict | WorkflowState,
    thread_id: str,
    progress_callback: Callable | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Stream workflow with progress callbacks.
    
    Enhanced streaming that tracks node transitions and calls a progress
    callback with percentage complete.
    
    Args:
        workflow: Compiled workflow graph
        initial_state: Initial state or form data
        thread_id: Thread identifier
        progress_callback: Optional callback(node, percentage, message)
        
    Yields:
        StreamEvent objects
    """
    from src.graphs.research_workflow import WORKFLOW_NODES
    
    total_nodes = len(WORKFLOW_NODES)
    completed_nodes = set()
    
    async for event in stream_research_workflow(
        workflow,
        initial_state,
        thread_id,
        [StreamMode.UPDATES, StreamMode.MESSAGES, StreamMode.CUSTOM],
    ):
        # Track node completion
        if event.node and event.node not in completed_nodes:
            completed_nodes.add(event.node)
            percentage = len(completed_nodes) / total_nodes * 100
            
            if progress_callback:
                progress_callback(
                    event.node,
                    percentage,
                    get_progress_message(event.node)
                )
        
        yield event


# =============================================================================
# UI Integration Helpers
# =============================================================================


def format_for_sse(event: StreamEvent) -> str:
    """
    Format a StreamEvent for Server-Sent Events.
    
    Args:
        event: StreamEvent to format
        
    Returns:
        SSE-formatted string
    """
    import json
    
    data = json.dumps(event.to_dict())
    return f"data: {data}\n\n"


def format_for_websocket(event: StreamEvent) -> dict:
    """
    Format a StreamEvent for WebSocket transmission.
    
    Args:
        event: StreamEvent to format
        
    Returns:
        Dictionary for JSON serialization
    """
    return event.to_dict()


async def collect_stream_to_state(
    workflow: StateGraph,
    initial_state: dict | WorkflowState,
    thread_id: str,
) -> WorkflowState:
    """
    Execute workflow and collect final state.
    
    Convenience function that runs the streaming workflow and returns
    the final state once complete.
    
    Args:
        workflow: Compiled workflow graph
        initial_state: Initial state
        thread_id: Thread identifier
        
    Returns:
        Final workflow state
    """
    final_state = None
    
    async for event in stream_research_workflow(
        workflow,
        initial_state,
        thread_id,
        [StreamMode.VALUES],
    ):
        if event.mode == StreamMode.VALUES and event.data:
            final_state = event.data
    
    return final_state
