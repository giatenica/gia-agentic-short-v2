"""Time travel and debugging utilities for the research workflow.

This module provides tools for inspecting workflow state, navigating
state history, and replaying from checkpoints.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator

from langgraph.graph import StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.state.schema import WorkflowState

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class StateSnapshot:
    """Snapshot of workflow state at a point in time.
    
    Attributes:
        checkpoint_id: Unique identifier for this checkpoint
        thread_id: Thread this checkpoint belongs to
        created_at: When this checkpoint was created
        node: Current node at this checkpoint
        next_nodes: Next node(s) to execute
        values: State values at this checkpoint
        metadata: Additional checkpoint metadata
    """
    checkpoint_id: str
    thread_id: str
    created_at: datetime | None
    node: str | None
    next_nodes: list[str]
    values: dict
    metadata: dict


@dataclass
class WorkflowStatus:
    """Current status of a workflow execution.
    
    Attributes:
        thread_id: Thread identifier
        current_node: Node currently executing (or last executed)
        next_nodes: Node(s) scheduled to execute next
        status: Workflow status from state
        is_interrupted: Whether workflow is paused for human input
        checkpoint_count: Number of checkpoints saved
        error: Error message if workflow failed
    """
    thread_id: str
    current_node: str | None
    next_nodes: list[str]
    status: str | None
    is_interrupted: bool
    checkpoint_count: int
    error: str | None


# =============================================================================
# State Inspection
# =============================================================================


def inspect_workflow_state(
    workflow: StateGraph,
    thread_id: str,
) -> WorkflowStatus:
    """
    Inspect the current state of a workflow execution.
    
    Args:
        workflow: Compiled workflow graph
        thread_id: Thread identifier to inspect
        
    Returns:
        WorkflowStatus with current execution state
        
    Example:
        status = inspect_workflow_state(workflow, "thread-123")
        print(f"Current node: {status.current_node}")
        print(f"Is interrupted: {status.is_interrupted}")
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state_snapshot = workflow.get_state(config)
        
        if state_snapshot is None:
            return WorkflowStatus(
                thread_id=thread_id,
                current_node=None,
                next_nodes=[],
                status=None,
                is_interrupted=False,
                checkpoint_count=0,
                error="No state found for thread",
            )
        
        # Extract values
        values = state_snapshot.values or {}
        
        # Get status from state
        status = values.get("status")
        if hasattr(status, "value"):
            status = status.value
        
        # Check for errors
        errors = values.get("errors", [])
        error_msg = None
        if errors:
            if isinstance(errors[0], str):
                error_msg = errors[0]
            elif hasattr(errors[0], "message"):
                error_msg = errors[0].message
        
        # Count checkpoints
        checkpoint_count = len(list(workflow.get_state_history(config)))
        
        # Check for interrupts
        is_interrupted = bool(getattr(state_snapshot, "next", None))
        
        return WorkflowStatus(
            thread_id=thread_id,
            current_node=getattr(state_snapshot, "next", [None])[0] if getattr(state_snapshot, "next", None) else None,
            next_nodes=list(getattr(state_snapshot, "next", []) or []),
            status=status,
            is_interrupted=is_interrupted,
            checkpoint_count=checkpoint_count,
            error=error_msg,
        )
        
    except Exception as e:
        logger.error(f"Error inspecting state: {e}")
        return WorkflowStatus(
            thread_id=thread_id,
            current_node=None,
            next_nodes=[],
            status=None,
            is_interrupted=False,
            checkpoint_count=0,
            error=str(e),
        )


def get_state_history(
    workflow: StateGraph,
    thread_id: str,
    limit: int | None = None,
) -> list[StateSnapshot]:
    """
    Get the state history for a workflow execution.
    
    Args:
        workflow: Compiled workflow graph
        thread_id: Thread identifier
        limit: Maximum number of snapshots to return (most recent first)
        
    Returns:
        List of StateSnapshot objects, most recent first
        
    Example:
        history = get_state_history(workflow, "thread-123", limit=10)
        for snapshot in history:
            print(f"{snapshot.created_at}: {snapshot.node}")
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    snapshots = []
    
    try:
        for i, state in enumerate(workflow.get_state_history(config)):
            if limit and i >= limit:
                break
            
            # Extract checkpoint info
            checkpoint_config = getattr(state, "config", {})
            checkpoint_id = checkpoint_config.get("configurable", {}).get("checkpoint_id", f"checkpoint-{i}")
            
            # Get created_at if available
            created_at = getattr(state, "created_at", None)
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except ValueError:
                    created_at = None
            
            # Get current/next node
            next_nodes = list(getattr(state, "next", []) or [])
            current_node = next_nodes[0] if next_nodes else None
            
            snapshots.append(StateSnapshot(
                checkpoint_id=checkpoint_id,
                thread_id=thread_id,
                created_at=created_at,
                node=current_node,
                next_nodes=next_nodes,
                values=state.values or {},
                metadata=getattr(state, "metadata", {}) or {},
            ))
            
    except Exception as e:
        logger.error(f"Error getting state history: {e}")
    
    return snapshots


def get_state_at_node(
    workflow: StateGraph,
    thread_id: str,
    node_name: str,
) -> StateSnapshot | None:
    """
    Get the state snapshot when a specific node was active.
    
    Args:
        workflow: Compiled workflow graph
        thread_id: Thread identifier
        node_name: Name of the node to find
        
    Returns:
        StateSnapshot at that node, or None if not found
    """
    for snapshot in get_state_history(workflow, thread_id):
        if snapshot.node == node_name:
            return snapshot
    return None


# =============================================================================
# Replay and Time Travel
# =============================================================================


def replay_from_checkpoint(
    workflow: StateGraph,
    thread_id: str,
    checkpoint_index: int = 0,
    new_thread_id: str | None = None,
) -> dict | None:
    """
    Replay workflow execution from a specific checkpoint.
    
    This enables "time travel" debugging by restarting execution from
    a previous state.
    
    Args:
        workflow: Compiled workflow graph
        thread_id: Original thread identifier
        checkpoint_index: Index in history (0 = most recent)
        new_thread_id: Optional new thread ID for the replay
        
    Returns:
        Final state after replay, or None if failed
        
    Example:
        # Replay from 2nd most recent checkpoint
        result = replay_from_checkpoint(workflow, "thread-123", checkpoint_index=1)
    """
    history = get_state_history(workflow, thread_id)
    
    if checkpoint_index >= len(history):
        logger.error(f"Checkpoint index {checkpoint_index} out of range (max: {len(history) - 1})")
        return None
    
    snapshot = history[checkpoint_index]
    
    # Use new thread ID or create one based on original
    target_thread = new_thread_id or f"{thread_id}-replay-{checkpoint_index}"
    config = {"configurable": {"thread_id": target_thread}}
    
    logger.info(f"Replaying from checkpoint {checkpoint_index} (node: {snapshot.node})")
    
    try:
        # Start from the snapshot's state
        result = workflow.invoke(snapshot.values, config)
        return result
    except Exception as e:
        logger.error(f"Replay failed: {e}")
        return None


def fork_from_state(
    workflow: StateGraph,
    thread_id: str,
    new_thread_id: str,
    state_modifications: dict | None = None,
) -> dict | None:
    """
    Fork a workflow from its current state with optional modifications.
    
    Creates a new execution branch from the current state, optionally
    modifying state values before continuing.
    
    Args:
        workflow: Compiled workflow graph
        thread_id: Source thread identifier
        new_thread_id: New thread identifier for the fork
        state_modifications: Optional state changes to apply
        
    Returns:
        Final state after fork execution
        
    Example:
        # Fork and modify the research question
        result = fork_from_state(
            workflow, 
            "thread-123", 
            "thread-123-fork",
            {"refined_query": "New research question..."}
        )
    """
    status = inspect_workflow_state(workflow, thread_id)
    
    if status.error:
        logger.error(f"Cannot fork from errored state: {status.error}")
        return None
    
    # Get current state values
    history = get_state_history(workflow, thread_id, limit=1)
    if not history:
        logger.error("No state to fork from")
        return None
    
    current_values = history[0].values.copy()
    
    # Apply modifications
    if state_modifications:
        current_values.update(state_modifications)
    
    # Execute on new thread
    config = {"configurable": {"thread_id": new_thread_id}}
    
    try:
        result = workflow.invoke(current_values, config)
        return result
    except Exception as e:
        logger.error(f"Fork failed: {e}")
        return None


# =============================================================================
# Workflow Inspector Class
# =============================================================================


class WorkflowInspector:
    """
    High-level interface for workflow inspection and debugging.
    
    Provides a stateful interface for exploring workflow execution
    history and performing debugging operations.
    
    Example:
        inspector = WorkflowInspector(workflow, "thread-123")
        
        # Check current status
        print(inspector.status)
        
        # Explore history
        for snapshot in inspector.history():
            print(f"{snapshot.node}: {snapshot.values.get('status')}")
        
        # Replay from a checkpoint
        result = inspector.replay(checkpoint_index=2)
    """
    
    def __init__(self, workflow: StateGraph, thread_id: str):
        """
        Initialize inspector for a specific workflow thread.
        
        Args:
            workflow: Compiled workflow graph
            thread_id: Thread to inspect
        """
        self.workflow = workflow
        self.thread_id = thread_id
    
    @property
    def status(self) -> WorkflowStatus:
        """Get current workflow status."""
        return inspect_workflow_state(self.workflow, self.thread_id)
    
    @property
    def current_state(self) -> dict | None:
        """Get current state values."""
        history = get_state_history(self.workflow, self.thread_id, limit=1)
        return history[0].values if history else None
    
    def history(self, limit: int | None = None) -> list[StateSnapshot]:
        """Get state history."""
        return get_state_history(self.workflow, self.thread_id, limit)
    
    def state_at_node(self, node_name: str) -> StateSnapshot | None:
        """Get state when a specific node was active."""
        return get_state_at_node(self.workflow, self.thread_id, node_name)
    
    def replay(
        self,
        checkpoint_index: int = 0,
        new_thread_id: str | None = None,
    ) -> dict | None:
        """Replay from a checkpoint."""
        return replay_from_checkpoint(
            self.workflow,
            self.thread_id,
            checkpoint_index,
            new_thread_id,
        )
    
    def fork(
        self,
        new_thread_id: str,
        modifications: dict | None = None,
    ) -> dict | None:
        """Fork to a new thread with optional modifications."""
        return fork_from_state(
            self.workflow,
            self.thread_id,
            new_thread_id,
            modifications,
        )
    
    def print_history(self, limit: int = 10) -> None:
        """Print formatted history to console."""
        print(f"\n=== Workflow History: {self.thread_id} ===\n")
        
        for i, snapshot in enumerate(self.history(limit)):
            status = snapshot.values.get("status", "unknown")
            if hasattr(status, "value"):
                status = status.value
            
            print(f"[{i}] Node: {snapshot.node or 'N/A'}")
            print(f"    Status: {status}")
            if snapshot.created_at:
                print(f"    Time: {snapshot.created_at}")
            print()
    
    def print_state(self, include_full: bool = False) -> None:
        """Print current state to console."""
        status = self.status
        
        print(f"\n=== Workflow Status: {self.thread_id} ===\n")
        print(f"Current Node: {status.current_node or 'N/A'}")
        print(f"Next Nodes: {status.next_nodes or 'None'}")
        print(f"Status: {status.status or 'unknown'}")
        print(f"Interrupted: {status.is_interrupted}")
        print(f"Checkpoints: {status.checkpoint_count}")
        
        if status.error:
            print(f"Error: {status.error}")
        
        if include_full and self.current_state:
            print("\n--- Full State ---")
            for key, value in self.current_state.items():
                if key not in ["messages"]:  # Skip verbose fields
                    print(f"  {key}: {type(value).__name__}")
