"""Error handlers for graceful error management.

This module provides error handling functions for tools, nodes,
and API calls, enabling graceful degradation and proper logging.
"""

import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Callable
from functools import wraps

from src.errors.exceptions import (
    GIAError,
    NodeExecutionError,
    ToolExecutionError,
    APIError,
    RateLimitError,
    ContextOverflowError,
    DataValidationError,
    SearchError,
    AnalysisError,
    WritingError,
)
from src.state.models import WorkflowError as WorkflowErrorModel

logger = logging.getLogger(__name__)


# =============================================================================
# Error Response Creation
# =============================================================================


def create_error_response(
    error: Exception,
    node: str | None = None,
    include_traceback: bool = False,
) -> dict[str, Any]:
    """Create a standardized error response dictionary.
    
    Args:
        error: The exception that occurred
        node: Node where the error occurred
        include_traceback: Whether to include full traceback
        
    Returns:
        Standardized error response dictionary
    """
    # Extract info from GIAError types
    if isinstance(error, GIAError):
        response = {
            "error_type": error.__class__.__name__,
            "message": error.message,
            "details": error.details,
            "recoverable": error.recoverable,
        }
    else:
        response = {
            "error_type": error.__class__.__name__,
            "message": str(error),
            "details": {},
            "recoverable": True,  # Assume recoverable for unknown errors
        }
    
    if node:
        response["node"] = node
    
    response["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    if include_traceback:
        response["traceback"] = traceback.format_exc()
    
    return response


def create_workflow_error_model(
    error: Exception,
    node: str,
    category: str | None = None,
) -> WorkflowErrorModel:
    """Create a WorkflowError model from an exception.
    
    Args:
        error: The exception that occurred
        node: Node where the error occurred
        category: Error category (auto-detected if not provided)
        
    Returns:
        WorkflowErrorModel for state tracking
    """
    # Auto-detect category
    if category is None:
        category = _detect_error_category(error)
    
    # Extract details
    if isinstance(error, GIAError):
        message = error.message
        recoverable = error.recoverable
        details = error.details
    else:
        message = str(error)
        recoverable = True
        details = {"original_type": error.__class__.__name__}
    
    return WorkflowErrorModel(
        node=node,
        category=category,
        message=message,
        recoverable=recoverable,
        details=details,
    )


def _detect_error_category(error: Exception) -> str:
    """Detect error category from exception type.
    
    Args:
        error: The exception
        
    Returns:
        Category string
    """
    if isinstance(error, RateLimitError):
        return "rate_limit"
    elif isinstance(error, ContextOverflowError):
        return "context_overflow"
    elif isinstance(error, APIError):
        return "api_error"
    elif isinstance(error, SearchError):
        return "search_error"
    elif isinstance(error, DataValidationError):
        return "validation_error"
    elif isinstance(error, AnalysisError):
        return "analysis_error"
    elif isinstance(error, WritingError):
        return "writing_error"
    elif isinstance(error, NodeExecutionError):
        return "node_error"
    elif isinstance(error, ToolExecutionError):
        return "tool_error"
    elif isinstance(error, GIAError):
        return "gia_error"
    elif isinstance(error, (TimeoutError, ConnectionError)):
        return "connection_error"
    else:
        return "unknown_error"


# =============================================================================
# Error Logging
# =============================================================================


def log_error_with_context(
    error: Exception,
    node: str | None = None,
    context: dict[str, Any] | None = None,
    level: int = logging.ERROR,
) -> None:
    """Log an error with full context.
    
    Args:
        error: The exception that occurred
        node: Node where the error occurred
        context: Additional context to log
        level: Logging level (default: ERROR)
    """
    # Build log message
    parts = [f"Error: {error.__class__.__name__}: {error}"]
    
    if node:
        parts.append(f"Node: {node}")
    
    if isinstance(error, GIAError):
        if error.details:
            parts.append(f"Details: {error.details}")
        parts.append(f"Recoverable: {error.recoverable}")
    
    if context:
        parts.append(f"Context: {context}")
    
    message = " | ".join(parts)
    
    logger.log(level, message)
    
    # Log traceback at debug level
    logger.debug(f"Traceback:\n{traceback.format_exc()}")


# =============================================================================
# Tool Error Handling
# =============================================================================


def handle_tool_error(error: Exception, tool_name: str | None = None) -> str:
    """Handle a tool error and return a user-friendly message.
    
    This function is designed to be used with LangGraph's ToolNode
    handle_tool_errors parameter.
    
    Args:
        error: The exception that occurred
        tool_name: Name of the tool that failed
        
    Returns:
        User-friendly error message string
    """
    log_error_with_context(error, node=f"tool:{tool_name}")
    
    # Rate limit errors
    if isinstance(error, RateLimitError):
        retry_msg = ""
        if error.retry_after:
            retry_msg = f" Try again in {error.retry_after:.0f} seconds."
        return f"Service rate limited ({error.service}).{retry_msg}"
    
    # Context overflow
    if isinstance(error, ContextOverflowError):
        return "Content too large for processing. Try with less text."
    
    # Search errors
    if isinstance(error, SearchError):
        source = getattr(error, 'source', 'search service')
        return f"Search failed on {source}. Results may be incomplete."
    
    # Analysis errors
    if isinstance(error, AnalysisError):
        return f"Analysis failed: {error.message}"
    
    # Generic API errors
    if isinstance(error, APIError):
        return f"External service error ({error.service}). Please try again."
    
    # Connection errors
    if isinstance(error, (TimeoutError, ConnectionError)):
        return "Connection issue. Please try again."
    
    # Tool execution errors
    if isinstance(error, ToolExecutionError):
        return f"Tool error: {error.message}"
    
    # Default fallback
    return f"An error occurred: {str(error)[:100]}"


# =============================================================================
# Node Error Handling
# =============================================================================


def handle_node_error(
    error: Exception,
    node: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Handle a node error and return state updates.
    
    This function processes node errors and returns the appropriate
    state updates for error tracking.
    
    Args:
        error: The exception that occurred
        node: Node where the error occurred
        state: Current workflow state
        
    Returns:
        Dictionary of state updates
    """
    log_error_with_context(error, node=node)
    
    # Create workflow error model
    error_model = create_workflow_error_model(error, node)
    
    # Get existing errors
    existing_errors = state.get("errors", [])
    
    # Build state updates
    updates: dict[str, Any] = {
        "errors": existing_errors + [error_model],
    }
    
    # Add error-specific updates
    if isinstance(error, ContextOverflowError):
        # Flag for content reduction
        updates["_needs_content_reduction"] = True
    
    if isinstance(error, RateLimitError):
        # Flag for rate limiting
        updates["_rate_limited"] = True
        if error.retry_after:
            updates["_retry_after"] = error.retry_after
    
    return updates


def handle_api_error(
    error: Exception,
    service: str,
    operation: str,
) -> str:
    """Handle an API error and return a user-friendly message.
    
    Args:
        error: The exception that occurred
        service: Name of the API service
        operation: Description of the operation being performed
        
    Returns:
        User-friendly error message
    """
    log_error_with_context(
        error,
        context={"service": service, "operation": operation}
    )
    
    # Rate limit
    if isinstance(error, RateLimitError):
        return f"{service} rate limit reached during {operation}. Will retry."
    
    # Context overflow
    if isinstance(error, ContextOverflowError):
        return f"Content too large for {operation}. Reducing content."
    
    # Connection issues
    if isinstance(error, (TimeoutError, ConnectionError)):
        return f"Connection to {service} failed during {operation}. Retrying."
    
    # Generic API error
    if isinstance(error, APIError):
        return f"{service} error during {operation}: {error.message}"
    
    return f"Error during {operation}: {str(error)[:100]}"


# =============================================================================
# Error Handler Class
# =============================================================================


class ErrorHandler:
    """Centralized error handler for workflow operations.
    
    Provides consistent error handling across all nodes and tools,
    with configurable behavior for different error types.
    
    Example:
        handler = ErrorHandler(node="literature_reviewer")
        
        try:
            result = do_something()
        except Exception as e:
            state_updates = handler.handle(e, current_state)
    """
    
    def __init__(
        self,
        node: str,
        max_errors: int = 3,
        fail_on_unrecoverable: bool = True,
    ):
        """Initialize the error handler.
        
        Args:
            node: Name of the node using this handler
            max_errors: Maximum errors before failing
            fail_on_unrecoverable: Whether to fail on unrecoverable errors
        """
        self.node = node
        self.max_errors = max_errors
        self.fail_on_unrecoverable = fail_on_unrecoverable
        self._error_count = 0
    
    def handle(
        self,
        error: Exception,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle an error and return state updates.
        
        Args:
            error: The exception that occurred
            state: Current workflow state
            
        Returns:
            Dictionary of state updates
        """
        self._error_count += 1
        
        # Get state updates
        updates = handle_node_error(error, self.node, state)
        
        # Check if we should fail
        if self.should_fail(error, state):
            updates["_should_fallback"] = True
        
        return updates
    
    def should_fail(
        self,
        error: Exception,
        state: dict[str, Any],
    ) -> bool:
        """Determine if the workflow should fail/fallback.
        
        Args:
            error: The exception that occurred
            state: Current workflow state
            
        Returns:
            True if should fail/fallback
        """
        # Check unrecoverable errors
        if self.fail_on_unrecoverable:
            if isinstance(error, GIAError) and not error.recoverable:
                logger.warning(f"Unrecoverable error in {self.node}")
                return True
        
        # Check error count
        total_errors = len(state.get("errors", [])) + 1
        if total_errors >= self.max_errors:
            logger.warning(
                f"Max errors ({self.max_errors}) reached in {self.node}"
            )
            return True
        
        return False
    
    def reset(self) -> None:
        """Reset the error counter."""
        self._error_count = 0


# =============================================================================
# Decorator for Error Handling
# =============================================================================


def with_error_handling(
    node: str,
    max_errors: int = 3,
):
    """Decorator to add error handling to a node function.
    
    Args:
        node: Name of the node
        max_errors: Maximum errors before failing
        
    Returns:
        Decorated function
        
    Example:
        @with_error_handling("literature_reviewer")
        def literature_reviewer_node(state):
            # ... node logic
            return updates
    """
    def decorator(func: Callable) -> Callable:
        handler = ErrorHandler(node=node, max_errors=max_errors)
        
        @wraps(func)
        def wrapper(state: dict[str, Any], *args, **kwargs):
            try:
                return func(state, *args, **kwargs)
            except Exception as e:
                return handler.handle(e, state)
        
        return wrapper
    return decorator
