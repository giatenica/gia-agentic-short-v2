"""Error handling and recovery for GIA Agentic v2.

This module provides:
- Custom exception types for workflow errors
- RetryPolicy configurations for different error scenarios
- Error handlers for graceful degradation
- Recovery strategies for partial workflow completion
"""

from src.errors.exceptions import (
    GIAError,
    WorkflowError,
    NodeExecutionError,
    ToolExecutionError,
    APIError,
    RateLimitError,
    ContextOverflowError,
    DataValidationError,
    SearchError,
    LiteratureSearchError,
    AnalysisError,
    WritingError,
    ReviewError,
)
from src.errors.policies import (
    create_api_retry_policy,
    create_search_retry_policy,
    create_analysis_retry_policy,
    DEFAULT_RETRY_POLICY,
    AGGRESSIVE_RETRY_POLICY,
    CONSERVATIVE_RETRY_POLICY,
)
from src.errors.handlers import (
    handle_tool_error,
    handle_node_error,
    handle_api_error,
    create_error_response,
    log_error_with_context,
    ErrorHandler,
)
from src.errors.recovery import (
    RecoveryStrategy,
    RecoveryAction,
    determine_recovery_strategy,
    execute_recovery,
    can_continue_workflow,
    get_partial_output,
    create_fallback_content,
)

__all__ = [
    # Exceptions
    "GIAError",
    "WorkflowError",
    "NodeExecutionError",
    "ToolExecutionError",
    "APIError",
    "RateLimitError",
    "ContextOverflowError",
    "DataValidationError",
    "SearchError",
    "LiteratureSearchError",
    "AnalysisError",
    "WritingError",
    "ReviewError",
    # Policies
    "create_api_retry_policy",
    "create_search_retry_policy",
    "create_analysis_retry_policy",
    "DEFAULT_RETRY_POLICY",
    "AGGRESSIVE_RETRY_POLICY",
    "CONSERVATIVE_RETRY_POLICY",
    # Handlers
    "handle_tool_error",
    "handle_node_error",
    "handle_api_error",
    "create_error_response",
    "log_error_with_context",
    "ErrorHandler",
    # Recovery
    "RecoveryStrategy",
    "RecoveryAction",
    "determine_recovery_strategy",
    "execute_recovery",
    "can_continue_workflow",
    "get_partial_output",
    "create_fallback_content",
]
