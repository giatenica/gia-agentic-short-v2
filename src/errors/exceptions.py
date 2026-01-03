"""Custom exception types for GIA Agentic v2.

This module defines a hierarchy of exceptions for categorizing errors
throughout the research workflow, enabling targeted error handling
and recovery strategies.
"""

from typing import Any


class GIAError(Exception):
    """Base exception for all GIA Agentic errors.
    
    Attributes:
        message: Human-readable error description
        details: Additional error context
        recoverable: Whether the workflow can recover from this error
    """
    
    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
    ):
        self.message = message
        self.details = details or {}
        self.recoverable = recoverable
        super().__init__(message)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
        }


# =============================================================================
# Workflow-Level Errors
# =============================================================================


class WorkflowError(GIAError):
    """Error at the workflow orchestration level.
    
    Raised when there are issues with workflow state, routing,
    or inter-node communication.
    """
    
    def __init__(
        self,
        message: str,
        node: str | None = None,
        state_key: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
    ):
        details = details or {}
        if node:
            details["node"] = node
        if state_key:
            details["state_key"] = state_key
        super().__init__(message, details, recoverable)
        self.node = node
        self.state_key = state_key


class NodeExecutionError(GIAError):
    """Error during node execution.
    
    Raised when a node fails to complete its task.
    """
    
    def __init__(
        self,
        message: str,
        node: str,
        phase: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
    ):
        details = details or {}
        details["node"] = node
        if phase:
            details["phase"] = phase
        super().__init__(message, details, recoverable)
        self.node = node
        self.phase = phase


# =============================================================================
# Tool Execution Errors
# =============================================================================


class ToolExecutionError(GIAError):
    """Error during tool execution.
    
    Raised when a tool (search, analysis, etc.) fails.
    """
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        tool_input: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
    ):
        details = details or {}
        details["tool_name"] = tool_name
        if tool_input:
            details["tool_input"] = tool_input
        super().__init__(message, details, recoverable)
        self.tool_name = tool_name
        self.tool_input = tool_input


# =============================================================================
# API-Related Errors
# =============================================================================


class APIError(GIAError):
    """Error from external API calls.
    
    Base class for API-related errors including rate limits,
    authentication failures, and service unavailability.
    """
    
    def __init__(
        self,
        message: str,
        service: str,
        status_code: int | None = None,
        response_body: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
    ):
        details = details or {}
        details["service"] = service
        if status_code:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body[:500]  # Truncate
        super().__init__(message, details, recoverable)
        self.service = service
        self.status_code = status_code


class RateLimitError(APIError):
    """Rate limit exceeded on an external API.
    
    Attributes:
        retry_after: Seconds to wait before retrying (if provided)
    """
    
    def __init__(
        self,
        message: str,
        service: str,
        retry_after: float | None = None,
        status_code: int = 429,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(
            message,
            service=service,
            status_code=status_code,
            details=details,
            recoverable=True,  # Rate limits are always recoverable
        )
        self.retry_after = retry_after


class ContextOverflowError(APIError):
    """Context window exceeded for LLM API.
    
    Raised when the input is too large for the model's context window.
    """
    
    def __init__(
        self,
        message: str,
        service: str = "anthropic",
        token_count: int | None = None,
        max_tokens: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        if token_count:
            details["token_count"] = token_count
        if max_tokens:
            details["max_tokens"] = max_tokens
        super().__init__(
            message,
            service=service,
            status_code=400,
            details=details,
            recoverable=True,  # Can recover by reducing content
        )
        self.token_count = token_count
        self.max_tokens = max_tokens


# =============================================================================
# Domain-Specific Errors
# =============================================================================


class DataValidationError(GIAError):
    """Error validating input data.
    
    Raised when form data, uploaded files, or intermediate
    state fails validation.
    """
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        constraint: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
    ):
        details = details or {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)[:100]  # Truncate
        if constraint:
            details["constraint"] = constraint
        super().__init__(message, details, recoverable)
        self.field = field
        self.constraint = constraint


class SearchError(GIAError):
    """Error during search operations.
    
    Base class for literature and web search errors.
    """
    
    def __init__(
        self,
        message: str,
        query: str | None = None,
        source: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
    ):
        details = details or {}
        if query:
            details["query"] = query[:200]  # Truncate
        if source:
            details["source"] = source
        super().__init__(message, details, recoverable)
        self.query = query
        self.source = source


class LiteratureSearchError(SearchError):
    """Error during academic literature search.
    
    Raised when searches on Semantic Scholar, arXiv, or
    other academic databases fail.
    """
    
    def __init__(
        self,
        message: str,
        query: str | None = None,
        source: str = "unknown",
        papers_found: int = 0,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
    ):
        details = details or {}
        details["papers_found"] = papers_found
        super().__init__(message, query, source, details, recoverable)
        self.papers_found = papers_found


class AnalysisError(GIAError):
    """Error during data analysis.
    
    Raised when statistical analysis, data transformation,
    or interpretation fails.
    """
    
    def __init__(
        self,
        message: str,
        analysis_type: str | None = None,
        dataset: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
    ):
        details = details or {}
        if analysis_type:
            details["analysis_type"] = analysis_type
        if dataset:
            details["dataset"] = dataset
        super().__init__(message, details, recoverable)
        self.analysis_type = analysis_type
        self.dataset = dataset


class WritingError(GIAError):
    """Error during content writing.
    
    Raised when section writing, style enforcement,
    or content generation fails.
    """
    
    def __init__(
        self,
        message: str,
        section: str | None = None,
        word_count: int | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
    ):
        details = details or {}
        if section:
            details["section"] = section
        if word_count:
            details["word_count"] = word_count
        super().__init__(message, details, recoverable)
        self.section = section
        self.word_count = word_count


class ReviewError(GIAError):
    """Error during review process.
    
    Raised when critical review, scoring, or
    revision generation fails.
    """
    
    def __init__(
        self,
        message: str,
        review_type: str | None = None,
        criteria: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
    ):
        details = details or {}
        if review_type:
            details["review_type"] = review_type
        if criteria:
            details["criteria"] = criteria
        super().__init__(message, details, recoverable)
        self.review_type = review_type
        self.criteria = criteria
