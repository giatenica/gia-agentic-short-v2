"""RetryPolicy configurations for different error scenarios.

This module provides pre-configured RetryPolicy instances for
various types of operations, with appropriate backoff strategies
and retry conditions.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Type

from src.errors.exceptions import (
    RateLimitError,
    APIError,
    ContextOverflowError,
    LiteratureSearchError,
    AnalysisError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# RetryPolicy Configuration
# =============================================================================


@dataclass
class RetryPolicy:
    """Configuration for retry behavior on errors.
    
    This is a custom implementation that mirrors LangGraph's RetryPolicy
    interface for use with our error handling system.
    
    Attributes:
        max_attempts: Maximum number of retry attempts (including initial)
        initial_interval: Initial delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff
        max_interval: Maximum delay between retries in seconds
        jitter: Whether to add random jitter to delays
        retry_on: Tuple of exception types to retry on
        should_retry: Optional custom function to determine if should retry
    """
    
    max_attempts: int = 3
    initial_interval: float = 1.0
    backoff_factor: float = 2.0
    max_interval: float = 60.0
    jitter: bool = True
    retry_on: tuple[Type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )
    should_retry: Callable[[Exception, int], bool] | None = None
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay before next retry.
        
        Uses exponential backoff with optional jitter.
        
        Args:
            attempt: The current attempt number (0-indexed)
            
        Returns:
            Delay in seconds before next retry
        """
        delay = min(
            self.initial_interval * (self.backoff_factor ** attempt),
            self.max_interval
        )
        
        if self.jitter:
            # Add 0-50% random jitter
            delay = delay * (1 + random.random() * 0.5)
        
        return delay
    
    def should_attempt_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if a retry should be attempted.
        
        Args:
            error: The exception that occurred
            attempt: The current attempt number (0-indexed)
            
        Returns:
            True if retry should be attempted
        """
        # Check attempt limit
        if attempt >= self.max_attempts - 1:
            return False
        
        # Check custom retry function if provided
        if self.should_retry is not None:
            return self.should_retry(error, attempt)
        
        # Check if error type is retryable
        return isinstance(error, self.retry_on)


# =============================================================================
# Pre-configured Policies
# =============================================================================


def create_api_retry_policy(
    max_attempts: int = 3,
    initial_interval: float = 1.0,
) -> RetryPolicy:
    """Create a retry policy optimized for API calls.
    
    Handles rate limits, timeouts, and transient API errors.
    
    Args:
        max_attempts: Maximum retry attempts
        initial_interval: Initial delay in seconds
        
    Returns:
        Configured RetryPolicy
    """
    return RetryPolicy(
        max_attempts=max_attempts,
        initial_interval=initial_interval,
        backoff_factor=2.0,
        max_interval=30.0,
        jitter=True,
        retry_on=(
            RateLimitError,
            TimeoutError,
            ConnectionError,
        ),
        should_retry=_should_retry_api_error,
    )


def create_search_retry_policy(
    max_attempts: int = 3,
    initial_interval: float = 2.0,
) -> RetryPolicy:
    """Create a retry policy for search operations.
    
    Handles search service failures and rate limits.
    
    Args:
        max_attempts: Maximum retry attempts
        initial_interval: Initial delay in seconds
        
    Returns:
        Configured RetryPolicy
    """
    return RetryPolicy(
        max_attempts=max_attempts,
        initial_interval=initial_interval,
        backoff_factor=2.0,
        max_interval=60.0,
        jitter=True,
        retry_on=(
            LiteratureSearchError,
            RateLimitError,
            TimeoutError,
            ConnectionError,
        ),
        should_retry=_should_retry_search_error,
    )


def create_analysis_retry_policy(
    max_attempts: int = 2,
    initial_interval: float = 1.0,
) -> RetryPolicy:
    """Create a retry policy for analysis operations.
    
    More conservative - analysis errors are often not transient.
    
    Args:
        max_attempts: Maximum retry attempts
        initial_interval: Initial delay in seconds
        
    Returns:
        Configured RetryPolicy
    """
    return RetryPolicy(
        max_attempts=max_attempts,
        initial_interval=initial_interval,
        backoff_factor=1.5,
        max_interval=10.0,
        jitter=False,
        retry_on=(
            AnalysisError,
            TimeoutError,
        ),
        should_retry=_should_retry_analysis_error,
    )


# =============================================================================
# Default Policies
# =============================================================================


DEFAULT_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=30.0,
    jitter=True,
    retry_on=(
        RateLimitError,
        TimeoutError,
        ConnectionError,
    ),
)
"""Default retry policy for general operations."""


AGGRESSIVE_RETRY_POLICY = RetryPolicy(
    max_attempts=5,
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=60.0,
    jitter=True,
    retry_on=(
        RateLimitError,
        TimeoutError,
        ConnectionError,
        APIError,
    ),
)
"""Aggressive retry policy for critical operations."""


CONSERVATIVE_RETRY_POLICY = RetryPolicy(
    max_attempts=2,
    initial_interval=2.0,
    backoff_factor=1.5,
    max_interval=10.0,
    jitter=False,
    retry_on=(
        RateLimitError,
        TimeoutError,
    ),
)
"""Conservative retry policy for non-critical operations."""


# =============================================================================
# Retry Decision Functions
# =============================================================================


def _should_retry_api_error(error: Exception, attempt: int) -> bool:
    """Determine if an API error should be retried.
    
    Args:
        error: The exception that occurred
        attempt: Current attempt number
        
    Returns:
        True if should retry
    """
    # Always retry rate limits
    if isinstance(error, RateLimitError):
        logger.info(
            f"Rate limit hit, will retry (attempt {attempt + 1}). "
            f"Retry after: {getattr(error, 'retry_after', 'unknown')}"
        )
        return True
    
    # Retry timeouts and connection errors
    if isinstance(error, (TimeoutError, ConnectionError)):
        logger.info(f"Connection issue, will retry (attempt {attempt + 1})")
        return True
    
    # Don't retry context overflow - needs content reduction
    if isinstance(error, ContextOverflowError):
        logger.warning("Context overflow - not retrying, needs content reduction")
        return False
    
    # Don't retry generic API errors after first attempt
    if isinstance(error, APIError) and attempt > 0:
        logger.warning(f"API error persists after {attempt + 1} attempts, not retrying")
        return False
    
    return True


def _should_retry_search_error(error: Exception, attempt: int) -> bool:
    """Determine if a search error should be retried.
    
    Args:
        error: The exception that occurred
        attempt: Current attempt number
        
    Returns:
        True if should retry
    """
    # Always retry rate limits
    if isinstance(error, RateLimitError):
        return True
    
    # Retry connection issues
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    
    # For literature search errors, check if it's a service issue
    if isinstance(error, LiteratureSearchError):
        # If no papers were found, might be a query issue - don't retry
        if getattr(error, 'papers_found', None) == 0:
            logger.info("No papers found - likely query issue, not retrying")
            return False
        return attempt < 2  # Retry once
    
    return False


def _should_retry_analysis_error(error: Exception, attempt: int) -> bool:
    """Determine if an analysis error should be retried.
    
    Args:
        error: The exception that occurred
        attempt: Current attempt number
        
    Returns:
        True if should retry
    """
    # Only retry timeouts for analysis
    if isinstance(error, TimeoutError):
        return attempt < 1
    
    # Analysis errors are usually not transient
    if isinstance(error, AnalysisError):
        logger.info("Analysis error - likely requires intervention, not retrying")
        return False
    
    return False


# =============================================================================
# Node-Specific Policy Mapping
# =============================================================================


NODE_RETRY_POLICIES: dict[str, RetryPolicy] = {
    "intake": CONSERVATIVE_RETRY_POLICY,
    "data_explorer": DEFAULT_RETRY_POLICY,
    "literature_reviewer": create_search_retry_policy(max_attempts=3),
    "literature_synthesizer": create_api_retry_policy(max_attempts=3),
    "gap_identifier": create_api_retry_policy(max_attempts=3),
    "planner": create_api_retry_policy(max_attempts=3),
    "data_analyst": create_analysis_retry_policy(max_attempts=2),
    "conceptual_synthesizer": create_api_retry_policy(max_attempts=3),
    "writer": create_api_retry_policy(max_attempts=3),
    "reviewer": create_api_retry_policy(max_attempts=3),
}
"""Mapping of node names to their retry policies."""


def get_retry_policy_for_node(node_name: str) -> RetryPolicy:
    """Get the retry policy configured for a specific node.
    
    Args:
        node_name: Name of the node
        
    Returns:
        Retry policy for the node, or DEFAULT_RETRY_POLICY if not configured
    """
    return NODE_RETRY_POLICIES.get(node_name, DEFAULT_RETRY_POLICY)
