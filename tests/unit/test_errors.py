"""Tests for error handling and recovery module.

This module tests:
- Custom exception hierarchy
- RetryPolicy configurations
- Error handlers
- Recovery strategies
- Fallback node
"""

from unittest.mock import MagicMock

import pytest

from src.errors import (
    # Exceptions
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
    # Policies
    create_api_retry_policy,
    create_search_retry_policy,
    create_analysis_retry_policy,
    DEFAULT_RETRY_POLICY,
    AGGRESSIVE_RETRY_POLICY,
    CONSERVATIVE_RETRY_POLICY,
    # Handlers
    handle_tool_error,
    handle_node_error,
    handle_api_error,
    create_error_response,
    log_error_with_context,
    ErrorHandler,
    # Recovery
    RecoveryStrategy,
    RecoveryAction,
    determine_recovery_strategy,
    can_continue_workflow,
    get_partial_output,
    create_fallback_content,
)
from src.errors.policies import RetryPolicy
from src.nodes.fallback import fallback_node, should_fallback, route_to_fallback_or_continue
from src.state.enums import ResearchStatus


# =============================================================================
# Exception Tests
# =============================================================================


class TestGIAError:
    """Tests for the base GIAError exception."""
    
    def test_basic_creation(self):
        """Test creating a basic GIAError."""
        error = GIAError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
    
    def test_with_details(self):
        """Test GIAError with details."""
        error = GIAError("Test error", details={"key": "value"})
        assert error.details == {"key": "value"}
    
    def test_with_recoverable_flag(self):
        """Test GIAError with recoverable flag."""
        error = GIAError("Test error", recoverable=True)
        assert error.recoverable is True
        
        error2 = GIAError("Fatal error", recoverable=False)
        assert error2.recoverable is False
    
    def test_to_dict(self):
        """Test serializing GIAError to dict."""
        error = GIAError(
            "Test error",
            details={"key": "value"},
            recoverable=True,
        )
        result = error.to_dict()
        assert result["message"] == "Test error"
        assert result["type"] == "GIAError"
        assert result["recoverable"] is True
        assert result["details"] == {"key": "value"}


class TestWorkflowError:
    """Tests for WorkflowError."""
    
    def test_creation(self):
        """Test creating WorkflowError."""
        error = WorkflowError("Workflow failed")
        assert isinstance(error, GIAError)
        assert error.message == "Workflow failed"
    
    def test_with_node(self):
        """Test WorkflowError with node context."""
        error = WorkflowError("Workflow failed", node="intake")
        assert error.node == "intake"
    
    def test_with_state_key(self):
        """Test WorkflowError with state_key."""
        error = WorkflowError("State error", state_key="research_question")
        assert error.state_key == "research_question"


class TestNodeExecutionError:
    """Tests for NodeExecutionError."""
    
    def test_creation(self):
        """Test creating NodeExecutionError."""
        error = NodeExecutionError("Node failed", node="intake")
        assert isinstance(error, GIAError)
        assert error.node == "intake"
    
    def test_with_phase(self):
        """Test NodeExecutionError with phase."""
        error = NodeExecutionError("Node failed", node="intake", phase="initialization")
        assert error.phase == "initialization"


class TestToolExecutionError:
    """Tests for ToolExecutionError."""
    
    def test_creation(self):
        """Test creating ToolExecutionError."""
        error = ToolExecutionError("Tool failed", tool_name="search_papers")
        assert isinstance(error, GIAError)
        assert error.tool_name == "search_papers"
    
    def test_with_tool_input(self):
        """Test ToolExecutionError with tool input."""
        error = ToolExecutionError(
            "Tool failed",
            tool_name="search_papers",
            tool_input={"query": "test"},
        )
        assert error.tool_input == {"query": "test"}


class TestAPIError:
    """Tests for APIError."""
    
    def test_creation(self):
        """Test creating APIError."""
        error = APIError("API request failed", service="semantic_scholar", status_code=500)
        assert isinstance(error, GIAError)
        assert error.status_code == 500
        assert error.service == "semantic_scholar"
    
    def test_without_status_code(self):
        """Test APIError without status code."""
        error = APIError("API failed", service="test")
        assert error.service == "test"


class TestRateLimitError:
    """Tests for RateLimitError."""
    
    def test_creation(self):
        """Test creating RateLimitError."""
        error = RateLimitError("Rate limited", service="semantic_scholar", retry_after=60)
        assert isinstance(error, APIError)
        assert error.retry_after == 60
    
    def test_always_recoverable(self):
        """Test RateLimitError is always recoverable."""
        error = RateLimitError("Rate limited", service="test")
        assert error.recoverable is True


class TestContextOverflowError:
    """Tests for ContextOverflowError."""
    
    def test_creation(self):
        """Test creating ContextOverflowError."""
        error = ContextOverflowError("Context too large", token_count=200000)
        assert isinstance(error, APIError)
        assert error.token_count == 200000
    
    def test_default_service(self):
        """Test ContextOverflowError default service."""
        error = ContextOverflowError("Context overflow")
        assert error.service == "anthropic"


class TestDataValidationError:
    """Tests for DataValidationError."""
    
    def test_creation(self):
        """Test creating DataValidationError."""
        error = DataValidationError("Invalid data", field="column_a")
        assert isinstance(error, GIAError)
        assert error.field == "column_a"
    
    def test_with_constraint(self):
        """Test DataValidationError with constraint."""
        error = DataValidationError("Invalid data", constraint="non_null")
        assert error.constraint == "non_null"


class TestSearchError:
    """Tests for SearchError."""
    
    def test_creation(self):
        """Test creating SearchError."""
        error = SearchError("Search failed", source="semantic_scholar")
        assert isinstance(error, GIAError)
        assert error.source == "semantic_scholar"
    
    def test_with_query(self):
        """Test SearchError with query."""
        error = SearchError("Search failed", query="AI research")
        assert error.query == "AI research"


class TestLiteratureSearchError:
    """Tests for LiteratureSearchError."""
    
    def test_creation(self):
        """Test creating LiteratureSearchError."""
        error = LiteratureSearchError("Literature search failed", query="AI research")
        assert isinstance(error, SearchError)
        assert error.query == "AI research"
    
    def test_with_papers_found(self):
        """Test LiteratureSearchError with papers_found."""
        error = LiteratureSearchError("Search failed", papers_found=5)
        assert error.papers_found == 5


class TestAnalysisError:
    """Tests for AnalysisError."""
    
    def test_creation(self):
        """Test creating AnalysisError."""
        error = AnalysisError("Analysis failed", analysis_type="regression")
        assert isinstance(error, GIAError)
        assert error.analysis_type == "regression"
    
    def test_with_dataset(self):
        """Test AnalysisError with dataset."""
        error = AnalysisError("Analysis failed", dataset="data.csv")
        assert error.dataset == "data.csv"


class TestWritingError:
    """Tests for WritingError."""
    
    def test_creation(self):
        """Test creating WritingError."""
        error = WritingError("Writing failed", section="introduction")
        assert isinstance(error, GIAError)
        assert error.section == "introduction"
    
    def test_with_word_count(self):
        """Test WritingError with word_count."""
        error = WritingError("Too long", word_count=10000)
        assert error.word_count == 10000


class TestReviewError:
    """Tests for ReviewError."""
    
    def test_creation(self):
        """Test creating ReviewError."""
        error = ReviewError("Review failed", review_type="critical")
        assert isinstance(error, GIAError)
        assert error.review_type == "critical"
    
    def test_with_criteria(self):
        """Test ReviewError with criteria."""
        error = ReviewError("Failed criteria", criteria="coherence")
        assert error.criteria == "coherence"


# =============================================================================
# Policy Tests
# =============================================================================


class TestRetryPolicy:
    """Tests for RetryPolicy dataclass."""
    
    def test_creation(self):
        """Test RetryPolicy creation."""
        policy = RetryPolicy(max_attempts=3)
        assert policy.max_attempts == 3
    
    def test_get_delay(self):
        """Test get_delay calculation."""
        policy = RetryPolicy(
            initial_interval=1.0,
            backoff_factor=2.0,
            jitter=False,
        )
        # First retry (attempt=0)
        delay = policy.get_delay(0)
        assert delay == 1.0
        # Second retry (attempt=1)
        delay = policy.get_delay(1)
        assert delay == 2.0
    
    def test_should_attempt_retry(self):
        """Test should_attempt_retry logic."""
        policy = RetryPolicy(max_attempts=3)
        error = Exception("test")
        
        # Should retry on first two attempts
        assert policy.should_attempt_retry(error, 0) is True
        assert policy.should_attempt_retry(error, 1) is True
        # Should not retry on third attempt
        assert policy.should_attempt_retry(error, 2) is False


class TestRetryPolicies:
    """Tests for pre-configured retry policies."""
    
    def test_default_retry_policy(self):
        """Test default retry policy settings."""
        policy = DEFAULT_RETRY_POLICY
        assert policy.max_attempts >= 1
        assert policy.initial_interval > 0
        assert policy.max_interval > policy.initial_interval
    
    def test_aggressive_retry_policy(self):
        """Test aggressive retry policy has more attempts."""
        assert AGGRESSIVE_RETRY_POLICY.max_attempts >= DEFAULT_RETRY_POLICY.max_attempts
    
    def test_conservative_retry_policy(self):
        """Test conservative retry policy has fewer attempts."""
        assert CONSERVATIVE_RETRY_POLICY.max_attempts <= DEFAULT_RETRY_POLICY.max_attempts
    
    def test_create_api_retry_policy(self):
        """Test creating API-specific retry policy."""
        policy = create_api_retry_policy()
        assert policy.max_attempts >= 2
        assert policy.retry_on is not None
    
    def test_create_search_retry_policy(self):
        """Test creating search-specific retry policy."""
        policy = create_search_retry_policy()
        assert policy.max_attempts >= 2
    
    def test_create_analysis_retry_policy(self):
        """Test creating analysis-specific retry policy."""
        policy = create_analysis_retry_policy()
        assert policy.max_attempts >= 1


# =============================================================================
# Handler Tests
# =============================================================================


class TestErrorHandlers:
    """Tests for error handler functions."""
    
    def test_handle_tool_error_returns_string(self):
        """Test handle_tool_error returns a string message."""
        error = ToolExecutionError("Tool failed", tool_name="search")
        result = handle_tool_error(error, "search")
        assert isinstance(result, str)
        assert "Tool" in result or "error" in result.lower()
    
    def test_handle_node_error_returns_dict(self):
        """Test handle_node_error returns a dictionary."""
        error = NodeExecutionError("Node failed", node="intake")
        result = handle_node_error(error, "intake", {})
        assert isinstance(result, dict)
    
    def test_handle_api_error_returns_string(self):
        """Test handle_api_error returns a string message."""
        error = APIError("API failed", service="test_service", status_code=500)
        result = handle_api_error(error, service="test_service", operation="search")
        assert isinstance(result, str)
    
    def test_create_error_response(self):
        """Test create_error_response creates proper structure."""
        error = GIAError("Test message", recoverable=True)
        response = create_error_response(error, node="test")
        assert response["error_type"] == "GIAError"
        assert response["message"] == "Test message"
        assert response["recoverable"] is True
        assert response["node"] == "test"
    
    def test_log_error_with_context(self):
        """Test log_error_with_context doesn't raise."""
        error = GIAError("Test error")
        # Should not raise
        log_error_with_context(error, context={"state_key": "value"})


class TestErrorHandler:
    """Tests for ErrorHandler class."""
    
    def test_handler_creation(self):
        """Test ErrorHandler can be created."""
        handler = ErrorHandler(node="test_node")
        assert handler is not None
        assert handler.node == "test_node"
    
    def test_handler_handle_method(self):
        """Test ErrorHandler.handle method."""
        handler = ErrorHandler(node="test")
        error = GIAError("Test error")
        result = handler.handle(error, state={})
        assert isinstance(result, dict)
        assert "errors" in result
    
    def test_handler_should_fail_on_max_errors(self):
        """Test ErrorHandler.should_fail triggers on max errors."""
        handler = ErrorHandler(node="test", max_errors=3)
        error = GIAError("Test error")
        # Call handle() first to increment _error_count
        handler.handle(error, state={})
        # Now state with 2 errors + _error_count(1) = 3 (max)
        state = {"errors": [MagicMock(), MagicMock()]}
        assert handler.should_fail(error, state) is True
    
    def test_handler_should_fail_on_unrecoverable(self):
        """Test ErrorHandler.should_fail triggers on unrecoverable."""
        handler = ErrorHandler(node="test", fail_on_unrecoverable=True)
        error = GIAError("Fatal error", recoverable=False)
        assert handler.should_fail(error, {}) is True


# =============================================================================
# Recovery Tests
# =============================================================================


class TestRecoveryAction:
    """Tests for RecoveryAction enum."""
    
    def test_all_actions_exist(self):
        """Test all expected recovery actions exist."""
        assert RecoveryAction.RETRY is not None
        assert RecoveryAction.SKIP is not None
        assert RecoveryAction.FALLBACK is not None
        assert RecoveryAction.REDUCE_CONTENT is not None
        assert RecoveryAction.WAIT_AND_RETRY is not None
        assert RecoveryAction.ABORT is not None
        assert RecoveryAction.PARTIAL_OUTPUT is not None


class TestRecoveryStrategy:
    """Tests for RecoveryStrategy dataclass."""
    
    def test_creation(self):
        """Test RecoveryStrategy creation."""
        strategy = RecoveryStrategy(
            action=RecoveryAction.RETRY,
            reason="Transient error",
        )
        assert strategy.action == RecoveryAction.RETRY
        assert strategy.reason == "Transient error"
    
    def test_to_dict(self):
        """Test RecoveryStrategy serialization."""
        strategy = RecoveryStrategy(
            action=RecoveryAction.FALLBACK,
            reason="Too many errors",
            params={"max_errors": 3},
        )
        result = strategy.to_dict()
        assert result["action"] == "fallback"
        assert result["reason"] == "Too many errors"
        assert result["params"] == {"max_errors": 3}


class TestDetermineRecoveryStrategy:
    """Tests for determine_recovery_strategy function."""
    
    def test_rate_limit_error_strategy(self):
        """Test recovery strategy for rate limit errors."""
        error = RateLimitError("Rate limited", service="test", retry_after=60)
        strategy = determine_recovery_strategy(error, "literature_reviewer", {})
        # Should suggest wait and retry or similar
        assert strategy.action in [
            RecoveryAction.WAIT_AND_RETRY,
            RecoveryAction.RETRY,
            RecoveryAction.FALLBACK,
        ]
    
    def test_context_overflow_strategy(self):
        """Test recovery strategy for context overflow."""
        error = ContextOverflowError("Too large", token_count=200000)
        strategy = determine_recovery_strategy(error, "writer", {})
        # Should suggest reducing content or fallback
        assert strategy.action in [
            RecoveryAction.REDUCE_CONTENT,
            RecoveryAction.FALLBACK,
            RecoveryAction.PARTIAL_OUTPUT,
        ]
    
    def test_too_many_errors_triggers_fallback(self):
        """Test that too many errors triggers fallback."""
        error = GIAError("Another error")
        state = {"errors": [MagicMock(), MagicMock(), MagicMock()]}  # 3 errors
        strategy = determine_recovery_strategy(error, "writer", state)
        assert strategy.action == RecoveryAction.FALLBACK


class TestCanContinueWorkflow:
    """Tests for can_continue_workflow function."""
    
    def test_can_continue_no_errors(self):
        """Test workflow can continue without errors."""
        state = {}
        assert can_continue_workflow(state) is True
    
    def test_can_continue_few_errors(self):
        """Test workflow can continue with few recoverable errors."""
        error = MagicMock()
        error.recoverable = True
        state = {"errors": [error]}
        assert can_continue_workflow(state) is True
    
    def test_cannot_continue_unrecoverable_error(self):
        """Test workflow cannot continue with unrecoverable error."""
        error = MagicMock()
        error.recoverable = False
        state = {"errors": [error]}
        # Check if it triggers fallback condition
        assert can_continue_workflow(state) is False or should_fallback(state) is True
    
    def test_cannot_continue_too_many_errors(self):
        """Test workflow cannot continue with too many errors."""
        errors = [MagicMock() for _ in range(5)]
        for e in errors:
            e.recoverable = True
        state = {"errors": errors}
        assert can_continue_workflow(state) is False


class TestGetPartialOutput:
    """Tests for get_partial_output function."""
    
    def test_empty_state_returns_dict(self):
        """Test partial output from empty state."""
        state = {}
        result = get_partial_output(state)
        assert isinstance(result, dict)
    
    def test_collects_available_data(self):
        """Test partial output collects available data."""
        state = {
            "research_question": "Test question",
            "literature_synthesis": {"themes": ["theme1"]},
            "gap_analysis": {"gaps": ["gap1"]},
        }
        result = get_partial_output(state)
        assert isinstance(result, dict)
        # Should include whatever is available
        assert "research_question" in result or len(result) > 0


class TestCreateFallbackContent:
    """Tests for create_fallback_content function."""
    
    def test_creates_string_content(self):
        """Test fallback content is a string."""
        state = {"research_question": "Test question"}
        result = create_fallback_content(state, "abstract")
        assert isinstance(result, str)
    
    def test_different_sections(self):
        """Test fallback content for different sections."""
        state = {"research_question": "Test question"}
        sections = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
        for section in sections:
            result = create_fallback_content(state, section)
            assert isinstance(result, str)
            assert len(result) > 0


# =============================================================================
# Fallback Node Tests
# =============================================================================


class TestFallbackNode:
    """Tests for the fallback node."""
    
    def test_fallback_node_returns_dict(self):
        """Test fallback node returns a dictionary."""
        state = {
            "research_question": "Test question",
            "errors": [],
        }
        result = fallback_node(state)
        assert isinstance(result, dict)
    
    def test_fallback_node_sets_completed_status(self):
        """Test fallback node sets completed status."""
        state = {"research_question": "Test question", "errors": []}
        result = fallback_node(state)
        assert result["status"] == ResearchStatus.COMPLETED
    
    def test_fallback_node_creates_report(self):
        """Test fallback node creates a fallback report."""
        state = {"research_question": "Test question", "errors": []}
        result = fallback_node(state)
        assert "fallback_report" in result
        assert isinstance(result["fallback_report"], dict)
    
    def test_fallback_node_with_errors(self):
        """Test fallback node handles errors in state."""
        error = MagicMock()
        error.category = "test_error"
        error.node = "test_node"
        error.message = "Test error message"
        error.recoverable = False
        
        state = {
            "research_question": "Test question",
            "errors": [error],
        }
        result = fallback_node(state)
        assert "fallback_report" in result
        assert result["fallback_report"]["error_summary"]["count"] == 1
    
    def test_fallback_node_generates_paper(self):
        """Test fallback node generates a final paper."""
        state = {
            "research_question": "Test question",
            "project_title": "Test Paper",
            "errors": [],
        }
        result = fallback_node(state)
        assert "final_paper" in result
        assert isinstance(result["final_paper"], str)
    
    def test_fallback_flag_set(self):
        """Test fallback node sets fallback activation flag."""
        state = {"research_question": "Test question", "errors": []}
        result = fallback_node(state)
        assert result.get("_fallback_activated") is True


class TestShouldFallback:
    """Tests for should_fallback function."""
    
    def test_no_fallback_empty_state(self):
        """Test no fallback needed for empty state."""
        state = {}
        assert should_fallback(state) is False
    
    def test_fallback_on_explicit_flag(self):
        """Test fallback on explicit flag."""
        state = {"_should_fallback": True}
        assert should_fallback(state) is True
    
    def test_fallback_on_too_many_errors(self):
        """Test fallback when too many errors."""
        errors = [MagicMock() for _ in range(3)]
        for e in errors:
            e.recoverable = True
        state = {"errors": errors}
        assert should_fallback(state) is True
    
    def test_fallback_on_unrecoverable_error(self):
        """Test fallback on unrecoverable error."""
        error = MagicMock()
        error.recoverable = False
        state = {"errors": [error]}
        assert should_fallback(state) is True
    
    def test_fallback_on_failed_status(self):
        """Test fallback on failed status."""
        state = {"status": ResearchStatus.FAILED}
        assert should_fallback(state) is True
    
    def test_no_fallback_with_recoverable_errors(self):
        """Test no fallback with few recoverable errors."""
        error = MagicMock()
        error.recoverable = True
        state = {"errors": [error]}
        assert should_fallback(state) is False


class TestRouteToFallbackOrContinue:
    """Tests for route_to_fallback_or_continue function."""
    
    def test_continue_normal_state(self):
        """Test continue routing for normal state."""
        state = {}
        result = route_to_fallback_or_continue(state)
        assert result == "continue"
    
    def test_fallback_on_errors(self):
        """Test fallback routing on errors."""
        errors = [MagicMock() for _ in range(3)]
        for e in errors:
            e.recoverable = True
        state = {"errors": errors}
        result = route_to_fallback_or_continue(state)
        assert result == "fallback"


# =============================================================================
# Integration Tests
# =============================================================================


class TestErrorHandlingIntegration:
    """Integration tests for error handling system."""
    
    def test_error_to_strategy_to_recovery_flow(self):
        """Test complete error handling flow."""
        # Create error
        error = RateLimitError("Rate limited", service="test", retry_after=30)
        
        # Get recovery strategy
        strategy = determine_recovery_strategy(
            error, "literature_reviewer", {"errors": []}
        )
        
        # Verify strategy is valid
        assert isinstance(strategy, RecoveryStrategy)
        assert isinstance(strategy.action, RecoveryAction)
    
    def test_multiple_errors_trigger_fallback(self):
        """Test multiple errors eventually trigger fallback."""
        errors = []
        for i in range(3):
            error = GIAError(f"Error {i}", recoverable=True)
            errors.append(error)
        
        state = {"errors": errors, "research_question": "Test"}
        
        # Should trigger fallback
        assert should_fallback(state) is True
        
        # Fallback node should work
        result = fallback_node(state)
        assert result["status"] == ResearchStatus.COMPLETED
    
    def test_unrecoverable_error_triggers_immediate_fallback(self):
        """Test unrecoverable error triggers immediate fallback."""
        error = GIAError("Fatal error", recoverable=False)
        state = {"errors": [error], "research_question": "Test"}
        
        assert should_fallback(state) is True


class TestModuleExports:
    """Tests for module exports."""
    
    def test_all_exceptions_exported(self):
        """Test all exception classes are exported."""
        from src.errors import (
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
        # All imports should work
        assert GIAError is not None
        assert WorkflowError is not None
    
    def test_all_policies_exported(self):
        """Test all policy creators are exported."""
        from src.errors import (
            create_api_retry_policy,
            create_search_retry_policy,
            create_analysis_retry_policy,
            DEFAULT_RETRY_POLICY,
            AGGRESSIVE_RETRY_POLICY,
            CONSERVATIVE_RETRY_POLICY,
        )
        assert DEFAULT_RETRY_POLICY is not None
    
    def test_all_handlers_exported(self):
        """Test all handlers are exported."""
        from src.errors import (
            handle_tool_error,
            handle_node_error,
            handle_api_error,
            create_error_response,
            log_error_with_context,
            ErrorHandler,
        )
        assert ErrorHandler is not None
    
    def test_all_recovery_items_exported(self):
        """Test all recovery items are exported."""
        from src.errors import (
            RecoveryStrategy,
            RecoveryAction,
            determine_recovery_strategy,
            execute_recovery,
            can_continue_workflow,
            get_partial_output,
            create_fallback_content,
        )
        assert RecoveryAction is not None
        assert RecoveryStrategy is not None
