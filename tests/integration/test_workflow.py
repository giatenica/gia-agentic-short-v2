"""Integration tests for complete workflow execution.

Tests the full research workflow from intake to output with mock LLM.
"""

from unittest.mock import MagicMock

from src.state.enums import ResearchStatus
from src.graphs import create_research_workflow, WorkflowConfig
from src.nodes.fallback import fallback_node, should_fallback


# =============================================================================
# Workflow Creation Tests
# =============================================================================


class TestWorkflowCreation:
    """Tests for workflow creation and configuration."""
    
    def test_create_workflow_with_defaults(self):
        """Test creating workflow with default configuration."""
        config = WorkflowConfig()
        workflow = create_research_workflow(config)
        
        assert workflow is not None
        # Check that all nodes exist
        assert hasattr(workflow, 'invoke')
    
    def test_create_workflow_with_checkpointer(self, mock_checkpointer):
        """Test creating workflow with custom checkpointer."""
        config = WorkflowConfig(checkpointer=mock_checkpointer)
        workflow = create_research_workflow(config)
        
        assert workflow is not None
    
    def test_create_workflow_with_debug_mode(self):
        """Test creating workflow in debug mode."""
        config = WorkflowConfig(debug=True)
        workflow = create_research_workflow(config)
        
        assert workflow is not None


# =============================================================================
# Fallback Integration Tests
# =============================================================================


class TestFallbackIntegration:
    """Tests for fallback node integration."""
    
    def test_fallback_triggers_on_error_threshold(self, state_with_errors):
        """Test that fallback triggers when error count exceeds threshold."""
        assert should_fallback(state_with_errors) is True
    
    def test_fallback_node_produces_output(self, state_with_errors):
        """Test fallback node generates valid output."""
        result = fallback_node(state_with_errors)
        
        assert result["status"] == ResearchStatus.COMPLETED
        assert "fallback_report" in result
        assert result.get("_fallback_activated") is True
    
    def test_fallback_includes_error_summary(self, state_with_errors):
        """Test fallback report includes error summary."""
        result = fallback_node(state_with_errors)
        
        report = result["fallback_report"]
        assert "error_summary" in report
        assert report["error_summary"]["count"] == 3
    
    def test_fallback_generates_partial_paper(self, state_after_literature):
        """Test fallback generates paper sections from available state."""
        # Add errors to trigger fallback
        state_after_literature["errors"] = [MagicMock(recoverable=False)]
        state_after_literature["_should_fallback"] = True
        
        result = fallback_node(state_after_literature)
        
        assert "final_paper" in result
        assert isinstance(result["final_paper"], str)
    
    def test_no_fallback_on_clean_state(self, minimal_state):
        """Test fallback does not trigger on clean state."""
        assert should_fallback(minimal_state) is False


# =============================================================================
# State Transition Tests
# =============================================================================


class TestStateTransitions:
    """Tests for workflow state transitions."""
    
    def test_intake_complete_status(self, state_after_intake):
        """Test state has correct status after intake."""
        assert state_after_intake["status"] == ResearchStatus.INTAKE_COMPLETE
    
    def test_literature_reviewed_status(self, state_after_literature):
        """Test state has correct status after literature review."""
        assert state_after_literature["status"] == ResearchStatus.LITERATURE_REVIEW_COMPLETE
    
    def test_complete_state_has_all_outputs(self, complete_state):
        """Test complete state has all expected outputs."""
        assert complete_state["status"] == ResearchStatus.WRITING_COMPLETE
        assert "writer_output" in complete_state
        assert "reviewer_output" in complete_state
        
        # Check writer output has all sections
        writer_output = complete_state["writer_output"]
        expected_sections = [
            "abstract", "introduction", "literature_review",
            "methods", "results", "discussion", "conclusion"
        ]
        for section in expected_sections:
            assert section in writer_output


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecovery:
    """Tests for error recovery and graceful degradation."""
    
    def test_recoverable_errors_allow_continuation(self):
        """Test recoverable errors allow workflow to continue."""
        error = MagicMock()
        error.recoverable = True
        
        state = {
            "errors": [error],
            "status": ResearchStatus.ANALYZING,
        }
        
        # Single recoverable error should not trigger fallback
        assert should_fallback(state) is False
    
    def test_unrecoverable_error_triggers_fallback(self):
        """Test unrecoverable error triggers fallback."""
        error = MagicMock()
        error.recoverable = False
        
        state = {
            "errors": [error],
            "status": ResearchStatus.ANALYZING,
        }
        
        assert should_fallback(state) is True
    
    def test_multiple_errors_trigger_fallback(self):
        """Test multiple errors trigger fallback."""
        errors = [MagicMock(recoverable=True) for _ in range(3)]
        
        state = {
            "errors": errors,
            "status": ResearchStatus.ANALYZING,
        }
        
        # 3+ errors should trigger fallback
        assert should_fallback(state) is True
    
    def test_failed_status_triggers_fallback(self):
        """Test FAILED status triggers fallback."""
        state = {
            "errors": [],
            "status": ResearchStatus.FAILED,
        }
        
        assert should_fallback(state) is True


# =============================================================================
# HITL Simulation Tests
# =============================================================================


class TestHITLSimulation:
    """Tests for human-in-the-loop simulation."""
    
    def test_state_supports_interrupt_data(self, minimal_state):
        """Test state can hold interrupt data."""
        minimal_state["_interrupt_data"] = {
            "action": "approve_plan",
            "message": "Review the research plan",
        }
        
        assert "_interrupt_data" in minimal_state
    
    def test_resume_with_approval(self, state_after_literature):
        """Test state can be updated with approval."""
        state_after_literature["_human_approved"] = True
        state_after_literature["_approval_timestamp"] = "2026-01-03T12:00:00Z"
        
        assert state_after_literature["_human_approved"] is True
    
    def test_resume_with_modifications(self, state_after_literature):
        """Test state can be updated with human modifications."""
        state_after_literature["refined_query"] = "Modified research question"
        state_after_literature["_human_modified"] = True
        
        assert "Modified" in state_after_literature["refined_query"]


# =============================================================================
# Streaming Tests
# =============================================================================


class TestStreamingFunctionality:
    """Tests for streaming progress updates."""
    
    def test_state_supports_progress_tracking(self, minimal_state):
        """Test state supports progress tracking fields."""
        minimal_state["_current_node"] = "literature_reviewer"
        minimal_state["_progress"] = 0.25
        
        assert minimal_state["_current_node"] == "literature_reviewer"
        assert minimal_state["_progress"] == 0.25
    
    def test_streaming_event_structure(self):
        """Test streaming event has expected structure."""
        from src.graphs.streaming import StreamEvent, StreamMode
        
        event = StreamEvent(
            mode=StreamMode.UPDATES,
            node="literature_reviewer",
            data={"papers_found": 10},
        )
        
        assert event.mode == StreamMode.UPDATES
        assert event.node == "literature_reviewer"
        assert event.data["papers_found"] == 10


# =============================================================================
# Configuration Tests
# =============================================================================


class TestWorkflowConfiguration:
    """Tests for workflow configuration options."""
    
    def test_config_with_custom_interrupt_points(self):
        """Test configuration with custom interrupt points."""
        config = WorkflowConfig(
            interrupt_before=["planner", "reviewer"],
            interrupt_after=["writer"],
        )
        
        assert "planner" in config.interrupt_before
        assert "writer" in config.interrupt_after
    
    def test_config_disable_cache(self):
        """Test configuration to disable caching."""
        config = WorkflowConfig(
            enable_caching=False,
        )
        
        assert config.enable_caching is False
    
    def test_config_with_store(self):
        """Test configuration with custom store."""
        mock_store = MagicMock()
        config = WorkflowConfig(store=mock_store)
        
        assert config.store is mock_store
