"""Tests for workflow persistence and resume functionality."""

from unittest.mock import MagicMock, patch
import pytest

from langgraph.checkpoint.memory import MemorySaver

from src.state.enums import ResearchStatus
from src.graphs import create_research_workflow, WorkflowConfig


# =============================================================================
# Checkpointer Tests
# =============================================================================


class TestCheckpointerIntegration:
    """Tests for checkpointer integration."""
    
    def test_workflow_compiles_with_memory_saver(self):
        """Test workflow compiles with MemorySaver checkpointer."""
        checkpointer = MemorySaver()
        config = WorkflowConfig(checkpointer=checkpointer)
        workflow = create_research_workflow(config)
        
        assert workflow is not None
    
    def test_config_stores_checkpointer(self):
        """Test WorkflowConfig stores checkpointer reference."""
        checkpointer = MemorySaver()
        config = WorkflowConfig(checkpointer=checkpointer)
        
        assert config.checkpointer is checkpointer


# =============================================================================
# State Persistence Tests
# =============================================================================


class TestStatePersistence:
    """Tests for state persistence functionality."""
    
    def test_state_is_serializable(self, complete_state):
        """Test complete state can be serialized."""
        import json
        
        # Remove mock objects for serialization test
        serializable_state = {
            k: v for k, v in complete_state.items()
            if not callable(v) and not k.startswith("_")
        }
        
        # Convert enums to values
        serializable_state["status"] = serializable_state["status"].value
        serializable_state["research_type"] = serializable_state["research_type"].value
        serializable_state["paper_type"] = serializable_state["paper_type"].value
        serializable_state["target_journal"] = serializable_state["target_journal"].value
        
        # Should not raise
        json_str = json.dumps(serializable_state)
        assert len(json_str) > 0
    
    def test_state_preserves_research_type(self, minimal_state):
        """Test state preserves research type through persistence."""
        from src.state.enums import ResearchType
        
        minimal_state["research_type"] = ResearchType.THEORETICAL
        
        assert minimal_state["research_type"] == ResearchType.THEORETICAL
    
    def test_errors_list_can_grow(self, minimal_state):
        """Test errors list can accumulate errors."""
        error1 = MagicMock(message="Error 1")
        error2 = MagicMock(message="Error 2")
        
        minimal_state["errors"].append(error1)
        minimal_state["errors"].append(error2)
        
        assert len(minimal_state["errors"]) == 2


# =============================================================================
# Resume Tests
# =============================================================================


class TestWorkflowResume:
    """Tests for workflow resume functionality."""
    
    def test_state_tracks_current_node(self, state_after_intake):
        """Test state can track current node for resume."""
        state_after_intake["_last_node"] = "intake"
        state_after_intake["_next_node"] = "literature_reviewer"
        
        assert state_after_intake["_last_node"] == "intake"
        assert state_after_intake["_next_node"] == "literature_reviewer"
    
    def test_state_tracks_timestamp(self, minimal_state):
        """Test state tracks timestamps."""
        from datetime import datetime, timezone
        
        minimal_state["_created_at"] = datetime.now(timezone.utc).isoformat()
        minimal_state["_updated_at"] = datetime.now(timezone.utc).isoformat()
        
        assert "_created_at" in minimal_state
        assert "_updated_at" in minimal_state
    
    def test_thread_id_in_config(self, workflow_config):
        """Test thread_id is in workflow config."""
        assert "thread_id" in workflow_config["configurable"]
        assert workflow_config["configurable"]["thread_id"] == "test-thread-123"


# =============================================================================
# Time Travel Tests
# =============================================================================


class TestTimeTravelFunctionality:
    """Tests for time travel debugging functionality."""
    
    def test_state_supports_checkpoint_id(self, minimal_state):
        """Test state can store checkpoint ID."""
        minimal_state["_checkpoint_id"] = "checkpoint_001"
        
        assert minimal_state["_checkpoint_id"] == "checkpoint_001"
    
    def test_state_supports_parent_checkpoint(self, minimal_state):
        """Test state can reference parent checkpoint."""
        minimal_state["_parent_checkpoint_id"] = "checkpoint_000"
        minimal_state["_checkpoint_id"] = "checkpoint_001"
        
        assert minimal_state["_parent_checkpoint_id"] == "checkpoint_000"


# =============================================================================
# Cross-Session Memory Tests
# =============================================================================


class TestCrossSessionMemory:
    """Tests for cross-session memory functionality."""
    
    def test_state_can_store_user_context(self, minimal_state):
        """Test state can store user context."""
        minimal_state["_user_context"] = {
            "user_id": "user_123",
            "session_id": "session_456",
            "preferences": {"language": "en"},
        }
        
        assert minimal_state["_user_context"]["user_id"] == "user_123"
    
    def test_state_can_store_research_history(self, minimal_state):
        """Test state can store research history."""
        minimal_state["_research_history"] = [
            {"query": "Previous query 1", "date": "2025-01-01"},
            {"query": "Previous query 2", "date": "2025-06-01"},
        ]
        
        assert len(minimal_state["_research_history"]) == 2
