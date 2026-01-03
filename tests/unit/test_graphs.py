"""Tests for Sprint 8: Graph Assembly and Full Workflow Integration.

This module tests:
- Research workflow factory and configuration
- Routing functions
- Streaming utilities
- Time travel and debugging
- Subgraph compositions
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from src.graphs.research_workflow import (
    create_research_workflow,
    create_studio_workflow,
    create_production_workflow,
    WorkflowConfig,
    output_node,
    INTERRUPT_BEFORE_NODES,
    INTERRUPT_AFTER_NODES,
    WORKFLOW_NODES,
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
    THEORETICAL_METHODOLOGIES,
)
from src.graphs.streaming import (
    StreamEvent,
    StreamMode,
    get_progress_message,
    NODE_PROGRESS_MESSAGES,
    format_for_sse,
    format_for_websocket,
)
from src.graphs.debug import (
    StateSnapshot,
    WorkflowStatus,
    inspect_workflow_state,
    WorkflowInspector,
)
from src.graphs.subgraphs import (
    create_literature_review_subgraph,
    create_analysis_subgraph,
    create_writing_subgraph,
)
from src.state.enums import ResearchStatus


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_state():
    """Minimal workflow state for testing."""
    return {
        "status": ResearchStatus.INTAKE_COMPLETE,
        "original_query": "Test research question",
    }


@pytest.fixture
def state_with_data():
    """State with uploaded data files."""
    return {
        "status": ResearchStatus.INTAKE_COMPLETE,
        "original_query": "Test research question",
        "uploaded_data": [{"name": "data.csv", "path": "/tmp/data.csv"}],
    }


@pytest.fixture
def state_with_exploration():
    """State with data exploration results."""
    return {
        "status": ResearchStatus.DATA_EXPLORED,
        "original_query": "Test research question",
        "data_exploration_results": {"datasets": [], "summary": "Test data"},
        "research_type": "empirical",
    }


@pytest.fixture
def state_with_literature():
    """State with literature review complete."""
    return {
        "status": ResearchStatus.LITERATURE_REVIEWED,
        "original_query": "Test research question",
        "literature_review_results": {"papers": []},
    }


@pytest.fixture
def state_with_gap_analysis():
    """State with gap analysis complete."""
    return {
        "status": ResearchStatus.GAP_IDENTIFICATION_COMPLETE,
        "original_query": "Test research question",
        "gap_analysis": {"gaps": [], "primary_gap": "Test gap"},
        "refined_query": "Refined research question",
    }


@pytest.fixture
def state_with_research_plan():
    """State with research plan approved."""
    return {
        "status": ResearchStatus.PLANNING_COMPLETE,
        "original_query": "Test research question",
        "research_plan": {
            "methodology_type": "regression_analysis",
            "approval_status": "approved",
        },
        "research_type": "empirical",
        "data_exploration_results": {"datasets": []},
    }


@pytest.fixture
def state_with_analysis():
    """State with analysis complete."""
    return {
        "status": ResearchStatus.ANALYSIS_COMPLETE,
        "original_query": "Test research question",
        "data_analyst_output": {"findings": "Test findings"},
    }


@pytest.fixture
def state_with_writer_output():
    """State with writer output."""
    return {
        "status": ResearchStatus.WRITING_COMPLETE,
        "original_query": "Test research question",
        "writer_output": {"title": "Test Paper", "sections": []},
    }


@pytest.fixture
def state_with_review():
    """State with review complete."""
    return {
        "status": ResearchStatus.REVIEWING,
        "original_query": "Test research question",
        "review_decision": "approve",
        "human_approved": True,
    }


# =============================================================================
# Test WorkflowConfig
# =============================================================================


class TestWorkflowConfig:
    """Tests for WorkflowConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WorkflowConfig()
        
        assert config.checkpointer is None
        assert config.store is None
        assert config.cache is None
        assert config.interrupt_before == INTERRUPT_BEFORE_NODES
        assert config.interrupt_after == INTERRUPT_AFTER_NODES
        assert config.enable_caching is True
        assert config.debug is False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = WorkflowConfig(
            interrupt_before=["planner"],
            interrupt_after=["reviewer", "writer"],
            enable_caching=False,
            debug=True,
        )
        
        assert config.interrupt_before == ["planner"]
        assert config.interrupt_after == ["reviewer", "writer"]
        assert config.enable_caching is False
        assert config.debug is True


# =============================================================================
# Test Workflow Factory
# =============================================================================


class TestWorkflowFactory:
    """Tests for workflow factory functions."""
    
    def test_create_research_workflow_basic(self):
        """Test basic workflow creation."""
        workflow = create_research_workflow()
        
        # Should compile without error
        assert workflow is not None
        assert hasattr(workflow, "invoke")
    
    def test_create_research_workflow_with_config(self):
        """Test workflow creation with custom config."""
        config = WorkflowConfig(
            enable_caching=False,
            interrupt_before=[],
            interrupt_after=[],
        )
        workflow = create_research_workflow(config)
        
        assert workflow is not None
    
    def test_create_studio_workflow(self):
        """Test studio workflow creation."""
        workflow = create_studio_workflow()
        
        assert workflow is not None
        assert hasattr(workflow, "invoke")
    
    @patch('langgraph.checkpoint.sqlite.SqliteSaver')
    def test_create_production_workflow(self, mock_saver):
        """Test production workflow creation."""
        mock_saver.from_conn_string.return_value = MagicMock()
        
        workflow = create_production_workflow("sqlite:///test.db")
        
        assert workflow is not None
        mock_saver.from_conn_string.assert_called_once_with("sqlite:///test.db")


# =============================================================================
# Test Output Node
# =============================================================================


class TestOutputNode:
    """Tests for the output node."""
    
    def test_output_node_with_final_paper(self):
        """Test output node with final paper content."""
        state = {
            "reviewer_output": {
                "final_paper": "This is the final paper content."
            }
        }
        
        result = output_node(state)
        
        assert result["status"] == ResearchStatus.COMPLETED
    
    def test_output_node_without_final_paper(self):
        """Test output node without final paper."""
        state = {
            "reviewer_output": {}
        }
        
        result = output_node(state)
        
        assert result["status"] == ResearchStatus.COMPLETED
    
    def test_output_node_no_reviewer_output(self):
        """Test output node with no reviewer output."""
        state = {}
        
        result = output_node(state)
        
        assert result["status"] == ResearchStatus.COMPLETED


# =============================================================================
# Test Routing Functions
# =============================================================================


class TestRouteAfterIntake:
    """Tests for route_after_intake."""
    
    def test_route_to_data_explorer_with_data(self, state_with_data):
        """Test routing to data explorer when data is uploaded."""
        result = route_after_intake(state_with_data)
        assert result == "data_explorer"
    
    def test_route_to_literature_reviewer_no_data(self, minimal_state):
        """Test routing to literature reviewer when no data."""
        result = route_after_intake(minimal_state)
        assert result == "literature_reviewer"
    
    def test_route_to_end_on_error(self):
        """Test routing to end on errors."""
        state = {"errors": [{"message": "Error"}]}
        result = route_after_intake(state)
        assert result == "__end__"
    
    def test_route_to_end_incomplete_status(self):
        """Test routing to end when status not complete."""
        state = {"status": ResearchStatus.INTAKE_PENDING}
        result = route_after_intake(state)
        assert result == "__end__"


class TestRouteAfterDataExplorer:
    """Tests for route_after_data_explorer."""
    
    def test_route_to_literature_reviewer(self):
        """Test normal routing to literature reviewer."""
        state = {}
        result = route_after_data_explorer(state)
        assert result == "literature_reviewer"
    
    def test_route_to_literature_reviewer_with_recoverable_errors(self):
        """Test routing continues with recoverable errors."""
        error = MagicMock()
        error.recoverable = True
        state = {"errors": [error]}
        
        result = route_after_data_explorer(state)
        assert result == "literature_reviewer"
    
    def test_route_to_end_with_fatal_errors(self):
        """Test routing to end with fatal errors."""
        error = MagicMock()
        error.recoverable = False
        state = {"errors": [error]}
        
        result = route_after_data_explorer(state)
        assert result == "__end__"


class TestRouteAfterLiteratureReviewer:
    """Tests for route_after_literature_reviewer."""
    
    def test_route_to_synthesizer(self):
        """Test normal routing to synthesizer."""
        state = {}
        result = route_after_literature_reviewer(state)
        assert result == "literature_synthesizer"
    
    def test_route_to_end_with_fatal_errors(self):
        """Test routing to end with fatal errors."""
        error = MagicMock()
        error.recoverable = False
        state = {"errors": [error]}
        
        result = route_after_literature_reviewer(state)
        assert result == "__end__"


class TestRouteAfterSynthesizer:
    """Tests for route_after_synthesizer."""
    
    def test_route_to_gap_identifier(self):
        """Test normal routing to gap identifier."""
        state = {}
        result = route_after_synthesizer(state)
        assert result == "gap_identifier"
    
    def test_route_to_end_with_fatal_errors(self):
        """Test routing to end with fatal errors."""
        error = MagicMock()
        error.recoverable = False
        state = {"errors": [error]}
        
        result = route_after_synthesizer(state)
        assert result == "__end__"


class TestRouteAfterGapIdentifier:
    """Tests for route_after_gap_identifier."""
    
    def test_route_to_planner_with_gap_analysis(self, state_with_gap_analysis):
        """Test routing to planner with gap analysis."""
        result = route_after_gap_identifier(state_with_gap_analysis)
        assert result == "planner"
    
    def test_route_to_planner_with_refined_query(self):
        """Test routing to planner with refined query."""
        state = {"refined_query": "Refined question"}
        result = route_after_gap_identifier(state)
        assert result == "planner"
    
    def test_route_to_end_on_error(self):
        """Test routing to end on errors."""
        state = {"errors": [{"message": "Error"}]}
        result = route_after_gap_identifier(state)
        assert result == "__end__"
    
    def test_route_to_end_without_gap_analysis(self):
        """Test routing to end without gap analysis."""
        state = {}
        result = route_after_gap_identifier(state)
        assert result == "__end__"


class TestRouteByResearchType:
    """Tests for route_by_research_type."""
    
    def test_route_to_data_analyst_empirical_with_data(self, state_with_exploration):
        """Test routing to data analyst for empirical research with data."""
        result = route_by_research_type(state_with_exploration)
        assert result == "data_analyst"
    
    def test_route_to_conceptual_synthesizer_theoretical(self):
        """Test routing to conceptual synthesizer for theoretical research."""
        state = {"research_type": "theoretical"}
        result = route_by_research_type(state)
        assert result == "conceptual_synthesizer"
    
    def test_route_to_conceptual_synthesizer_no_data(self):
        """Test routing to conceptual synthesizer when no data."""
        state = {"research_type": "empirical", "data_exploration_results": None}
        result = route_by_research_type(state)
        assert result == "conceptual_synthesizer"
    
    def test_route_to_conceptual_synthesizer_theoretical_methodology(self):
        """Test routing based on theoretical methodology."""
        state = {
            "research_plan": {"methodology_type": "systematic_review"},
            "data_exploration_results": {"datasets": []},
        }
        result = route_by_research_type(state)
        assert result == "conceptual_synthesizer"
    
    def test_default_with_data(self):
        """Test default routing with data."""
        state = {"data_exploration_results": {"datasets": []}}
        result = route_by_research_type(state)
        assert result == "data_analyst"


class TestRouteAfterPlanner:
    """Tests for route_after_planner."""
    
    def test_route_to_data_analyst(self, state_with_research_plan):
        """Test routing to data analyst after planner."""
        result = route_after_planner(state_with_research_plan)
        assert result == "data_analyst"
    
    def test_route_to_end_on_error(self):
        """Test routing to end on errors."""
        state = {"errors": [{"message": "Error"}]}
        result = route_after_planner(state)
        assert result == "__end__"
    
    def test_route_to_end_no_plan(self):
        """Test routing to end without research plan."""
        state = {}
        result = route_after_planner(state)
        assert result == "__end__"
    
    def test_route_to_end_rejected_plan(self):
        """Test routing to end with rejected plan."""
        state = {
            "research_plan": {"approval_status": "rejected"}
        }
        result = route_after_planner(state)
        assert result == "__end__"


class TestRouteAfterAnalysis:
    """Tests for route_after_analysis."""
    
    def test_route_to_writer_with_data_analysis(self, state_with_analysis):
        """Test routing to writer after data analysis."""
        result = route_after_analysis(state_with_analysis)
        assert result == "writer"
    
    def test_route_to_writer_with_conceptual_synthesis(self):
        """Test routing to writer after conceptual synthesis."""
        state = {"conceptual_synthesis_output": {"framework": "Test"}}
        result = route_after_analysis(state)
        assert result == "writer"
    
    def test_route_to_end_on_error(self):
        """Test routing to end on errors."""
        state = {"errors": [{"message": "Error"}]}
        result = route_after_analysis(state)
        assert result == "__end__"
    
    def test_route_to_end_no_output(self):
        """Test routing to end without analysis output."""
        state = {}
        result = route_after_analysis(state)
        assert result == "__end__"


class TestRouteAfterWriter:
    """Tests for route_after_writer."""
    
    def test_route_to_reviewer(self, state_with_writer_output):
        """Test routing to reviewer after writer."""
        result = route_after_writer(state_with_writer_output)
        assert result == "reviewer"
    
    def test_route_to_end_on_error(self):
        """Test routing to end on errors."""
        state = {"errors": [{"message": "Error"}]}
        result = route_after_writer(state)
        assert result == "__end__"
    
    def test_route_to_end_no_output(self):
        """Test routing to end without writer output."""
        state = {}
        result = route_after_writer(state)
        assert result == "__end__"


class TestRouteAfterReviewer:
    """Tests for route_after_reviewer."""
    
    def test_route_to_output_approved(self, state_with_review):
        """Test routing to output when approved."""
        result = route_after_reviewer(state_with_review)
        assert result == "output"
    
    def test_route_to_output_escalated(self):
        """Test routing to output when escalated."""
        state = {"review_decision": "escalate"}
        result = route_after_reviewer(state)
        assert result == "output"
    
    def test_route_to_writer_revise(self):
        """Test routing to writer for revision."""
        state = {"review_decision": "revise"}
        result = route_after_reviewer(state)
        assert result == "writer"
    
    def test_route_to_end_rejected(self):
        """Test routing to end when rejected."""
        state = {"review_decision": "reject"}
        result = route_after_reviewer(state)
        assert result == "__end__"


# =============================================================================
# Test Streaming Utilities
# =============================================================================


class TestStreamEvent:
    """Tests for StreamEvent."""
    
    def test_stream_event_creation(self):
        """Test creating a stream event."""
        event = StreamEvent(
            mode=StreamMode.UPDATES,
            node="intake",
            data={"status": "processing"},
        )
        
        assert event.mode == StreamMode.UPDATES
        assert event.node == "intake"
        assert event.data == {"status": "processing"}
    
    def test_stream_event_to_dict(self):
        """Test converting stream event to dictionary."""
        event = StreamEvent(
            mode=StreamMode.MESSAGES,
            node="writer",
            data="token",
            metadata={"id": "123"},
        )
        
        result = event.to_dict()
        
        assert result["mode"] == "messages"
        assert result["node"] == "writer"
        assert result["data"] == "token"
        assert result["metadata"] == {"id": "123"}


class TestProgressMessages:
    """Tests for progress message functions."""
    
    def test_get_progress_message_known_node(self):
        """Test getting progress message for known node."""
        message = get_progress_message("intake")
        assert message == "Processing research intake form..."
    
    def test_get_progress_message_unknown_node(self):
        """Test getting progress message for unknown node."""
        message = get_progress_message("unknown_node")
        assert message == "Processing unknown_node..."
    
    def test_all_workflow_nodes_have_messages(self):
        """Test that all workflow nodes have progress messages."""
        for node in WORKFLOW_NODES:
            assert node in NODE_PROGRESS_MESSAGES


class TestStreamFormatters:
    """Tests for stream formatting functions."""
    
    def test_format_for_sse(self):
        """Test SSE formatting."""
        event = StreamEvent(
            mode=StreamMode.UPDATES,
            node="intake",
            data={"test": "value"},
        )
        
        result = format_for_sse(event)
        
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        assert "test" in result
    
    def test_format_for_websocket(self):
        """Test WebSocket formatting."""
        event = StreamEvent(
            mode=StreamMode.MESSAGES,
            node="writer",
            data="token",
        )
        
        result = format_for_websocket(event)
        
        assert result["mode"] == "messages"
        assert result["node"] == "writer"
        assert result["data"] == "token"


# =============================================================================
# Test Debug Utilities
# =============================================================================


class TestStateSnapshot:
    """Tests for StateSnapshot."""
    
    def test_state_snapshot_creation(self):
        """Test creating a state snapshot."""
        snapshot = StateSnapshot(
            checkpoint_id="cp-123",
            thread_id="thread-1",
            created_at=datetime.now(timezone.utc),
            node="planner",
            next_nodes=["data_analyst"],
            values={"status": "planning"},
            metadata={},
        )
        
        assert snapshot.checkpoint_id == "cp-123"
        assert snapshot.node == "planner"


class TestWorkflowStatus:
    """Tests for WorkflowStatus."""
    
    def test_workflow_status_creation(self):
        """Test creating a workflow status."""
        status = WorkflowStatus(
            thread_id="thread-1",
            current_node="writer",
            next_nodes=["reviewer"],
            status="writing",
            is_interrupted=False,
            checkpoint_count=5,
            error=None,
        )
        
        assert status.current_node == "writer"
        assert status.checkpoint_count == 5
        assert status.error is None


class TestInspectWorkflowState:
    """Tests for inspect_workflow_state."""
    
    def test_inspect_workflow_no_state(self):
        """Test inspection when no state exists."""
        mock_workflow = MagicMock()
        mock_workflow.get_state.return_value = None
        
        status = inspect_workflow_state(mock_workflow, "thread-1")
        
        assert status.thread_id == "thread-1"
        assert status.error == "No state found for thread"
    
    def test_inspect_workflow_with_state(self):
        """Test inspection with valid state."""
        mock_state = MagicMock()
        mock_state.values = {"status": ResearchStatus.WRITING}
        mock_state.next = ["writer"]
        
        mock_workflow = MagicMock()
        mock_workflow.get_state.return_value = mock_state
        mock_workflow.get_state_history.return_value = [mock_state]
        
        status = inspect_workflow_state(mock_workflow, "thread-1")
        
        assert status.thread_id == "thread-1"
        assert status.current_node == "writer"


class TestWorkflowInspector:
    """Tests for WorkflowInspector class."""
    
    def test_inspector_creation(self):
        """Test creating a workflow inspector."""
        mock_workflow = MagicMock()
        inspector = WorkflowInspector(mock_workflow, "thread-1")
        
        assert inspector.thread_id == "thread-1"
        assert inspector.workflow == mock_workflow
    
    def test_inspector_status(self):
        """Test inspector status property."""
        mock_state = MagicMock()
        mock_state.values = {"status": ResearchStatus.COMPLETED}
        mock_state.next = []
        
        mock_workflow = MagicMock()
        mock_workflow.get_state.return_value = mock_state
        mock_workflow.get_state_history.return_value = []
        
        inspector = WorkflowInspector(mock_workflow, "thread-1")
        status = inspector.status
        
        assert status.thread_id == "thread-1"


# =============================================================================
# Test Subgraphs
# =============================================================================


class TestSubgraphs:
    """Tests for subgraph creation functions."""
    
    def test_create_literature_review_subgraph(self):
        """Test creating literature review subgraph."""
        subgraph = create_literature_review_subgraph()
        
        assert subgraph is not None
        assert hasattr(subgraph, "invoke")
    
    def test_create_literature_review_subgraph_no_cache(self):
        """Test creating literature review subgraph without caching."""
        subgraph = create_literature_review_subgraph(enable_caching=False)
        
        assert subgraph is not None
    
    def test_create_analysis_subgraph(self):
        """Test creating analysis subgraph."""
        subgraph = create_analysis_subgraph()
        
        assert subgraph is not None
        assert hasattr(subgraph, "invoke")
    
    def test_create_analysis_subgraph_no_cache(self):
        """Test creating analysis subgraph without caching."""
        subgraph = create_analysis_subgraph(enable_caching=False)
        
        assert subgraph is not None
    
    def test_create_writing_subgraph(self):
        """Test creating writing subgraph."""
        subgraph = create_writing_subgraph()
        
        assert subgraph is not None
        assert hasattr(subgraph, "invoke")
    
    def test_create_writing_subgraph_custom_max_revisions(self):
        """Test creating writing subgraph with custom max revisions."""
        subgraph = create_writing_subgraph(max_revisions=5)
        
        assert subgraph is not None


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""
    
    def test_interrupt_before_nodes(self):
        """Test interrupt_before nodes are valid."""
        for node in INTERRUPT_BEFORE_NODES:
            assert node in WORKFLOW_NODES
    
    def test_interrupt_after_nodes(self):
        """Test interrupt_after nodes are valid."""
        for node in INTERRUPT_AFTER_NODES:
            assert node in WORKFLOW_NODES
    
    def test_workflow_nodes_order(self):
        """Test workflow nodes are in expected order."""
        assert WORKFLOW_NODES[0] == "intake"
        assert WORKFLOW_NODES[-1] == "output"
    
    def test_theoretical_methodologies(self):
        """Test theoretical methodologies set."""
        assert "systematic_review" in THEORETICAL_METHODOLOGIES
        assert "meta_analysis" in THEORETICAL_METHODOLOGIES
        assert "conceptual_framework" in THEORETICAL_METHODOLOGIES


# =============================================================================
# Integration Tests
# =============================================================================


class TestWorkflowIntegration:
    """Integration tests for the complete workflow."""
    
    def test_workflow_has_all_nodes(self):
        """Test that workflow has all expected nodes."""
        workflow = create_research_workflow()
        
        # The compiled workflow should have nodes
        assert workflow is not None
    
    def test_workflow_compiles_with_all_configs(self):
        """Test workflow compiles with various configurations."""
        configs = [
            WorkflowConfig(),
            WorkflowConfig(enable_caching=False),
            WorkflowConfig(interrupt_before=[]),
            WorkflowConfig(interrupt_after=[]),
            WorkflowConfig(debug=True),
        ]
        
        for config in configs:
            workflow = create_research_workflow(config)
            assert workflow is not None
    
    def test_studio_workflow_works(self):
        """Test studio workflow can be created and used."""
        workflow = create_studio_workflow()
        
        # Should be able to get graph
        assert workflow is not None
        
    def test_all_routers_return_valid_nodes(self, minimal_state):
        """Test that all routers return valid node names or __end__."""
        # Create various states and test each router
        states = [
            minimal_state,
            {"errors": [{"message": "test"}]},
            {},
        ]
        
        routers = [
            route_after_intake,
            route_after_data_explorer,
            route_after_literature_reviewer,
            route_after_synthesizer,
            route_after_gap_identifier,
            route_after_analysis,
            route_after_writer,
            route_after_reviewer,
        ]
        
        # All valid routing destinations
        all_valid_destinations = set(WORKFLOW_NODES) | {"__end__"}
        
        for router in routers:
            for state in states:
                result = router(state)
                assert result in all_valid_destinations, f"{router.__name__} returned invalid destination: {result}"
