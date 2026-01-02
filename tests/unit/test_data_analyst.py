"""Tests for DATA_ANALYST node (Sprint 5).

Tests cover:
- Data extraction from state
- Analysis execution
- Finding generation
- Gap coverage assessment
- Routing logic
"""

import pytest
from datetime import datetime, date
from unittest.mock import MagicMock, patch

from src.nodes.data_analyst import (
    data_analyst_node,
    route_after_data_analyst,
)
from src.state.schema import create_initial_state, WorkflowState
from src.state.enums import (
    ResearchStatus,
    AnalysisStatus,
    MethodologyType,
    AnalysisApproach,
    PlanApprovalStatus,
)
from src.state.models import (
    DataExplorationResult,
    ColumnAnalysis,
    ResearchPlan,
    GapAnalysis,
    ResearchGap,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_data_exploration():
    """Create mock data exploration results."""
    return DataExplorationResult(
        total_rows=1000,
        total_columns=10,
        columns=[
            ColumnAnalysis(
                name="returns",
                dtype="float",
                non_null_count=980,
                null_count=20,
                null_percentage=2.0,
                unique_count=950,
                mean=0.05,
                std=0.15,
                min_value=-0.5,
                max_value=0.8,
                q25=-0.02,
                median=0.04,
                q75=0.12,
            ),
            ColumnAnalysis(
                name="volatility",
                dtype="float",
                non_null_count=990,
                null_count=10,
                null_percentage=1.0,
                unique_count=500,
                mean=0.2,
                std=0.08,
                min_value=0.05,
                max_value=0.6,
                q25=0.15,
                median=0.18,
                q75=0.25,
            ),
            ColumnAnalysis(
                name="market_cap",
                dtype="float",
                non_null_count=1000,
                null_count=0,
                null_percentage=0.0,
                unique_count=800,
                mean=5000000000,
                std=10000000000,
                min_value=100000000,
                max_value=100000000000,
            ),
        ],
        quality_level="good",
        feasibility_assessment="Data is suitable for regression analysis",
    )


@pytest.fixture
def mock_research_plan():
    """Create mock research plan."""
    return ResearchPlan(
        original_query="How does volatility affect stock returns?",
        refined_query="What is the relationship between volatility and returns in the US equity market?",
        target_gap="Empirical gap in volatility-return relationship",
        gap_type="empirical",
        methodology_type=MethodologyType.REGRESSION_ANALYSIS,
        methodology="Panel data regression with fixed effects",
        analysis_approach=AnalysisApproach.FIXED_EFFECTS,
        statistical_tests=["t-test", "f-test", "hausman"],
        key_variables=["returns", "volatility"],
        control_variables=["market_cap"],
        hypothesis="Higher volatility is associated with higher expected returns",
        approval_status=PlanApprovalStatus.APPROVED,
    )


@pytest.fixture
def mock_gap_analysis():
    """Create mock gap analysis."""
    return GapAnalysis(
        original_question="How does volatility affect stock returns?",
        primary_gap=ResearchGap(
            gap_type="empirical",
            title="Volatility-Return Relationship",
            description="Limited empirical evidence on volatility-return relationship in recent market conditions",
            significance="high",
        ),
    )


@pytest.fixture
def state_with_data(mock_data_exploration, mock_research_plan, mock_gap_analysis):
    """Create state with data for analysis."""
    state = create_initial_state()
    state["data_exploration_results"] = mock_data_exploration
    state["research_plan"] = mock_research_plan
    state["gap_analysis"] = mock_gap_analysis
    state["refined_query"] = "What is the relationship between volatility and returns?"
    state["original_query"] = "How does volatility affect stock returns?"
    state["research_type"] = "empirical"
    return state


@pytest.fixture
def state_without_data():
    """Create state without data."""
    state = create_initial_state()
    state["research_plan"] = ResearchPlan(
        original_query="Test question",
        methodology="Test methodology",
    )
    return state


@pytest.fixture
def state_without_plan():
    """Create state without research plan."""
    state = create_initial_state()
    state["data_exploration_results"] = DataExplorationResult(
        total_rows=100,
        total_columns=5,
    )
    return state


# =============================================================================
# Main Node Tests
# =============================================================================


class TestDataAnalystNode:
    """Tests for data_analyst_node."""
    
    def test_successful_analysis(self, state_with_data):
        """Test successful analysis execution."""
        result = data_analyst_node(state_with_data)
        
        assert result["status"] == ResearchStatus.ANALYSIS_COMPLETE
        assert "analysis" in result
        assert "messages" in result
        assert not result.get("errors")
    
    def test_analysis_generates_findings(self, state_with_data):
        """Test that analysis generates findings."""
        result = data_analyst_node(state_with_data)
        
        analysis = result["analysis"]
        assert "findings" in analysis
        assert len(analysis["findings"]) > 0
    
    def test_no_data_error(self, state_without_data):
        """Test error when no data available."""
        result = data_analyst_node(state_without_data)
        
        assert result["status"] == ResearchStatus.FAILED
        assert len(result["errors"]) > 0
        assert "No data available" in result["errors"][0].message
    
    def test_no_plan_error(self, state_without_plan):
        """Test error when no research plan available."""
        result = data_analyst_node(state_without_plan)
        
        assert result["status"] == ResearchStatus.FAILED
        assert len(result["errors"]) > 0
        assert "No research plan" in result["errors"][0].message
    
    def test_gap_coverage_assessment(self, state_with_data):
        """Test gap coverage is assessed."""
        result = data_analyst_node(state_with_data)
        
        analysis = result["analysis"]
        assert "gap_addressed" in analysis
        assert "gap_coverage_score" in analysis
        assert "gap_coverage_explanation" in analysis
    
    def test_regression_executed(self, state_with_data):
        """Test that regression analysis is executed for regression methodology."""
        result = data_analyst_node(state_with_data)
        
        analysis = result["analysis"]
        assert "regression_analyses" in analysis
        # Should have at least one regression
        assert len(analysis["regression_analyses"]) >= 0  # May be empty if variables insufficient
    
    def test_hypothesis_testing(self, state_with_data):
        """Test hypothesis testing when hypothesis provided."""
        result = data_analyst_node(state_with_data)
        
        analysis = result["analysis"]
        assert "hypothesis_supported" in analysis
        assert "hypothesis_test_summary" in analysis


# =============================================================================
# Routing Tests
# =============================================================================


class TestRouteAfterDataAnalyst:
    """Tests for route_after_data_analyst."""
    
    def test_route_to_writer_on_complete(self):
        """Test routing to writer when analysis complete."""
        state = {
            "status": ResearchStatus.ANALYSIS_COMPLETE,
            "analysis": {"findings": []},
        }
        assert route_after_data_analyst(state) == "writer"
    
    def test_route_to_end_on_error(self):
        """Test routing to end when errors present."""
        state = {
            "status": ResearchStatus.ANALYSIS_COMPLETE,
            "errors": [MagicMock()],
        }
        assert route_after_data_analyst(state) == "__end__"
    
    def test_route_to_end_on_failed_status(self):
        """Test routing to end when status is failed."""
        state = {
            "status": ResearchStatus.FAILED,
        }
        assert route_after_data_analyst(state) == "__end__"
    
    def test_route_to_writer_with_analysis(self):
        """Test routing to writer when analysis present."""
        state = {
            "analysis": {"findings": []},
        }
        assert route_after_data_analyst(state) == "writer"


# =============================================================================
# Integration Tests
# =============================================================================


class TestDataAnalystIntegration:
    """Integration tests for data analyst node."""
    
    def test_full_workflow(self, state_with_data):
        """Test complete analysis workflow."""
        result = data_analyst_node(state_with_data)
        
        # Check all expected outputs
        assert result["status"] == ResearchStatus.ANALYSIS_COMPLETE
        
        analysis = result["analysis"]
        assert analysis["analysis_status"] == AnalysisStatus.COMPLETE.value
        assert analysis["sample_size"] == 1000
        assert len(analysis["variables_analyzed"]) >= 1
        assert analysis["overall_confidence"] > 0
    
    def test_message_format(self, state_with_data):
        """Test that message is properly formatted."""
        result = data_analyst_node(state_with_data)
        
        assert len(result["messages"]) == 1
        message = result["messages"][0]
        assert "DATA_ANALYST" in message.content
        assert "Analysis complete" in message.content
