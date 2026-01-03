"""Unit tests for Sprint 14: Data Acquisition Node.

Tests the intelligent data acquisition functionality.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.state.enums import (
    DataRequirementPriority,
    AcquisitionStatus,
    ResearchStatus,
)
from src.state.models import (
    DataRequirement,
    DataAcquisitionTask,
    AcquisitionFailure,
    AcquiredDataset,
    TimeRange,
)
from src.nodes.data_acquisition import (
    data_acquisition_node,
    route_after_acquisition,
    should_skip_acquisition,
    _parse_data_requirements_from_plan,
    _check_requirements_against_uploads,
    _find_acquisition_source,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def base_state():
    """Create a base workflow state for testing."""
    return {
        "original_query": "Analyze stock returns for AAPL",
        "status": ResearchStatus.PLANNING,
        "loaded_datasets": [],
        "research_plan": None,
        "messages": [],
    }


@pytest.fixture
def state_with_plan(base_state):
    """Create a state with a research plan."""
    base_state["research_plan"] = {
        "methodology": "regression analysis on stock returns",
        "methodology_type": "empirical",
        "data_requirements": [
            {
                "variable_name": "stock_prices",
                "data_type": "stock_prices",
                "required_fields": ["close", "volume"],
                "entities": ["AAPL"],
                "priority": "required",
            }
        ],
    }
    return base_state


@pytest.fixture
def state_theoretical(base_state):
    """Create a state for theoretical research."""
    base_state["research_plan"] = {
        "methodology": "conceptual framework development",
        "methodology_type": "analytical_model",
    }
    return base_state


@pytest.fixture
def data_requirement():
    """Create a sample data requirement."""
    return DataRequirement(
        variable_name="stock_prices",
        data_type="stock_prices",
        description="Daily stock prices for AAPL",
        required_fields=["close", "volume"],
        entities=["AAPL"],
        priority=DataRequirementPriority.REQUIRED,
        time_range=TimeRange(
            start_date="2020-01-01",
            end_date="2023-12-31",
        ),
    )


# =============================================================================
# Data Requirement Parsing Tests
# =============================================================================


class TestDataRequirementParsing:
    """Tests for parsing data requirements from research plans."""
    
    def test_parse_explicit_requirements(self, state_with_plan):
        """Explicit requirements should be parsed correctly."""
        requirements = _parse_data_requirements_from_plan(state_with_plan)
        
        assert len(requirements) >= 1
        assert requirements[0].variable_name == "stock_prices"
        assert requirements[0].data_type == "stock_prices"
        assert "AAPL" in requirements[0].entities
    
    def test_parse_no_plan_returns_empty(self, base_state):
        """No plan should return empty requirements."""
        requirements = _parse_data_requirements_from_plan(base_state)
        assert len(requirements) == 0
    
    def test_parse_infers_from_methodology(self, base_state):
        """Requirements should be inferred from methodology if not explicit."""
        base_state["research_plan"] = {
            "methodology": "stock price analysis using regression",
        }
        requirements = _parse_data_requirements_from_plan(base_state)
        
        # Should infer stock data requirement
        assert len(requirements) >= 1
        assert any(r.data_type == "stock_prices" for r in requirements)


# =============================================================================
# Requirement Checking Tests
# =============================================================================


class TestRequirementChecking:
    """Tests for checking requirements against uploaded data."""
    
    @patch("src.nodes.data_acquisition.get_registry")
    def test_check_satisfied_by_upload(self, mock_registry, data_requirement):
        """Requirements should be marked satisfied if upload matches."""
        # Create a mock DataFrame with matching columns
        mock_df = MagicMock()
        mock_df.columns = ["date", "open", "high", "low", "close", "volume"]
        
        mock_reg = MagicMock()
        mock_reg.get.return_value = mock_df
        mock_registry.return_value = mock_reg
        
        satisfied, unsatisfied = _check_requirements_against_uploads(
            [data_requirement],
            ["stock_data.csv"],
            {},
        )
        
        assert data_requirement.requirement_id in satisfied
        assert len(unsatisfied) == 0
    
    @patch("src.nodes.data_acquisition.get_registry")
    def test_check_unsatisfied_no_match(self, mock_registry, data_requirement):
        """Requirements should be unsatisfied if no upload matches."""
        # Create a mock DataFrame without required columns
        mock_df = MagicMock()
        mock_df.columns = ["id", "name", "description"]
        
        mock_reg = MagicMock()
        mock_reg.get.return_value = mock_df
        mock_registry.return_value = mock_reg
        
        satisfied, unsatisfied = _check_requirements_against_uploads(
            [data_requirement],
            ["other_data.csv"],
            {},
        )
        
        assert len(satisfied) == 0
        assert data_requirement in unsatisfied


# =============================================================================
# Source Finding Tests
# =============================================================================


class TestSourceFinding:
    """Tests for finding acquisition sources."""
    
    def test_find_source_stock_data(self, data_requirement):
        """Stock data requirements should map to yfinance."""
        result = _find_acquisition_source(data_requirement)
        
        assert result is not None
        source, tool, params = result
        assert source == "yfinance"
        assert tool == "acquire_stock_data"
        assert params.get("ticker") == "AAPL"
    
    def test_find_source_economic_indicator(self):
        """Economic indicator requirements should map to FRED."""
        req = DataRequirement(
            variable_name="gdp",
            data_type="economic_indicator",
            entities=["GDP"],
        )
        
        result = _find_acquisition_source(req)
        
        assert result is not None
        source, tool, params = result
        assert source == "fred"
        assert tool == "acquire_economic_indicator"
    
    def test_find_source_crypto(self):
        """Crypto requirements should map to CoinGecko."""
        req = DataRequirement(
            variable_name="btc_price",
            data_type="crypto_prices",
            entities=["bitcoin"],
        )
        
        result = _find_acquisition_source(req)
        
        assert result is not None
        source, tool, params = result
        assert source == "coingecko"
        assert tool == "acquire_crypto_data"
    
    def test_find_source_unknown_type(self):
        """Unknown data types should return None."""
        req = DataRequirement(
            variable_name="custom_data",
            data_type="custom_unknown_type",
        )
        
        result = _find_acquisition_source(req)
        assert result is None


# =============================================================================
# Node Execution Tests
# =============================================================================


class TestDataAcquisitionNode:
    """Tests for the data acquisition node."""
    
    @patch("src.nodes.data_acquisition.acquire_stock_data")
    @patch("src.nodes.data_acquisition.get_registry")
    def test_node_acquires_missing_data(self, mock_registry, mock_acquire, state_with_plan):
        """Node should acquire data for unsatisfied requirements."""
        # Setup mock
        mock_acquire.invoke.return_value = {
            "status": "success",
            "dataset_name": "AAPL_stock",
        }
        
        mock_df = MagicMock()
        mock_df.columns = ["date", "close"]
        mock_df.__len__ = lambda self: 100
        
        mock_reg = MagicMock()
        mock_reg.get.return_value = mock_df
        mock_registry.return_value = mock_reg
        
        result = data_acquisition_node(state_with_plan)
        
        assert "data_acquisition_plan" in result
        assert "acquired_datasets" in result
        # Acquisition may succeed or fail depending on mock
    
    def test_node_no_requirements(self, base_state):
        """Node should handle states with no requirements gracefully."""
        base_state["research_plan"] = {}
        
        result = data_acquisition_node(base_state)
        
        assert "data_acquisition_plan" in result
        assert result["data_acquisition_plan"]["requirements"] == []


# =============================================================================
# Routing Tests
# =============================================================================


class TestRouting:
    """Tests for routing functions."""
    
    def test_route_no_failures_to_analyst(self, base_state):
        """No failures should route to data_analyst."""
        base_state["acquisition_failures"] = []
        
        result = route_after_acquisition(base_state)
        assert result == "data_analyst"
    
    def test_route_required_failure_to_interrupt(self, base_state):
        """Required data failures should route to human_interrupt."""
        base_state["acquisition_failures"] = [
            {
                "requirement": {
                    "variable_name": "stock_prices",
                    "priority": "required",
                },
                "error_messages": ["Failed to fetch data"],
            }
        ]
        
        result = route_after_acquisition(base_state)
        assert result == "human_interrupt"
    
    def test_route_optional_failure_continues(self, base_state):
        """Optional data failures should continue to data_analyst."""
        base_state["acquisition_failures"] = [
            {
                "requirement": {
                    "variable_name": "market_sentiment",
                    "priority": "optional",
                },
                "error_messages": ["Failed to fetch data"],
            }
        ]
        
        result = route_after_acquisition(base_state)
        assert result == "data_analyst"


# =============================================================================
# Skip Acquisition Tests
# =============================================================================


class TestSkipAcquisition:
    """Tests for skip acquisition logic."""
    
    def test_skip_no_plan(self, base_state):
        """Should skip if no research plan."""
        assert should_skip_acquisition(base_state) is True
    
    def test_skip_theoretical_research(self, state_theoretical):
        """Should skip for theoretical research."""
        state_theoretical["research_type"] = "theoretical"
        assert should_skip_acquisition(state_theoretical) is True
    
    def test_not_skip_empirical_research(self, state_with_plan):
        """Should not skip for empirical research with plan."""
        state_with_plan["research_type"] = "empirical"
        assert should_skip_acquisition(state_with_plan) is False


# =============================================================================
# Model Tests
# =============================================================================


class TestModels:
    """Tests for Sprint 14 models."""
    
    def test_time_range_model(self):
        """TimeRange model should work correctly."""
        tr = TimeRange(
            start_date="2020-01-01",
            end_date="2023-12-31",
        )
        assert tr.start_date == "2020-01-01"
        assert tr.end_date == "2023-12-31"
    
    def test_data_requirement_defaults(self):
        """DataRequirement should have sensible defaults."""
        req = DataRequirement(
            variable_name="test",
            data_type="stock_prices",
        )
        assert req.priority == DataRequirementPriority.REQUIRED
        assert req.required_fields == []
        assert req.requirement_id is not None
    
    def test_acquisition_task_model(self, data_requirement):
        """DataAcquisitionTask model should work correctly."""
        task = DataAcquisitionTask(
            requirement=data_requirement,
            source="yfinance",
            tool_to_use="acquire_stock_data",
            params={"ticker": "AAPL"},
        )
        assert task.status == AcquisitionStatus.PENDING
        assert task.source == "yfinance"
    
    def test_acquisition_failure_model(self, data_requirement):
        """AcquisitionFailure model should work correctly."""
        failure = AcquisitionFailure(
            requirement=data_requirement,
            attempted_sources=["yfinance", "alpha_vantage"],
            error_messages=["Rate limited", "API error"],
            user_action_needed="Please upload stock data manually",
        )
        assert len(failure.attempted_sources) == 2
        assert len(failure.error_messages) == 2
    
    def test_acquired_dataset_model(self):
        """AcquiredDataset model should work correctly."""
        dataset = AcquiredDataset(
            dataset_name="AAPL_stock",
            source="yfinance",
            requirement_id="req123",
            row_count=1000,
            column_count=6,
        )
        assert dataset.dataset_name == "AAPL_stock"
        assert dataset.row_count == 1000
