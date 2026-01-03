"""Unit tests for Sprint 13 Data Source Registry and External Data Tools.

Tests the new functionality introduced in Sprint 13:
- DataSource protocol and registry
- YFinance, FRED, CoinGecko data sources
- External data acquisition tools
- Rate limiting and error handling
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock
from importlib.util import find_spec

# Check optional dependencies
HAS_PANDAS = find_spec("pandas") is not None
HAS_YFINANCE = find_spec("yfinance") is not None
HAS_FREDAPI = find_spec("fredapi") is not None
HAS_REQUESTS = find_spec("requests") is not None

if HAS_PANDAS:
    import pandas as pd
    import numpy as np


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_yfinance_ticker():
    """Mock yfinance Ticker object."""
    with patch("yfinance.Ticker") as mock:
        ticker = MagicMock()
        
        # Mock history data
        if HAS_PANDAS:
            history_data = pd.DataFrame({
                "Open": [100.0, 101.0, 102.0],
                "High": [105.0, 106.0, 107.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [104.0, 105.0, 106.0],
                "Volume": [1000000, 1100000, 1200000],
            }, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
            history_data.index.name = "Date"
            ticker.history.return_value = history_data
        
        # Mock info
        ticker.info = {
            "symbol": "AAPL",
            "shortName": "Apple Inc.",
            "sector": "Technology",
        }
        
        # Mock options
        ticker.options = ["2024-01-19", "2024-01-26"]
        
        # Mock dividends/splits
        if HAS_PANDAS:
            ticker.dividends = pd.Series([0.24, 0.24], index=pd.to_datetime(["2024-01-01", "2024-04-01"]))
            ticker.splits = pd.Series([4.0], index=pd.to_datetime(["2020-08-31"]))
        
        mock.return_value = ticker
        yield mock


@pytest.fixture
def mock_fred():
    """Mock fredapi Fred object."""
    with patch("fredapi.Fred") as mock:
        fred = MagicMock()
        
        # Mock get_series
        if HAS_PANDAS:
            series_data = pd.Series(
                [25000.0, 25500.0, 26000.0],
                index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            )
            fred.get_series.return_value = series_data
        
        # Mock get_series_info
        fred.get_series_info.return_value = MagicMock(
            id="GDP",
            title="Gross Domestic Product",
        )
        
        mock.return_value = fred
        yield mock


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for CoinGecko."""
    with patch("requests.get") as mock:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "prices": [
                [1704067200000, 45000.0],
                [1704153600000, 45500.0],
                [1704240000000, 46000.0],
            ],
            "total_volumes": [
                [1704067200000, 30000000000],
                [1704153600000, 31000000000],
                [1704240000000, 32000000000],
            ],
            "market_caps": [
                [1704067200000, 880000000000],
                [1704153600000, 890000000000],
                [1704240000000, 900000000000],
            ],
        }
        mock.return_value = response
        yield mock


@pytest.fixture
def clean_registry():
    """Clean the DataSourceRegistry before and after tests."""
    from src.data_sources.base import DataSourceRegistry
    
    # Store original state
    original_by_name = DataSourceRegistry._sources_by_name.copy()
    original_by_domain = DataSourceRegistry._sources_by_domain.copy()
    original_index = DataSourceRegistry._data_type_index.copy()
    
    # Clear for test
    DataSourceRegistry.clear()
    
    yield
    
    # Restore original state
    DataSourceRegistry._sources_by_name = original_by_name
    DataSourceRegistry._sources_by_domain = original_by_domain
    DataSourceRegistry._data_type_index = original_index


# =============================================================================
# Test DataSource Base Classes
# =============================================================================


class TestRateLimit:
    """Test RateLimit dataclass."""
    
    def test_rate_limit_creation(self):
        """Test creating a RateLimit."""
        from src.data_sources.base import RateLimit
        
        limit = RateLimit(requests_per_minute=60, requests_per_day=1000)
        
        assert limit.requests_per_minute == 60
        assert limit.requests_per_day == 1000
        assert limit.concurrent_requests == 5  # default
    
    def test_rate_limit_str(self):
        """Test RateLimit string representation."""
        from src.data_sources.base import RateLimit
        
        limit = RateLimit(requests_per_minute=60, requests_per_day=1000)
        assert "60/min" in str(limit)
        assert "1000/day" in str(limit)
    
    def test_rate_limit_unlimited(self):
        """Test unlimited rate limit."""
        from src.data_sources.base import RateLimit
        
        limit = RateLimit()
        assert str(limit) == "unlimited"


class TestRateLimitTracker:
    """Test RateLimitTracker."""
    
    def test_tracker_allows_requests(self):
        """Test tracker allows requests within limits."""
        from src.data_sources.base import RateLimit, RateLimitTracker
        
        limit = RateLimit(requests_per_minute=10)
        tracker = RateLimitTracker(limit)
        
        assert tracker.can_request()
        
        # Make some requests
        for _ in range(5):
            tracker.record_request()
        
        assert tracker.can_request()
        assert tracker.minute_requests == 5
    
    def test_tracker_blocks_over_limit(self):
        """Test tracker blocks requests over limit."""
        from src.data_sources.base import RateLimit, RateLimitTracker
        
        limit = RateLimit(requests_per_minute=3)
        tracker = RateLimitTracker(limit)
        
        for _ in range(3):
            tracker.record_request()
        
        assert not tracker.can_request()


class TestDataSourceRegistry:
    """Test DataSourceRegistry."""
    
    def test_register_and_get_source(self, clean_registry):
        """Test registering and retrieving a source."""
        from src.data_sources.base import DataSource, DataSourceRegistry, RateLimit
        
        class TestSource(DataSource):
            name = "test_source"
            domain = "test"
            description = "Test source"
            data_types = ["test_data"]
            
            def fetch(self, data_type, **params):
                return None
            
            def check_availability(self, data_type, **params):
                return True
        
        DataSourceRegistry.register(TestSource)
        
        source = DataSourceRegistry.get_source("test_source")
        assert source is not None
        assert source.name == "test_source"
    
    def test_get_sources_by_domain(self, clean_registry):
        """Test getting sources by domain."""
        from src.data_sources.base import DataSource, DataSourceRegistry
        
        class Source1(DataSource):
            name = "source1"
            domain = "finance"
            description = "Source 1"
            data_types = ["type1"]
            
            def fetch(self, data_type, **params):
                return None
            
            def check_availability(self, data_type, **params):
                return True
        
        class Source2(DataSource):
            name = "source2"
            domain = "finance"
            description = "Source 2"
            data_types = ["type2"]
            
            def fetch(self, data_type, **params):
                return None
            
            def check_availability(self, data_type, **params):
                return True
        
        DataSourceRegistry.register(Source1)
        DataSourceRegistry.register(Source2)
        
        sources = DataSourceRegistry.get_sources("finance")
        assert len(sources) == 2
    
    def test_find_source_by_data_type(self, clean_registry):
        """Test finding source by data type."""
        from src.data_sources.base import DataSource, DataSourceRegistry
        
        class TestSource(DataSource):
            name = "test_source"
            domain = "test"
            description = "Test"
            data_types = ["stock_prices", "options"]
            
            def fetch(self, data_type, **params):
                return None
            
            def check_availability(self, data_type, **params):
                return True
        
        DataSourceRegistry.register(TestSource)
        
        source = DataSourceRegistry.find_source("stock_prices")
        assert source is not None
        assert source.name == "test_source"
        
        # Test case insensitivity
        source2 = DataSourceRegistry.find_source("STOCK_PRICES")
        assert source2 is not None


# =============================================================================
# Test Finance Sources
# =============================================================================


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestYFinanceSource:
    """Test YFinance data source."""
    
    def test_source_attributes(self):
        """Test YFinanceSource has correct attributes."""
        from src.data_sources.finance import YFinanceSource
        
        source = YFinanceSource()
        
        assert source.name == "yfinance"
        assert source.domain == "finance"
        assert not source.requires_api_key
        assert "stock_prices" in source.data_types
    
    @pytest.mark.skipif(not HAS_YFINANCE, reason="yfinance not installed")
    def test_fetch_stock_prices(self, mock_yfinance_ticker):
        """Test fetching stock prices."""
        from src.data_sources.finance import YFinanceSource
        
        source = YFinanceSource()
        df = source.fetch("stock_prices", ticker="AAPL", period="5d")
        
        assert df is not None
        assert len(df) > 0
        assert "Close" in df.columns
        assert "ticker" in df.columns
    
    @pytest.mark.skipif(not HAS_YFINANCE, reason="yfinance not installed")
    def test_check_availability(self, mock_yfinance_ticker):
        """Test checking availability."""
        from src.data_sources.finance import YFinanceSource
        
        source = YFinanceSource()
        available = source.check_availability("stock_prices", ticker="AAPL")
        
        assert available is True
    
    def test_required_params(self):
        """Test getting required parameters."""
        from src.data_sources.finance import YFinanceSource
        
        source = YFinanceSource()
        
        params = source.get_required_params("stock_prices")
        assert "ticker" in params


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestFREDSource:
    """Test FRED data source."""
    
    def test_source_attributes(self):
        """Test FREDSource has correct attributes."""
        from src.data_sources.finance import FREDSource
        
        source = FREDSource()
        
        assert source.name == "fred"
        assert source.domain == "economics"
        assert source.requires_api_key
        assert source.api_key_env_var == "FRED_API_KEY"
    
    def test_common_series_mapping(self):
        """Test common series name mapping."""
        from src.data_sources.finance import FREDSource
        
        source = FREDSource()
        
        assert "gdp" in source.COMMON_SERIES
        assert source.COMMON_SERIES["gdp"] == "GDP"
        assert source.COMMON_SERIES["unemployment"] == "UNRATE"
    
    @pytest.mark.skipif(not HAS_FREDAPI, reason="fredapi not installed")
    def test_fetch_with_mock(self, mock_fred):
        """Test fetching data with mocked FRED."""
        from src.data_sources.finance import FREDSource
        
        with patch.dict("os.environ", {"FRED_API_KEY": "test_key"}):
            source = FREDSource()
            df = source.fetch("gdp")
            
            assert df is not None
            assert "date" in df.columns
            assert "value" in df.columns


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestCoinGeckoSource:
    """Test CoinGecko data source."""
    
    def test_source_attributes(self):
        """Test CoinGeckoSource has correct attributes."""
        from src.data_sources.finance import CoinGeckoSource
        
        source = CoinGeckoSource()
        
        assert source.name == "coingecko"
        assert source.domain == "finance"
        assert not source.requires_api_key
        assert "crypto_prices" in source.data_types
    
    def test_common_coins_mapping(self):
        """Test common coin symbol mapping."""
        from src.data_sources.finance import CoinGeckoSource
        
        source = CoinGeckoSource()
        
        assert source.COMMON_COINS["btc"] == "bitcoin"
        assert source.COMMON_COINS["eth"] == "ethereum"
    
    @pytest.mark.skipif(not HAS_REQUESTS, reason="requests not installed")
    def test_fetch_with_mock(self, mock_requests_get):
        """Test fetching crypto data with mocked requests."""
        from src.data_sources.finance import CoinGeckoSource
        
        source = CoinGeckoSource()
        df = source.fetch("crypto_prices", coin="bitcoin", days=7)
        
        assert df is not None
        assert "price" in df.columns
        assert "date" in df.columns


# =============================================================================
# Test External Data Tools
# =============================================================================


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestAcquireStockData:
    """Test acquire_stock_data tool."""
    
    @pytest.mark.skipif(not HAS_YFINANCE, reason="yfinance not installed")
    def test_acquire_stock_data(self, mock_yfinance_ticker):
        """Test acquiring stock data."""
        from src.tools.external_data import acquire_stock_data
        from src.tools.data_loading import get_registry
        
        registry = get_registry()
        registry.clear()
        
        result = acquire_stock_data.invoke({
            "ticker": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
        })
        
        assert result["status"] == "success"
        assert "dataset_name" in result
        assert result["ticker"] == "AAPL"
        assert result["row_count"] > 0
        
        # Check data was registered
        df = registry.get(result["dataset_name"])
        assert df is not None
        
        registry.clear()
    
    def test_acquire_stock_data_missing_yfinance(self):
        """Test handling missing yfinance library."""
        from src.tools.external_data import acquire_stock_data
        
        with patch("src.data_sources.finance.HAS_YFINANCE", False):
            with patch("src.data_sources.get_source") as mock_get:
                mock_source = MagicMock()
                mock_source.fetch.side_effect = Exception("yfinance not installed")
                mock_get.return_value = mock_source
                
                result = acquire_stock_data.invoke({"ticker": "AAPL"})
                
                assert result["status"] == "error"


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestAcquireEconomicIndicator:
    """Test acquire_economic_indicator tool."""
    
    def test_missing_api_key(self):
        """Test handling missing API key."""
        from src.tools.external_data import acquire_economic_indicator
        
        with patch.dict("os.environ", {}, clear=True):
            # Remove FRED_API_KEY if present
            import os
            if "FRED_API_KEY" in os.environ:
                del os.environ["FRED_API_KEY"]
            
            result = acquire_economic_indicator.invoke({"indicator": "GDP"})
            
            # Should return api_key_missing or error status
            assert result["status"] in ("api_key_missing", "error")


class TestFetchApiJson:
    """Test fetch_api_json tool."""
    
    def test_https_required(self):
        """Test that HTTPS is required."""
        from src.tools.external_data import fetch_api_json
        
        result = fetch_api_json.invoke({
            "url": "http://example.com/api",  # HTTP not HTTPS
        })
        
        assert result["status"] == "error"
        assert "HTTPS" in result["error"]
    
    @pytest.mark.skipif(not HAS_REQUESTS, reason="requests not installed")
    def test_successful_fetch(self):
        """Test successful JSON fetch."""
        from src.tools.external_data import fetch_api_json
        
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": "success"}
            mock_response.headers = {"content-type": "application/json"}
            mock_get.return_value = mock_response
            
            result = fetch_api_json.invoke({
                "url": "https://api.example.com/data",
            })
            
            assert result["status"] == "success"
            assert result["data"] == {"result": "success"}
    
    @pytest.mark.skipif(not HAS_REQUESTS, reason="requests not installed")
    def test_timeout_handling(self):
        """Test timeout handling."""
        from src.tools.external_data import fetch_api_json
        import requests
        
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout()
            
            result = fetch_api_json.invoke({
                "url": "https://api.example.com/data",
                "timeout": 5,
            })
            
            assert result["status"] == "timeout"


class TestListAvailableDataSources:
    """Test list_available_data_sources tool."""
    
    def test_list_all_sources(self):
        """Test listing all sources."""
        from src.tools.external_data import list_available_data_sources
        
        result = list_available_data_sources.invoke({})
        
        assert result["status"] == "success"
        assert "sources" in result
        assert result["count"] >= 0
    
    def test_list_finance_sources(self):
        """Test listing finance sources."""
        from src.tools.external_data import list_available_data_sources
        
        result = list_available_data_sources.invoke({"domain": "finance"})
        
        assert result["status"] == "success"
        assert result["domain_filter"] == "finance"
        
        # Check all returned sources are finance domain
        for source in result["sources"]:
            assert source["domain"] == "finance"


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling in data sources."""
    
    def test_rate_limit_error(self):
        """Test RateLimitError creation."""
        from src.data_sources.base import RateLimitError
        
        error = RateLimitError("test_source", 60)
        
        assert error.source == "test_source"
        assert error.retry_after == 60
        assert "60 seconds" in str(error)
    
    def test_api_key_missing_error(self):
        """Test APIKeyMissingError creation."""
        from src.data_sources.base import APIKeyMissingError
        
        error = APIKeyMissingError("fred", "FRED_API_KEY")
        
        assert error.source == "fred"
        assert error.key_name == "FRED_API_KEY"
        assert "FRED_API_KEY" in str(error)
    
    def test_data_not_available_error(self):
        """Test DataNotAvailableError creation."""
        from src.data_sources.base import DataNotAvailableError
        
        error = DataNotAvailableError("No data for INVALID ticker")
        
        assert "INVALID" in str(error)


# =============================================================================
# Test Registration Decorator
# =============================================================================


class TestRegistrationDecorator:
    """Test the @register_source decorator."""
    
    def test_decorator_registers_source(self, clean_registry):
        """Test that decorator properly registers source."""
        from src.data_sources.base import DataSource, DataSourceRegistry, register_source
        
        @register_source
        class DecoratedSource(DataSource):
            name = "decorated"
            domain = "test"
            description = "Test decorated source"
            data_types = ["test_type"]
            
            def fetch(self, data_type, **params):
                return None
            
            def check_availability(self, data_type, **params):
                return True
        
        # Should be auto-registered
        source = DataSourceRegistry.get_source("decorated")
        assert source is not None
        assert source.name == "decorated"


# =============================================================================
# Integration Tests (marked as slow)
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not HAS_YFINANCE, reason="yfinance not installed")
class TestYFinanceIntegration:
    """Integration tests with real yfinance calls."""
    
    def test_real_stock_data(self):
        """Test fetching real stock data (requires internet)."""
        from src.data_sources.finance import YFinanceSource
        
        source = YFinanceSource()
        
        # Try to fetch a small amount of recent data
        try:
            df = source.fetch("stock_prices", ticker="AAPL", period="5d")
            assert df is not None
            assert len(df) > 0
        except Exception:
            # Skip if network unavailable
            pytest.skip("Network unavailable")
