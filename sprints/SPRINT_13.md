# Sprint 13: Data Source Registry & External APIs

**Status**: ✅ Complete  
**PR**: #29  
**Date**: 2026-01-03  

## Overview

Sprint 13 implements an extensible registry of external data sources and tools for autonomous data acquisition from finance, economics, and science APIs.

## New Components

### Data Source Registry (`src/data_sources/`)

A pluggable architecture for discovering and using external data sources.

#### Base Classes (`src/data_sources/base.py`)

| Class | Purpose |
|-------|---------|
| `DataSource` | Abstract base class for all data sources |
| `DataSourceRegistry` | Central registry for source discovery |
| `RateLimit` | Rate limiting configuration |
| `RateLimitTracker` | Track API request counts |
| `@register_source` | Decorator for auto-registration |

#### Exceptions

| Exception | Use Case |
|-----------|----------|
| `DataSourceError` | Base exception for source errors |
| `RateLimitError` | Rate limit exceeded |
| `APIKeyMissingError` | Required API key not configured |
| `DataNotAvailableError` | Requested data not found |

### Finance Sources (`src/data_sources/finance.py`)

| Source | Domain | API Key | Data Types |
|--------|--------|---------|------------|
| `YFinanceSource` | finance | No | stock_prices, options, fundamentals, dividends |
| `FREDSource` | economics | Yes | economic_indicator, gdp, unemployment, inflation |
| `CoinGeckoSource` | finance | No | crypto_prices, crypto_history, crypto_market |

### External Data Tools (`src/tools/external_data.py`)

| Tool | Description |
|------|-------------|
| `acquire_stock_data` | Fetch stock prices from Yahoo Finance |
| `acquire_economic_indicator` | Fetch FRED economic data |
| `acquire_crypto_data` | Fetch cryptocurrency prices |
| `fetch_api_json` | Generic REST API JSON fetcher |
| `list_available_data_sources` | Discover available sources |

## Usage Examples

### Fetching Stock Data

```python
from src.tools.external_data import acquire_stock_data

# Fetch 5 years of Apple stock prices
result = acquire_stock_data.invoke({
    "ticker": "AAPL",
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    "interval": "1d",
})

# Result contains:
# - dataset_name: registered name in DataRegistry
# - row_count: number of observations
# - date_range: start and end dates
# - columns: available columns (Open, High, Low, Close, Volume)
```

### Fetching Economic Indicators

```python
from src.tools.external_data import acquire_economic_indicator

# Fetch GDP data (requires FRED_API_KEY)
result = acquire_economic_indicator.invoke({
    "indicator": "gdp",  # or FRED series ID like "GDP"
    "start_date": "2010-01-01",
})

# Common indicators: gdp, unemployment, cpi, fed_funds, treasury_10y
```

### Using the Data Source Registry

```python
from src.data_sources import DataSourceRegistry, get_source, list_sources

# Find a source by data type
source = DataSourceRegistry.find_source("stock_prices")
# Returns: YFinanceSource

# Get a specific source
fred = get_source("fred")

# List all finance sources
finance_sources = list_sources("finance")

# Fetch data directly from source
df = source.fetch("stock_prices", ticker="AAPL", period="1y")
```

### Generic API Fetch

```python
from src.tools.external_data import fetch_api_json

# Fetch from any HTTPS JSON API
result = fetch_api_json.invoke({
    "url": "https://api.example.com/data",
    "method": "GET",
    "params": {"key": "value"},
    "timeout": 30,
})
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FRED_API_KEY` | For FRED | Get free key at https://fred.stlouisfed.org/docs/api/api_key.html |

Add to `.env`:
```bash
FRED_API_KEY=your_api_key_here
```

## Rate Limiting

Sources enforce rate limits automatically:

| Source | Rate Limit |
|--------|------------|
| YFinance | 60/min (informal) |
| FRED | 120/min |
| CoinGecko | 30/min |

## DataSource Protocol

Create custom data sources by implementing the protocol:

```python
from src.data_sources.base import DataSource, register_source, RateLimit

@register_source
class MyDataSource(DataSource):
    name = "my_source"
    domain = "finance"
    description = "My custom data source"
    requires_api_key = True
    api_key_env_var = "MY_API_KEY"
    rate_limit = RateLimit(requests_per_minute=60)
    data_types = ["my_data_type"]
    
    def fetch(self, data_type: str, **params) -> pd.DataFrame:
        # Implementation
        pass
    
    def check_availability(self, data_type: str, **params) -> bool:
        # Check if data is available
        return True
```

## Testing

26 new unit tests covering:
- DataSource protocol and registry
- Rate limiting and error handling
- YFinance, FRED, CoinGecko sources (mocked)
- External data tools
- Registration decorator

All 673 tests pass.

## Files Changed

### New Files
- `src/data_sources/__init__.py` - Module exports and convenience functions
- `src/data_sources/base.py` - DataSource protocol, registry, rate limiting
- `src/data_sources/finance.py` - YFinance, FRED, CoinGecko implementations
- `src/tools/external_data.py` - External data acquisition tools
- `tests/unit/test_sprint13_data_sources.py` - Unit tests
- `sprints/SPRINT_13.md` - This documentation

### Modified Files
- `src/config/settings.py` - Added FRED_API_KEY setting
- `src/tools/__init__.py` - Added external data tool exports

## Dependencies

Optional dependencies for data sources:
```toml
yfinance = ">=0.2.0"     # Stock data (optional)
fredapi = ">=0.5.0"      # FRED economic data (optional)
requests = ">=2.28.0"    # API requests (already installed)
```

These are optional - tools gracefully handle missing libraries.

## Success Criteria

- [x] `DataSourceRegistry.find_source("stock_prices")` returns YFinanceSource ✅
- [x] `acquire_stock_data("AAPL", "2020-01-01", "2024-12-31")` returns valid data ✅
- [x] Tools gracefully handle missing API keys and rate limits ✅
- [x] All 673 tests pass ✅

## Next Sprint

Sprint 14: Data Acquisition Node & Code Execution
- Data acquisition agent with intelligent source selection
- Code execution engine for custom data fetching
- Human-in-the-loop for unavailable data
