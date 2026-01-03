"""External data acquisition tools.

Sprint 13: Tools for fetching data from external APIs including
stock prices, economic indicators, and generic REST API access.

These tools integrate with the DataSourceRegistry and provide
a consistent interface for the data acquisition agent.
"""

from typing import Any, Literal
import logging

from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:
    import pandas as pd  # noqa: F401 - Used for HAS_PANDAS check
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from src.tools.data_loading import get_registry


# =============================================================================
# Input Schemas
# =============================================================================


class AcquireStockDataInput(BaseModel):
    """Input schema for acquire_stock_data tool."""
    ticker: str = Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOG')")
    start_date: str | None = Field(
        default=None,
        description="Start date in YYYY-MM-DD format. Defaults to 5 years ago."
    )
    end_date: str | None = Field(
        default=None,
        description="End date in YYYY-MM-DD format. Defaults to today."
    )
    interval: str = Field(
        default="1d",
        description="Data interval: '1d' (daily), '1wk' (weekly), '1mo' (monthly)"
    )
    dataset_name: str | None = Field(
        default=None,
        description="Name to register dataset with. Defaults to ticker_prices."
    )


class AcquireEconomicIndicatorInput(BaseModel):
    """Input schema for acquire_economic_indicator tool."""
    indicator: str = Field(
        description="FRED series ID or common name (e.g., 'GDP', 'UNRATE', 'fed_funds')"
    )
    start_date: str | None = Field(
        default=None,
        description="Start date in YYYY-MM-DD format"
    )
    end_date: str | None = Field(
        default=None,
        description="End date in YYYY-MM-DD format"
    )
    dataset_name: str | None = Field(
        default=None,
        description="Name to register dataset with"
    )


class AcquireCryptoDataInput(BaseModel):
    """Input schema for acquire_crypto_data tool."""
    coin: str = Field(description="Cryptocurrency ID or symbol (e.g., 'bitcoin', 'btc')")
    vs_currency: str = Field(default="usd", description="Quote currency")
    days: int = Field(default=365, description="Number of days of history")
    dataset_name: str | None = Field(default=None, description="Name to register dataset with")


class FetchAPIJsonInput(BaseModel):
    """Input schema for fetch_api_json tool."""
    url: str = Field(description="Full URL to fetch")
    method: Literal["GET", "POST"] = Field(default="GET", description="HTTP method")
    headers: dict[str, str] | None = Field(default=None, description="Request headers")
    params: dict[str, Any] | None = Field(default=None, description="Query parameters")
    json_body: dict[str, Any] | None = Field(default=None, description="JSON body for POST")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class ListDataSourcesInput(BaseModel):
    """Input schema for list_available_data_sources tool."""
    domain: str | None = Field(
        default=None,
        description="Filter by domain: 'finance', 'economics', 'health', 'science'"
    )


# =============================================================================
# Stock Data Tool
# =============================================================================


@tool(args_schema=AcquireStockDataInput)
def acquire_stock_data(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "1d",
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """
    Fetch historical stock price data from Yahoo Finance.
    
    Downloads OHLCV (Open, High, Low, Close, Volume) data and registers
    it in the DataRegistry for use in analysis.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOG', 'MSFT')
        start_date: Start date (YYYY-MM-DD). Defaults to 5 years ago.
        end_date: End date (YYYY-MM-DD). Defaults to today.
        interval: Data interval - '1d', '1wk', '1mo'
        dataset_name: Name to register the dataset with
        
    Returns:
        Summary of acquired data including row count, date range, columns
        
    Example:
        acquire_stock_data("AAPL", "2020-01-01", "2024-12-31")
    """
    if not HAS_PANDAS:
        return {"status": "error", "error": "pandas is required but not installed"}
    
    try:
        from src.data_sources import get_source
        from src.data_sources.base import DataSourceError, DataNotAvailableError
        
        source = get_source("yfinance")
        if not source:
            return {"status": "error", "error": "YFinance data source not available"}
        
        # Build params
        params = {
            "ticker": ticker.upper(),
            "interval": interval,
        }
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date
        
        # Fetch data
        df = source.fetch("stock_prices", **params)
        
        if df.empty:
            return {
                "status": "no_data",
                "message": f"No data found for {ticker}",
            }
        
        # Register in DataRegistry
        name = dataset_name or f"{ticker.lower()}_prices"
        registry = get_registry()
        registry.register(name, df)
        
        # Build summary
        date_col = "Date" if "Date" in df.columns else "date"
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        
        # Handle datetime vs date
        if hasattr(min_date, "strftime"):
            min_date_str = min_date.strftime("%Y-%m-%d")
            max_date_str = max_date.strftime("%Y-%m-%d")
        else:
            min_date_str = str(min_date)[:10]
            max_date_str = str(max_date)[:10]
        
        return {
            "status": "success",
            "dataset_name": name,
            "ticker": ticker.upper(),
            "row_count": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": min_date_str,
                "end": max_date_str,
            },
            "interval": interval,
            "sample_data": df.head(3).to_dict(orient="records"),
        }
        
    except DataNotAvailableError as e:
        return {
            "status": "not_available",
            "error": str(e),
            "ticker": ticker,
        }
    except DataSourceError as e:
        return {
            "status": "error",
            "error": str(e),
            "ticker": ticker,
        }
    except Exception as e:
        logging.error(f"acquire_stock_data failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


# =============================================================================
# Economic Indicator Tool
# =============================================================================


@tool(args_schema=AcquireEconomicIndicatorInput)
def acquire_economic_indicator(
    indicator: str,
    start_date: str | None = None,
    end_date: str | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """
    Fetch economic indicator data from FRED (Federal Reserve Economic Data).
    
    FRED provides 800,000+ economic time series including GDP, employment,
    inflation, interest rates, and more.
    
    Common indicators:
    - GDP, real_gdp, gdp_growth - Gross Domestic Product
    - unemployment, nonfarm_payrolls - Employment
    - cpi, cpi_core, pce - Inflation
    - fed_funds, treasury_10y, treasury_2y - Interest rates
    - usd_eur, usd_gbp - Exchange rates
    - Or use any FRED series ID directly
    
    Args:
        indicator: FRED series ID or common name
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        dataset_name: Name to register the dataset with
        
    Returns:
        Summary of acquired data
        
    Note:
        Requires FRED_API_KEY environment variable. Get free key at:
        https://fred.stlouisfed.org/docs/api/api_key.html
    """
    if not HAS_PANDAS:
        return {"status": "error", "error": "pandas is required but not installed"}
    
    try:
        from src.data_sources import get_source
        from src.data_sources.base import (
            DataSourceError,
            DataNotAvailableError,
            APIKeyMissingError,
        )
        
        source = get_source("fred")
        if not source:
            return {"status": "error", "error": "FRED data source not available"}
        
        # Build params
        params = {"series_id": indicator}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        # Fetch data
        df = source.fetch("economic_indicator", **params)
        
        if df.empty:
            return {
                "status": "no_data",
                "message": f"No data found for {indicator}",
            }
        
        # Register in DataRegistry
        name = dataset_name or f"fred_{indicator.lower()}"
        registry = get_registry()
        registry.register(name, df)
        
        # Build summary
        min_date = df["date"].min()
        max_date = df["date"].max()
        
        if hasattr(min_date, "strftime"):
            min_date_str = min_date.strftime("%Y-%m-%d")
            max_date_str = max_date.strftime("%Y-%m-%d")
        else:
            min_date_str = str(min_date)[:10]
            max_date_str = str(max_date)[:10]
        
        return {
            "status": "success",
            "dataset_name": name,
            "indicator": indicator,
            "series_id": df["series_id"].iloc[0] if "series_id" in df.columns else indicator,
            "row_count": len(df),
            "date_range": {
                "start": min_date_str,
                "end": max_date_str,
            },
            "latest_value": float(df["value"].iloc[-1]) if len(df) > 0 else None,
            "sample_data": df.tail(5).to_dict(orient="records"),
        }
        
    except APIKeyMissingError as e:
        return {
            "status": "api_key_missing",
            "error": str(e),
            "help": "Set FRED_API_KEY environment variable. Get free key at: "
                    "https://fred.stlouisfed.org/docs/api/api_key.html",
        }
    except DataNotAvailableError as e:
        return {
            "status": "not_available",
            "error": str(e),
            "indicator": indicator,
        }
    except DataSourceError as e:
        return {
            "status": "error",
            "error": str(e),
        }
    except Exception as e:
        logging.error(f"acquire_economic_indicator failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


# =============================================================================
# Cryptocurrency Data Tool
# =============================================================================


@tool(args_schema=AcquireCryptoDataInput)
def acquire_crypto_data(
    coin: str,
    vs_currency: str = "usd",
    days: int = 365,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """
    Fetch cryptocurrency price data from CoinGecko.
    
    Downloads historical price, volume, and market cap data.
    No API key required.
    
    Common coins: bitcoin (btc), ethereum (eth), solana (sol), cardano (ada)
    
    Args:
        coin: Cryptocurrency ID or symbol (e.g., 'bitcoin', 'btc')
        vs_currency: Quote currency (default: 'usd')
        days: Number of days of history (default: 365)
        dataset_name: Name to register the dataset with
        
    Returns:
        Summary of acquired data
    """
    if not HAS_PANDAS:
        return {"status": "error", "error": "pandas is required but not installed"}
    
    try:
        from src.data_sources import get_source
        from src.data_sources.base import DataSourceError, DataNotAvailableError
        
        source = get_source("coingecko")
        if not source:
            return {"status": "error", "error": "CoinGecko data source not available"}
        
        # Fetch data
        df = source.fetch(
            "crypto_prices",
            coin=coin,
            vs_currency=vs_currency,
            days=days,
        )
        
        if df.empty:
            return {
                "status": "no_data",
                "message": f"No data found for {coin}",
            }
        
        # Register in DataRegistry
        name = dataset_name or f"{coin.lower()}_prices"
        registry = get_registry()
        registry.register(name, df)
        
        return {
            "status": "success",
            "dataset_name": name,
            "coin": coin,
            "row_count": len(df),
            "vs_currency": vs_currency,
            "columns": list(df.columns),
            "latest_price": float(df["price"].iloc[-1]) if len(df) > 0 else None,
            "sample_data": df.tail(3).to_dict(orient="records"),
        }
        
    except DataNotAvailableError as e:
        return {
            "status": "not_available",
            "error": str(e),
            "coin": coin,
        }
    except DataSourceError as e:
        return {
            "status": "error",
            "error": str(e),
        }
    except Exception as e:
        logging.error(f"acquire_crypto_data failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


# =============================================================================
# Generic API Fetch Tool
# =============================================================================


@tool(args_schema=FetchAPIJsonInput)
def fetch_api_json(
    url: str,
    method: Literal["GET", "POST"] = "GET",
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Fetch JSON data from any REST API endpoint.
    
    A general-purpose tool for accessing REST APIs that return JSON.
    Use this when a specific data source tool is not available.
    
    Args:
        url: Full URL to fetch (must be https for security)
        method: HTTP method ('GET' or 'POST')
        headers: Optional request headers (e.g., Authorization)
        params: Query parameters for GET requests
        json_body: JSON body for POST requests
        timeout: Request timeout in seconds
        
    Returns:
        JSON response data or error information
        
    Security:
        - Only HTTPS URLs are allowed
        - Timeout is enforced to prevent hanging
        - Response size is limited
    """
    if not HAS_REQUESTS:
        return {"status": "error", "error": "requests library not installed"}
    
    # Security: Only allow HTTPS
    if not url.startswith("https://"):
        return {
            "status": "error",
            "error": "Only HTTPS URLs are allowed for security",
        }
    
    try:
        # Build request
        request_kwargs = {
            "timeout": min(timeout, 60),  # Cap at 60 seconds
            "headers": headers or {},
        }
        
        if method == "GET":
            request_kwargs["params"] = params
            response = requests.get(url, **request_kwargs)
        else:  # POST
            request_kwargs["json"] = json_body
            request_kwargs["params"] = params
            response = requests.post(url, **request_kwargs)
        
        # Check response size (limit to 10MB)
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:
            return {
                "status": "error",
                "error": "Response too large (>10MB)",
            }
        
        response.raise_for_status()
        
        # Parse JSON
        data = response.json()
        
        return {
            "status": "success",
            "status_code": response.status_code,
            "data": data,
            "headers": dict(response.headers),
        }
        
    except requests.exceptions.Timeout:
        return {
            "status": "timeout",
            "error": f"Request timed out after {timeout} seconds",
        }
    except requests.exceptions.HTTPError as e:
        return {
            "status": "http_error",
            "status_code": e.response.status_code if e.response else None,
            "error": str(e),
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "error": str(e),
        }
    except ValueError as e:
        return {
            "status": "parse_error",
            "error": f"Failed to parse JSON: {e}",
        }


# =============================================================================
# Discovery Tool
# =============================================================================


@tool(args_schema=ListDataSourcesInput)
def list_available_data_sources(
    domain: str | None = None,
) -> dict[str, Any]:
    """
    List available external data sources.
    
    Use this to discover what data sources are available for fetching
    external data. Sources are organized by domain.
    
    Args:
        domain: Filter by domain ('finance', 'economics', 'health', 'science')
                If not provided, lists all sources.
        
    Returns:
        List of available data sources with their capabilities
    """
    try:
        from src.data_sources import DataSourceRegistry, list_sources
        
        if domain:
            sources = list_sources(domain)
        else:
            sources = DataSourceRegistry.all_sources()
        
        source_info = []
        for source in sources:
            source_info.append({
                "name": source.name,
                "domain": source.domain,
                "description": source.description,
                "requires_api_key": source.requires_api_key,
                "api_key_env_var": source.api_key_env_var,
                "data_types": source.data_types,
                "rate_limit": str(source.rate_limit),
            })
        
        return {
            "status": "success",
            "count": len(source_info),
            "domain_filter": domain,
            "sources": source_info,
            "available_domains": DataSourceRegistry.list_domains(),
            "available_data_types": DataSourceRegistry.list_data_types(),
        }
        
    except Exception as e:
        logging.error(f"list_available_data_sources failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


# =============================================================================
# Tool Collection
# =============================================================================


def get_external_data_tools() -> list:
    """Get all external data acquisition tools.
    
    Returns:
        List of LangChain tools for external data acquisition
    """
    return [
        acquire_stock_data,
        acquire_economic_indicator,
        acquire_crypto_data,
        fetch_api_json,
        list_available_data_sources,
    ]
