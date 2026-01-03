"""Data Source Registry for external data acquisition.

Sprint 13: Provides an extensible registry of external data sources organized by domain
(finance, economics, health, science) with automatic source discovery.

Usage:
    from src.data_sources import DataSourceRegistry, get_source, list_sources
    
    # Find a source for stock data
    source = DataSourceRegistry.find_source("stock_prices")
    
    # List all finance sources
    sources = DataSourceRegistry.get_sources("finance")
    
    # Convenience functions
    yf = get_source("yfinance")
    all_finance = list_sources("finance")
"""

from src.data_sources.base import (
    DataSource,
    DataSourceRegistry,
    RateLimit,
    DataSourceError,
    RateLimitError,
    APIKeyMissingError,
)

# Import source implementations to trigger auto-registration
from src.data_sources import finance


def get_source(name: str) -> DataSource | None:
    """Get a data source by name.
    
    Args:
        name: Name of the data source (e.g., "yfinance", "fred")
        
    Returns:
        DataSource instance or None if not found
    """
    return DataSourceRegistry.get_source(name)


def list_sources(domain: str | None = None) -> list[DataSource]:
    """List available data sources.
    
    Args:
        domain: Optional domain filter (finance, economics, health, science)
        
    Returns:
        List of DataSource instances
    """
    if domain:
        return DataSourceRegistry.get_sources(domain)
    return DataSourceRegistry.all_sources()


def find_source_for_data(data_type: str) -> DataSource | None:
    """Find a data source that can provide the specified data type.
    
    Args:
        data_type: Type of data needed (e.g., "stock_prices", "gdp")
        
    Returns:
        First matching DataSource or None
    """
    return DataSourceRegistry.find_source(data_type)


__all__ = [
    # Base classes
    "DataSource",
    "DataSourceRegistry",
    "RateLimit",
    # Errors
    "DataSourceError",
    "RateLimitError",
    "APIKeyMissingError",
    # Convenience functions
    "get_source",
    "list_sources",
    "find_source_for_data",
]
