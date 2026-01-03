"""Base protocol and registry for external data sources.

Defines the DataSource protocol that all external data source implementations must follow,
and the DataSourceRegistry for discovering and managing data sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar
import logging

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =============================================================================
# Exceptions
# =============================================================================


class DataSourceError(Exception):
    """Base exception for data source errors."""
    pass


class RateLimitError(DataSourceError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, source: str, retry_after: int | None = None):
        self.source = source
        self.retry_after = retry_after
        msg = f"Rate limit exceeded for {source}"
        if retry_after:
            msg += f". Retry after {retry_after} seconds."
        super().__init__(msg)


class APIKeyMissingError(DataSourceError):
    """Raised when required API key is not configured."""
    
    def __init__(self, source: str, key_name: str):
        self.source = source
        self.key_name = key_name
        super().__init__(
            f"{source} requires API key '{key_name}'. "
            f"Set it in your environment or .env file."
        )


class DataNotAvailableError(DataSourceError):
    """Raised when requested data is not available."""
    pass


# =============================================================================
# Rate Limiting
# =============================================================================


@dataclass
class RateLimit:
    """Rate limit configuration for a data source.
    
    Attributes:
        requests_per_minute: Max requests allowed per minute (0 = unlimited)
        requests_per_day: Max requests allowed per day (0 = unlimited)
        concurrent_requests: Max concurrent requests allowed
    """
    requests_per_minute: int = 0  # 0 means unlimited
    requests_per_day: int = 0
    concurrent_requests: int = 5
    
    def __str__(self) -> str:
        parts = []
        if self.requests_per_minute:
            parts.append(f"{self.requests_per_minute}/min")
        if self.requests_per_day:
            parts.append(f"{self.requests_per_day}/day")
        return ", ".join(parts) if parts else "unlimited"


@dataclass
class RateLimitTracker:
    """Track rate limit usage for a data source."""
    
    rate_limit: RateLimit
    minute_requests: int = 0
    day_requests: int = 0
    minute_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    day_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def can_request(self) -> bool:
        """Check if a request can be made within rate limits."""
        self._reset_if_needed()
        
        if self.rate_limit.requests_per_minute > 0:
            if self.minute_requests >= self.rate_limit.requests_per_minute:
                return False
        
        if self.rate_limit.requests_per_day > 0:
            if self.day_requests >= self.rate_limit.requests_per_day:
                return False
        
        return True
    
    def record_request(self) -> None:
        """Record that a request was made."""
        self._reset_if_needed()
        self.minute_requests += 1
        self.day_requests += 1
    
    def seconds_until_available(self) -> int:
        """Get seconds until next request is available."""
        self._reset_if_needed()
        
        if self.can_request():
            return 0
        
        if self.rate_limit.requests_per_minute > 0:
            if self.minute_requests >= self.rate_limit.requests_per_minute:
                return int((self.minute_reset + timedelta(minutes=1) - datetime.now(timezone.utc)).total_seconds())
        
        if self.rate_limit.requests_per_day > 0:
            if self.day_requests >= self.rate_limit.requests_per_day:
                return int((self.day_reset + timedelta(days=1) - datetime.now(timezone.utc)).total_seconds())
        
        return 0
    
    def _reset_if_needed(self) -> None:
        """Reset counters if time windows have passed."""
        now = datetime.now(timezone.utc)
        
        if now - self.minute_reset > timedelta(minutes=1):
            self.minute_requests = 0
            self.minute_reset = now
        
        if now - self.day_reset > timedelta(days=1):
            self.day_requests = 0
            self.day_reset = now


# =============================================================================
# DataSource Protocol
# =============================================================================


class DataSource(ABC):
    """Abstract base class for external data sources.
    
    All data source implementations must inherit from this class and implement
    the required abstract methods.
    
    Attributes:
        name: Unique identifier for the source (e.g., "yfinance")
        domain: Data domain (finance, economics, health, science)
        description: Human-readable description
        requires_api_key: Whether an API key is required
        api_key_env_var: Environment variable name for API key
        rate_limit: Rate limiting configuration
        data_types: List of data types this source provides
    """
    
    name: ClassVar[str]
    domain: ClassVar[str]
    description: ClassVar[str]
    requires_api_key: ClassVar[bool] = False
    api_key_env_var: ClassVar[str | None] = None
    rate_limit: ClassVar[RateLimit] = RateLimit()
    data_types: ClassVar[list[str]] = []
    
    def __init__(self):
        """Initialize the data source."""
        self._rate_tracker = RateLimitTracker(self.rate_limit)
        self._logger = logging.getLogger(f"data_sources.{self.name}")
    
    @abstractmethod
    def fetch(
        self,
        data_type: str,
        **params: Any,
    ) -> "pd.DataFrame":
        """Fetch data from the source.
        
        Args:
            data_type: Type of data to fetch (e.g., "stock_prices")
            **params: Parameters specific to the data type
            
        Returns:
            DataFrame with the requested data
            
        Raises:
            DataSourceError: If fetch fails
            RateLimitError: If rate limit exceeded
            APIKeyMissingError: If required API key missing
        """
        pass
    
    @abstractmethod
    def check_availability(
        self,
        data_type: str,
        **params: Any,
    ) -> bool:
        """Check if the requested data is available.
        
        Args:
            data_type: Type of data to check
            **params: Parameters to check
            
        Returns:
            True if data is available, False otherwise
        """
        pass
    
    def get_required_params(self, data_type: str) -> list[str]:
        """Get required parameters for a data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            List of required parameter names
        """
        return []
    
    def get_api_key(self) -> str | None:
        """Get the API key from environment.
        
        Returns:
            API key string or None if not set
            
        Raises:
            APIKeyMissingError: If key is required but not set
        """
        import os
        
        if not self.api_key_env_var:
            return None
        
        key = os.getenv(self.api_key_env_var, "")
        
        if self.requires_api_key and not key:
            raise APIKeyMissingError(self.name, self.api_key_env_var)
        
        return key if key else None
    
    def _check_rate_limit(self) -> None:
        """Check and record rate limit usage.
        
        Raises:
            RateLimitError: If rate limit exceeded
        """
        if not self._rate_tracker.can_request():
            retry_after = self._rate_tracker.seconds_until_available()
            raise RateLimitError(self.name, retry_after)
        
        self._rate_tracker.record_request()
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, domain={self.domain!r}, "
            f"data_types={self.data_types})"
        )


# =============================================================================
# DataSourceRegistry
# =============================================================================


class DataSourceRegistry:
    """Registry for discovering and managing external data sources.
    
    Sources are organized by domain (finance, economics, health, science)
    and can be looked up by name or by the data types they provide.
    
    Sources auto-register when their module is imported.
    
    Usage:
        # Register a source
        DataSourceRegistry.register(MySource)
        
        # Find sources by domain
        finance_sources = DataSourceRegistry.get_sources("finance")
        
        # Find source by data type
        source = DataSourceRegistry.find_source("stock_prices")
        
        # Get source by name
        yf = DataSourceRegistry.get_source("yfinance")
    """
    
    _sources_by_name: ClassVar[dict[str, DataSource]] = {}
    _sources_by_domain: ClassVar[dict[str, list[DataSource]]] = {}
    _data_type_index: ClassVar[dict[str, list[DataSource]]] = {}
    
    @classmethod
    def register(cls, source_class: type[DataSource]) -> None:
        """Register a data source class.
        
        Args:
            source_class: DataSource subclass to register
        """
        # Instantiate the source
        source = source_class()
        
        # Register by name
        cls._sources_by_name[source.name.lower()] = source
        
        # Register by domain
        domain = source.domain.lower()
        if domain not in cls._sources_by_domain:
            cls._sources_by_domain[domain] = []
        cls._sources_by_domain[domain].append(source)
        
        # Index by data types
        for data_type in source.data_types:
            data_type_lower = data_type.lower()
            if data_type_lower not in cls._data_type_index:
                cls._data_type_index[data_type_lower] = []
            cls._data_type_index[data_type_lower].append(source)
        
        logging.getLogger("data_sources.registry").debug(
            f"Registered data source: {source.name} ({source.domain})"
        )
    
    @classmethod
    def get_source(cls, name: str) -> DataSource | None:
        """Get a data source by name.
        
        Args:
            name: Name of the source (case-insensitive)
            
        Returns:
            DataSource instance or None
        """
        return cls._sources_by_name.get(name.lower())
    
    @classmethod
    def get_sources(cls, domain: str) -> list[DataSource]:
        """Get all data sources for a domain.
        
        Args:
            domain: Domain name (finance, economics, health, science)
            
        Returns:
            List of DataSource instances
        """
        return cls._sources_by_domain.get(domain.lower(), [])
    
    @classmethod
    def find_source(cls, data_type: str) -> DataSource | None:
        """Find a data source that provides a specific data type.
        
        Returns the first source that doesn't require an API key,
        or the first available source if all require keys.
        
        Args:
            data_type: Type of data needed (e.g., "stock_prices")
            
        Returns:
            DataSource instance or None
        """
        sources = cls._data_type_index.get(data_type.lower(), [])
        
        if not sources:
            return None
        
        # Prefer sources that don't require API keys
        for source in sources:
            if not source.requires_api_key:
                return source
        
        # Fall back to first available
        return sources[0]
    
    @classmethod
    def find_all_sources(cls, data_type: str) -> list[DataSource]:
        """Find all data sources that provide a specific data type.
        
        Args:
            data_type: Type of data needed
            
        Returns:
            List of DataSource instances
        """
        return cls._data_type_index.get(data_type.lower(), [])
    
    @classmethod
    def all_sources(cls) -> list[DataSource]:
        """Get all registered data sources.
        
        Returns:
            List of all DataSource instances
        """
        return list(cls._sources_by_name.values())
    
    @classmethod
    def list_domains(cls) -> list[str]:
        """Get all registered domains.
        
        Returns:
            List of domain names
        """
        return list(cls._sources_by_domain.keys())
    
    @classmethod
    def list_data_types(cls) -> list[str]:
        """Get all available data types.
        
        Returns:
            List of data type names
        """
        return list(cls._data_type_index.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered sources (for testing)."""
        cls._sources_by_name.clear()
        cls._sources_by_domain.clear()
        cls._data_type_index.clear()


# =============================================================================
# Registration Decorator
# =============================================================================


def register_source(cls: type[DataSource]) -> type[DataSource]:
    """Decorator to auto-register a DataSource class.
    
    Usage:
        @register_source
        class MySource(DataSource):
            name = "mysource"
            ...
    """
    DataSourceRegistry.register(cls)
    return cls
