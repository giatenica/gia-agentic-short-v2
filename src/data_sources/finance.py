"""Finance domain data sources.

Provides data sources for financial data including:
- Stock prices and fundamentals (yfinance)
- Economic indicators (FRED)
- Cryptocurrency prices (CoinGecko)
"""

from __future__ import annotations

from datetime import datetime, date, timezone
from typing import Any, ClassVar

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from fredapi import Fred
    HAS_FREDAPI = True
except ImportError:
    HAS_FREDAPI = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from src.data_sources.base import (
    DataSource,
    RateLimit,
    DataSourceError,
    RateLimitError,
    APIKeyMissingError,
    DataNotAvailableError,
    register_source,
)


# =============================================================================
# Yahoo Finance Source
# =============================================================================


@register_source
class YFinanceSource(DataSource):
    """Yahoo Finance data source via yfinance library.
    
    Provides free access to:
    - Historical stock prices (OHLCV)
    - Options chains
    - Company fundamentals
    - Dividend history
    - Stock splits
    
    No API key required. Rate limits are informal.
    """
    
    name: ClassVar[str] = "yfinance"
    domain: ClassVar[str] = "finance"
    description: ClassVar[str] = "Yahoo Finance - Stock prices, options, fundamentals"
    requires_api_key: ClassVar[bool] = False
    rate_limit: ClassVar[RateLimit] = RateLimit(
        requests_per_minute=60,  # Informal limit
        requests_per_day=0,  # No hard daily limit
    )
    data_types: ClassVar[list[str]] = [
        "stock_prices",
        "stock_history",
        "options",
        "fundamentals",
        "dividends",
        "splits",
        "info",
    ]
    
    def fetch(
        self,
        data_type: str,
        **params: Any,
    ) -> "pd.DataFrame":
        """Fetch data from Yahoo Finance.
        
        Args:
            data_type: One of stock_prices, options, fundamentals, etc.
            **params: Data type specific parameters
            
        Returns:
            DataFrame with requested data
        """
        if not HAS_YFINANCE:
            raise DataSourceError("yfinance library not installed. Run: pip install yfinance")
        
        if not HAS_PANDAS:
            raise DataSourceError("pandas library not installed")
        
        self._check_rate_limit()
        
        if data_type in ("stock_prices", "stock_history"):
            return self._fetch_stock_prices(**params)
        elif data_type == "options":
            return self._fetch_options(**params)
        elif data_type == "fundamentals":
            return self._fetch_fundamentals(**params)
        elif data_type == "dividends":
            return self._fetch_dividends(**params)
        elif data_type == "splits":
            return self._fetch_splits(**params)
        elif data_type == "info":
            return self._fetch_info(**params)
        else:
            raise DataSourceError(f"Unknown data type: {data_type}")
    
    def check_availability(
        self,
        data_type: str,
        **params: Any,
    ) -> bool:
        """Check if ticker data is available."""
        if not HAS_YFINANCE:
            return False
        
        ticker = params.get("ticker") or params.get("symbol")
        if not ticker:
            return False
        
        try:
            t = yf.Ticker(ticker)
            info = t.info
            return info is not None and len(info) > 0
        except Exception:
            return False
    
    def get_required_params(self, data_type: str) -> list[str]:
        """Get required parameters for a data type."""
        if data_type in ("stock_prices", "stock_history"):
            return ["ticker"]
        elif data_type == "options":
            return ["ticker"]
        elif data_type in ("fundamentals", "dividends", "splits", "info"):
            return ["ticker"]
        return []
    
    def _fetch_stock_prices(
        self,
        ticker: str,
        start: str | date | None = None,
        end: str | date | None = None,
        period: str | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> "pd.DataFrame":
        """Fetch historical stock prices.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            start: Start date (YYYY-MM-DD or date object)
            end: End date (YYYY-MM-DD or date object)
            period: Alternative to start/end (e.g., "1y", "5y", "max")
            interval: Data frequency ("1d", "1wk", "1mo", "1h", etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        t = yf.Ticker(ticker)
        
        try:
            if period:
                df = t.history(period=period, interval=interval)
            else:
                # Default to last 5 years if no dates specified
                if not start:
                    start = "2019-01-01"
                if not end:
                    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                df = t.history(start=start, end=end, interval=interval)
            
            if df.empty:
                raise DataNotAvailableError(f"No data available for {ticker}")
            
            # Reset index to make Date a column
            df = df.reset_index()
            df["ticker"] = ticker
            
            return df
            
        except Exception as e:
            if "No data found" in str(e) or df.empty:
                raise DataNotAvailableError(f"No data available for {ticker}")
            raise DataSourceError(f"Failed to fetch {ticker}: {e}")
    
    def _fetch_options(
        self,
        ticker: str,
        expiration: str | None = None,
        **kwargs: Any,
    ) -> "pd.DataFrame":
        """Fetch options chain data."""
        t = yf.Ticker(ticker)
        
        try:
            if expiration:
                opt = t.option_chain(expiration)
            else:
                # Get the nearest expiration
                expirations = t.options
                if not expirations:
                    raise DataNotAvailableError(f"No options data for {ticker}")
                opt = t.option_chain(expirations[0])
            
            # Combine calls and puts
            calls = opt.calls.copy()
            calls["option_type"] = "call"
            puts = opt.puts.copy()
            puts["option_type"] = "put"
            
            df = pd.concat([calls, puts], ignore_index=True)
            df["ticker"] = ticker
            
            return df
            
        except Exception as e:
            raise DataSourceError(f"Failed to fetch options for {ticker}: {e}")
    
    def _fetch_fundamentals(
        self,
        ticker: str,
        **kwargs: Any,
    ) -> "pd.DataFrame":
        """Fetch company fundamentals."""
        t = yf.Ticker(ticker)
        
        try:
            # Get quarterly financials
            financials = t.quarterly_financials
            if financials is None or financials.empty:
                # Try annual
                financials = t.financials
            
            if financials is None or financials.empty:
                raise DataNotAvailableError(f"No fundamentals for {ticker}")
            
            # Transpose so dates are rows
            df = financials.T.reset_index()
            df = df.rename(columns={"index": "date"})
            df["ticker"] = ticker
            
            return df
            
        except Exception as e:
            raise DataSourceError(f"Failed to fetch fundamentals for {ticker}: {e}")
    
    def _fetch_dividends(
        self,
        ticker: str,
        **kwargs: Any,
    ) -> "pd.DataFrame":
        """Fetch dividend history."""
        t = yf.Ticker(ticker)
        
        try:
            dividends = t.dividends
            if dividends is None or dividends.empty:
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=["date", "dividend", "ticker"])
            
            df = dividends.reset_index()
            df.columns = ["date", "dividend"]
            df["ticker"] = ticker
            
            return df
            
        except Exception as e:
            raise DataSourceError(f"Failed to fetch dividends for {ticker}: {e}")
    
    def _fetch_splits(
        self,
        ticker: str,
        **kwargs: Any,
    ) -> "pd.DataFrame":
        """Fetch stock split history."""
        t = yf.Ticker(ticker)
        
        try:
            splits = t.splits
            if splits is None or splits.empty:
                return pd.DataFrame(columns=["date", "split_ratio", "ticker"])
            
            df = splits.reset_index()
            df.columns = ["date", "split_ratio"]
            df["ticker"] = ticker
            
            return df
            
        except Exception as e:
            raise DataSourceError(f"Failed to fetch splits for {ticker}: {e}")
    
    def _fetch_info(
        self,
        ticker: str,
        **kwargs: Any,
    ) -> "pd.DataFrame":
        """Fetch company info as single-row DataFrame."""
        t = yf.Ticker(ticker)
        
        try:
            info = t.info
            if not info:
                raise DataNotAvailableError(f"No info for {ticker}")
            
            # Convert dict to single-row DataFrame
            df = pd.DataFrame([info])
            df["ticker"] = ticker
            
            return df
            
        except Exception as e:
            raise DataSourceError(f"Failed to fetch info for {ticker}: {e}")


# =============================================================================
# FRED Source (Federal Reserve Economic Data)
# =============================================================================


@register_source
class FREDSource(DataSource):
    """Federal Reserve Economic Data (FRED) source.
    
    Provides access to 800,000+ economic time series including:
    - GDP, employment, inflation
    - Interest rates, exchange rates
    - Industrial production, retail sales
    
    Requires free API key from https://fred.stlouisfed.org/docs/api/api_key.html
    """
    
    name: ClassVar[str] = "fred"
    domain: ClassVar[str] = "economics"
    description: ClassVar[str] = "FRED - Federal Reserve Economic Data"
    requires_api_key: ClassVar[bool] = True
    api_key_env_var: ClassVar[str] = "FRED_API_KEY"
    rate_limit: ClassVar[RateLimit] = RateLimit(
        requests_per_minute=120,
        requests_per_day=0,  # No daily limit
    )
    data_types: ClassVar[list[str]] = [
        "economic_indicator",
        "gdp",
        "employment",
        "inflation",
        "interest_rates",
        "exchange_rates",
        "money_supply",
    ]
    
    # Common series IDs
    COMMON_SERIES: ClassVar[dict[str, str]] = {
        "gdp": "GDP",
        "real_gdp": "GDPCA",
        "gdp_growth": "A191RL1Q225SBEA",
        "unemployment": "UNRATE",
        "unemployment_claims": "ICSA",
        "nonfarm_payrolls": "PAYEMS",
        "cpi": "CPIAUCSL",
        "cpi_core": "CPILFESL",
        "pce": "PCE",
        "pce_core": "PCEPILFE",
        "fed_funds": "FEDFUNDS",
        "treasury_10y": "DGS10",
        "treasury_2y": "DGS2",
        "treasury_3mo": "DTB3",
        "prime_rate": "DPRIME",
        "usd_eur": "DEXUSEU",
        "usd_gbp": "DEXUSUK",
        "usd_jpy": "DEXJPUS",
        "industrial_production": "INDPRO",
        "retail_sales": "RSXFS",
        "housing_starts": "HOUST",
        "m1": "M1SL",
        "m2": "M2SL",
        "consumer_sentiment": "UMCSENT",
        "vix": "VIXCLS",
    }
    
    def fetch(
        self,
        data_type: str,
        **params: Any,
    ) -> "pd.DataFrame":
        """Fetch data from FRED.
        
        Args:
            data_type: "economic_indicator" or a common name like "gdp"
            **params: series_id, start_date, end_date
            
        Returns:
            DataFrame with date and value columns
        """
        if not HAS_FREDAPI:
            raise DataSourceError("fredapi library not installed. Run: pip install fredapi")
        
        if not HAS_PANDAS:
            raise DataSourceError("pandas library not installed")
        
        self._check_rate_limit()
        
        api_key = self.get_api_key()
        fred = Fred(api_key=api_key)
        
        # Resolve series ID
        series_id = params.get("series_id") or params.get("series")
        if not series_id:
            # Try to map common names
            series_id = self.COMMON_SERIES.get(data_type.lower())
        
        if not series_id:
            raise DataSourceError(
                f"No series_id provided and '{data_type}' is not a known series. "
                f"Known series: {list(self.COMMON_SERIES.keys())}"
            )
        
        start_date = params.get("start_date") or params.get("start")
        end_date = params.get("end_date") or params.get("end")
        
        try:
            series = fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
            
            if series is None or series.empty:
                raise DataNotAvailableError(f"No data for series {series_id}")
            
            df = series.reset_index()
            df.columns = ["date", "value"]
            df["series_id"] = series_id
            
            return df
            
        except Exception as e:
            if "Bad Request" in str(e) or "not found" in str(e).lower():
                raise DataNotAvailableError(f"Series {series_id} not found in FRED")
            raise DataSourceError(f"Failed to fetch {series_id}: {e}")
    
    def check_availability(
        self,
        data_type: str,
        **params: Any,
    ) -> bool:
        """Check if a series exists in FRED."""
        if not HAS_FREDAPI:
            return False
        
        try:
            api_key = self.get_api_key()
        except APIKeyMissingError:
            return False
        
        series_id = params.get("series_id") or self.COMMON_SERIES.get(data_type.lower())
        if not series_id:
            return False
        
        try:
            fred = Fred(api_key=api_key)
            info = fred.get_series_info(series_id)
            return info is not None
        except Exception:
            return False
    
    def get_required_params(self, data_type: str) -> list[str]:
        """Get required parameters."""
        # If it's a known series name, no params needed
        if data_type.lower() in self.COMMON_SERIES:
            return []
        return ["series_id"]
    
    def search_series(
        self,
        query: str,
        limit: int = 10,
    ) -> "pd.DataFrame":
        """Search for FRED series by keyword.
        
        Args:
            query: Search query
            limit: Max results to return
            
        Returns:
            DataFrame with series info
        """
        if not HAS_FREDAPI:
            raise DataSourceError("fredapi not installed")
        
        api_key = self.get_api_key()
        fred = Fred(api_key=api_key)
        
        try:
            results = fred.search(query, limit=limit)
            return results.reset_index()
        except Exception as e:
            raise DataSourceError(f"FRED search failed: {e}")


# =============================================================================
# CoinGecko Source (Cryptocurrency)
# =============================================================================


@register_source
class CoinGeckoSource(DataSource):
    """CoinGecko cryptocurrency data source.
    
    Provides free access to:
    - Cryptocurrency prices
    - Market caps
    - Trading volumes
    - Historical data
    
    No API key required for basic usage.
    """
    
    name: ClassVar[str] = "coingecko"
    domain: ClassVar[str] = "finance"
    description: ClassVar[str] = "CoinGecko - Cryptocurrency prices and market data"
    requires_api_key: ClassVar[bool] = False
    rate_limit: ClassVar[RateLimit] = RateLimit(
        requests_per_minute=30,
        requests_per_day=0,
    )
    data_types: ClassVar[list[str]] = [
        "crypto_prices",
        "crypto_history",
        "crypto_market",
    ]
    
    BASE_URL: ClassVar[str] = "https://api.coingecko.com/api/v3"
    
    # Common coin IDs
    COMMON_COINS: ClassVar[dict[str, str]] = {
        "btc": "bitcoin",
        "eth": "ethereum",
        "usdt": "tether",
        "bnb": "binancecoin",
        "xrp": "ripple",
        "ada": "cardano",
        "sol": "solana",
        "doge": "dogecoin",
    }
    
    def fetch(
        self,
        data_type: str,
        **params: Any,
    ) -> "pd.DataFrame":
        """Fetch cryptocurrency data from CoinGecko."""
        if not HAS_REQUESTS:
            raise DataSourceError("requests library not installed")
        
        if not HAS_PANDAS:
            raise DataSourceError("pandas library not installed")
        
        self._check_rate_limit()
        
        if data_type in ("crypto_prices", "crypto_history"):
            return self._fetch_price_history(**params)
        elif data_type == "crypto_market":
            return self._fetch_market_data(**params)
        else:
            raise DataSourceError(f"Unknown data type: {data_type}")
    
    def check_availability(
        self,
        data_type: str,
        **params: Any,
    ) -> bool:
        """Check if coin data is available."""
        if not HAS_REQUESTS:
            return False
        
        coin = params.get("coin") or params.get("coin_id")
        if not coin:
            return False
        
        coin_id = self.COMMON_COINS.get(coin.lower(), coin.lower())
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/coins/{coin_id}",
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_required_params(self, data_type: str) -> list[str]:
        """Get required parameters."""
        if data_type in ("crypto_prices", "crypto_history"):
            return ["coin"]
        return []
    
    def _fetch_price_history(
        self,
        coin: str,
        vs_currency: str = "usd",
        days: int = 365,
        **kwargs: Any,
    ) -> "pd.DataFrame":
        """Fetch historical price data for a coin."""
        coin_id = self.COMMON_COINS.get(coin.lower(), coin.lower())
        
        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": days,
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 429:
                raise RateLimitError(self.name, 60)
            
            response.raise_for_status()
            data = response.json()
            
            # Parse prices
            prices = data.get("prices", [])
            if not prices:
                raise DataNotAvailableError(f"No price data for {coin}")
            
            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["coin"] = coin_id
            df["vs_currency"] = vs_currency
            
            # Add volume and market cap if available
            if "total_volumes" in data:
                volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
                df["volume"] = volumes["volume"]
            
            if "market_caps" in data:
                caps = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
                df["market_cap"] = caps["market_cap"]
            
            return df[["date", "price", "volume", "market_cap", "coin", "vs_currency"]]
            
        except requests.exceptions.RequestException as e:
            raise DataSourceError(f"Failed to fetch {coin} prices: {e}")
    
    def _fetch_market_data(
        self,
        vs_currency: str = "usd",
        per_page: int = 100,
        page: int = 1,
        **kwargs: Any,
    ) -> "pd.DataFrame":
        """Fetch current market data for top coins."""
        url = f"{self.BASE_URL}/coins/markets"
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "sparkline": False,
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 429:
                raise RateLimitError(self.name, 60)
            
            response.raise_for_status()
            data = response.json()
            
            if not data:
                raise DataNotAvailableError("No market data available")
            
            df = pd.DataFrame(data)
            df["vs_currency"] = vs_currency
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise DataSourceError(f"Failed to fetch market data: {e}")
