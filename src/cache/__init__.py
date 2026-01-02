"""Node-level caching for LangGraph workflows.

Provides configurable caching to avoid redundant LLM computation during
development and testing. Uses SQLite for persistence across restarts.

Usage:
    from src.cache import get_cache, get_cache_policy
    
    # In graph compilation
    cache = get_cache()  # Returns SqliteCache or None based on settings
    graph = workflow.compile(cache=cache)
    
    # Per-node cache policies
    workflow.add_node(
        "expensive_node",
        node_function,
        cache_policy=get_cache_policy(ttl=3600)
    )

Configuration via environment variables:
    CACHE_ENABLED=true       Enable/disable caching (default: true)
    CACHE_PATH=./data/cache.db   SQLite cache file path
    CACHE_TTL_DEFAULT=1800   Default TTL in seconds (30 minutes)
    CACHE_TTL_LITERATURE=3600    Literature search cache TTL
    CACHE_TTL_SYNTHESIS=1800     Synthesis nodes cache TTL
    CACHE_TTL_GAP_ANALYSIS=1800  Gap analysis cache TTL
    CACHE_TTL_WRITER=600         Writer node cache TTL
"""

import logging
from pathlib import Path

from langgraph.types import CachePolicy

from src.config import settings

logger = logging.getLogger(__name__)

# Cache backend - lazily initialized
_cache_instance = None


def get_cache():
    """
    Get the cache backend instance.
    
    Returns SqliteCache if caching is enabled, None otherwise.
    The cache instance is lazily initialized and reused.
    
    Returns:
        SqliteCache instance or None if caching is disabled.
    """
    global _cache_instance
    
    if not settings.cache_enabled:
        logger.debug("Node caching is disabled")
        return None
    
    if _cache_instance is not None:
        return _cache_instance
    
    # Lazy import to avoid import errors if langgraph.cache is not available
    try:
        from langgraph.cache.sqlite import SqliteCache
    except ImportError:
        logger.warning(
            "langgraph.cache.sqlite not available. "
            "Node caching requires langgraph >= 0.2.0. "
            "Caching will be disabled."
        )
        return None
    
    # Ensure cache directory exists
    cache_path = Path(settings.cache_path)
    cache_dir = cache_path.parent
    
    if not cache_dir.exists():
        logger.info(f"Creating cache directory: {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initializing SQLite cache at: {cache_path}")
    _cache_instance = SqliteCache(path=str(cache_path))
    
    return _cache_instance


def get_cache_policy(ttl: int | None = None) -> CachePolicy | None:
    """
    Create a CachePolicy with the specified TTL.
    
    Args:
        ttl: Time-to-live in seconds. If None, uses settings.cache_ttl_default.
             If caching is disabled, returns None.
    
    Returns:
        CachePolicy instance or None if caching is disabled.
    """
    if not settings.cache_enabled:
        return None
    
    effective_ttl = ttl if ttl is not None else settings.cache_ttl_default
    
    return CachePolicy(ttl=effective_ttl)


def clear_cache() -> bool:
    """
    Clear all cached entries.
    
    Returns:
        True if cache was cleared, False if caching is disabled or unavailable.
    """
    cache = get_cache()
    if cache is None:
        return False
    
    try:
        # Clear all namespaces
        cache.clear(namespaces=None)
        logger.info("Node cache cleared")
        return True
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return False


def get_cache_stats() -> dict:
    """
    Get cache statistics (if available).
    
    Returns:
        Dictionary with cache statistics or empty dict if unavailable.
    """
    if not settings.cache_enabled:
        return {"enabled": False}
    
    cache_path = Path(settings.cache_path)
    stats = {
        "enabled": True,
        "path": str(cache_path),
        "exists": cache_path.exists(),
        "size_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
        "ttl_default": settings.cache_ttl_default,
        "ttl_literature": settings.cache_ttl_literature,
        "ttl_synthesis": settings.cache_ttl_synthesis,
        "ttl_gap_analysis": settings.cache_ttl_gap_analysis,
        "ttl_writer": settings.cache_ttl_writer,
    }
    
    return stats


# Convenience exports
__all__ = [
    "get_cache",
    "get_cache_policy",
    "clear_cache",
    "get_cache_stats",
    "CachePolicy",
]
