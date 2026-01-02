"""Unit tests for node-level caching module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.settings import Settings


class TestCacheSettings:
    """Tests for cache configuration in settings."""
    
    def test_cache_enabled_default(self):
        """Cache should be enabled by default."""
        settings = Settings()
        assert settings.cache_enabled is True
    
    def test_cache_path_default(self):
        """Cache path should default to data/node_cache.db."""
        settings = Settings()
        assert "node_cache.db" in settings.cache_path
        assert "data" in settings.cache_path
    
    def test_cache_ttl_default(self):
        """Default TTL should be 1800 seconds (30 minutes)."""
        settings = Settings()
        assert settings.cache_ttl_default == 1800
    
    def test_cache_ttl_literature(self):
        """Literature TTL should be 3600 seconds (1 hour)."""
        settings = Settings()
        assert settings.cache_ttl_literature == 3600
    
    def test_cache_ttl_synthesis(self):
        """Synthesis TTL should be 1800 seconds (30 minutes)."""
        settings = Settings()
        assert settings.cache_ttl_synthesis == 1800
    
    def test_cache_ttl_writer(self):
        """Writer TTL should be 600 seconds (10 minutes)."""
        settings = Settings()
        assert settings.cache_ttl_writer == 600
    
    def test_cache_ttl_gap_analysis(self):
        """Gap analysis TTL should be 1800 seconds (30 minutes)."""
        settings = Settings()
        assert settings.cache_ttl_gap_analysis == 1800


class TestCacheModule:
    """Tests for cache module functions."""
    
    def test_import_cache_module(self):
        """Cache module should be importable."""
        from src.cache import get_cache, get_cache_policy, clear_cache, get_cache_stats
        assert callable(get_cache)
        assert callable(get_cache_policy)
        assert callable(clear_cache)
        assert callable(get_cache_stats)
    
    def test_get_cache_policy_with_ttl(self):
        """get_cache_policy should create policy with specified TTL."""
        from src.cache import get_cache_policy
        from langgraph.types import CachePolicy
        
        policy = get_cache_policy(ttl=3600)
        
        # Policy should be created when caching is enabled
        if policy is not None:
            assert isinstance(policy, CachePolicy)
            assert policy.ttl == 3600
    
    def test_get_cache_policy_default_ttl(self):
        """get_cache_policy should use default TTL when not specified."""
        from src.cache import get_cache_policy
        from src.config import settings
        
        policy = get_cache_policy()
        
        if policy is not None:
            assert policy.ttl == settings.cache_ttl_default
    
    def test_get_cache_stats_returns_dict(self):
        """get_cache_stats should return a dictionary."""
        from src.cache import get_cache_stats
        
        stats = get_cache_stats()
        
        assert isinstance(stats, dict)
        assert "enabled" in stats
    
    def test_get_cache_stats_when_enabled(self):
        """get_cache_stats should include TTL values when enabled."""
        from src.cache import get_cache_stats
        from src.config import settings
        
        if settings.cache_enabled:
            stats = get_cache_stats()
            
            assert stats["enabled"] is True
            assert "ttl_default" in stats
            assert "ttl_literature" in stats
            assert "ttl_synthesis" in stats
            assert "ttl_writer" in stats


class TestCacheBackend:
    """Tests for cache backend initialization."""
    
    def test_get_cache_returns_sqlite_cache(self):
        """get_cache should return SqliteCache when enabled."""
        from src.cache import get_cache
        from src.config import settings
        
        if settings.cache_enabled:
            cache = get_cache()
            
            # Should return a cache instance
            assert cache is not None
            # Should have the cache interface methods
            assert hasattr(cache, "get") or hasattr(cache, "aget")
    
    def test_get_cache_creates_directory(self):
        """get_cache should create cache directory if it doesn't exist."""
        from src.cache import get_cache
        from src.config import settings
        
        if settings.cache_enabled:
            cache = get_cache()
            
            # Directory should exist
            cache_path = Path(settings.cache_path)
            assert cache_path.parent.exists()
    
    def test_get_cache_singleton(self):
        """get_cache should return the same instance on multiple calls."""
        from src.cache import get_cache
        from src.config import settings
        
        if settings.cache_enabled:
            cache1 = get_cache()
            cache2 = get_cache()
            
            assert cache1 is cache2


class TestCacheDisabled:
    """Tests for behavior when caching is disabled."""
    
    def test_get_cache_returns_none_when_disabled(self):
        """get_cache should return None when caching is disabled."""
        from src.cache import get_cache
        
        # Temporarily disable caching
        from src.config import settings
        original_enabled = settings.cache_enabled
        
        try:
            settings.cache_enabled = False
            
            # Reset the singleton
            import src.cache as cache_module
            cache_module._cache_instance = None
            
            cache = get_cache()
            assert cache is None
        finally:
            settings.cache_enabled = original_enabled
            cache_module._cache_instance = None
    
    def test_get_cache_policy_returns_none_when_disabled(self):
        """get_cache_policy should return None when caching is disabled."""
        from src.cache import get_cache_policy
        from src.config import settings
        
        original_enabled = settings.cache_enabled
        
        try:
            settings.cache_enabled = False
            
            policy = get_cache_policy(ttl=3600)
            assert policy is None
        finally:
            settings.cache_enabled = original_enabled
    
    def test_clear_cache_returns_false_when_disabled(self):
        """clear_cache should return False when caching is disabled."""
        from src.cache import clear_cache
        from src.config import settings
        
        original_enabled = settings.cache_enabled
        
        try:
            settings.cache_enabled = False
            
            # Reset the singleton
            import src.cache as cache_module
            cache_module._cache_instance = None
            
            result = clear_cache()
            assert result is False
        finally:
            settings.cache_enabled = original_enabled
            cache_module._cache_instance = None


class TestGraphCaching:
    """Tests for caching in the workflow graph."""
    
    def test_graph_imports_cache(self):
        """Graph module should import cache functions."""
        from studio.graphs import get_cache, get_cache_policy
        assert callable(get_cache)
        assert callable(get_cache_policy)
    
    def test_create_research_workflow_with_cache(self):
        """create_research_workflow should compile with cache."""
        from studio.graphs import create_research_workflow
        
        # Should not raise an error
        workflow = create_research_workflow()
        
        assert workflow is not None
        # Should have nodes
        assert len(workflow.nodes) > 0
