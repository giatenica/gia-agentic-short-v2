"""Application settings and environment configuration."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    default_model: str = "claude-sonnet-4-5-20250929"

    # LangSmith
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "") or os.getenv(
        "LANGCHAIN_SMITH_API_KEY", ""
    )
    langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "gia-agentic-short-v2")

    # Tavily
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # Google Serper (alternative search)
    serper_api_key: str = os.getenv("GOOGLE_SERPER_API_KEY", "")

    # External Data APIs (Sprint 13)
    fred_api_key: str = os.getenv("FRED_API_KEY", "")

    # Node-level caching configuration
    # Caches LLM responses to avoid redundant computation during development/testing
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    cache_path: str = os.getenv("CACHE_PATH", str(PROJECT_ROOT / "data" / "node_cache.db"))
    cache_ttl_default: int = int(os.getenv("CACHE_TTL_DEFAULT", "1800"))  # 30 minutes
    
    # Per-node TTL overrides (in seconds)
    cache_ttl_literature: int = int(os.getenv("CACHE_TTL_LITERATURE", "3600"))  # 1 hour
    cache_ttl_synthesis: int = int(os.getenv("CACHE_TTL_SYNTHESIS", "1800"))  # 30 minutes
    cache_ttl_gap_analysis: int = int(os.getenv("CACHE_TTL_GAP_ANALYSIS", "1800"))  # 30 minutes
    cache_ttl_writer: int = int(os.getenv("CACHE_TTL_WRITER", "600"))  # 10 minutes

    # Output artifacts
    # Where to write exported tables/figures and compiled LaTeX PDFs.
    output_dir: str = os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "data" / "outputs"))

    def __post_init__(self):
        """Configure LangSmith environment variables."""
        if self.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_TRACING"] = str(self.langsmith_tracing).lower()
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project

    def validate(self) -> list[str]:
        """Validate required settings are present."""
        errors = []
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is not set")
        if not self.langsmith_api_key:
            errors.append("LANGSMITH_API_KEY (or LANGCHAIN_SMITH_API_KEY) is not set")
        return errors


# Global settings instance
settings = Settings()
