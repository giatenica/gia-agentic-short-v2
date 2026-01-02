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
