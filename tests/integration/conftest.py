"""Fixtures for integration tests.

Provides mock LLM, workflow fixtures, and test data for integration testing.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.state.enums import (
    ResearchStatus,
    ResearchType,
    PaperType,
    TargetJournal,
)


# =============================================================================
# Mock LLM Response Fixtures
# =============================================================================


class MockLLMResponse:
    """Mock response from Claude API."""
    
    def __init__(self, content: str, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []
        
    @property
    def text(self) -> str:
        return self.content


class MockChatModel:
    """Mock Anthropic Chat model for testing."""
    
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or [
            "This is a mock LLM response for testing purposes."
        ]
        self.call_count = 0
        self.calls: list[dict] = []
    
    def invoke(self, messages: list, **kwargs) -> MockLLMResponse:
        """Synchronous invoke."""
        self.calls.append({"messages": messages, "kwargs": kwargs})
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return MockLLMResponse(response)
    
    async def ainvoke(self, messages: list, **kwargs) -> MockLLMResponse:
        """Async invoke."""
        return self.invoke(messages, **kwargs)
    
    def bind_tools(self, tools: list):
        """Return self with tools bound."""
        return self


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    return MockChatModel()


@pytest.fixture
def mock_llm_with_responses():
    """Factory fixture to create mock LLM with specific responses."""
    def _create(responses: list[str]) -> MockChatModel:
        return MockChatModel(responses=responses)
    return _create


# =============================================================================
# Workflow State Fixtures
# =============================================================================


@pytest.fixture
def minimal_state() -> dict[str, Any]:
    """Minimal valid workflow state for testing."""
    return {
        "original_query": "What is the impact of AI on productivity?",
        "project_title": "AI and Productivity Research",
        "research_type": ResearchType.EMPIRICAL,
        "paper_type": PaperType.FULL_PAPER,
        "target_journal": TargetJournal.JF,
        "status": ResearchStatus.INITIALIZED,
        "errors": [],
        "messages": [],
    }


@pytest.fixture
def state_after_intake() -> dict[str, Any]:
    """State after intake node has processed."""
    return {
        "original_query": "What is the impact of AI on productivity?",
        "refined_query": "How does artificial intelligence adoption affect firm-level productivity?",
        "project_title": "AI and Productivity Research",
        "research_type": ResearchType.EMPIRICAL,
        "paper_type": PaperType.FULL_PAPER,
        "target_journal": TargetJournal.JF,
        "status": ResearchStatus.INTAKE_COMPLETE,
        "intake_output": {
            "validated": True,
            "research_question": "How does artificial intelligence adoption affect firm-level productivity?",
            "key_concepts": ["artificial intelligence", "productivity", "firm performance"],
        },
        "errors": [],
        "messages": [],
    }


@pytest.fixture
def state_after_literature() -> dict[str, Any]:
    """State after literature review."""
    return {
        "original_query": "What is the impact of AI on productivity?",
        "refined_query": "How does AI adoption affect firm productivity?",
        "project_title": "AI and Productivity Research",
        "research_type": ResearchType.EMPIRICAL,
        "paper_type": PaperType.FULL_PAPER,
        "target_journal": TargetJournal.JF,
        "status": ResearchStatus.LITERATURE_REVIEW_COMPLETE,
        "literature_review_results": {
            "papers_found": 25,
            "key_themes": [
                "AI adoption barriers",
                "Productivity measurement",
                "Firm heterogeneity",
            ],
            "seminal_works": [
                {"title": "AI and Productivity", "authors": ["Smith, J."], "year": 2023},
            ],
        },
        "literature_synthesis": {
            "themes": ["AI adoption", "Productivity metrics", "Firm size effects"],
            "gaps_identified": ["Limited SME studies", "Long-term effects unclear"],
        },
        "errors": [],
        "messages": [],
    }


@pytest.fixture
def state_with_errors() -> dict[str, Any]:
    """State with accumulated errors for fallback testing."""
    mock_error1 = MagicMock()
    mock_error1.message = "Search API failed"
    mock_error1.recoverable = True
    mock_error1.node = "literature_reviewer"
    
    mock_error2 = MagicMock()
    mock_error2.message = "Rate limited"
    mock_error2.recoverable = True
    mock_error2.node = "literature_reviewer"
    
    mock_error3 = MagicMock()
    mock_error3.message = "Context overflow"
    mock_error3.recoverable = False
    mock_error3.node = "writer"
    
    return {
        "original_query": "Test query",
        "project_title": "Test Project",
        "research_type": ResearchType.THEORETICAL,
        "status": ResearchStatus.ANALYZING,
        "errors": [mock_error1, mock_error2, mock_error3],
        "messages": [],
    }


@pytest.fixture
def complete_state() -> dict[str, Any]:
    """Complete state ready for final output."""
    return {
        "original_query": "What is the impact of AI on productivity?",
        "refined_query": "How does AI adoption affect firm productivity?",
        "project_title": "AI and Productivity: A Firm-Level Analysis",
        "research_type": ResearchType.EMPIRICAL,
        "paper_type": PaperType.FULL_PAPER,
        "target_journal": TargetJournal.JF,
        "status": ResearchStatus.WRITING_COMPLETE,
        "literature_review_results": {"papers_found": 25},
        "literature_synthesis": {"themes": ["AI", "Productivity"]},
        "gap_analysis": {"primary_gap": "SME studies lacking"},
        "research_plan": {"methodology": "Panel regression"},
        "data_analyst_output": {"findings": ["AI increases productivity by 15%"]},
        "writer_output": {
            "abstract": "This paper examines...",
            "introduction": "The rise of AI...",
            "literature_review": "Prior research has...",
            "methods": "We employ panel data...",
            "results": "Our findings indicate...",
            "discussion": "These results suggest...",
            "conclusion": "In conclusion...",
        },
        "reviewer_output": {
            "decision": "APPROVE",
            "overall_score": 8.5,
        },
        "errors": [],
        "messages": [],
    }


# =============================================================================
# Mock Service Fixtures
# =============================================================================


@pytest.fixture
def mock_semantic_scholar():
    """Mock Semantic Scholar API responses."""
    return {
        "data": [
            {
                "paperId": "abc123",
                "title": "AI and Productivity",
                "abstract": "This paper studies...",
                "year": 2023,
                "citationCount": 50,
                "authors": [{"name": "John Smith"}],
            },
            {
                "paperId": "def456",
                "title": "Machine Learning in Business",
                "abstract": "We examine...",
                "year": 2022,
                "citationCount": 30,
                "authors": [{"name": "Jane Doe"}],
            },
        ],
        "total": 2,
    }


@pytest.fixture
def mock_tavily_search():
    """Mock Tavily search results."""
    return {
        "results": [
            {
                "title": "AI Productivity Research",
                "url": "https://example.com/ai-productivity",
                "content": "Recent studies show AI increases productivity...",
                "score": 0.95,
            },
        ],
        "query": "AI productivity research",
    }


# =============================================================================
# Workflow Fixtures
# =============================================================================


@pytest.fixture
def mock_checkpointer():
    """Mock checkpointer for testing persistence."""
    checkpointer = MagicMock()
    checkpointer.get = MagicMock(return_value=None)
    checkpointer.put = MagicMock()
    checkpointer.list = MagicMock(return_value=[])
    return checkpointer


@pytest.fixture
def workflow_config():
    """Standard workflow configuration for testing."""
    return {
        "configurable": {
            "thread_id": "test-thread-123",
        }
    }


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_research_questions() -> list[dict]:
    """Sample research questions for evaluation."""
    return [
        {
            "id": "rq_001",
            "query": "What factors influence cryptocurrency adoption?",
            "type": "empirical",
            "expected_themes": ["adoption", "cryptocurrency", "blockchain"],
        },
        {
            "id": "rq_002",
            "query": "How do ESG factors affect stock returns?",
            "type": "empirical",
            "expected_themes": ["ESG", "returns", "sustainable investing"],
        },
        {
            "id": "rq_003",
            "query": "What is the theoretical framework for digital transformation?",
            "type": "theoretical",
            "expected_themes": ["digital transformation", "theory", "framework"],
        },
    ]


@pytest.fixture
def sample_paper_data() -> dict:
    """Sample paper data for citation testing."""
    return {
        "title": "The Impact of AI on Financial Services",
        "authors": ["Smith, John", "Doe, Jane"],
        "year": 2024,
        "journal": "Journal of Finance",
        "volume": "79",
        "issue": "3",
        "pages": "1245-1289",
        "doi": "10.1111/jofi.12345",
    }
