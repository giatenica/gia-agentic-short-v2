"""Unit tests for LITERATURE_REVIEWER and LITERATURE_SYNTHESIZER nodes."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from src.state.enums import ResearchStatus
from src.state.models import SearchQuery, SearchResult
from src.state.schema import create_initial_state
from src.nodes.literature_reviewer import (
    literature_reviewer_node,
    generate_search_queries,
    execute_searches_sync,
    process_search_results,
    extract_methodology_precedents,
)
from src.nodes.literature_synthesizer import (
    literature_synthesizer_node,
    extract_themes,
    identify_gaps,
    synthesize_literature,
    generate_contribution_statement,
    refine_research_question,
)


# =============================================================================
# Literature Reviewer Node Tests
# =============================================================================


class TestGenerateSearchQueries:
    """Tests for search query generation."""
    
    @patch("src.nodes.literature_reviewer.ChatAnthropic")
    def test_generate_queries_basic(self, mock_claude):
        """Test generating search queries from research question."""
        mock_response = MagicMock()
        mock_response.content = """
QUERY: machine learning stock prediction
TYPE: academic
PRIORITY: 1

QUERY: deep learning finance
TYPE: theory
PRIORITY: 2

QUERY: neural networks financial forecasting 2023
TYPE: recent
PRIORITY: 3
"""
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_claude.return_value = mock_model
        
        queries = generate_search_queries(
            "How can machine learning improve stock price prediction?",
            ["stock prices", "ML models"],
        )
        
        assert len(queries) >= 1
        assert all(isinstance(q, SearchQuery) for q in queries)
        assert queries[0].query_text != ""
    
    @patch("src.nodes.literature_reviewer.ChatAnthropic")
    def test_generate_queries_fallback(self, mock_claude):
        """Test fallback when parsing fails."""
        mock_response = MagicMock()
        mock_response.content = "Unable to generate queries"
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_claude.return_value = mock_model
        
        queries = generate_search_queries(
            "Research question here",
            [],
        )
        
        # Should fallback to original question as query
        assert len(queries) >= 1
        assert queries[0].query_text == "Research question here"


class TestProcessSearchResults:
    """Tests for search result processing."""
    
    def test_process_results_basic(self):
        """Test processing raw search results."""
        raw_results = [
            {
                "title": "High Impact Paper",
                "url": "https://example.com/1",
                "abstract": "Important findings",
                "citation_count": 500,
                "source": "semantic_scholar",
            },
            {
                "title": "Regular Paper",
                "url": "https://example.com/2",
                "abstract": "Some findings",
                "citation_count": 20,
                "source": "semantic_scholar",
            },
        ]
        
        processed, seminal = process_search_results(raw_results, "query-1")
        
        assert len(processed) <= len(raw_results)
        # High impact paper should be in seminal works
        seminal_titles = [s.get("title") for s in seminal]
        assert "High Impact Paper" in seminal_titles
    
    def test_process_empty_results(self):
        """Test processing empty results."""
        processed, seminal = process_search_results([], "query-1")
        
        assert processed == []
        assert seminal == []


class TestExtractMethodologyPrecedents:
    """Tests for methodology precedent extraction."""
    
    def test_extract_methodology_from_results(self):
        """Test extracting methodology precedents."""
        results = [
            SearchResult(
                query_id="q1",
                title="A Novel Methodology for Analysis",
                url="https://example.com/1",
                snippet="This paper proposes a new regression methodology for panel data...",
                citation_count=100,
                venue="Top Journal",
            ),
            SearchResult(
                query_id="q1",
                title="Regular Paper",
                url="https://example.com/2",
                snippet="We find interesting results about markets...",
                citation_count=50,
            ),
        ]
        
        precedents = extract_methodology_precedents(results)
        
        # Should include paper with methodology in snippet
        assert any("Methodology" in p for p in precedents)
    
    def test_extract_methodology_empty(self):
        """Test extraction with no methodology papers."""
        results = [
            SearchResult(
                query_id="q1",
                title="Findings Paper",
                url="https://example.com",
                snippet="We present our findings on market behavior...",
            ),
        ]
        
        precedents = extract_methodology_precedents(results)
        
        # Should return empty list if no methodology papers
        assert len(precedents) == 0


class TestLiteratureReviewerNode:
    """Tests for the main literature reviewer node."""
    
    @patch("src.nodes.literature_reviewer.execute_searches_sync")
    @patch("src.nodes.literature_reviewer.generate_search_queries")
    def test_node_success(self, mock_queries, mock_search):
        """Test successful literature review."""
        mock_queries.return_value = [
            SearchQuery(query_text="test query", source_type="academic"),
        ]
        mock_search.return_value = [
            {
                "title": "Test Paper",
                "url": "https://example.com",
                "abstract": "Test abstract",
                "citation_count": 50,
                "source": "semantic_scholar",
            },
        ]
        
        state = create_initial_state()
        state["original_query"] = "What is the impact of X on Y?"
        
        result = literature_reviewer_node(state)
        
        assert result["status"] == ResearchStatus.LITERATURE_REVIEW_COMPLETE
        assert "search_results" in result
        assert "messages" in result
    
    def test_node_no_query(self):
        """Test node with no research question."""
        state = create_initial_state()
        state["original_query"] = ""
        
        result = literature_reviewer_node(state)
        
        assert result["status"] == ResearchStatus.FAILED
        assert len(result["errors"]) > 0
    
    @patch("src.nodes.literature_reviewer.execute_searches_sync")
    @patch("src.nodes.literature_reviewer.generate_search_queries")
    def test_node_no_results(self, mock_queries, mock_search):
        """Test node when search returns no results."""
        mock_queries.return_value = [
            SearchQuery(query_text="very specific query"),
        ]
        mock_search.return_value = []
        
        state = create_initial_state()
        state["original_query"] = "Very niche topic"
        
        result = literature_reviewer_node(state)
        
        assert result["status"] == ResearchStatus.LITERATURE_REVIEW_COMPLETE
        assert result["search_results"] == []


# =============================================================================
# Literature Synthesizer Node Tests
# =============================================================================


class TestExtractThemes:
    """Tests for theme extraction."""
    
    @patch("src.nodes.literature_synthesizer.ChatAnthropic")
    def test_extract_themes_basic(self, mock_claude):
        """Test extracting themes from search results."""
        mock_response = MagicMock()
        mock_response.content = """
THEME: Machine Learning in Finance
DESCRIPTION: Application of ML techniques to financial prediction
EVIDENCE: 1, 2, 3

THEME: Market Efficiency
DESCRIPTION: Studies examining efficient market hypothesis
EVIDENCE: 4, 5
"""
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_claude.return_value = mock_model
        
        results = [
            SearchResult(
                query_id="q1",
                title="ML in Finance",
                url="https://example.com",
                snippet="Using machine learning for stock prediction",
            ),
        ]
        
        themes = extract_themes(results)
        
        assert len(themes) >= 1
        assert any("Machine Learning" in t for t in themes)
    
    def test_extract_themes_empty(self):
        """Test theme extraction with no results."""
        themes = extract_themes([])
        
        assert themes == []


class TestIdentifyGaps:
    """Tests for research gap identification."""
    
    @patch("src.nodes.literature_synthesizer.ChatAnthropic")
    def test_identify_gaps_basic(self, mock_claude):
        """Test identifying research gaps."""
        mock_response = MagicMock()
        mock_response.content = """
GAP: Limited studies on emerging markets
TYPE: contextual
OPPORTUNITY: Extend analysis to developing countries

GAP: Lack of real-time prediction models
TYPE: methodological
OPPORTUNITY: Develop streaming ML approaches
"""
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_claude.return_value = mock_model
        
        results = [
            SearchResult(
                query_id="q1",
                title="Developed Market Study",
                url="https://example.com",
                snippet="Analysis of US stock markets",
            ),
        ]
        
        gaps = identify_gaps(
            "Impact of ML on stock prediction",
            results,
            ["ML in Finance"],
        )
        
        assert len(gaps) >= 1
        assert any("emerging" in g.lower() or "market" in g.lower() for g in gaps)


class TestSynthesizeLiterature:
    """Tests for literature synthesis."""
    
    @patch("src.nodes.literature_synthesizer.ChatAnthropic")
    def test_synthesize_basic(self, mock_claude):
        """Test synthesizing literature."""
        mock_response = MagicMock()
        mock_response.content = """
1. STATE OF THE FIELD

The field of machine learning in finance has grown substantially.
Research has focused on prediction accuracy and model interpretability.

2. KEY FINDINGS

- ML models outperform traditional methods
- Feature engineering is crucial
- Deep learning shows promise

3. THEORETICAL FRAMEWORKS

- Efficient Market Hypothesis
- Behavioral Finance

4. METHODOLOGICAL APPROACHES

- Regression analysis
- Neural networks
- Ensemble methods

5. CONTRIBUTION OPPORTUNITIES

- Real-time prediction systems
- Emerging market applications
"""
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_claude.return_value = mock_model
        
        results = [
            SearchResult(
                query_id="q1",
                title="Test Paper",
                url="https://example.com",
                snippet="Test abstract",
            ),
        ]
        
        synthesis = synthesize_literature(
            "Research question",
            results,
            ["Theme 1"],
            ["Gap 1"],
        )
        
        assert "summary" in synthesis
        assert "full_synthesis" in synthesis
    
    def test_synthesize_empty_results(self):
        """Test synthesis with no results."""
        synthesis = synthesize_literature(
            "Research question",
            [],
            [],
            [],
        )
        
        assert "Insufficient" in synthesis["summary"]


class TestGenerateContributionStatement:
    """Tests for contribution statement generation."""
    
    @patch("src.nodes.literature_synthesizer.ChatAnthropic")
    def test_generate_contribution(self, mock_claude):
        """Test generating contribution statement."""
        mock_response = MagicMock()
        mock_response.content = "This research addresses the gap in emerging market analysis by developing novel ML models."
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_claude.return_value = mock_model
        
        contribution = generate_contribution_statement(
            "Impact of ML on emerging market prediction",
            ["Gap in emerging market studies"],
            {"contribution_opportunities": ["Extend to emerging markets"]},
        )
        
        assert len(contribution) > 0
        assert "emerging" in contribution.lower() or "gap" in contribution.lower()


class TestRefineResearchQuestion:
    """Tests for research question refinement."""
    
    @patch("src.nodes.literature_synthesizer.ChatAnthropic")
    def test_refine_question(self, mock_claude):
        """Test refining research question."""
        mock_response = MagicMock()
        mock_response.content = "How do ensemble machine learning methods improve stock price prediction in emerging Asian markets?"
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_claude.return_value = mock_model
        
        refined = refine_research_question(
            "How can ML improve stock prediction?",
            ["Gap in emerging markets", "Gap in ensemble methods"],
            {"state_of_field": "Current research focuses on developed markets"},
        )
        
        assert len(refined) > 0
    
    @patch("src.nodes.literature_synthesizer.ChatAnthropic")
    def test_refine_keeps_original_if_good(self, mock_claude):
        """Test that original question is kept if already good."""
        original = "A very specific and well-formed research question?"
        mock_response = MagicMock()
        mock_response.content = original  # Returns unchanged
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_claude.return_value = mock_model
        
        refined = refine_research_question(original, [], {})
        
        assert refined == original


class TestLiteratureSynthesizerNode:
    """Tests for the main literature synthesizer node."""
    
    @patch("src.nodes.literature_synthesizer.refine_research_question")
    @patch("src.nodes.literature_synthesizer.generate_contribution_statement")
    @patch("src.nodes.literature_synthesizer.synthesize_literature")
    @patch("src.nodes.literature_synthesizer.identify_gaps")
    @patch("src.nodes.literature_synthesizer.extract_themes")
    def test_node_success(
        self,
        mock_themes,
        mock_gaps,
        mock_synthesize,
        mock_contribution,
        mock_refine,
    ):
        """Test successful literature synthesis."""
        mock_themes.return_value = ["Theme 1", "Theme 2"]
        mock_gaps.return_value = ["Gap 1", "Gap 2"]
        mock_synthesize.return_value = {
            "summary": "Summary",
            "state_of_field": "State",
            "key_findings": ["Finding 1"],
            "theoretical_frameworks": ["Framework 1"],
            "methodological_approaches": ["Method 1"],
            "contribution_opportunities": ["Opportunity 1"],
            "full_synthesis": "Full text",
        }
        mock_contribution.return_value = "Contribution statement"
        mock_refine.return_value = "Refined question"
        
        state = create_initial_state()
        state["original_query"] = "What is the impact of X on Y?"
        state["search_results"] = [
            SearchResult(
                query_id="q1",
                title="Test Paper",
                url="https://example.com",
                snippet="Test abstract",
            ),
        ]
        
        result = literature_synthesizer_node(state)
        
        assert result["status"] == ResearchStatus.GAP_IDENTIFICATION_COMPLETE
        assert result["literature_themes"] == ["Theme 1", "Theme 2"]
        assert result["identified_gaps"] == ["Gap 1", "Gap 2"]
        assert result["contribution_statement"] == "Contribution statement"
        assert result["refined_query"] == "Refined question"
    
    def test_node_no_query(self):
        """Test node with no research question."""
        state = create_initial_state()
        state["original_query"] = ""
        
        result = literature_synthesizer_node(state)
        
        assert result["status"] == ResearchStatus.FAILED
        assert len(result["errors"]) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestLiteratureReviewWorkflow:
    """Integration tests for literature review workflow."""
    
    @patch("src.nodes.literature_synthesizer.ChatAnthropic")
    @patch("src.nodes.literature_reviewer.execute_searches_sync")
    @patch("src.nodes.literature_reviewer.ChatAnthropic")
    def test_reviewer_to_synthesizer_flow(
        self,
        mock_reviewer_claude,
        mock_search,
        mock_synthesizer_claude,
    ):
        """Test data flow from reviewer to synthesizer."""
        # Setup reviewer mocks
        mock_query_response = MagicMock()
        mock_query_response.content = "QUERY: test\nTYPE: academic\nPRIORITY: 1"
        mock_reviewer_model = MagicMock()
        mock_reviewer_model.invoke.return_value = mock_query_response
        mock_reviewer_claude.return_value = mock_reviewer_model
        
        mock_search.return_value = [
            {
                "title": "Test Paper",
                "url": "https://example.com",
                "abstract": "Test abstract",
                "citation_count": 100,
                "source": "semantic_scholar",
            },
        ]
        
        # Setup synthesizer mocks
        mock_synth_response = MagicMock()
        mock_synth_response.content = "THEME: Test Theme\nDESCRIPTION: Test"
        mock_synthesizer_model = MagicMock()
        mock_synthesizer_model.invoke.return_value = mock_synth_response
        mock_synthesizer_claude.return_value = mock_synthesizer_model
        
        # Run workflow
        state = create_initial_state()
        state["original_query"] = "Test research question?"
        
        # Step 1: Literature review
        reviewer_result = literature_reviewer_node(state)
        assert reviewer_result["status"] == ResearchStatus.LITERATURE_REVIEW_COMPLETE
        
        # Step 2: Update state and synthesize
        state.update(reviewer_result)
        synthesizer_result = literature_synthesizer_node(state)
        
        # Verify final state
        assert synthesizer_result["status"] == ResearchStatus.GAP_IDENTIFICATION_COMPLETE
        assert "literature_synthesis" in synthesizer_result
        assert "literature_themes" in synthesizer_result
