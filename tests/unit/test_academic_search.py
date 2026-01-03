"""Unit tests for academic search and citation analysis tools."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date

from src.state.models import SearchResult
from src.tools.academic_search import (
    semantic_scholar_search,
    arxiv_search,
    tavily_academic_search,
    convert_to_search_result,
    merge_search_results,
    rank_by_citations,
    identify_seminal_works,
    ACADEMIC_SEARCH_TOOLS,
)
from src.tools.citation_analysis import (
    get_citing_papers,
    get_references,
    get_paper_details,
    get_author_papers,
    calculate_citation_metrics,
    CITATION_ANALYSIS_TOOLS,
)


# =============================================================================
# Academic Search Tool Tests
# =============================================================================


class TestSemanticScholarSearch:
    """Tests for Semantic Scholar search tool."""
    
    @patch("src.tools.academic_search.httpx.Client")
    def test_successful_search(self, mock_client_class):
        """Test successful Semantic Scholar search."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "total": 2,
            "data": [
                {
                    "paperId": "abc123",
                    "title": "Test Paper 1",
                    "abstract": "This is a test abstract",
                    "year": 2023,
                    "authors": [{"name": "John Doe"}],
                    "citationCount": 50,
                    "url": "https://example.com/paper1",
                    "venue": "Test Journal",
                    "externalIds": {"DOI": "10.1234/test"},
                },
                {
                    "paperId": "def456",
                    "title": "Test Paper 2",
                    "abstract": "Another abstract",
                    "year": 2022,
                    "authors": [{"name": "Jane Smith"}],
                    "citationCount": 100,
                    "url": "https://example.com/paper2",
                    "venue": "Other Journal",
                    "externalIds": {},
                },
            ],
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = semantic_scholar_search.invoke({"query": "machine learning"})
        
        assert "error" not in result
        assert result["source"] == "semantic_scholar"
        assert result["total_results"] == 2
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Test Paper 1"
        assert result["results"][0]["citation_count"] == 50
        assert result["results"][0]["doi"] == "10.1234/test"
    
    @patch("src.tools.academic_search.httpx.Client")
    def test_search_with_filters(self, mock_client_class):
        """Test search with year and field filters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"total": 0, "data": []}
        mock_response.raise_for_status = MagicMock()
        
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = semantic_scholar_search.invoke({
            "query": "finance",
            "year_start": 2020,
            "year_end": 2023,
            "fields_of_study": ["Economics", "Finance"],
        })
        
        assert "error" not in result
        # Verify the API was called with filters
        call_args = mock_client.get.call_args
        assert "year" in call_args.kwargs["params"]
    
    @patch("src.tools.academic_search.time.sleep")
    @patch("src.tools.academic_search.httpx.Client")
    def test_search_api_error(self, mock_client_class, mock_sleep):
        """Test handling of API errors."""
        import httpx
        
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limited", request=MagicMock(), response=mock_response
        )
        
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = semantic_scholar_search.invoke({"query": "test"})
        
        assert "error" in result
        assert result["results"] == []
        # Verify retries happened (sleep was called)
        assert mock_sleep.call_count >= 1


class TestArxivSearch:
    """Tests for arXiv search tool."""
    
    @patch("src.tools.academic_search.httpx.Client")
    def test_successful_search(self, mock_client_class):
        """Test successful arXiv search."""
        xml_response = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
            <entry>
                <id>http://arxiv.org/abs/2301.00001</id>
                <title>Test arXiv Paper</title>
                <summary>Test abstract for arXiv paper</summary>
                <author><name>John Researcher</name></author>
                <published>2023-01-15T00:00:00Z</published>
                <arxiv:primary_category term="cs.LG"/>
            </entry>
        </feed>
        """
        
        mock_response = MagicMock()
        mock_response.text = xml_response
        mock_response.raise_for_status = MagicMock()
        
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = arxiv_search.invoke({"query": "machine learning"})
        
        assert "error" not in result
        assert result["source"] == "arxiv"
        assert len(result["results"]) >= 1
        assert result["results"][0]["title"] == "Test arXiv Paper"
        assert "2301.00001" in result["results"][0]["arxiv_id"]
    
    @patch("src.tools.academic_search.httpx.Client")
    def test_search_with_categories(self, mock_client_class):
        """Test search with category filter."""
        xml_response = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>
        """
        
        mock_response = MagicMock()
        mock_response.text = xml_response
        mock_response.raise_for_status = MagicMock()
        
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = arxiv_search.invoke({
            "query": "transformer",
            "categories": ["cs.LG", "cs.AI"],
        })
        
        assert "error" not in result


class TestTavilyAcademicSearch:
    """Tests for Tavily academic search tool."""
    
    @patch("langchain_tavily.TavilySearch")
    def test_successful_search(self, mock_tavily_class):
        """Test successful Tavily search."""
        mock_tavily = MagicMock()
        mock_tavily.invoke.return_value = [
            {
                "title": "Academic Paper",
                "url": "https://scholar.google.com/paper",
                "content": "Paper content",
            }
        ]
        mock_tavily_class.return_value = mock_tavily
        
        result = tavily_academic_search.invoke({"query": "corporate governance"})
        
        assert "error" not in result
        assert result["source"] == "tavily_academic"
        assert len(result["results"]) >= 1
    
    def test_tavily_not_configured(self):
        """Test handling when Tavily is not configured (no API key)."""
        # This test verifies the error handling when Tavily raises an exception
        # We can't easily mock ImportError since the import happens inside the function
        # Instead, test the tool with an invalid configuration
        with patch("langchain_tavily.TavilySearch") as mock_tavily_class:
            mock_tavily_class.side_effect = Exception("API key required")
            result = tavily_academic_search.invoke({"query": "test"})
            
            assert "error" in result
            assert result["results"] == []


class TestSearchResultProcessing:
    """Tests for search result processing utilities."""
    
    def test_convert_to_search_result(self):
        """Test converting raw result to SearchResult model."""
        raw = {
            "title": "Test Paper",
            "url": "https://example.com",
            "abstract": "Test abstract",
            "publication_date": "2023-06-15",
            "authors": ["Author 1", "Author 2"],
            "citation_count": 25,
            "venue": "Test Journal",
            "doi": "10.1234/test",
            "source": "semantic_scholar",
        }
        
        result = convert_to_search_result(raw, "query-123")
        
        assert isinstance(result, SearchResult)
        assert result.title == "Test Paper"
        assert result.query_id == "query-123"
        assert result.citation_count == 25
        assert result.published_date == date(2023, 6, 15)
    
    def test_merge_search_results(self):
        """Test merging and deduplicating results."""
        results1 = [
            {"title": "Paper A", "url": "url1"},
            {"title": "Paper B", "url": "url2"},
        ]
        results2 = [
            {"title": "Paper A", "url": "url3"},  # Duplicate by title
            {"title": "Paper C", "url": "url4"},
        ]
        
        merged = merge_search_results([results1, results2])
        
        assert len(merged) == 3
        titles = [r["title"] for r in merged]
        assert titles.count("Paper A") == 1
    
    def test_rank_by_citations(self):
        """Test ranking results by citation count."""
        results = [
            {"title": "Low cited", "citation_count": 5},
            {"title": "High cited", "citation_count": 500},
            {"title": "Medium cited", "citation_count": 50},
            {"title": "No citations", "citation_count": None},
        ]
        
        ranked = rank_by_citations(results)
        
        assert ranked[0]["title"] == "High cited"
        assert ranked[1]["title"] == "Medium cited"
        assert ranked[2]["title"] == "Low cited"
    
    def test_identify_seminal_works(self):
        """Test identifying seminal works by citation threshold."""
        results = [
            {"title": "Seminal 1", "citation_count": 500},
            {"title": "Seminal 2", "citation_count": 200},
            {"title": "Regular", "citation_count": 50},
            {"title": "New", "citation_count": 5},
        ]
        
        seminal = identify_seminal_works(results, citation_threshold=100)
        
        assert len(seminal) == 2
        assert all(r["citation_count"] >= 100 for r in seminal)


# =============================================================================
# Citation Analysis Tool Tests
# =============================================================================


class TestGetCitingPapers:
    """Tests for getting citing papers."""
    
    @patch("src.tools.citation_analysis.httpx.Client")
    def test_successful_get_citing(self, mock_client_class):
        """Test successfully getting citing papers."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "citingPaper": {
                        "paperId": "cit1",
                        "title": "Citing Paper 1",
                        "abstract": "Abstract",
                        "year": 2023,
                        "authors": [{"name": "Author"}],
                        "citationCount": 10,
                        "url": "https://example.com",
                        "venue": "Journal",
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = get_citing_papers.invoke({"paper_id": "abc123"})
        
        assert "error" not in result
        assert result["source_paper_id"] == "abc123"
        assert len(result["citing_papers"]) == 1
        assert result["citing_papers"][0]["title"] == "Citing Paper 1"


class TestGetReferences:
    """Tests for getting paper references."""
    
    @patch("src.tools.citation_analysis.httpx.Client")
    def test_successful_get_references(self, mock_client_class):
        """Test successfully getting references."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "citedPaper": {
                        "paperId": "ref1",
                        "title": "Referenced Paper 1",
                        "abstract": "Abstract",
                        "year": 2020,
                        "authors": [{"name": "Author"}],
                        "citationCount": 100,
                        "url": "https://example.com",
                        "venue": "Conference",
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = get_references.invoke({"paper_id": "abc123"})
        
        assert "error" not in result
        assert result["source_paper_id"] == "abc123"
        assert len(result["references"]) == 1
        assert result["references"][0]["citation_count"] == 100


class TestGetPaperDetails:
    """Tests for getting paper details."""
    
    @patch("src.tools.citation_analysis.httpx.Client")
    def test_successful_get_details(self, mock_client_class):
        """Test successfully getting paper details."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "paperId": "abc123",
            "title": "Detailed Paper",
            "abstract": "Full abstract text",
            "year": 2023,
            "authors": [{"name": "Author 1", "authorId": "auth1"}],
            "citationCount": 50,
            "referenceCount": 30,
            "url": "https://example.com",
            "venue": "Top Journal",
            "publicationDate": "2023-03-15",
            "fieldsOfStudy": ["Computer Science", "Economics"],
            "externalIds": {"DOI": "10.1234/test", "ArXiv": "2301.00001"},
            "tldr": {"text": "This paper does X"},
            "isOpenAccess": True,
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = get_paper_details.invoke({"paper_id": "abc123"})
        
        assert "error" not in result
        assert result["title"] == "Detailed Paper"
        assert result["doi"] == "10.1234/test"
        assert result["arxiv_id"] == "2301.00001"
        assert result["is_open_access"] is True
        assert result["tldr"] == "This paper does X"


class TestGetAuthorPapers:
    """Tests for getting author's papers."""
    
    @patch("src.tools.citation_analysis.httpx.Client")
    def test_successful_get_author_papers(self, mock_client_class):
        """Test successfully getting author papers."""
        mock_author_response = MagicMock()
        mock_author_response.json.return_value = {
            "authorId": "auth123",
            "name": "Famous Researcher",
            "affiliations": ["MIT"],
            "paperCount": 100,
            "citationCount": 5000,
            "hIndex": 30,
        }
        mock_author_response.raise_for_status = MagicMock()
        
        mock_papers_response = MagicMock()
        mock_papers_response.json.return_value = {
            "data": [
                {
                    "paperId": "paper1",
                    "title": "Paper 1",
                    "year": 2023,
                    "citationCount": 50,
                    "venue": "Journal",
                    "url": "https://example.com",
                }
            ]
        }
        mock_papers_response.raise_for_status = MagicMock()
        
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = [mock_author_response, mock_papers_response]
        mock_client_class.return_value = mock_client
        
        result = get_author_papers.invoke({"author_id": "auth123"})
        
        assert "error" not in result
        assert result["name"] == "Famous Researcher"
        assert result["h_index"] == 30
        assert len(result["papers"]) == 1


class TestCitationMetrics:
    """Tests for citation metrics calculation."""
    
    def test_calculate_metrics(self):
        """Test calculating citation metrics."""
        papers = [
            {"title": "Paper 1", "citation_count": 100},
            {"title": "Paper 2", "citation_count": 50},
            {"title": "Paper 3", "citation_count": 10},
            {"title": "Paper 4", "citation_count": 5},
            {"title": "Paper 5", "citation_count": None},
        ]
        
        metrics = calculate_citation_metrics(papers)
        
        assert metrics["total_papers"] == 5
        assert metrics["total_citations"] == 165
        assert metrics["max_citations"] == 100
        assert metrics["highly_cited_papers"] == 3  # >= 10 citations
    
    def test_empty_papers(self):
        """Test metrics with empty list."""
        metrics = calculate_citation_metrics([])
        
        assert metrics["total_papers"] == 0
        assert metrics["total_citations"] == 0


class TestToolExports:
    """Tests for tool exports."""
    
    def test_academic_search_tools_list(self):
        """Test that ACADEMIC_SEARCH_TOOLS contains expected tools."""
        assert len(ACADEMIC_SEARCH_TOOLS) == 3
        tool_names = [t.name for t in ACADEMIC_SEARCH_TOOLS]
        assert "semantic_scholar_search" in tool_names
        assert "arxiv_search" in tool_names
        assert "tavily_academic_search" in tool_names
    
    def test_citation_analysis_tools_list(self):
        """Test that CITATION_ANALYSIS_TOOLS contains expected tools."""
        assert len(CITATION_ANALYSIS_TOOLS) == 4
        tool_names = [t.name for t in CITATION_ANALYSIS_TOOLS]
        assert "get_citing_papers" in tool_names
        assert "get_references" in tool_names
        assert "get_paper_details" in tool_names
        assert "get_author_papers" in tool_names
