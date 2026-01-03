"""Academic search tools for literature review.

These tools search academic databases for peer-reviewed papers,
preprints, and scholarly content. They are used by the LITERATURE_REVIEWER
node to gather literature before research planning.

Supported databases:
- Semantic Scholar: Peer-reviewed papers with citation graphs
- arXiv: Preprints in physics, math, CS, and related fields
- Tavily: Broad academic coverage via web search
"""

import time
import xml.etree.ElementTree as ET

import httpx
from datetime import datetime
from typing import Any

from langchain_core.tools import tool

from src.state.models import SearchResult


# =============================================================================
# Rate Limit Configuration
# =============================================================================

MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
BACKOFF_MULTIPLIER = 2.0


def _make_request_with_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, Any],
    max_retries: int = MAX_RETRIES,
) -> httpx.Response:
    """Make HTTP request with exponential backoff retry for rate limits."""
    backoff = INITIAL_BACKOFF
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            response = client.get(url, params=params)
            # If we get a 429, wait and retry
            if response.status_code == 429:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= BACKOFF_MULTIPLIER
                    continue
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries:
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
                last_exception = e
                continue
            raise
        except httpx.RequestError as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
                continue
            raise
    
    if last_exception:
        raise last_exception
    raise RuntimeError("Request failed after retries")


# =============================================================================
# Semantic Scholar Search
# =============================================================================

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"


@tool
def semantic_scholar_search(
    query: str,
    limit: int = 10,
    year_start: int | None = None,
    year_end: int | None = None,
    fields_of_study: list[str] | None = None,
) -> dict[str, Any]:
    """
    Search Semantic Scholar for peer-reviewed academic papers.
    
    Semantic Scholar provides high-quality academic search with citation data,
    abstracts, and links to full papers. Best for finding peer-reviewed research.
    
    Args:
        query: Search query text (research topic or keywords).
        limit: Maximum number of results to return (default 10, max 100).
        year_start: Filter papers published after this year (optional).
        year_end: Filter papers published before this year (optional).
        fields_of_study: Filter by fields like "Computer Science", "Economics" (optional).
        
    Returns:
        Dictionary with search results including titles, abstracts, citations, and URLs.
        
    Example:
        >>> semantic_scholar_search("machine learning finance", limit=5)
    """
    try:
        # Build query parameters
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "paperId,title,abstract,year,authors,citationCount,url,venue,publicationDate,fieldsOfStudy,externalIds",
        }
        
        # Add year filter if specified
        if year_start or year_end:
            year_filter = ""
            if year_start:
                year_filter += f"{year_start}-"
            else:
                year_filter += "1900-"
            if year_end:
                year_filter += str(year_end)
            else:
                year_filter += str(datetime.utcnow().year)
            params["year"] = year_filter
        
        # Add fields of study filter
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        # Make API request with retry for rate limits
        with httpx.Client(timeout=30.0) as client:
            response = _make_request_with_retry(
                client,
                f"{SEMANTIC_SCHOLAR_API}/paper/search",
                params,
            )
            data = response.json()
        
        # Parse results
        results = []
        for paper in data.get("data", []):
            # Extract DOI if available
            doi = None
            external_ids = paper.get("externalIds", {})
            if external_ids:
                doi = external_ids.get("DOI")
            
            # Build author list
            authors = [
                author.get("name", "Unknown")
                for author in paper.get("authors", [])[:5]  # Limit to 5 authors
            ]
            
            # Parse publication date
            pub_date = None
            if paper.get("publicationDate"):
                try:
                    pub_date = paper["publicationDate"]
                except (ValueError, TypeError):
                    # Ignore invalid date formats and leave pub_date as None
                    pass
            
            results.append({
                "paper_id": paper.get("paperId"),
                "title": paper.get("title", "Untitled"),
                "abstract": paper.get("abstract", ""),
                "year": paper.get("year"),
                "authors": authors,
                "citation_count": paper.get("citationCount", 0),
                "url": paper.get("url", f"https://www.semanticscholar.org/paper/{paper.get('paperId')}"),
                "venue": paper.get("venue", ""),
                "doi": doi,
                "publication_date": pub_date,
                "fields_of_study": paper.get("fieldsOfStudy", []),
                "source": "semantic_scholar",
            })
        
        return {
            "query": query,
            "total_results": data.get("total", len(results)),
            "results": results,
            "source": "semantic_scholar",
        }
        
    except httpx.HTTPStatusError as e:
        return {
            "error": f"Semantic Scholar API error: {e.response.status_code}",
            "query": query,
            "results": [],
            "source": "semantic_scholar",
        }
    except httpx.RequestError as e:
        return {
            "error": f"Request failed: {str(e)}",
            "query": query,
            "results": [],
            "source": "semantic_scholar",
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "query": query,
            "results": [],
            "source": "semantic_scholar",
        }


# =============================================================================
# arXiv Search
# =============================================================================

ARXIV_API = "https://export.arxiv.org/api/query"


@tool
def arxiv_search(
    query: str,
    limit: int = 10,
    sort_by: str = "relevance",
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """
    Search arXiv for preprints and working papers.
    
    arXiv hosts preprints in physics, mathematics, computer science, 
    quantitative biology, quantitative finance, statistics, and more.
    Good for finding cutting-edge research before peer review.
    
    Args:
        query: Search query text.
        limit: Maximum number of results (default 10, max 100).
        sort_by: Sort order - "relevance", "lastUpdatedDate", or "submittedDate".
        categories: arXiv categories to filter (e.g., ["cs.AI", "q-fin.ST"]).
        
    Returns:
        Dictionary with search results including titles, abstracts, and arXiv IDs.
        
    Example:
        >>> arxiv_search("transformer attention mechanism", limit=5, categories=["cs.LG"])
    """
    try:
        # Build search query
        search_query = query
        if categories:
            cat_filter = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({query}) AND ({cat_filter})"
        
        # Sort mapping
        sort_map = {
            "relevance": "relevance",
            "lastUpdatedDate": "lastUpdatedDate",
            "submittedDate": "submittedDate",
        }
        sort_order = sort_map.get(sort_by, "relevance")
        
        # Build parameters
        params = {
            "search_query": f"all:{search_query}",
            "start": 0,
            "max_results": min(limit, 100),
            "sortBy": sort_order,
            "sortOrder": "descending",
        }
        
        # Make API request (follow redirects for arXiv)
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(ARXIV_API, params=params)
            response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.text)
        
        # Define namespaces
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        
        results = []
        for entry in root.findall("atom:entry", ns):
            # Extract arXiv ID from the id URL
            arxiv_id_elem = entry.find("atom:id", ns)
            arxiv_url = arxiv_id_elem.text if arxiv_id_elem is not None else ""
            arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else ""
            
            # Get title
            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else "Untitled"
            
            # Get abstract
            summary_elem = entry.find("atom:summary", ns)
            abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""
            
            # Get authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name_elem = author.find("atom:name", ns)
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            # Get publication date
            published_elem = entry.find("atom:published", ns)
            pub_date = published_elem.text[:10] if published_elem is not None else None
            
            # Get categories
            categories_found = []
            for category in entry.findall("arxiv:primary_category", ns):
                cat_term = category.get("term")
                if cat_term:
                    categories_found.append(cat_term)
            for category in entry.findall("atom:category", ns):
                cat_term = category.get("term")
                if cat_term and cat_term not in categories_found:
                    categories_found.append(cat_term)
            
            # Get PDF link
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""
            
            results.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract[:1000] + "..." if len(abstract) > 1000 else abstract,
                "authors": authors[:5],  # Limit to 5 authors
                "publication_date": pub_date,
                "url": arxiv_url,
                "pdf_url": pdf_url,
                "categories": categories_found,
                "source": "arxiv",
            })
        
        return {
            "query": query,
            "total_results": len(results),
            "results": results,
            "source": "arxiv",
        }
        
    except httpx.HTTPStatusError as e:
        return {
            "error": f"arXiv API error: {e.response.status_code}",
            "query": query,
            "results": [],
            "source": "arxiv",
        }
    except httpx.RequestError as e:
        return {
            "error": f"Request failed: {str(e)}",
            "query": query,
            "results": [],
            "source": "arxiv",
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "query": query,
            "results": [],
            "source": "arxiv",
        }


# =============================================================================
# Tavily Academic Search (via general Tavily with academic focus)
# =============================================================================

@tool
def tavily_academic_search(
    query: str,
    limit: int = 10,
    include_domains: list[str] | None = None,
) -> dict[str, Any]:
    """
    Search for academic content using Tavily with academic domain focus.
    
    Uses Tavily's search API with domain filtering to target academic sources
    like Google Scholar, university sites, and academic publishers.
    
    Args:
        query: Search query text.
        limit: Maximum number of results (default 10).
        include_domains: Domains to include (defaults to academic domains).
        
    Returns:
        Dictionary with search results from academic web sources.
        
    Example:
        >>> tavily_academic_search("corporate governance board diversity")
    """
    try:
        from langchain_tavily import TavilySearch
        from src.config import settings
        
        # Default to academic domains if not specified
        if include_domains is None:
            include_domains = [
                "scholar.google.com",
                "arxiv.org",
                "ssrn.com",
                "researchgate.net",
                "academia.edu",
                "jstor.org",
                "sciencedirect.com",
                "springer.com",
                "wiley.com",
                "nature.com",
                "ncbi.nlm.nih.gov",
                "ieee.org",
                "acm.org",
            ]
        
        # Initialize Tavily search
        tavily = TavilySearch(
            api_key=settings.tavily_api_key,
            max_results=min(limit, 20),
            include_domains=include_domains,
        )
        
        # Execute search
        response = tavily.invoke(query)
        
        # Parse results based on response type
        results = []
        if isinstance(response, str):
            # Simple string response
            results.append({
                "title": "Search Results",
                "url": "",
                "snippet": response,
                "source": "tavily_academic",
            })
        elif isinstance(response, list):
            # List of results
            for item in response:
                if isinstance(item, dict):
                    results.append({
                        "title": item.get("title", "Untitled"),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", item.get("snippet", "")),
                        "source": "tavily_academic",
                    })
        elif isinstance(response, dict):
            # Single dict response
            for item in response.get("results", [response]):
                results.append({
                    "title": item.get("title", "Untitled"),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", item.get("snippet", "")),
                    "source": "tavily_academic",
                })
        
        return {
            "query": query,
            "total_results": len(results),
            "results": results,
            "source": "tavily_academic",
        }
        
    except ImportError:
        return {
            "error": "Tavily not configured. Install langchain-tavily and set TAVILY_API_KEY.",
            "query": query,
            "results": [],
            "source": "tavily_academic",
        }
    except Exception as e:
        return {
            "error": f"Tavily search error: {str(e)}",
            "query": query,
            "results": [],
            "source": "tavily_academic",
        }


# =============================================================================
# Utility Functions
# =============================================================================


def convert_to_search_result(
    raw_result: dict[str, Any],
    query_id: str,
) -> SearchResult:
    """
    Convert a raw search result to a SearchResult model.
    
    Args:
        raw_result: Raw result dict from any search tool.
        query_id: ID of the query that produced this result.
        
    Returns:
        SearchResult model instance.
    """
    from datetime import date as date_type
    
    # Parse publication date if present
    pub_date = None
    date_str = raw_result.get("publication_date") or raw_result.get("pub_date")
    if date_str:
        try:
            if isinstance(date_str, str):
                pub_date = date_type.fromisoformat(date_str[:10])
            elif isinstance(date_str, date_type):
                pub_date = date_str
        except (ValueError, TypeError):
            # Ignore invalid or malformed publication dates and leave pub_date as None
            pass
    
    return SearchResult(
        query_id=query_id,
        title=raw_result.get("title", "Untitled"),
        url=raw_result.get("url", ""),
        snippet=raw_result.get("abstract") or raw_result.get("snippet", ""),
        source_type=raw_result.get("source", "unknown"),
        published_date=pub_date,
        authors=raw_result.get("authors", []),
        citation_count=raw_result.get("citation_count"),
        venue=raw_result.get("venue"),
        doi=raw_result.get("doi"),
        relevance_score=raw_result.get("relevance_score", 0.5),
    )


def merge_search_results(
    results_lists: list[list[dict[str, Any]]],
    deduplicate_by: str = "title",
) -> list[dict[str, Any]]:
    """
    Merge and deduplicate results from multiple search sources.
    
    Args:
        results_lists: List of result lists from different sources.
        deduplicate_by: Field to use for deduplication (title, url, or doi).
        
    Returns:
        Merged and deduplicated list of results.
    """
    seen = set()
    merged = []
    
    for results in results_lists:
        for result in results:
            # Get deduplication key
            key = result.get(deduplicate_by, "").lower().strip()
            
            # Skip if we've seen this before
            if key and key in seen:
                continue
            
            seen.add(key)
            merged.append(result)
    
    return merged


def rank_by_citations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Rank search results by citation count (highest first).
    
    Args:
        results: List of search results.
        
    Returns:
        Results sorted by citation count (descending).
    """
    return sorted(
        results,
        key=lambda x: x.get("citation_count", 0) or 0,
        reverse=True,
    )


def identify_seminal_works(
    results: list[dict[str, Any]],
    citation_threshold: int = 100,
) -> list[dict[str, Any]]:
    """
    Identify seminal works based on citation count.
    
    Args:
        results: List of search results with citation counts.
        citation_threshold: Minimum citations to be considered seminal.
        
    Returns:
        List of results meeting the citation threshold.
    """
    return [
        r for r in results
        if (r.get("citation_count") or 0) >= citation_threshold
    ]


# Export tools for use in agents
ACADEMIC_SEARCH_TOOLS = [
    semantic_scholar_search,
    arxiv_search,
    tavily_academic_search,
]
