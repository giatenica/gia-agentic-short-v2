"""Citation analysis tools for literature review.

These tools analyze citation relationships between papers using
the Semantic Scholar API. They help identify influential papers,
understand research lineage, and map citation networks.
"""

import httpx
from typing import Any

from langchain_core.tools import tool


SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"


# =============================================================================
# Citation Graph Tools
# =============================================================================


@tool
def get_citing_papers(
    paper_id: str,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Get papers that cite a given paper (forward citations).
    
    Retrieves the list of papers that have cited the specified paper,
    helping to understand its impact and how the research has been built upon.
    
    Args:
        paper_id: Semantic Scholar paper ID or DOI (e.g., "10.1000/xyz123").
        limit: Maximum number of citing papers to return (default 20, max 100).
        
    Returns:
        Dictionary with citing papers including their titles, authors, and citation counts.
        
    Example:
        >>> get_citing_papers("649def34f8be52c8b66281af98ae884c09aef38b")
    """
    try:
        params = {
            "fields": "paperId,title,abstract,year,authors,citationCount,url,venue",
            "limit": min(limit, 100),
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/citations",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("data", []):
            citing_paper = item.get("citingPaper", {})
            if citing_paper:
                authors = [
                    author.get("name", "Unknown")
                    for author in citing_paper.get("authors", [])[:5]
                ]
                results.append({
                    "paper_id": citing_paper.get("paperId"),
                    "title": citing_paper.get("title", "Untitled"),
                    "abstract": citing_paper.get("abstract", "")[:500],
                    "year": citing_paper.get("year"),
                    "authors": authors,
                    "citation_count": citing_paper.get("citationCount", 0),
                    "url": citing_paper.get("url", ""),
                    "venue": citing_paper.get("venue", ""),
                })
        
        return {
            "source_paper_id": paper_id,
            "total_citations": len(results),
            "citing_papers": results,
        }
        
    except httpx.HTTPStatusError as e:
        return {
            "error": f"API error: {e.response.status_code}",
            "source_paper_id": paper_id,
            "citing_papers": [],
        }
    except httpx.RequestError as e:
        return {
            "error": f"Request failed: {str(e)}",
            "source_paper_id": paper_id,
            "citing_papers": [],
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "source_paper_id": paper_id,
            "citing_papers": [],
        }


@tool
def get_references(
    paper_id: str,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Get papers referenced by a given paper (backward citations).
    
    Retrieves the list of papers that the specified paper cites,
    helping to understand its research foundation and related work.
    
    Args:
        paper_id: Semantic Scholar paper ID or DOI.
        limit: Maximum number of references to return (default 20, max 100).
        
    Returns:
        Dictionary with referenced papers including their titles, authors, and citation counts.
        
    Example:
        >>> get_references("649def34f8be52c8b66281af98ae884c09aef38b")
    """
    try:
        params = {
            "fields": "paperId,title,abstract,year,authors,citationCount,url,venue",
            "limit": min(limit, 100),
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/references",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("data", []):
            cited_paper = item.get("citedPaper", {})
            if cited_paper:
                authors = [
                    author.get("name", "Unknown")
                    for author in cited_paper.get("authors", [])[:5]
                ]
                results.append({
                    "paper_id": cited_paper.get("paperId"),
                    "title": cited_paper.get("title", "Untitled"),
                    "abstract": cited_paper.get("abstract", "")[:500] if cited_paper.get("abstract") else "",
                    "year": cited_paper.get("year"),
                    "authors": authors,
                    "citation_count": cited_paper.get("citationCount", 0),
                    "url": cited_paper.get("url", ""),
                    "venue": cited_paper.get("venue", ""),
                })
        
        return {
            "source_paper_id": paper_id,
            "total_references": len(results),
            "references": results,
        }
        
    except httpx.HTTPStatusError as e:
        return {
            "error": f"API error: {e.response.status_code}",
            "source_paper_id": paper_id,
            "references": [],
        }
    except httpx.RequestError as e:
        return {
            "error": f"Request failed: {str(e)}",
            "source_paper_id": paper_id,
            "references": [],
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "source_paper_id": paper_id,
            "references": [],
        }


@tool
def get_paper_details(
    paper_id: str,
) -> dict[str, Any]:
    """
    Get detailed information about a specific paper.
    
    Retrieves comprehensive metadata for a paper including abstract,
    authors, citation count, venue, and external identifiers.
    
    Args:
        paper_id: Semantic Scholar paper ID, DOI, or arXiv ID.
                  For DOI use format "DOI:10.xxx/yyy"
                  For arXiv use format "ARXIV:2301.xxxxx"
        
    Returns:
        Dictionary with detailed paper information.
        
    Example:
        >>> get_paper_details("DOI:10.18653/v1/N18-1202")
    """
    try:
        fields = [
            "paperId",
            "title",
            "abstract",
            "year",
            "authors",
            "citationCount",
            "referenceCount",
            "url",
            "venue",
            "publicationDate",
            "fieldsOfStudy",
            "externalIds",
            "tldr",
            "isOpenAccess",
            "openAccessPdf",
        ]
        
        params = {"fields": ",".join(fields)}
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}",
                params=params,
            )
            response.raise_for_status()
            paper = response.json()
        
        # Extract external IDs
        external_ids = paper.get("externalIds", {})
        
        # Extract authors
        authors = [
            {
                "name": author.get("name", "Unknown"),
                "author_id": author.get("authorId"),
            }
            for author in paper.get("authors", [])
        ]
        
        # Extract TLDR if available
        tldr = None
        if paper.get("tldr"):
            tldr = paper["tldr"].get("text")
        
        # Extract open access PDF
        pdf_url = None
        if paper.get("openAccessPdf"):
            pdf_url = paper["openAccessPdf"].get("url")
        
        return {
            "paper_id": paper.get("paperId"),
            "title": paper.get("title", "Untitled"),
            "abstract": paper.get("abstract", ""),
            "year": paper.get("year"),
            "authors": authors,
            "citation_count": paper.get("citationCount", 0),
            "reference_count": paper.get("referenceCount", 0),
            "url": paper.get("url", ""),
            "venue": paper.get("venue", ""),
            "publication_date": paper.get("publicationDate"),
            "fields_of_study": paper.get("fieldsOfStudy", []),
            "doi": external_ids.get("DOI"),
            "arxiv_id": external_ids.get("ArXiv"),
            "tldr": tldr,
            "is_open_access": paper.get("isOpenAccess", False),
            "pdf_url": pdf_url,
        }
        
    except httpx.HTTPStatusError as e:
        return {
            "error": f"API error: {e.response.status_code}",
            "paper_id": paper_id,
        }
    except httpx.RequestError as e:
        return {
            "error": f"Request failed: {str(e)}",
            "paper_id": paper_id,
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "paper_id": paper_id,
        }


@tool
def get_author_papers(
    author_id: str,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Get papers by a specific author.
    
    Retrieves the publication history of an author, useful for
    following key researchers in a field.
    
    Args:
        author_id: Semantic Scholar author ID.
        limit: Maximum number of papers to return (default 20, max 100).
        
    Returns:
        Dictionary with author information and their papers.
        
    Example:
        >>> get_author_papers("1741101")  # Yoshua Bengio's author ID
    """
    try:
        # First get author info
        author_fields = "authorId,name,affiliations,paperCount,citationCount,hIndex"
        
        with httpx.Client(timeout=30.0) as client:
            # Get author details
            author_response = client.get(
                f"{SEMANTIC_SCHOLAR_API}/author/{author_id}",
                params={"fields": author_fields},
            )
            author_response.raise_for_status()
            author_data = author_response.json()
            
            # Get author's papers
            paper_fields = "paperId,title,year,citationCount,venue,url"
            papers_response = client.get(
                f"{SEMANTIC_SCHOLAR_API}/author/{author_id}/papers",
                params={"fields": paper_fields, "limit": min(limit, 100)},
            )
            papers_response.raise_for_status()
            papers_data = papers_response.json()
        
        papers = []
        for item in papers_data.get("data", []):
            papers.append({
                "paper_id": item.get("paperId"),
                "title": item.get("title", "Untitled"),
                "year": item.get("year"),
                "citation_count": item.get("citationCount", 0),
                "venue": item.get("venue", ""),
                "url": item.get("url", ""),
            })
        
        return {
            "author_id": author_id,
            "name": author_data.get("name", "Unknown"),
            "affiliations": author_data.get("affiliations", []),
            "paper_count": author_data.get("paperCount", 0),
            "citation_count": author_data.get("citationCount", 0),
            "h_index": author_data.get("hIndex"),
            "papers": papers,
        }
        
    except httpx.HTTPStatusError as e:
        return {
            "error": f"API error: {e.response.status_code}",
            "author_id": author_id,
            "papers": [],
        }
    except httpx.RequestError as e:
        return {
            "error": f"Request failed: {str(e)}",
            "author_id": author_id,
            "papers": [],
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "author_id": author_id,
            "papers": [],
        }


# =============================================================================
# Citation Analysis Functions
# =============================================================================


def build_citation_network(
    seed_paper_ids: list[str],
    depth: int = 1,
    max_papers: int = 50,
) -> dict[str, Any]:
    """
    Build a citation network starting from seed papers.
    
    Creates a graph of papers connected by citation relationships,
    useful for understanding the research landscape.
    
    Note: This function makes multiple API calls in sequence. For large
    networks, consider implementing rate limiting or exponential backoff
    to avoid API rate limits.
    
    Args:
        seed_paper_ids: List of paper IDs to start from.
        depth: How many levels of citations to follow (1 = direct only).
        max_papers: Maximum total papers to include in the network.
        
    Returns:
        Dictionary with nodes (papers) and edges (citations).
    """
    nodes = {}  # paper_id -> paper info
    edges = []  # list of (citing_id, cited_id)
    
    papers_to_process = list(seed_paper_ids)
    current_depth = 0
    
    while papers_to_process and len(nodes) < max_papers and current_depth < depth:
        next_level = []
        
        for paper_id in papers_to_process:
            if paper_id in nodes:
                continue
            
            # Get paper details
            details_result = get_paper_details.invoke(paper_id)
            if "error" not in details_result:
                nodes[paper_id] = {
                    "id": paper_id,
                    "title": details_result.get("title"),
                    "year": details_result.get("year"),
                    "citation_count": details_result.get("citation_count", 0),
                }
            
            # Get references (backward citations)
            refs_result = get_references.invoke({"paper_id": paper_id, "limit": 10})
            for ref in refs_result.get("references", []):
                ref_id = ref.get("paper_id")
                if ref_id:
                    edges.append((paper_id, ref_id))
                    if ref_id not in nodes and len(nodes) < max_papers:
                        next_level.append(ref_id)
            
            if len(nodes) >= max_papers:
                break
        
        papers_to_process = next_level
        current_depth += 1
    
    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }


def find_common_citations(
    paper_ids: list[str],
    min_overlap: int = 2,
) -> list[dict[str, Any]]:
    """
    Find papers that are cited by multiple papers in a list.
    
    Helps identify foundational works that multiple papers build upon.
    
    Note: This function makes multiple API calls in sequence. For large
    paper lists, consider implementing rate limiting or exponential backoff
    to avoid API rate limits.
    
    Args:
        paper_ids: List of paper IDs to analyze.
        min_overlap: Minimum number of papers that must cite a reference.
        
    Returns:
        List of commonly cited papers sorted by frequency.
    """
    citation_counts: dict[str, dict[str, Any]] = {}
    
    for paper_id in paper_ids:
        refs_result = get_references.invoke({"paper_id": paper_id, "limit": 50})
        for ref in refs_result.get("references", []):
            ref_id = ref.get("paper_id")
            if ref_id:
                if ref_id not in citation_counts:
                    citation_counts[ref_id] = {
                        "paper_id": ref_id,
                        "title": ref.get("title"),
                        "year": ref.get("year"),
                        "citation_count": ref.get("citation_count", 0),
                        "cited_by_count": 0,
                    }
                citation_counts[ref_id]["cited_by_count"] += 1
    
    # Filter and sort by overlap
    common = [
        info for info in citation_counts.values()
        if info["cited_by_count"] >= min_overlap
    ]
    
    return sorted(common, key=lambda x: x["cited_by_count"], reverse=True)


def calculate_citation_metrics(
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate citation metrics for a collection of papers.
    
    Args:
        papers: List of paper dictionaries with citation_count field.
        
    Returns:
        Dictionary with metrics like total citations, average, h-index equivalent.
    """
    if not papers:
        return {
            "total_papers": 0,
            "total_citations": 0,
            "average_citations": 0,
            "median_citations": 0,
            "max_citations": 0,
            "highly_cited_papers": 0,
        }
    
    citation_counts = [p.get("citation_count", 0) or 0 for p in papers]
    citation_counts.sort(reverse=True)
    
    total = sum(citation_counts)
    n = len(citation_counts)
    median_idx = n // 2
    
    # Count papers with 10+ citations
    highly_cited = sum(1 for c in citation_counts if c >= 10)
    
    return {
        "total_papers": n,
        "total_citations": total,
        "average_citations": round(total / n, 2) if n > 0 else 0,
        "median_citations": citation_counts[median_idx] if n > 0 else 0,
        "max_citations": citation_counts[0] if citation_counts else 0,
        "highly_cited_papers": highly_cited,
    }


# Export tools for use in agents
CITATION_ANALYSIS_TOOLS = [
    get_citing_papers,
    get_references,
    get_paper_details,
    get_author_papers,
]
