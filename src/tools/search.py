"""Search tools using Tavily API."""

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from src.config import settings


# Initialize Tavily search tool
web_search_tool = TavilySearch(
    max_results=5,
    tavily_api_key=settings.tavily_api_key,
)


@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily API.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default: 5).

    Returns:
        Search results as a formatted string.
    """
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(query=query, max_results=max_results)

    results = []
    for result in response.get("results", []):
        results.append(
            f"Title: {result.get('title', 'N/A')}\n"
            f"URL: {result.get('url', 'N/A')}\n"
            f"Content: {result.get('content', 'N/A')}\n"
        )

    return "\n---\n".join(results) if results else "No results found."
