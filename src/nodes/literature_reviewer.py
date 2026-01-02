"""LITERATURE_REVIEWER node for systematic literature search.

This node:
1. Generates search queries from the research question
2. Searches multiple academic databases in parallel
3. Deduplicates and ranks results
4. Identifies seminal works and recent developments
5. Prepares literature for synthesis

Key LangGraph features used:
- Parallel search execution via async operations
- State updates with accumulated results
"""

import asyncio
from datetime import datetime
from typing import Any

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage

from src.config import settings
from src.state.enums import ResearchStatus
from src.state.models import SearchQuery, SearchResult, WorkflowError
from src.state.schema import WorkflowState
from src.tools.academic_search import (
    semantic_scholar_search,
    arxiv_search,
    tavily_academic_search,
    merge_search_results,
    rank_by_citations,
    convert_to_search_result,
)
from src.tools.citation_analysis import get_paper_details


# =============================================================================
# Query Generation
# =============================================================================


def generate_search_queries(
    research_question: str,
    key_variables: list[str],
    model_name: str | None = None,
) -> list[SearchQuery]:
    """
    Generate diverse search queries from a research question.
    
    Uses Claude to decompose the research question into multiple
    search queries targeting different aspects and databases.
    
    Args:
        research_question: The original research question.
        key_variables: Key variables or concepts to include.
        model_name: Model to use (defaults to settings.default_model).
        
    Returns:
        List of SearchQuery objects to execute.
    """
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2048,
        api_key=settings.anthropic_api_key,
    )
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    key_vars_str = ", ".join(key_variables) if key_variables else "not specified"
    
    prompt = f"""Current date: {current_date}

Research Question: {research_question}
Key Variables: {key_vars_str}

Generate 5-8 search queries to find relevant academic literature. Include:
1. The exact research question (for precise matches)
2. Broader theoretical queries (for foundational literature)
3. Methodological queries (for methodology papers)
4. Recent development queries (last 3 years)
5. Key author/seminal work queries if applicable

Format each query as:
QUERY: <search text>
TYPE: <academic|preprint|methodology|theory|recent>
PRIORITY: <1-5 where 1 is highest priority>

Focus on queries that would work well with Semantic Scholar and arXiv APIs.
Avoid overly long queries. Each query should be 3-10 words."""

    response = model.invoke([HumanMessage(content=prompt)])
    
    # Parse response into SearchQuery objects
    queries = []
    lines = response.content.split("\n")
    
    current_query = {}
    for line in lines:
        line = line.strip()
        if line.startswith("QUERY:"):
            if current_query.get("query_text"):
                queries.append(SearchQuery(**current_query))
            current_query = {"query_text": line[6:].strip()}
        elif line.startswith("TYPE:"):
            current_query["source_type"] = line[5:].strip().lower()
        elif line.startswith("PRIORITY:"):
            try:
                current_query["priority"] = int(line[9:].strip())
            except ValueError:
                current_query["priority"] = 3
    
    # Add last query
    if current_query.get("query_text"):
        queries.append(SearchQuery(**current_query))
    
    # Ensure we have at least the original question as a query
    if not queries:
        queries = [
            SearchQuery(
                query_text=research_question,
                source_type="academic",
                priority=1,
            )
        ]
    
    return queries


# =============================================================================
# Parallel Search Execution
# =============================================================================


async def execute_search_async(
    query: SearchQuery,
    search_type: str = "academic",
) -> dict[str, Any]:
    """
    Execute a single search query asynchronously.
    
    Args:
        query: The search query to execute.
        search_type: Type of search (semantic_scholar, arxiv, tavily).
        
    Returns:
        Search results dictionary.
    """
    try:
        # Run synchronous tool in thread pool
        loop = asyncio.get_event_loop()
        
        if search_type == "semantic_scholar":
            result = await loop.run_in_executor(
                None,
                lambda: semantic_scholar_search.invoke({"query": query.query_text, "limit": 15})
            )
        elif search_type == "arxiv":
            result = await loop.run_in_executor(
                None,
                lambda: arxiv_search.invoke({"query": query.query_text, "limit": 10})
            )
        else:  # tavily
            result = await loop.run_in_executor(
                None,
                lambda: tavily_academic_search.invoke({"query": query.query_text, "limit": 10})
            )
        
        return {
            "query_id": query.query_id,
            "search_type": search_type,
            "results": result.get("results", []),
            "error": result.get("error"),
        }
    except Exception as e:
        return {
            "query_id": query.query_id,
            "search_type": search_type,
            "results": [],
            "error": str(e),
        }


async def execute_all_searches(
    queries: list[SearchQuery],
) -> list[dict[str, Any]]:
    """
    Execute searches across all databases for all queries.
    
    Args:
        queries: List of search queries to execute.
        
    Returns:
        Combined list of all search results.
    """
    tasks = []
    
    # Limit to top 5 queries by priority
    sorted_queries = sorted(queries, key=lambda q: q.priority)[:5]
    
    for query in sorted_queries:
        # Search Semantic Scholar for all queries
        tasks.append(execute_search_async(query, "semantic_scholar"))
        
        # Search arXiv for preprint/recent queries
        if query.source_type in ["preprint", "recent", "methodology"]:
            tasks.append(execute_search_async(query, "arxiv"))
        
        # Search Tavily for broader coverage on theory queries
        if query.source_type in ["theory", "academic"]:
            tasks.append(execute_search_async(query, "tavily"))
    
    # Execute all searches in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and flatten
    valid_results = []
    for r in results:
        if isinstance(r, dict) and not r.get("error"):
            valid_results.extend(r.get("results", []))
    
    return valid_results


def execute_searches_sync(queries: list[SearchQuery]) -> list[dict[str, Any]]:
    """
    Synchronous wrapper for parallel search execution.
    
    Args:
        queries: List of search queries.
        
    Returns:
        Combined search results.
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If in async context, create new loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, execute_all_searches(queries))
                return future.result()
        else:
            return loop.run_until_complete(execute_all_searches(queries))
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(execute_all_searches(queries))


# =============================================================================
# Result Processing
# =============================================================================


def process_search_results(
    raw_results: list[dict[str, Any]],
    primary_query_id: str,
    min_relevance: float = 0.3,
) -> tuple[list[SearchResult], list[dict[str, Any]]]:
    """
    Process and filter raw search results.
    
    Args:
        raw_results: Raw results from search tools.
        primary_query_id: ID of the primary query for result association.
        min_relevance: Minimum relevance threshold (0-1).
        
    Returns:
        Tuple of (processed SearchResult list, seminal works list).
    """
    # Deduplicate by title
    unique_results = merge_search_results([raw_results])
    
    # Convert to SearchResult models
    processed = []
    for raw in unique_results:
        try:
            result = convert_to_search_result(raw, primary_query_id)
            # Basic relevance filter
            if result.relevance_score >= min_relevance or result.citation_count:
                processed.append(result)
        except Exception:
            # Skip invalid results
            continue
    
    # Rank by citations
    ranked_dicts = rank_by_citations([r.model_dump() for r in processed])
    
    # Identify seminal works (highly cited papers)
    seminal_works = [
        r for r in ranked_dicts 
        if (r.get("citation_count") or 0) >= 100
    ][:10]  # Top 10 seminal works
    
    # Convert back to SearchResult objects
    final_results = []
    for r in ranked_dicts[:50]:  # Keep top 50 results
        try:
            final_results.append(SearchResult(**r))
        except Exception:
            continue
    
    return final_results, seminal_works


def extract_methodology_precedents(
    results: list[SearchResult],
    max_precedents: int = 5,
) -> list[str]:
    """
    Extract methodology precedents from search results.
    
    Looks for papers that describe methodologies used in the research domain.
    
    Args:
        results: Processed search results.
        max_precedents: Maximum precedents to return.
        
    Returns:
        List of methodology precedent descriptions.
    """
    precedents = []
    
    for result in results:
        snippet_lower = result.snippet.lower()
        
        # Look for methodology indicators
        methodology_terms = [
            "methodology", "method", "approach", "framework",
            "technique", "analysis", "empirical", "quantitative",
            "qualitative", "regression", "panel data", "cross-sectional",
        ]
        
        if any(term in snippet_lower for term in methodology_terms):
            precedent = f"{result.title}"
            if result.venue:
                precedent += f" ({result.venue})"
            if result.citation_count:
                precedent += f" - {result.citation_count} citations"
            precedents.append(precedent)
            
            if len(precedents) >= max_precedents:
                break
    
    return precedents


# =============================================================================
# Main Node Function
# =============================================================================


def literature_reviewer_node(state: WorkflowState) -> dict[str, Any]:
    """
    LITERATURE_REVIEWER node: Systematic literature search.
    
    Searches multiple academic databases in parallel, deduplicates results,
    identifies seminal works, and prepares literature for synthesis.
    
    Args:
        state: Current workflow state with original_query.
        
    Returns:
        State updates with search_results, seed_literature, methodology_precedents.
    """
    original_query = state.get("original_query", "")
    key_variables = state.get("key_variables", [])
    
    if not original_query:
        return {
            "status": ResearchStatus.FAILED,
            "errors": [WorkflowError(
                node="literature_reviewer",
                category="validation",
                message="No research question provided",
                recoverable=False,
            )],
            "messages": [AIMessage(
                content="Cannot perform literature review without a research question."
            )],
        }
    
    try:
        # Step 1: Generate search queries
        queries = generate_search_queries(original_query, key_variables)
        
        # Step 2: Execute parallel searches
        raw_results = execute_searches_sync(queries)
        
        if not raw_results:
            return {
                "status": ResearchStatus.LITERATURE_REVIEW_COMPLETE,
                "search_results": [],
                "seed_literature": [],
                "methodology_precedents": [],
                "messages": [AIMessage(
                    content="Literature search completed but found no results. "
                    "This may indicate a very novel research area or network issues."
                )],
                "checkpoints": [
                    f"{datetime.utcnow().isoformat()}: Literature review - no results found"
                ],
            }
        
        # Step 3: Process results
        primary_query_id = queries[0].query_id if queries else "default"
        processed_results, seminal_works = process_search_results(
            raw_results, 
            primary_query_id
        )
        
        # Step 4: Extract methodology precedents
        methodology_precedents = extract_methodology_precedents(processed_results)
        
        # Step 5: Prepare seed literature (top seminal works)
        seed_literature = []
        for work in seminal_works[:5]:
            seed_lit = {
                "title": work.get("title"),
                "authors": work.get("authors", []),
                "year": work.get("year"),
                "citation_count": work.get("citation_count"),
                "url": work.get("url"),
                "relevance": "seminal work",
            }
            seed_literature.append(seed_lit)
        
        # Generate summary message
        message = generate_summary_message(
            total_results=len(processed_results),
            seminal_count=len(seminal_works),
            methodology_count=len(methodology_precedents),
            queries_executed=len(queries),
        )
        
        return {
            "status": ResearchStatus.LITERATURE_REVIEW_COMPLETE,
            "search_results": processed_results,
            "seed_literature": seed_literature,
            "methodology_precedents": methodology_precedents,
            "messages": [AIMessage(content=message)],
            "checkpoints": [
                f"{datetime.utcnow().isoformat()}: Literature review complete - "
                f"{len(processed_results)} papers found, {len(seminal_works)} seminal works"
            ],
            "updated_at": datetime.utcnow(),
        }
        
    except Exception as e:
        return {
            "status": ResearchStatus.FAILED,
            "errors": [WorkflowError(
                node="literature_reviewer",
                category="search",
                message=f"Literature search failed: {str(e)}",
                recoverable=True,
            )],
            "messages": [AIMessage(
                content=f"Literature review encountered an error: {str(e)}. "
                "The workflow will continue but may have limited literature context."
            )],
        }


def generate_summary_message(
    total_results: int,
    seminal_count: int,
    methodology_count: int,
    queries_executed: int,
) -> str:
    """Generate a human-readable summary of the literature review."""
    summary = f"""Literature Review Complete

Search Summary:
- Executed {queries_executed} search queries across academic databases
- Found {total_results} relevant papers (deduplicated)
- Identified {seminal_count} highly-cited seminal works
- Extracted {methodology_count} methodology precedents

The literature has been analyzed and is ready for synthesis. Key findings include:
- Seminal works that establish the theoretical foundation
- Recent papers (last 3 years) showing current developments
- Methodology papers providing research design precedents

Next step: Literature synthesis to identify themes and research gaps."""
    
    return summary


# =============================================================================
# Subgraph for Complex Literature Review (Optional Extension)
# =============================================================================


def create_literature_review_subgraph():
    """
    Create a subgraph for more complex literature review workflows.
    
    This subgraph can be used for deep literature reviews that need:
    - Citation network analysis
    - Multiple rounds of snowball searching
    - Full paper retrieval and summarization
    
    Returns:
        Compiled LangGraph subgraph.
    """
    from langgraph.graph import StateGraph, START, END
    from typing_extensions import TypedDict
    
    class LitReviewState(TypedDict):
        """State for literature review subgraph."""
        query: str
        queries: list[SearchQuery]
        raw_results: list[dict]
        processed_results: list[SearchResult]
        seminal_works: list[dict]
        iteration: int
        max_iterations: int
    
    def generate_queries_node(state: LitReviewState) -> dict:
        """Generate search queries."""
        queries = generate_search_queries(state["query"], [])
        return {"queries": queries}
    
    def search_node(state: LitReviewState) -> dict:
        """Execute searches."""
        results = execute_searches_sync(state["queries"])
        return {"raw_results": results}
    
    def process_node(state: LitReviewState) -> dict:
        """Process results."""
        query_id = state["queries"][0].query_id if state["queries"] else "default"
        processed, seminal = process_search_results(state["raw_results"], query_id)
        return {
            "processed_results": processed,
            "seminal_works": seminal,
            "iteration": state.get("iteration", 0) + 1,
        }
    
    def should_continue(state: LitReviewState) -> str:
        """Decide whether to do another iteration."""
        if state.get("iteration", 0) >= state.get("max_iterations", 1):
            return END
        if len(state.get("processed_results", [])) >= 30:
            return END
        return "generate_queries"
    
    # Build subgraph
    graph = StateGraph(LitReviewState)
    graph.add_node("generate_queries", generate_queries_node)
    graph.add_node("search", search_node)
    graph.add_node("process", process_node)
    
    graph.add_edge(START, "generate_queries")
    graph.add_edge("generate_queries", "search")
    graph.add_edge("search", "process")
    graph.add_conditional_edges("process", should_continue, ["generate_queries", END])
    
    return graph.compile()
