"""Tools module for agent capabilities."""

from src.tools.search import tavily_search, web_search_tool
from src.tools.basic import get_current_time, calculate
from src.tools.data_exploration import (
    parse_csv_file,
    parse_excel_file,
    detect_schema,
    generate_summary_stats,
    detect_missing_values,
    detect_outliers,
    assess_data_quality,
    DATA_EXPLORATION_TOOLS,
)
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
    build_citation_network,
    find_common_citations,
    calculate_citation_metrics,
    CITATION_ANALYSIS_TOOLS,
)

__all__ = [
    # Search tools
    "tavily_search",
    "web_search_tool",
    # Basic tools
    "get_current_time",
    "calculate",
    # Data exploration tools
    "parse_csv_file",
    "parse_excel_file",
    "detect_schema",
    "generate_summary_stats",
    "detect_missing_values",
    "detect_outliers",
    "assess_data_quality",
    "DATA_EXPLORATION_TOOLS",
    # Academic search tools
    "semantic_scholar_search",
    "arxiv_search",
    "tavily_academic_search",
    "convert_to_search_result",
    "merge_search_results",
    "rank_by_citations",
    "identify_seminal_works",
    "ACADEMIC_SEARCH_TOOLS",
    # Citation analysis tools
    "get_citing_papers",
    "get_references",
    "get_paper_details",
    "get_author_papers",
    "build_citation_network",
    "find_common_citations",
    "calculate_citation_metrics",
    "CITATION_ANALYSIS_TOOLS",
]
