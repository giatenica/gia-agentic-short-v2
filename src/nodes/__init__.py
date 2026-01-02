"""LangGraph nodes for GIA Agentic v2 workflow."""

from src.nodes.intake import intake_node, parse_intake_form, validate_intake
from src.nodes.data_explorer import data_explorer_node
from src.nodes.literature_reviewer import (
    literature_reviewer_node,
    generate_search_queries,
    execute_searches_sync,
    process_search_results,
    extract_methodology_precedents,
    create_literature_review_subgraph,
)
from src.nodes.literature_synthesizer import (
    literature_synthesizer_node,
    extract_themes,
    identify_gaps,
    synthesize_literature,
    generate_contribution_statement,
    refine_research_question,
)

__all__ = [
    # Intake node
    "intake_node",
    "parse_intake_form",
    "validate_intake",
    # Data explorer node
    "data_explorer_node",
    # Literature reviewer node
    "literature_reviewer_node",
    "generate_search_queries",
    "execute_searches_sync",
    "process_search_results",
    "extract_methodology_precedents",
    "create_literature_review_subgraph",
    # Literature synthesizer node
    "literature_synthesizer_node",
    "extract_themes",
    "identify_gaps",
    "synthesize_literature",
    "generate_contribution_statement",
    "refine_research_question",
]
