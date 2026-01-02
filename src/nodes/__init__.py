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
    identify_gaps as synthesizer_identify_gaps,
    synthesize_literature,
    generate_contribution_statement as synthesizer_generate_contribution,
    refine_research_question as synthesizer_refine_question,
)
from src.nodes.gap_identifier import (
    gap_identifier_node,
    identify_gaps,
    select_primary_gap,
    create_refined_question,
    create_contribution_statement,
    prepare_approval_request,
    process_approval_response,
    should_refine_further,
    route_after_gap_identifier,
    get_gap_identifier_tools,
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
    "synthesizer_identify_gaps",
    "synthesize_literature",
    "synthesizer_generate_contribution",
    "synthesizer_refine_question",
    # Gap identifier node
    "gap_identifier_node",
    "identify_gaps",
    "select_primary_gap",
    "create_refined_question",
    "create_contribution_statement",
    "prepare_approval_request",
    "process_approval_response",
    "should_refine_further",
    "route_after_gap_identifier",
    "get_gap_identifier_tools",
]
