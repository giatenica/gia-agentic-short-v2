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
from src.nodes.planner import (
    planner_node,
    route_after_planner,
)
from src.nodes.data_analyst import (
    data_analyst_node,
    route_after_data_analyst,
)
from src.nodes.conceptual_synthesizer import (
    conceptual_synthesizer_node,
    route_after_conceptual_synthesizer,
)
from src.nodes.writer import (
    writer_node,
    should_continue_writing,
    get_section_writer,
    build_section_context,
    SECTION_ORDER,
)
from src.nodes.reviewer import (
    reviewer_node,
    route_after_reviewer,
)
from src.nodes.fallback import (
    fallback_node,
    should_fallback,
    route_to_fallback_or_continue,
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
    # Planner node
    "planner_node",
    "route_after_planner",
    # Data analyst node (Sprint 5)
    "data_analyst_node",
    "route_after_data_analyst",
    # Conceptual synthesizer node (Sprint 5)
    "conceptual_synthesizer_node",
    "route_after_conceptual_synthesizer",
    # Writer node (Sprint 6)
    "writer_node",
    "should_continue_writing",
    "get_section_writer",
    "build_section_context",
    "SECTION_ORDER",
    # Reviewer node (Sprint 7)
    "reviewer_node",
    "route_after_reviewer",
    # Fallback node (Sprint 9)
    "fallback_node",
    "should_fallback",
    "route_to_fallback_or_continue",
]
