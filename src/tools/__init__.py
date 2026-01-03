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
from src.tools.gap_analysis import (
    compare_coverage,
    identify_methodological_gaps,
    identify_empirical_gaps,
    identify_theoretical_gaps,
    assess_gap_significance,
    perform_gap_analysis,
    compare_coverage_tool,
    identify_gaps_tool,
)
from src.tools.contribution import (
    generate_contribution_statement,
    position_in_literature,
    differentiate_from_prior,
    refine_research_question,
    generate_contribution_tool,
    refine_question_tool,
)
from src.tools.analysis import (
    execute_descriptive_stats,
    compute_correlation_matrix,
    run_ttest,
    run_anova,
    run_chi_square,
    run_mann_whitney,
    run_kruskal_wallis,
    run_ols_regression,
    run_logistic_regression,
    generate_finding,
    assess_gap_coverage,
    execute_robustness_check,
    # Backward compatibility
    execute_hypothesis_test,
    execute_regression_analysis,
    get_analysis_tools,
)
from src.tools.synthesis import (
    extract_key_concepts,
    define_concept,
    map_concept_relationships,
    define_relationship,
    generate_propositions,
    define_proposition,
    build_conceptual_framework,
    ground_in_theory,
    assess_theoretical_contribution,
    get_synthesis_tools,
)
from src.tools.llm_interpretation import (
    interpret_regression,
    interpret_hypothesis_test,
    summarize_findings,
    generate_methods_section,
    get_interpretation_tools,
    LLM_INTERPRETATION_TOOLS,
)
from src.tools.data_loading import (
    load_data,
    query_data,
    list_datasets,
    get_dataset_info,
    sample_data,
    get_data_loading_tools,
    DATA_LOADING_TOOLS,
)
from src.tools.data_profiling import (
    profile_dataset,
    describe_dataset,
    describe_variable,
    get_profiling_tools,
    DATA_PROFILING_TOOLS,
)
from src.tools.data_transformation import (
    filter_data,
    select_columns,
    aggregate_data,
    merge_datasets,
    create_variable,
    handle_missing,
    encode_categorical,
    pivot_data,
    melt_data,
    get_transformation_tools,
    DATA_TRANSFORMATION_TOOLS,
)

# Tool collections
GAP_ANALYSIS_TOOLS = [
    compare_coverage_tool,
    identify_gaps_tool,
]

CONTRIBUTION_TOOLS = [
    generate_contribution_tool,
    refine_question_tool,
]

ANALYSIS_TOOLS = get_analysis_tools()
SYNTHESIS_TOOLS = get_synthesis_tools()

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
    # Gap analysis tools
    "compare_coverage",
    "identify_methodological_gaps",
    "identify_empirical_gaps",
    "identify_theoretical_gaps",
    "assess_gap_significance",
    "perform_gap_analysis",
    "compare_coverage_tool",
    "identify_gaps_tool",
    "GAP_ANALYSIS_TOOLS",
    # Contribution tools
    "generate_contribution_statement",
    "position_in_literature",
    "differentiate_from_prior",
    "refine_research_question",
    "generate_contribution_tool",
    "refine_question_tool",
    "CONTRIBUTION_TOOLS",
    # Analysis tools (Sprint 5) - real implementations with scipy/statsmodels
    "execute_descriptive_stats",
    "compute_correlation_matrix",
    "run_ttest",
    "run_anova",
    "run_chi_square",
    "run_mann_whitney",
    "run_kruskal_wallis",
    "run_ols_regression",
    "run_logistic_regression",
    "generate_finding",
    "assess_gap_coverage",
    "execute_robustness_check",
    # Backward compatibility wrappers
    "execute_hypothesis_test",
    "execute_regression_analysis",
    "get_analysis_tools",
    "ANALYSIS_TOOLS",
    # Synthesis tools (Sprint 5)
    "extract_key_concepts",
    "define_concept",
    "map_concept_relationships",
    "define_relationship",
    "generate_propositions",
    "define_proposition",
    "build_conceptual_framework",
    "ground_in_theory",
    "assess_theoretical_contribution",
    "get_synthesis_tools",
    "SYNTHESIS_TOOLS",
    # LLM interpretation tools
    "interpret_regression",
    "interpret_hypothesis_test",
    "summarize_findings",
    "generate_methods_section",
    "get_interpretation_tools",
    "LLM_INTERPRETATION_TOOLS",
    # Data loading tools
    "load_data",
    "query_data",
    "list_datasets",
    "get_dataset_info",
    "sample_data",
    "get_data_loading_tools",
    "DATA_LOADING_TOOLS",
    # Data profiling tools
    "profile_dataset",
    "describe_dataset",
    "describe_variable",
    "get_profiling_tools",
    "DATA_PROFILING_TOOLS",
    # Data transformation tools
    "filter_data",
    "select_columns",
    "aggregate_data",
    "merge_datasets",
    "create_variable",
    "handle_missing",
    "encode_categorical",
    "pivot_data",
    "melt_data",
    "get_transformation_tools",
    "DATA_TRANSFORMATION_TOOLS",
]
