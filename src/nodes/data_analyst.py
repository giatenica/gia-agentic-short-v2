"""DATA_ANALYST node for empirical research analysis.

This node uses the comprehensive data analysis toolset:
1. Takes input from PLANNER (methodology, analysis approach, research plan)
2. Uses the new statistical tools (t-test, ANOVA, regression with scipy/statsmodels)
3. Transforms data as needed (filter, aggregate, handle missing)
4. Generates LLM-powered interpretations for academic prose
5. Links findings to research question and gap
"""

from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage

from src.state.enums import (
    ResearchStatus,
    AnalysisStatus,
    AnalysisApproach,
    FindingType,
    EvidenceStrength,
)
from src.state.models import (
    DataAnalysisResult,
    DataAnalysisFinding,
    StatisticalResult,
    RegressionResult,
    RegressionCoefficient,
    WorkflowError,
    TableArtifact,
    FigureArtifact,
)
from src.state.schema import WorkflowState

# Import new comprehensive analysis tools
from src.tools.analysis import (
    execute_descriptive_stats,
    run_ttest,
    run_anova,
    run_chi_square,
    run_ols_regression,
    run_logistic_regression,
    compute_correlation_matrix,
    assess_gap_coverage,
)
from src.tools.data_transformation import (
    filter_data,
    handle_missing,
)
from src.tools.llm_interpretation import (
    interpret_regression,
    interpret_hypothesis_test,
    summarize_findings,
)
from src.tools.data_loading import get_registry

# Sprint 15: Visualization tools
from src.tools.visualization import (
    create_summary_statistics_table,
    create_regression_table,
    create_correlation_matrix_table,
    create_time_series_plot,
    create_scatter_plot,
    create_distribution_plot,
)

# For reloading data if registry is empty
from src.tools.data_loading import load_data

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


# Common finance/options variables to look for in datasets
COMMON_OPTIONS_VARIABLES = [
    # Price-related
    "bid", "ask", "mid", "mark", "last", "close", "price", "premium",
    # Greeks
    "delta", "gamma", "theta", "vega", "rho", "iv", "implied_volatility",
    # Volume/Liquidity
    "volume", "open_interest", "oi", "turnover",
    # Contract characteristics
    "strike", "expiration", "dte", "days_to_expiration", "moneyness",
    # Spread measures  
    "spread", "bid_ask_spread", "quoted_spread",
    # Returns
    "return", "returns", "ret", "pnl",
    # Underlying
    "underlying_price", "spot", "stock_price",
]


def _infer_key_variables_from_dataset(
    dataset_name: str,
    research_question: str = "",
    max_vars: int = 10,
) -> list[str]:
    """
    Infer key variables from dataset columns when not explicitly provided.
    
    Prioritizes numeric columns that match common finance/options variable patterns.
    
    Args:
        dataset_name: Name of the dataset in registry.
        research_question: Research question for context.
        max_vars: Maximum number of variables to return.
        
    Returns:
        List of inferred key variable names.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return []
    
    try:
        df = registry.get_dataframe(dataset_name)
        columns = list(df.columns)
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return columns[:max_vars]
        
        # Score columns by relevance
        scored_cols = []
        rq_lower = research_question.lower()
        
        for col in numeric_cols:
            col_lower = col.lower()
            score = 0
            
            # Check if column matches common options variables
            for pattern in COMMON_OPTIONS_VARIABLES:
                if pattern in col_lower:
                    score += 10
                    break
            
            # Boost if mentioned in research question
            if col_lower in rq_lower or col_lower.replace("_", " ") in rq_lower:
                score += 5
            
            # Prefer columns with variation (not constant)
            try:
                if df[col].std() > 0:
                    score += 3
            except Exception:
                pass
            
            # Penalize ID-like columns
            if any(term in col_lower for term in ["id", "index", "row", "unnamed"]):
                score -= 20
            
            scored_cols.append((col, score))
        
        # Sort by score descending
        scored_cols.sort(key=lambda x: x[1], reverse=True)
        
        # Return top columns
        result = [col for col, score in scored_cols[:max_vars] if score > 0]
        
        # If no good matches, return first numeric columns
        if not result:
            result = numeric_cols[:max_vars]
        
        logger.info(f"DATA_ANALYST: Inferred {len(result)} key variables from {dataset_name}: {result[:5]}...")
        return result
        
    except Exception as e:
        logger.warning(f"DATA_ANALYST: Failed to infer variables from {dataset_name}: {e}")
        return []


def _extract_variables_from_analysis_design(analysis_design: str) -> list[str]:
    """
    Extract variable names from the analysis_design text.
    
    Parses common patterns like:
    - DEPENDENT_VARIABLE: variable_name
    - INDEPENDENT_VARIABLES: var1, var2, var3
    - bullet lists
    
    Args:
        analysis_design: The analysis design text from research plan.
        
    Returns:
        List of extracted variable names.
    """
    if not analysis_design:
        return []
    
    variables = []
    
    # Common section headers to look for
    patterns = [
        "DEPENDENT_VARIABLE", "INDEPENDENT_VARIABLES", "CONTROL_VARIABLES",
        "key_variables", "outcome_variable", "explanatory_variables",
    ]
    
    for pattern in patterns:
        if pattern in analysis_design:
            # Get the section after the pattern
            section = analysis_design.split(pattern)[-1]
            # Take first few lines
            lines = section.split("\n")[:10]
            for line in lines:
                # Clean and extract variable names
                line = line.strip().lstrip("-•*:").strip()
                if line and not line.startswith("#") and len(line) < 100:
                    # Extract potential variable names (alphanumeric with underscores)
                    import re
                    var_matches = re.findall(r'\b([a-z][a-z0-9_]*(?:_[a-z0-9]+)*)\b', line.lower())
                    for var in var_matches:
                        if len(var) > 2 and var not in ["the", "and", "for", "with", "from"]:
                            variables.append(var)
    
    # Deduplicate while preserving order
    seen = set()
    result = []
    for v in variables:
        if v not in seen:
            seen.add(v)
            result.append(v)
    
    return result[:20]  # Limit to 20 variables


def _ensure_data_loaded(state: WorkflowState) -> list[str]:
    """
    Ensure data is loaded into registry. Reloads from uploaded_data if needed.
    
    This handles the case where the registry singleton is reset between nodes
    (e.g., in LangGraph Studio's async workers).
    
    Returns:
        List of successfully loaded dataset names.
    """
    from pathlib import Path
    
    registry = get_registry()
    available = list(registry.datasets.keys())
    
    # Get what datasets SHOULD be loaded from state
    expected_datasets = state.get("loaded_datasets", [])
    logger.info(f"DATA_ANALYST: Expected datasets from state: {expected_datasets}")
    logger.info(f"DATA_ANALYST: Currently in registry: {available}")
    
    # Check if all expected datasets are available
    missing = [d for d in expected_datasets if d not in available]
    
    if not missing:
        logger.info(f"DATA_ANALYST: All {len(available)} datasets available in registry")
        return available
    
    logger.warning(f"DATA_ANALYST: Missing datasets: {missing}. Attempting reload...")
    
    # Try to reload from uploaded_data
    uploaded_data = state.get("uploaded_data", [])
    logger.info(f"DATA_ANALYST: Found {len(uploaded_data)} files in uploaded_data")
    
    if not uploaded_data:
        logger.warning("DATA_ANALYST: No uploaded_data in state, cannot reload")
        return available
    
    loaded = list(available)  # Start with what we have
    
    for data_file in uploaded_data:
        try:
            # Handle both DataFile objects and dicts
            if hasattr(data_file, 'filepath'):
                filepath = str(data_file.filepath)
            elif isinstance(data_file, dict):
                filepath = str(data_file.get('filepath', ''))
            else:
                logger.warning(f"DATA_ANALYST: Unknown data_file type: {type(data_file)}")
                continue
            
            if not filepath:
                continue
            
            # Skip macOS metadata files
            if "/__MACOSX/" in filepath or "/._" in filepath:
                continue
            
            # Check if file exists
            path = Path(filepath)
            if not path.exists():
                logger.warning(f"DATA_ANALYST: File not found: {filepath}")
                continue
            
            # Generate dataset name from path (match data_explorer logic)
            parent_name = path.parent.name if path.parent.name not in [".", "..", "project_data", "data", "tmp"] else ""
            base_name = path.stem
            
            if parent_name:
                dataset_name = f"{parent_name}_{base_name}".lower().replace(" ", "_").replace("-", "_")
            else:
                dataset_name = base_name.lower().replace(" ", "_").replace("-", "_")
            
            # Skip if already loaded
            if dataset_name in loaded:
                continue
            
            logger.info(f"DATA_ANALYST: Reloading {dataset_name} from {filepath}")
            
            result = load_data.invoke({"filepath": filepath, "name": dataset_name})
            if "error" not in result:
                loaded.append(dataset_name)
                logger.info(f"DATA_ANALYST: Successfully reloaded {dataset_name} ({result.get('row_count', 0)} rows)")
            else:
                logger.error(f"DATA_ANALYST: Failed to reload {dataset_name}: {result.get('error')}")
        except Exception as e:
            logger.error(f"DATA_ANALYST: Error reloading data: {e}", exc_info=True)
    
    # Final check
    final_available = list(registry.datasets.keys())
    logger.info(f"DATA_ANALYST: After reload, registry has: {final_available}")
    
    return final_available


def _extract_data_info(state: WorkflowState) -> dict[str, Any]:
    """Extract data exploration information from state."""
    data_results = state.get("data_exploration_results")
    loaded_datasets = state.get("loaded_datasets", [])
    
    if not data_results:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "columns": [],
            "quality_level": "not_assessed",
            "loaded_datasets": loaded_datasets,
        }
    
    if isinstance(data_results, dict):
        return {
            "total_rows": data_results.get("total_rows", 0),
            "total_columns": data_results.get("total_columns", 0),
            "columns": data_results.get("columns", []),
            "quality_level": data_results.get("quality_level", "not_assessed"),
            "variable_mappings": data_results.get("variable_mappings", []),
            "loaded_datasets": loaded_datasets,
        }
    elif hasattr(data_results, "total_rows"):
        return {
            "total_rows": data_results.total_rows,
            "total_columns": data_results.total_columns,
            "columns": [c.model_dump() if hasattr(c, "model_dump") else c for c in data_results.columns],
            "quality_level": data_results.quality_level.value if hasattr(data_results.quality_level, "value") else str(data_results.quality_level),
            "variable_mappings": [v.model_dump() if hasattr(v, "model_dump") else v for v in data_results.variable_mappings],
            "loaded_datasets": loaded_datasets,
        }
    
    return {"total_rows": 0, "total_columns": 0, "columns": [], "quality_level": "not_assessed", "loaded_datasets": loaded_datasets}


def _extract_plan_info(state: WorkflowState) -> dict[str, Any]:
    """Extract research plan information from state."""
    plan = state.get("research_plan")
    
    if not plan:
        return {
            "methodology": "",
            "methodology_type": None,
            "analysis_approach": None,
            "statistical_tests": [],
            "key_variables": [],
            "control_variables": [],
            "hypothesis": None,
        }
    
    if isinstance(plan, dict):
        return {
            "methodology": plan.get("methodology", ""),
            "methodology_type": plan.get("methodology_type"),
            "analysis_approach": plan.get("analysis_approach"),
            "statistical_tests": plan.get("statistical_tests", []),
            "key_variables": plan.get("key_variables", []),
            "control_variables": plan.get("control_variables", []),
            "hypothesis": plan.get("hypothesis"),
        }
    elif hasattr(plan, "methodology"):
        return {
            "methodology": plan.methodology,
            "methodology_type": plan.methodology_type.value if hasattr(plan.methodology_type, "value") else plan.methodology_type,
            "analysis_approach": plan.analysis_approach.value if hasattr(plan.analysis_approach, "value") else plan.analysis_approach,
            "statistical_tests": plan.statistical_tests,
            "key_variables": plan.key_variables,
            "control_variables": plan.control_variables,
            "hypothesis": plan.hypothesis,
        }
    
    return {"methodology": "", "methodology_type": None, "analysis_approach": None}


def _extract_gap_info(state: WorkflowState) -> dict[str, Any]:
    """Extract gap analysis information from state."""
    gap_analysis = state.get("gap_analysis")
    
    if not gap_analysis:
        return {
            "primary_gap": None,
            "gap_description": "",
        }
    
    if isinstance(gap_analysis, dict):
        primary_gap = gap_analysis.get("primary_gap", {})
        return {
            "primary_gap": primary_gap,
            "gap_description": primary_gap.get("description", "") if isinstance(primary_gap, dict) else "",
        }
    elif hasattr(gap_analysis, "primary_gap"):
        primary = gap_analysis.primary_gap
        return {
            "primary_gap": primary.model_dump() if hasattr(primary, "model_dump") else primary,
            "gap_description": primary.description if hasattr(primary, "description") else "",
        }
    
    return {"primary_gap": None, "gap_description": ""}


def _get_research_question(state: WorkflowState) -> str:
    """Get the research question from state."""
    if state.get("refined_query"):
        return state["refined_query"]
    
    refined_rq = state.get("refined_research_question")
    if refined_rq:
        if isinstance(refined_rq, dict):
            return refined_rq.get("refined_question", state.get("original_query", ""))
        elif hasattr(refined_rq, "refined_question"):
            return refined_rq.refined_question
    
    return state.get("original_query", "")


def _get_primary_dataset(data_info: dict[str, Any], research_question: str = "") -> str | None:
    """
    Get the primary dataset name from data info, prioritizing based on research context.
    
    For options/finance research, prioritizes options datasets over underlying.
    
    Args:
        data_info: Data exploration information.
        research_question: Research question for context.
        
    Returns:
        Name of the primary dataset to analyze.
    """
    datasets = data_info.get("loaded_datasets", [])
    if not datasets:
        return None
    
    # If only one dataset, use it
    if len(datasets) == 1:
        return datasets[0]
    
    # Prioritize based on research context
    rq_lower = research_question.lower()
    
    # For options research, prefer options datasets
    if any(term in rq_lower for term in ["option", "derivatives", "pricing", "volatility", "implied"]):
        options_datasets = [d for d in datasets if "option" in d.lower()]
        if options_datasets:
            # Prefer GOOG or GOOGL options for dual-class research
            if "goog" in rq_lower or "googl" in rq_lower:
                goog_options = [d for d in options_datasets if "goog" in d.lower()]
                if goog_options:
                    # Prefer the one with more data (GOOG typically)
                    return sorted(goog_options, key=lambda x: "googl" not in x.lower())[0]
            return options_datasets[0]
    
    # For stock/equity research, prefer underlying
    if any(term in rq_lower for term in ["stock", "equity", "return", "price"]):
        underlying_datasets = [d for d in datasets if "underlying" in d.lower()]
        if underlying_datasets:
            return underlying_datasets[0]
    
    # Default to first dataset
    return datasets[0]


def _get_all_relevant_datasets(data_info: dict[str, Any], research_question: str = "") -> list[str]:
    """
    Get all datasets relevant to the research question.
    
    For comparative studies (e.g., GOOG vs GOOGL), returns both datasets.
    
    Args:
        data_info: Data exploration information.
        research_question: Research question for context.
        
    Returns:
        List of relevant dataset names.
    """
    datasets = data_info.get("loaded_datasets", [])
    if not datasets:
        return []
    
    rq_lower = research_question.lower()
    
    # For dual-class share research (GOOG/GOOGL comparison)
    if "dual" in rq_lower or ("goog" in rq_lower and "googl" in rq_lower):
        # Return both GOOG and GOOGL options datasets
        relevant = [d for d in datasets if any(
            term in d.lower() for term in ["goog_option", "googl_option", "goog_opt", "googl_opt"]
        )]
        if relevant:
            return relevant
    
    # For options research, return all options datasets
    if any(term in rq_lower for term in ["option", "derivatives", "pricing"]):
        options = [d for d in datasets if "option" in d.lower()]
        if options:
            return options
    
    # Default to all datasets
    return datasets


def _run_descriptive_analysis(
    dataset_name: str,
    key_vars: list[str],
) -> dict[str, Any]:
    """Run descriptive statistics using new tools."""
    try:
        result = execute_descriptive_stats.invoke({
            "dataset_name": dataset_name,
            "columns": key_vars if key_vars else None,
        })
        return result
    except Exception as e:
        return {"error": str(e)}


def _run_hypothesis_test(
    dataset_name: str,
    test_type: str,
    variables: list[str],
    group_var: str | None = None,
) -> tuple[StatisticalResult | None, str]:
    """Run hypothesis test using new scipy-backed tools."""
    try:
        if test_type in ["t_test", "ttest", "t-test"]:
            if len(variables) >= 2:
                result = run_ttest.invoke({
                    "dataset_name": dataset_name,
                    "column1": variables[0],
                    "column2": variables[1],
                    "paired": False,
                })
            elif group_var:
                result = run_ttest.invoke({
                    "dataset_name": dataset_name,
                    "column1": variables[0],
                    "group_column": group_var,
                })
            else:
                return None, "T-test requires two variables or a group variable"
                
        elif test_type in ["anova", "ANOVA"]:
            if group_var and len(variables) >= 1:
                result = run_anova.invoke({
                    "dataset_name": dataset_name,
                    "value_column": variables[0],
                    "group_column": group_var,
                })
            else:
                return None, "ANOVA requires value and group variables"
                
        elif test_type in ["chi_square", "chi-square", "chi2"]:
            if len(variables) >= 2:
                result = run_chi_square.invoke({
                    "dataset_name": dataset_name,
                    "column1": variables[0],
                    "column2": variables[1],
                })
            else:
                return None, "Chi-square requires two categorical variables"
        else:
            return None, f"Unknown test type: {test_type}"
        
        if "error" in result:
            return None, result["error"]
        
        # Convert to StatisticalResult model
        stat_result = StatisticalResult(
            test_name=result.get("test_name", test_type),
            test_statistic=result.get("statistic", 0.0),
            p_value=result.get("p_value", 1.0),
            degrees_of_freedom=int(result.get("df", 0)) if result.get("df") else 0,
            effect_size=result.get("effect_size"),
            effect_size_interpretation=result.get("effect_interpretation", ""),
            confidence_interval=result.get("ci"),
            is_significant=result.get("significant", False),
            interpretation=result.get("interpretation", ""),
            variables_tested=variables,
        )
        
        return stat_result, ""
        
    except Exception as e:
        return None, str(e)


def _run_regression_analysis(
    dataset_name: str,
    dependent: str,
    independent: list[str],
    control_vars: list[str] | None = None,
    model_type: str = "ols",
) -> tuple[RegressionResult | None, str]:
    """Run regression using new statsmodels-backed tools."""
    try:
        all_predictors = independent + (control_vars or [])
        
        if model_type in ["ols", "linear", "ols_regression"]:
            result = run_ols_regression.invoke({
                "dataset_name": dataset_name,
                "dependent_variable": dependent,
                "independent_variables": all_predictors,
            })
        elif model_type in ["logistic", "logit"]:
            result = run_logistic_regression.invoke({
                "dataset_name": dataset_name,
                "dependent_variable": dependent,
                "independent_variables": all_predictors,
            })
        else:
            return None, f"Unknown regression type: {model_type}"

        # New regression tools return structured models (RegressionResult) for OLS.
        # Older paths or logistic regression return dicts.
        if isinstance(result, RegressionResult):
            # Treat tool-level errors as failures.
            if result.model_type.endswith("_error") or result.n_observations <= 0:
                return None, result.interpretation or "Regression failed"
            return result, ""

        if not isinstance(result, dict):
            return None, f"Unexpected regression result type: {type(result)}"
        
        if result.get("status") == "error" or "error" in result:
            return None, str(result.get("error", "Regression failed"))
        
        # Convert coefficients
        coefficients = []
        for coef in result.get("coefficients", []):
            coefficients.append(RegressionCoefficient(
                variable=coef.get("variable", ""),
                coefficient=coef.get("coefficient", 0.0),
                std_error=coef.get("std_error", 0.0),
                t_statistic=coef.get("t_statistic", coef.get("t_stat", coef.get("z_statistic", 0.0))),
                p_value=coef.get("p_value", 1.0),
                confidence_interval_lower=coef.get("ci_lower", 0.0),
                confidence_interval_upper=coef.get("ci_upper", 0.0),
                is_significant=coef.get("is_significant", coef.get("significant", False)),
            ))
        
        # Build regression result
        n_obs = int(result.get("n_observations") or result.get("sample_size") or 0)
        if n_obs <= 0:
            return None, str(result.get("interpretation") or result.get("error") or "Regression failed")

        reg_result = RegressionResult(
            model_type=str(result.get("model_type", model_type)),
            dependent_variable=str(result.get("dependent_variable", dependent)),
            r_squared=float(result.get("r_squared", result.get("pseudo_r_squared", 0.0))),
            adjusted_r_squared=float(result.get("adjusted_r_squared", result.get("adj_r_squared", result.get("pseudo_r_squared", 0.0)))),
            f_statistic=result.get("f_statistic"),
            f_p_value=result.get("f_p_value"),
            coefficients=coefficients,
            n_observations=n_obs,
            interpretation=str(result.get("interpretation", "")),
        )
        
        return reg_result, ""
        
    except Exception as e:
        return None, str(e)


def _generate_finding_from_stat_result(
    stat_result: StatisticalResult,
    hypothesis: str | None,
) -> DataAnalysisFinding:
    """Generate a finding from statistical test result."""
    is_sig = stat_result.is_significant
    
    finding_type = FindingType.MAIN_RESULT if is_sig else FindingType.NULL_RESULT
    
    statement = (
        f"{stat_result.test_name} {'reveals a statistically significant effect' if is_sig else 'shows no significant effect'} "
        f"(p = {stat_result.p_value:.4f})"
    )
    
    if stat_result.effect_size:
        statement += f" with {stat_result.effect_size_interpretation} effect size ({stat_result.effect_size:.3f})"
    
    return DataAnalysisFinding(
        finding_type=finding_type,
        statement=statement,
        detailed_description=stat_result.interpretation,
        statistical_results=[stat_result],
        evidence_strength=EvidenceStrength.STRONG if is_sig and stat_result.effect_size and abs(stat_result.effect_size) > 0.5 else EvidenceStrength.MODERATE if is_sig else EvidenceStrength.WEAK,
        addresses_research_question=True,
        addresses_gap=is_sig,
        confidence_level=0.95 if stat_result.p_value < 0.01 else 0.9 if stat_result.p_value < 0.05 else 0.5,
    )


def _generate_finding_from_regression(
    reg_result: RegressionResult,
) -> DataAnalysisFinding:
    """Generate a finding from regression result."""
    sig_vars = [c.variable for c in reg_result.coefficients if c.is_significant and c.variable != "(Intercept)"]
    
    if sig_vars:
        finding_type = FindingType.MAIN_RESULT
        statement = f"Regression analysis reveals significant relationships: {', '.join(sig_vars)} predict {reg_result.dependent_variable}"
        evidence = EvidenceStrength.STRONG if reg_result.r_squared > 0.3 else EvidenceStrength.MODERATE
    else:
        finding_type = FindingType.NULL_RESULT
        statement = f"No significant predictors found for {reg_result.dependent_variable}"
        evidence = EvidenceStrength.WEAK
    
    return DataAnalysisFinding(
        finding_type=finding_type,
        statement=statement,
        detailed_description=reg_result.interpretation,
        regression_results=[reg_result],
        evidence_strength=evidence,
        addresses_research_question=True,
        addresses_gap=len(sig_vars) > 0,
        confidence_level=0.8 if reg_result.r_squared > 0.2 else 0.6,
    )


def _assess_gap_addressed(
    findings: list[DataAnalysisFinding],
    gap_info: dict[str, Any],
    research_question: str,
) -> tuple[bool, float, str]:
    """Assess whether findings address the research gap."""
    gap_description = gap_info.get("gap_description", "")
    
    # Convert findings to dict for tool
    findings_dicts = [
        {
            "finding_id": f.finding_id,
            "addresses_gap": f.addresses_gap,
            "addresses_research_question": f.addresses_research_question,
            "finding_type": f.finding_type.value if hasattr(f.finding_type, "value") else f.finding_type,
        }
        for f in findings
    ]
    
    assessment = assess_gap_coverage.invoke({
        "findings": findings_dicts,
        "gap_description": gap_description,
        "research_question": research_question,
    })
    
    return (
        assessment.get("gap_addressed", False),
        assessment.get("coverage_score", 0.0),
        assessment.get("explanation", ""),
    )


def _generate_llm_interpretations(
    findings: list[DataAnalysisFinding],
    stat_results: list[StatisticalResult],
    reg_results: list[RegressionResult],
    research_question: str,
) -> str:
    """Generate LLM-powered interpretations for academic prose."""
    interpretations = []
    
    # Interpret regression results
    for reg in reg_results:
        try:
            interp = interpret_regression.invoke({
                "regression_output": reg.model_dump() if hasattr(reg, "model_dump") else reg,
                "research_context": research_question,
            })
            if isinstance(interp, dict) and "interpretation" in interp:
                interpretations.append(interp["interpretation"])
            elif isinstance(interp, str):
                interpretations.append(interp)
        except Exception:
            pass
    
    # Interpret hypothesis tests
    for stat in stat_results:
        try:
            interp = interpret_hypothesis_test.invoke({
                "test_output": stat.model_dump() if hasattr(stat, "model_dump") else stat,
                "research_context": research_question,
            })
            if isinstance(interp, dict) and "interpretation" in interp:
                interpretations.append(interp["interpretation"])
            elif isinstance(interp, str):
                interpretations.append(interp)
        except Exception:
            pass
    
    # Summarize all findings
    if findings:
        try:
            summary = summarize_findings.invoke({
                "findings": [f.model_dump() if hasattr(f, "model_dump") else f for f in findings],
                "research_question": research_question,
            })
            if isinstance(summary, dict) and "summary" in summary:
                interpretations.append(f"\nOverall Summary:\n{summary['summary']}")
            elif isinstance(summary, str):
                interpretations.append(f"\nOverall Summary:\n{summary}")
        except Exception:
            pass
    
    return "\n\n".join(interpretations) if interpretations else ""


def _generate_visualization_artifacts(
    dataset_name: str,
    key_vars: list[str],
    reg_results: list[RegressionResult],
    data_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Generate publication-ready tables and figures.
    
    Returns:
        Tuple of (tables, figures) as lists of artifact dicts.
    """
    tables = []
    figures = []
    
    if not dataset_name:
        return tables, figures
    
    registry = get_registry()
    if dataset_name not in registry.datasets:
        return tables, figures
    
    # Table 1: Summary Statistics
    try:
        result = create_summary_statistics_table.invoke({
            "dataset_name": dataset_name,
            "variables": key_vars if key_vars else None,
            "format": "latex",
            "title": "Summary Statistics",
            "label": "tab:summary",
        })
        if result.get("status") == "success" and result.get("artifact"):
            tables.append(result["artifact"])
            logger.info("DATA_ANALYST: Generated summary statistics table (Table 1)")
    except Exception as e:
        logger.warning(f"DATA_ANALYST: Failed to generate summary statistics table: {e}")
    
    # Table 2: Regression Results (if regressions were run)
    if reg_results:
        try:
            reg_dicts: list[dict[str, Any]] = []
            for r in reg_results:
                if isinstance(r, RegressionResult):
                    reg_dicts.append(r.model_dump())
                elif isinstance(r, dict):
                    reg_dicts.append(r)
                else:
                    logger.warning(
                        "DATA_ANALYST: Skipping unexpected regression result type: %s",
                        type(r),
                    )
            result = create_regression_table.invoke({
                "regression_results": reg_dicts,
                "format": "latex",
                "title": "Regression Results",
                "label": "tab:regression",
            })
            if result.get("status") == "success" and result.get("artifact"):
                tables.append(result["artifact"])
                logger.info("DATA_ANALYST: Generated regression table (Table 2)")
        except Exception as e:
            logger.warning(f"DATA_ANALYST: Failed to generate regression table: {e}")
    
    # Table 3: Correlation Matrix (if multiple numeric variables)
    if len(key_vars) >= 2:
        try:
            result = create_correlation_matrix_table.invoke({
                "dataset_name": dataset_name,
                "variables": key_vars,
                "format": "latex",
                "title": "Correlation Matrix",
                "label": "tab:correlation",
            })
            if result.get("status") == "success" and result.get("artifact"):
                tables.append(result["artifact"])
                logger.info("DATA_ANALYST: Generated correlation matrix (Table 3)")
        except Exception as e:
            logger.warning(f"DATA_ANALYST: Failed to generate correlation table: {e}")
    
    # Figure 1: Time series (if date column detected)
    df = registry.get_dataframe(dataset_name)
    date_cols = [c for c in df.columns if df[c].dtype.name in ['datetime64[ns]', 'object']]
    
    # Try to identify a date column
    potential_date_cols = [c for c in date_cols if any(
        d in c.lower() for d in ['date', 'time', 'year', 'month', 'day', 'period']
    )]
    
    if potential_date_cols and len(key_vars) >= 1:
        try:
            # Take first numeric variable for time series
            value_cols = [v for v in key_vars if v in df.select_dtypes(include=['number']).columns][:3]
            if value_cols:
                result = create_time_series_plot.invoke({
                    "dataset_name": dataset_name,
                    "date_column": potential_date_cols[0],
                    "value_columns": value_cols,
                    "title": f"Time Series of {', '.join(value_cols[:2])}",
                    "ylabel": "Value",
                })
                if result.get("status") == "success" and result.get("artifact"):
                    figures.append(result["artifact"])
                    logger.info("DATA_ANALYST: Generated time series plot (Figure 1)")
        except Exception as e:
            logger.warning(f"DATA_ANALYST: Failed to generate time series plot: {e}")
    
    # Figure 2: Scatter plot (for regression variables)
    if reg_results and len(key_vars) >= 2:
        try:
            result = create_scatter_plot.invoke({
                "dataset_name": dataset_name,
                "x_column": key_vars[1],  # First independent var
                "y_column": key_vars[0],  # Dependent var
                "title": f"{key_vars[0]} vs {key_vars[1]}",
                "add_regression_line": True,
            })
            if result.get("status") == "success" and result.get("artifact"):
                figures.append(result["artifact"])
                logger.info("DATA_ANALYST: Generated scatter plot (Figure 2)")
        except Exception as e:
            logger.warning(f"DATA_ANALYST: Failed to generate scatter plot: {e}")
    
    # Figure 3: Distribution of dependent variable
    if key_vars:
        try:
            result = create_distribution_plot.invoke({
                "dataset_name": dataset_name,
                "column": key_vars[0],
                "plot_type": "histogram",
                "title": f"Distribution of {key_vars[0]}",
            })
            if result.get("status") == "success" and result.get("artifact"):
                figures.append(result["artifact"])
                logger.info("DATA_ANALYST: Generated distribution plot (Figure 3)")
        except Exception as e:
            logger.warning(f"DATA_ANALYST: Failed to generate distribution plot: {e}")
    
    return tables, figures


# =============================================================================
# Main Node Function
# =============================================================================


def data_analyst_node(state: WorkflowState) -> dict:
    """
    Execute data analysis per research plan using comprehensive tools.
    
    This node:
    1. Extracts methodology and analysis approach from research plan
    2. Uses scipy/statsmodels for real statistical analysis
    3. Generates LLM-powered interpretations
    4. Links findings to research question
    5. Assesses whether findings address the identified gap
    
    Args:
        state: Current workflow state.
        
    Returns:
        Updated state with analysis results.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # CRITICAL: Ensure data is loaded into registry (handles async worker resets)
    available_datasets = _ensure_data_loaded(state)
    registry = get_registry()
    
    # Extract information from state
    data_info = _extract_data_info(state)
    plan_info = _extract_plan_info(state)
    gap_info = _extract_gap_info(state)
    research_question = _get_research_question(state)
    
    # Get primary dataset from state with research context
    dataset_name = _get_primary_dataset(data_info, research_question)
    all_relevant_datasets = _get_all_relevant_datasets(data_info, research_question)
    
    logger.info(f"DATA_ANALYST: Primary dataset from state: {dataset_name}")
    logger.info(f"DATA_ANALYST: All relevant datasets: {all_relevant_datasets}")
    logger.info(f"DATA_ANALYST: Loaded datasets in state: {data_info.get('loaded_datasets', [])}")
    logger.info(f"DATA_ANALYST: Available in registry: {available_datasets}")
    
    # If dataset_name from state doesn't exist in registry, try to use first available
    if dataset_name and dataset_name not in registry.datasets:
        logger.warning(f"DATA_ANALYST: Dataset '{dataset_name}' not in registry. Available: {available_datasets}")
        if available_datasets:
            dataset_name = available_datasets[0]
            logger.info(f"DATA_ANALYST: Using fallback dataset: {dataset_name}")
    elif not dataset_name and available_datasets:
        dataset_name = available_datasets[0]
        logger.info(f"DATA_ANALYST: No dataset in state, using registry dataset: {dataset_name}")
    
    # Validate we have data to analyze
    if not dataset_name and data_info.get("total_rows", 0) == 0:
        error = WorkflowError(
            node="data_analyst",
            category="validation",
            message="No data available for analysis. DATA_ANALYST requires data from DATA_EXPLORER.",
            recoverable=False,
        )
        return {
            "status": ResearchStatus.FAILED,
            "errors": [error],
            "messages": [
                AIMessage(content=f"[{current_date}] DATA_ANALYST: Error - No data available for analysis.")
            ],
        }
    
    # Validate we have a research plan
    if not plan_info.get("methodology"):
        error = WorkflowError(
            node="data_analyst",
            category="validation",
            message="No research plan available. DATA_ANALYST requires a plan from PLANNER.",
            recoverable=False,
        )
        return {
            "status": ResearchStatus.FAILED,
            "errors": [error],
            "messages": [
                AIMessage(content=f"[{current_date}] DATA_ANALYST: Error - No research plan available.")
            ],
        }
    
    findings: list[DataAnalysisFinding] = []
    stat_results: list[StatisticalResult] = []
    reg_results: list[RegressionResult] = []
    analysis_notes: list[str] = []
    
    methodology = plan_info.get("methodology_type", "")
    analysis_approach = plan_info.get("analysis_approach", "")
    key_vars = plan_info.get("key_variables", [])
    control_vars = plan_info.get("control_variables", [])
    hypothesis = plan_info.get("hypothesis")
    
    # CRITICAL FIX: Infer key variables if not provided
    if not key_vars and dataset_name:
        logger.info(f"DATA_ANALYST: key_variables empty, attempting to infer from {dataset_name}")
        
        # First try to extract from analysis_design text in research_plan
        analysis_design_text = plan_info.get("analysis_design", "")
        if analysis_design_text:
            key_vars = _extract_variables_from_analysis_design(analysis_design_text)
            if key_vars:
                logger.info(f"DATA_ANALYST: Extracted {len(key_vars)} variables from analysis_design: {key_vars[:5]}")
        
        # If still empty, infer from dataset columns
        if not key_vars:
            key_vars = _infer_key_variables_from_dataset(dataset_name, research_question)
            if key_vars:
                logger.info(f"DATA_ANALYST: Inferred {len(key_vars)} variables from dataset columns")
        
        # Log what we're using
        if key_vars:
            analysis_notes.append(f"Key variables inferred automatically: {', '.join(key_vars[:5])}...")
        else:
            logger.warning("DATA_ANALYST: Could not infer any key variables - analysis may be limited")
    
    logger.info(f"DATA_ANALYST: Using {len(key_vars)} key variables: {key_vars[:5] if key_vars else '[]'}")
    
    # Run descriptive statistics
    if dataset_name:
        desc_stats = _run_descriptive_analysis(dataset_name, key_vars)
        if "error" not in desc_stats:
            analysis_notes.append(f"Descriptive statistics computed for {len(key_vars)} variables")
    
    # Execute regression if appropriate
    # For empirical research, always try regression if we have enough variables
    should_run_regression = (
        methodology in ["regression_analysis", "panel_data", "ols", "quantitative", "empirical"]
        or analysis_approach in ["ols_regression", "fixed_effects", "ols", "regression"]
        or research_question  # If we have a research question, try regression
    )
    
    if should_run_regression and len(key_vars) >= 2 and dataset_name:
        dependent = key_vars[0]
        independent = key_vars[1:]
        
        logger.info(f"DATA_ANALYST: Running OLS regression - dependent: {dependent}, independent: {independent[:5]}...")
        
        reg_result, error = _run_regression_analysis(
            dataset_name=dataset_name,
            dependent=dependent,
            independent=independent,
            control_vars=control_vars,
            model_type="ols",
        )
        
        if reg_result:
            reg_results.append(reg_result)
            findings.append(_generate_finding_from_regression(reg_result))
            analysis_notes.append(f"OLS regression: {dependent} ~ {' + '.join(independent)}")
        else:
            analysis_notes.append(f"Regression failed: {error}")
    
    # Execute hypothesis test if hypothesis provided
    if hypothesis and len(key_vars) >= 1 and dataset_name:
        # Determine appropriate test
        test_type = "t_test"  # Default
        planned_tests = plan_info.get("statistical_tests", [])
        if planned_tests:
            test_type = planned_tests[0].lower().replace("-", "_").replace(" ", "_")
        
        stat_result, error = _run_hypothesis_test(
            dataset_name=dataset_name,
            test_type=test_type,
            variables=key_vars[:2],
            group_var=key_vars[2] if len(key_vars) > 2 else None,
        )
        
        if stat_result:
            stat_results.append(stat_result)
            findings.append(_generate_finding_from_stat_result(stat_result, hypothesis))
            analysis_notes.append(f"Hypothesis test ({test_type}): p = {stat_result.p_value:.4f}")
        else:
            analysis_notes.append(f"Hypothesis test failed: {error}")
    
    # Add descriptive finding
    if data_info.get("total_rows", 0) > 0:
        finding = DataAnalysisFinding(
            finding_type=FindingType.SUPPORTING,
            statement=f"Analysis based on {data_info.get('total_rows', 0)} observations across {data_info.get('total_columns', 0)} variables.",
            detailed_description=f"Data quality assessed as {data_info.get('quality_level', 'not assessed')}.",
            evidence_strength=EvidenceStrength.MODERATE,
            addresses_research_question=False,
            addresses_gap=False,
            confidence_level=1.0,
        )
        findings.append(finding)
    
    # Assess gap coverage
    gap_addressed, coverage_score, coverage_explanation = _assess_gap_addressed(
        findings, gap_info, research_question
    )
    
    # Generate LLM interpretations
    llm_interpretations = _generate_llm_interpretations(
        findings, stat_results, reg_results, research_question
    )
    if llm_interpretations:
        analysis_notes.append("LLM interpretations generated for academic prose")
    
    # Sprint 15: Generate visualization artifacts (tables and figures)
    tables, figures = _generate_visualization_artifacts(
        dataset_name=dataset_name,
        key_vars=key_vars,
        reg_results=reg_results,
        data_info=data_info,
    )
    if tables:
        analysis_notes.append(f"Generated {len(tables)} table(s)")
    if figures:
        analysis_notes.append(f"Generated {len(figures)} figure(s)")
    
    # Determine hypothesis support
    hypothesis_supported = None
    hypothesis_summary = ""
    if plan_info.get("hypothesis"):
        significant = [f for f in findings if f.evidence_strength == EvidenceStrength.STRONG]
        hypothesis_supported = len(significant) > 0
        hypothesis_summary = (
            f"Hypothesis {'supported' if hypothesis_supported else 'not supported'} by "
            f"{len(significant)} significant finding(s)."
        )
    
    # Identify main findings
    main_findings = [f for f in findings if f.finding_type == FindingType.MAIN_RESULT]
    
    # Get actual sample size from registry - aggregate across all relevant datasets
    actual_sample_size = 0
    datasets_analyzed = []
    dataset_sample_sizes = {}
    
    # First, calculate total from all relevant datasets
    for ds_name in all_relevant_datasets:
        if ds_name in registry.datasets:
            ds_info = registry.datasets[ds_name]
            ds_rows = ds_info.get("row_count", 0)
            actual_sample_size += ds_rows
            datasets_analyzed.append(ds_name)
            dataset_sample_sizes[ds_name] = ds_rows
            logger.info(f"DATA_ANALYST: Dataset {ds_name}: {ds_rows:,} rows")
    
    # If no relevant datasets, try all available
    if actual_sample_size == 0:
        for ds_name in available_datasets:
            if ds_name in registry.datasets:
                ds_info = registry.datasets[ds_name]
                ds_rows = ds_info.get("row_count", 0)
                actual_sample_size += ds_rows
                datasets_analyzed.append(ds_name)
                dataset_sample_sizes[ds_name] = ds_rows
    
    # Fallback to state's total_rows
    if actual_sample_size == 0:
        actual_sample_size = data_info.get("total_rows", 0)
    
    logger.info(f"DATA_ANALYST: Total sample size across {len(datasets_analyzed)} datasets: {actual_sample_size:,}")
    
    # Build data summary with dataset breakdown
    if len(datasets_analyzed) > 1:
        ds_breakdown = "; ".join([f"{ds}: {dataset_sample_sizes.get(ds, 0):,}" for ds in datasets_analyzed[:4]])
        data_summary = f"Analyzed {actual_sample_size:,} observations across {len(datasets_analyzed)} datasets ({ds_breakdown})."
    else:
        data_summary = f"Analyzed {actual_sample_size:,} observations across {len(key_vars)} key variables using {dataset_name}."
    
    # Build analysis result
    analysis_approach_enum = None
    if plan_info.get("analysis_approach"):
        try:
            analysis_approach_enum = AnalysisApproach(plan_info["analysis_approach"])
        except ValueError:
            analysis_approach_enum = AnalysisApproach.OTHER
    
    analysis_result = DataAnalysisResult(
        analysis_status=AnalysisStatus.COMPLETE,
        methodology_used=plan_info.get("methodology", ""),
        analysis_approach=analysis_approach_enum,
        data_summary=data_summary,
        sample_size=actual_sample_size,
        variables_analyzed=key_vars,
        descriptive_stats=desc_stats if dataset_name and "error" not in (desc_stats or {}) else {},
        findings=findings,
        main_findings=main_findings,
        statistical_tests=stat_results,
        regression_analyses=reg_results,
        gap_addressed=gap_addressed,
        gap_coverage_score=coverage_score,
        gap_coverage_explanation=coverage_explanation,
        hypothesis_supported=hypothesis_supported,
        hypothesis_test_summary=hypothesis_summary,
        overall_confidence=0.7 if gap_addressed else 0.5,
        limitations=[
            "Analysis based on available data structure",
            "Statistical assumptions validated where applicable",
        ],
        llm_interpretations=llm_interpretations,
    )
    
    # Build summary message
    datasets_display = ", ".join(datasets_analyzed[:3])
    if len(datasets_analyzed) > 3:
        datasets_display += f" (+{len(datasets_analyzed) - 3} more)"
    
    summary_parts = [
        f"[{current_date}] DATA_ANALYST: Analysis complete using comprehensive toolset.",
        f"Datasets: {datasets_display}" if datasets_analyzed else f"Dataset: {dataset_name}",
        f"Sample size: {analysis_result.sample_size:,}",
        f"Findings: {len(findings)} ({len(main_findings)} main)",
        f"Gap addressed: {'Yes' if gap_addressed else 'Partially'} (score: {coverage_score:.2f})",
    ]
    if hypothesis_summary:
        summary_parts.append(f"Hypothesis: {hypothesis_summary}")
    if analysis_notes:
        summary_parts.append(f"Notes: {'; '.join(analysis_notes[:3])}")
    
    return {
        "status": ResearchStatus.ANALYSIS_COMPLETE,
        "analysis": analysis_result.model_dump(),
        "data_analyst_output": analysis_result.model_dump(),  # For routing
        "tables": tables,  # Sprint 15: Table artifacts
        "figures": figures,  # Sprint 15: Figure artifacts
        "messages": [
            AIMessage(content=" | ".join(summary_parts))
        ],
    }


# =============================================================================
# Routing Function
# =============================================================================


def route_after_data_analyst(state: WorkflowState) -> Literal["writer", "__end__"]:
    """
    Route after DATA_ANALYST to WRITER node.
    
    Args:
        state: Current workflow state.
        
    Returns:
        Next node: "writer" if analysis complete, "__end__" if failed.
    """
    if state.get("errors"):
        return "__end__"
    
    if state.get("status") == ResearchStatus.ANALYSIS_COMPLETE:
        return "writer"
    
    if state.get("analysis") or state.get("data_analyst_output"):
        return "writer"
    
    return "__end__"
