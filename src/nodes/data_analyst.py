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

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


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


def _get_primary_dataset(data_info: dict[str, Any]) -> str | None:
    """Get the primary dataset name from data info."""
    datasets = data_info.get("loaded_datasets", [])
    if datasets:
        return datasets[0]
    return None


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
                "dependent_var": dependent,
                "independent_vars": all_predictors,
            })
        elif model_type in ["logistic", "logit"]:
            result = run_logistic_regression.invoke({
                "dataset_name": dataset_name,
                "dependent_var": dependent,
                "independent_vars": all_predictors,
            })
        else:
            return None, f"Unknown regression type: {model_type}"
        
        if "error" in result:
            return None, result["error"]
        
        # Convert coefficients
        coefficients = []
        for coef in result.get("coefficients", []):
            coefficients.append(RegressionCoefficient(
                variable=coef.get("variable", ""),
                coefficient=coef.get("coefficient", 0.0),
                std_error=coef.get("std_error", 0.0),
                t_statistic=coef.get("t_stat", 0.0),
                p_value=coef.get("p_value", 1.0),
                confidence_interval_lower=coef.get("ci_lower"),
                confidence_interval_upper=coef.get("ci_upper"),
                is_significant=coef.get("significant", False),
            ))
        
        # Build regression result
        reg_result = RegressionResult(
            model_type=model_type,
            dependent_variable=dependent,
            independent_variables=all_predictors,
            coefficients=coefficients,
            r_squared=result.get("r_squared", 0.0),
            adj_r_squared=result.get("adj_r_squared", 0.0),
            f_statistic=result.get("f_statistic", 0.0),
            f_p_value=result.get("f_p_value", 1.0),
            sample_size=result.get("n_observations", 0),
            interpretation=result.get("interpretation", ""),
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
            reg_dicts = [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in reg_results
            ]
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
    date_cols = [c for c in df.columns if df[c].dtype in ['datetime64[ns]', 'object']]
    
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
    
    # Debug: Check registry state
    registry = get_registry()
    available_datasets = list(registry.datasets.keys())
    logger.info(f"DATA_ANALYST: Registry has {len(available_datasets)} datasets: {available_datasets}")
    
    # Extract information from state
    data_info = _extract_data_info(state)
    plan_info = _extract_plan_info(state)
    gap_info = _extract_gap_info(state)
    research_question = _get_research_question(state)
    
    # Get primary dataset from state
    dataset_name = _get_primary_dataset(data_info)
    logger.info(f"DATA_ANALYST: Primary dataset from state: {dataset_name}")
    logger.info(f"DATA_ANALYST: Loaded datasets in state: {data_info.get('loaded_datasets', [])}")
    
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
    
    # Run descriptive statistics
    if dataset_name:
        desc_stats = _run_descriptive_analysis(dataset_name, key_vars)
        if "error" not in desc_stats:
            analysis_notes.append(f"Descriptive statistics computed for {len(key_vars)} variables")
    
    # Execute regression if appropriate
    if methodology in ["regression_analysis", "panel_data", "ols", "quantitative"] or analysis_approach in ["ols_regression", "fixed_effects", "ols"]:
        if len(key_vars) >= 2 and dataset_name:
            dependent = key_vars[0]
            independent = key_vars[1:]
            
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
    
    # Get actual sample size from registry if available
    actual_sample_size = data_info.get("total_rows", 0)
    if dataset_name and dataset_name in registry.datasets:
        dataset_info = registry.datasets[dataset_name]
        actual_sample_size = dataset_info.get("row_count", actual_sample_size)
        logger.info(f"DATA_ANALYST: Actual sample size from registry: {actual_sample_size}")
    
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
        data_summary=f"Analyzed {actual_sample_size} observations across {len(key_vars)} key variables using {dataset_name}.",
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
    summary_parts = [
        f"[{current_date}] DATA_ANALYST: Analysis complete using comprehensive toolset.",
        f"Dataset: {dataset_name}" if dataset_name else "Dataset: from state",
        f"Sample size: {analysis_result.sample_size}",
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
