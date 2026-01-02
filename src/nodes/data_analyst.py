"""DATA_ANALYST node for empirical research analysis.

This node:
1. Takes input from PLANNER (methodology, analysis approach, research plan)
2. Executes data analysis following the research plan methodology
3. Generates statistical results and findings
4. Links findings to research question and gap
5. Assesses whether findings address the identified gap
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
    WorkflowError,
)
from src.state.schema import WorkflowState
from src.tools.analysis import (
    execute_hypothesis_test,
    execute_regression_analysis,
    assess_gap_coverage,
)


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_data_info(state: WorkflowState) -> dict[str, Any]:
    """Extract data exploration information from state."""
    data_results = state.get("data_exploration_results")
    
    if not data_results:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "columns": [],
            "quality_level": "not_assessed",
        }
    
    if isinstance(data_results, dict):
        return {
            "total_rows": data_results.get("total_rows", 0),
            "total_columns": data_results.get("total_columns", 0),
            "columns": data_results.get("columns", []),
            "quality_level": data_results.get("quality_level", "not_assessed"),
            "variable_mappings": data_results.get("variable_mappings", []),
        }
    elif hasattr(data_results, "total_rows"):
        return {
            "total_rows": data_results.total_rows,
            "total_columns": data_results.total_columns,
            "columns": [c.model_dump() if hasattr(c, "model_dump") else c for c in data_results.columns],
            "quality_level": data_results.quality_level.value if hasattr(data_results.quality_level, "value") else str(data_results.quality_level),
            "variable_mappings": [v.model_dump() if hasattr(v, "model_dump") else v for v in data_results.variable_mappings],
        }
    
    return {"total_rows": 0, "total_columns": 0, "columns": [], "quality_level": "not_assessed"}


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


def _generate_descriptive_analysis(
    data_info: dict[str, Any],
    plan_info: dict[str, Any],
) -> dict[str, Any]:
    """Generate descriptive statistics summary."""
    columns = data_info.get("columns", [])
    key_vars = plan_info.get("key_variables", [])
    
    # Filter to key variables if specified
    if key_vars:
        columns = [c for c in columns if c.get("name") in key_vars]
    
    descriptive = {}
    for col in columns:
        col_name = col.get("name", "unknown")
        dtype = col.get("dtype", "")
        
        if dtype in ["numeric", "integer", "float"]:
            descriptive[col_name] = {
                "mean": col.get("mean"),
                "std": col.get("std"),
                "min": col.get("min_value"),
                "max": col.get("max_value"),
                "median": col.get("median"),
                "n": col.get("non_null_count", 0),
            }
    
    return descriptive


def _execute_planned_analysis(
    data_info: dict[str, Any],
    plan_info: dict[str, Any],
    gap_info: dict[str, Any],
    research_question: str,
) -> tuple[list[DataAnalysisFinding], list[StatisticalResult], list[RegressionResult]]:
    """Execute the analysis specified in the research plan."""
    findings = []
    stat_results = []
    reg_results = []
    
    methodology = plan_info.get("methodology_type", "")
    analysis_approach = plan_info.get("analysis_approach", "")
    key_vars = plan_info.get("key_variables", [])
    control_vars = plan_info.get("control_variables", [])
    hypothesis = plan_info.get("hypothesis")
    
    # Execute regression if appropriate
    if methodology in ["regression_analysis", "panel_data", "ols"] or analysis_approach in ["ols_regression", "fixed_effects"]:
        if len(key_vars) >= 2:
            dependent = key_vars[0]
            independent = key_vars[1:]
            
            reg_result = execute_regression_analysis.invoke({
                "model_type": analysis_approach or "ols",
                "dependent_variable": dependent,
                "independent_variables": independent,
                "control_variables": control_vars,
                "data_info": data_info,
            })
            reg_results.append(reg_result)
            
            # Generate finding from regression
            sig_vars = [c.variable for c in reg_result.coefficients if c.is_significant and c.variable != "(Intercept)"]
            if sig_vars:
                finding = DataAnalysisFinding(
                    finding_type=FindingType.MAIN_RESULT,
                    statement=f"Regression analysis reveals significant relationships between {', '.join(sig_vars)} and {dependent}.",
                    detailed_description=reg_result.interpretation,
                    regression_results=[reg_result],
                    evidence_strength=EvidenceStrength.STRONG if reg_result.r_squared > 0.3 else EvidenceStrength.MODERATE,
                    addresses_research_question=True,
                    addresses_gap=True,
                    confidence_level=0.8,
                )
                findings.append(finding)
    
    # Execute hypothesis test if hypothesis provided
    if hypothesis and len(key_vars) >= 1:
        stat_result = execute_hypothesis_test.invoke({
            "test_type": "t_test",
            "hypothesis": hypothesis,
            "variables": key_vars[:2],
        })
        stat_results.append(stat_result)
        
        # Generate finding from hypothesis test
        finding = DataAnalysisFinding(
            finding_type=FindingType.MAIN_RESULT if stat_result.is_significant else FindingType.NULL_RESULT,
            statement=(
                f"Hypothesis test {'supports' if stat_result.is_significant else 'does not support'} "
                f"the proposed hypothesis (p={stat_result.p_value:.4f})."
            ),
            detailed_description=stat_result.interpretation,
            statistical_results=[stat_result],
            evidence_strength=EvidenceStrength.STRONG if stat_result.is_significant else EvidenceStrength.WEAK,
            addresses_research_question=True,
            addresses_gap=stat_result.is_significant,
            confidence_level=0.9 if stat_result.is_significant else 0.5,
        )
        findings.append(finding)
    
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
    
    return findings, stat_results, reg_results


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


# =============================================================================
# Main Node Function
# =============================================================================


def data_analyst_node(state: WorkflowState) -> dict:
    """
    Execute data analysis per research plan.
    
    This node:
    1. Extracts methodology and analysis approach from research plan
    2. Executes appropriate statistical analyses
    3. Generates findings with statistical backing
    4. Links findings to research question
    5. Assesses whether findings address the identified gap
    
    Args:
        state: Current workflow state.
        
    Returns:
        Updated state with analysis results.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Extract information from state
    data_info = _extract_data_info(state)
    plan_info = _extract_plan_info(state)
    gap_info = _extract_gap_info(state)
    research_question = _get_research_question(state)
    
    # Validate we have data to analyze
    if data_info.get("total_rows", 0) == 0:
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
    
    # Generate descriptive statistics
    descriptive_stats = _generate_descriptive_analysis(data_info, plan_info)
    
    # Execute planned analysis
    findings, stat_results, reg_results = _execute_planned_analysis(
        data_info, plan_info, gap_info, research_question
    )
    
    # Assess gap coverage
    gap_addressed, coverage_score, coverage_explanation = _assess_gap_addressed(
        findings, gap_info, research_question
    )
    
    # Determine hypothesis support
    hypothesis_supported = None
    hypothesis_summary = ""
    if plan_info.get("hypothesis"):
        # Check if any significant findings support the hypothesis
        significant = [f for f in findings if f.evidence_strength == EvidenceStrength.STRONG]
        hypothesis_supported = len(significant) > 0
        hypothesis_summary = (
            f"Hypothesis {'supported' if hypothesis_supported else 'not supported'} by "
            f"{len(significant)} significant finding(s)."
        )
    
    # Identify main findings
    main_findings = [f for f in findings if f.finding_type == FindingType.MAIN_RESULT]
    
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
        data_summary=f"Analyzed {data_info.get('total_rows', 0)} observations across {len(plan_info.get('key_variables', []))} key variables.",
        sample_size=data_info.get("total_rows", 0),
        variables_analyzed=plan_info.get("key_variables", []),
        descriptive_stats=descriptive_stats,
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
            "Statistical assumptions may require further validation",
        ],
    )
    
    # Build summary message
    summary_parts = [
        f"[{current_date}] DATA_ANALYST: Analysis complete.",
        f"Sample size: {analysis_result.sample_size}",
        f"Findings: {len(findings)} ({len(main_findings)} main)",
        f"Gap addressed: {'Yes' if gap_addressed else 'Partially'} (score: {coverage_score:.2f})",
    ]
    if hypothesis_summary:
        summary_parts.append(f"Hypothesis: {hypothesis_summary}")
    
    return {
        "status": ResearchStatus.ANALYSIS_COMPLETE,
        "analysis": analysis_result.model_dump(),
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
    
    if state.get("analysis"):
        return "writer"
    
    return "__end__"
