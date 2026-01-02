"""Analysis design tools for the PLANNER node.

This module provides tools for designing research analysis approaches,
including statistical methods, variable selection, and output planning.
"""

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.config import settings
from src.state.enums import (
    MethodologyType,
    AnalysisApproach,
)


# =============================================================================
# Paper Section Templates
# =============================================================================

PAPER_SECTIONS_BY_TYPE: dict[str, list[str]] = {
    "short_article": [
        "Abstract",
        "Introduction",
        "Data and Methodology",
        "Results",
        "Discussion and Conclusion",
        "References",
    ],
    "full_paper": [
        "Abstract",
        "Introduction",
        "Literature Review",
        "Hypothesis Development",
        "Data",
        "Methodology",
        "Empirical Results",
        "Robustness Tests",
        "Discussion",
        "Conclusion",
        "References",
        "Appendix",
    ],
    "theoretical": [
        "Abstract",
        "Introduction",
        "Model Setup",
        "Theoretical Analysis",
        "Model Implications",
        "Discussion",
        "Conclusion",
        "References",
        "Proofs (Appendix)",
    ],
    "literature_review": [
        "Abstract",
        "Introduction",
        "Methodology",
        "Literature Overview",
        "Thematic Analysis",
        "Synthesis and Gaps",
        "Future Research Directions",
        "Conclusion",
        "References",
    ],
    "case_study": [
        "Abstract",
        "Introduction",
        "Background",
        "Case Description",
        "Analysis",
        "Discussion",
        "Implications",
        "Conclusion",
        "References",
    ],
}

# Statistical tests by methodology
STATISTICAL_TESTS_BY_METHODOLOGY: dict[MethodologyType, list[str]] = {
    MethodologyType.REGRESSION_ANALYSIS: [
        "OLS regression",
        "Heteroscedasticity tests (White, Breusch-Pagan)",
        "Multicollinearity tests (VIF)",
        "Normality tests (Jarque-Bera)",
        "F-tests for joint significance",
        "t-tests for coefficient significance",
    ],
    MethodologyType.PANEL_DATA: [
        "Hausman test (fixed vs random effects)",
        "F-test for fixed effects",
        "Breusch-Pagan LM test",
        "Cluster-robust standard errors",
        "Serial correlation tests",
        "Unit root tests",
    ],
    MethodologyType.EVENT_STUDY: [
        "Abnormal return calculation",
        "Cumulative abnormal returns (CAR)",
        "Cross-sectional t-tests",
        "Sign tests",
        "Generalized sign tests",
        "Rank tests",
    ],
    MethodologyType.DIFFERENCE_IN_DIFFERENCES: [
        "Parallel trends test",
        "Placebo tests",
        "Triple difference (DDD)",
        "Cluster-robust standard errors",
        "Synthetic control comparison",
    ],
    MethodologyType.INSTRUMENTAL_VARIABLES: [
        "First-stage F-statistic",
        "Weak instrument tests",
        "Overidentification tests (Sargan/Hansen)",
        "Endogeneity tests (Hausman/Wu)",
        "Reduced form estimates",
    ],
    MethodologyType.TIME_SERIES: [
        "Unit root tests (ADF, PP, KPSS)",
        "Cointegration tests (Johansen, Engle-Granger)",
        "Granger causality tests",
        "Autocorrelation tests (Durbin-Watson, LM)",
        "ARCH/GARCH tests",
    ],
    MethodologyType.META_ANALYSIS: [
        "Heterogeneity tests (Q-statistic, I²)",
        "Publication bias tests (Funnel plot, Egger's test)",
        "Sensitivity analysis",
        "Subgroup analysis",
        "Meta-regression",
    ],
}


# =============================================================================
# Analysis Design Functions
# =============================================================================


def design_quantitative_analysis(
    methodology_type: MethodologyType,
    research_question: str,
    key_variables: list[str] | None = None,
    data_info: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Design a quantitative analysis approach.
    
    Args:
        methodology_type: The selected methodology
        research_question: The research question to address
        key_variables: Key variables identified
        data_info: Information about available data
        model_name: Model to use for design
        
    Returns:
        Quantitative analysis design.
    """
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2500,
        api_key=settings.anthropic_api_key,
    )
    
    # Get relevant statistical tests
    tests = STATISTICAL_TESTS_BY_METHODOLOGY.get(
        methodology_type,
        ["Descriptive statistics", "Standard hypothesis tests"]
    )
    
    system_prompt = """You are an expert quantitative research methodologist.
Design a comprehensive quantitative analysis plan.

Your response should include:
1. DEPENDENT_VARIABLE: The main outcome variable
2. INDEPENDENT_VARIABLES: Key explanatory variables
3. CONTROL_VARIABLES: Variables to control for
4. MODEL_SPECIFICATION: The main regression/analysis model
5. ROBUSTNESS_TESTS: Additional tests for robustness
6. EXPECTED_TABLES: Tables to produce
7. EXPECTED_FIGURES: Figures to produce

Be specific and practical. Use academic finance conventions."""

    human_prompt = f"""Design a quantitative analysis for:

Research Question: {research_question}

Methodology: {methodology_type.value}

Key Variables Identified: {', '.join(key_variables or ['Not specified'])}

Data Information:
{_format_data_info(data_info)}

Relevant Statistical Tests:
{chr(10).join(f'- {t}' for t in tests)}

Design the analysis approach."""

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])
    
    content = response.content
    
    # Parse response sections
    result = {
        "analysis_type": "quantitative",
        "methodology": methodology_type.value,
        "statistical_tests": tests,
        "full_design": content,
    }
    
    # Extract structured components
    for field in ["DEPENDENT_VARIABLE", "MODEL_SPECIFICATION"]:
        if f"{field}:" in content:
            value = content.split(f"{field}:")[1].split("\n")[0].strip()
            result[field.lower()] = value
    
    # Extract lists
    for field in ["INDEPENDENT_VARIABLES", "CONTROL_VARIABLES", "ROBUSTNESS_TESTS", "EXPECTED_TABLES", "EXPECTED_FIGURES"]:
        if f"{field}:" in content:
            section = content.split(f"{field}:")[1]
            # Get until next section or end
            next_sections = ["DEPENDENT", "INDEPENDENT", "CONTROL", "MODEL", "ROBUSTNESS", "EXPECTED", "\n\n"]
            end_idx = len(section)
            for ns in next_sections:
                idx = section.find(ns)
                if idx > 0 and idx < end_idx:
                    end_idx = idx
            items = section[:end_idx].strip().split("\n")
            result[field.lower()] = [
                i.strip().lstrip("-•*").strip() 
                for i in items 
                if i.strip() and not i.strip().startswith(tuple(next_sections))
            ]
    
    return result


def design_qualitative_analysis(
    methodology_type: MethodologyType,
    research_question: str,
    data_sources: list[str] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Design a qualitative analysis approach.
    
    Args:
        methodology_type: The selected methodology
        research_question: The research question to address
        data_sources: Sources of qualitative data
        model_name: Model to use for design
        
    Returns:
        Qualitative analysis design.
    """
    model = ChatAnthropic(
        model=model_name or "claude-3-5-haiku-latest",
        temperature=0,
        max_tokens=2000,
        api_key=settings.anthropic_api_key,
    )
    
    system_prompt = """You are an expert qualitative research methodologist.
Design a comprehensive qualitative analysis plan.

Your response should include:
1. DATA_COLLECTION: How data will be gathered
2. CODING_APPROACH: How data will be coded/analyzed
3. THEMES_TO_EXPLORE: Initial themes to investigate
4. VALIDITY_MEASURES: How to ensure validity
5. PRESENTATION_FORMAT: How findings will be presented

Be specific and practical."""

    human_prompt = f"""Design a qualitative analysis for:

Research Question: {research_question}
Methodology: {methodology_type.value}
Data Sources: {', '.join(data_sources or ['Not specified'])}

Design the analysis approach."""

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])
    
    return {
        "analysis_type": "qualitative",
        "methodology": methodology_type.value,
        "full_design": response.content,
    }


def design_mixed_methods(
    research_question: str,
    quantitative_component: dict[str, Any],
    qualitative_component: dict[str, Any],
    integration_strategy: str = "sequential",
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Design a mixed methods analysis approach.
    
    Args:
        research_question: The research question
        quantitative_component: Quantitative analysis design
        qualitative_component: Qualitative analysis design
        integration_strategy: How to integrate (sequential, concurrent)
        model_name: Model to use for design
        
    Returns:
        Mixed methods analysis design.
    """
    return {
        "analysis_type": "mixed_methods",
        "integration_strategy": integration_strategy,
        "quantitative": quantitative_component,
        "qualitative": qualitative_component,
        "integration_points": [
            "Data collection phase",
            "Analysis phase",
            "Interpretation phase",
        ],
    }


def map_variables_to_analysis(
    research_question: str,
    available_variables: list[str],
    methodology_type: MethodologyType,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Map available variables to analysis components.
    
    Args:
        research_question: The research question
        available_variables: List of available variables
        methodology_type: The methodology being used
        model_name: Model to use for mapping
        
    Returns:
        Variable mapping for analysis.
    """
    model = ChatAnthropic(
        model=model_name or "claude-3-5-haiku-latest",
        temperature=0,
        max_tokens=1500,
        api_key=settings.anthropic_api_key,
    )
    
    system_prompt = """You are a research design expert.
Map the available variables to analysis roles.

Respond with:
DEPENDENT: [variable(s) that measure the outcome]
INDEPENDENT: [main explanatory variables]
CONTROL: [control variables]
INSTRUMENT: [potential instruments if applicable]
UNMAPPED: [variables that don't fit current analysis]"""

    human_prompt = f"""Map these variables for the analysis:

Research Question: {research_question}
Methodology: {methodology_type.value}

Available Variables:
{chr(10).join(f'- {v}' for v in available_variables)}

Assign each variable to its analysis role."""

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])
    
    content = response.content
    
    result = {
        "dependent": [],
        "independent": [],
        "control": [],
        "instrument": [],
        "unmapped": [],
    }
    
    for role in result.keys():
        key = role.upper()
        if f"{key}:" in content:
            section = content.split(f"{key}:")[1].split("\n")[0]
            vars_str = section.strip()
            # Parse comma or newline separated
            vars_list = [v.strip() for v in vars_str.replace(",", "\n").split("\n") if v.strip()]
            result[role] = vars_list
    
    return result


def determine_paper_sections(
    paper_type: str,
    methodology_type: MethodologyType,
    research_type: str = "empirical",
) -> list[str]:
    """
    Determine the expected sections for the paper.
    
    Args:
        paper_type: Type of paper (short_article, full_paper, etc.)
        methodology_type: The methodology being used
        research_type: Type of research (empirical, theoretical)
        
    Returns:
        List of expected paper sections.
    """
    # Get base sections by paper type
    sections = PAPER_SECTIONS_BY_TYPE.get(
        paper_type.lower().replace(" ", "_"),
        PAPER_SECTIONS_BY_TYPE["full_paper"]
    )
    
    # Adjust for theoretical papers
    if research_type.lower() == "theoretical":
        sections = PAPER_SECTIONS_BY_TYPE.get("theoretical", sections)
    
    # Adjust for literature reviews
    if methodology_type in [MethodologyType.SYSTEMATIC_REVIEW, MethodologyType.META_ANALYSIS, MethodologyType.NARRATIVE_REVIEW]:
        sections = PAPER_SECTIONS_BY_TYPE.get("literature_review", sections)
    
    return sections


def define_success_criteria(
    gap_analysis: dict[str, Any],
    methodology_type: MethodologyType,
    research_question: str,
    model_name: str | None = None,
) -> list[str]:
    """
    Define success criteria for the research.
    
    Args:
        gap_analysis: Gap analysis results
        methodology_type: The methodology being used
        research_question: The research question
        model_name: Model to use for criteria generation
        
    Returns:
        List of success criteria.
    """
    model = ChatAnthropic(
        model=model_name or "claude-3-5-haiku-latest",
        temperature=0,
        max_tokens=1500,
        api_key=settings.anthropic_api_key,
    )
    
    primary_gap = gap_analysis.get("primary_gap", {})
    gap_type = primary_gap.get("gap_type", "empirical")
    gap_description = primary_gap.get("description", "")
    
    system_prompt = """You are a research planning expert.
Define clear, measurable success criteria for this research project.

Provide 4-6 specific success criteria that:
1. Are measurable/verifiable
2. Address the identified gap
3. Are achievable within the methodology
4. Lead to a publishable contribution

Format as a numbered list."""

    human_prompt = f"""Define success criteria for:

Research Question: {research_question}
Methodology: {methodology_type.value}
Gap Being Addressed: {gap_type} - {gap_description[:300]}

What criteria indicate successful completion?"""

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])
    
    # Parse numbered list
    content = response.content
    criteria = []
    for line in content.split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            # Remove number/bullet prefix
            criterion = line.lstrip("0123456789.-) ").strip()
            if criterion and len(criterion) > 10:
                criteria.append(criterion)
    
    return criteria[:6] if criteria else [
        "Complete analysis addressing the identified research gap",
        "Produce statistically significant results (if applicable)",
        "Generate findings suitable for peer-reviewed publication",
        "Provide clear contribution to existing literature",
    ]


def _format_data_info(data_info: dict[str, Any] | None) -> str:
    """Format data information for prompts."""
    if not data_info:
        return "No data information provided"
    
    lines = []
    if data_info.get("total_rows"):
        lines.append(f"- Total rows: {data_info['total_rows']}")
    if data_info.get("total_columns"):
        lines.append(f"- Total columns: {data_info['total_columns']}")
    if data_info.get("quality_level"):
        lines.append(f"- Quality level: {data_info['quality_level']}")
    if data_info.get("columns"):
        col_names = [c.get("name", "") for c in data_info["columns"][:10]]
        lines.append(f"- Sample columns: {', '.join(col_names)}")
    
    return "\n".join(lines) if lines else "Basic data available"


# =============================================================================
# Tool Definitions for LangGraph
# =============================================================================


@tool
def design_analysis_approach(
    methodology: str,
    research_question: str,
    analysis_type: str = "quantitative",
) -> dict[str, Any]:
    """
    Design the analysis approach for the research.
    
    Args:
        methodology: The selected methodology
        research_question: The research question
        analysis_type: Type of analysis (quantitative, qualitative, mixed)
        
    Returns:
        Analysis design specification.
    """
    try:
        method_type = MethodologyType(methodology)
    except ValueError:
        method_type = MethodologyType.OTHER
    
    if analysis_type.lower() == "qualitative":
        return design_qualitative_analysis(
            methodology_type=method_type,
            research_question=research_question,
        )
    else:
        return design_quantitative_analysis(
            methodology_type=method_type,
            research_question=research_question,
        )


@tool
def get_paper_sections(
    paper_type: str,
    methodology: str,
    research_type: str = "empirical",
) -> list[str]:
    """
    Get the expected sections for the research paper.
    
    Args:
        paper_type: Type of paper (short_article, full_paper, etc.)
        methodology: The research methodology
        research_type: Type of research (empirical, theoretical)
        
    Returns:
        List of paper sections.
    """
    try:
        method_type = MethodologyType(methodology)
    except ValueError:
        method_type = MethodologyType.OTHER
    
    return determine_paper_sections(
        paper_type=paper_type,
        methodology_type=method_type,
        research_type=research_type,
    )


@tool
def get_success_criteria(
    research_question: str,
    methodology: str,
    gap_type: str = "empirical",
) -> list[str]:
    """
    Get success criteria for the research.
    
    Args:
        research_question: The research question
        methodology: The research methodology
        gap_type: Type of gap being addressed
        
    Returns:
        List of success criteria.
    """
    try:
        method_type = MethodologyType(methodology)
    except ValueError:
        method_type = MethodologyType.OTHER
    
    gap_analysis = {
        "primary_gap": {
            "gap_type": gap_type,
            "description": f"Research gap addressed by: {research_question}",
        }
    }
    
    return define_success_criteria(
        gap_analysis=gap_analysis,
        methodology_type=method_type,
        research_question=research_question,
    )
