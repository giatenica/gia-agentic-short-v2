"""Methodology selection and validation tools for the PLANNER node.

This module provides tools for selecting appropriate research methodologies
based on the research type, identified gaps, available data, and literature
precedents.
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
# Methodology Selection Mappings
# =============================================================================

# Map research types to appropriate methodologies
RESEARCH_TYPE_METHODOLOGIES: dict[str, list[MethodologyType]] = {
    "empirical": [
        MethodologyType.REGRESSION_ANALYSIS,
        MethodologyType.PANEL_DATA,
        MethodologyType.EVENT_STUDY,
        MethodologyType.DIFFERENCE_IN_DIFFERENCES,
        MethodologyType.INSTRUMENTAL_VARIABLES,
        MethodologyType.PROPENSITY_SCORE_MATCHING,
        MethodologyType.TIME_SERIES,
        MethodologyType.CROSS_SECTIONAL,
    ],
    "theoretical": [
        MethodologyType.ANALYTICAL_MODEL,
        MethodologyType.SIMULATION,
        MethodologyType.CONCEPTUAL_FRAMEWORK,
    ],
    "mixed": [
        MethodologyType.SEQUENTIAL_MIXED,
        MethodologyType.CONCURRENT_MIXED,
    ],
    "literature_review": [
        MethodologyType.SYSTEMATIC_REVIEW,
        MethodologyType.META_ANALYSIS,
        MethodologyType.NARRATIVE_REVIEW,
    ],
    "case_study": [
        MethodologyType.CASE_STUDY,
        MethodologyType.CONTENT_ANALYSIS,
        MethodologyType.THEMATIC_ANALYSIS,
    ],
}

# Map gap types to preferred methodologies
GAP_TYPE_METHODOLOGIES: dict[str, list[MethodologyType]] = {
    "methodological": [
        MethodologyType.ANALYTICAL_MODEL,
        MethodologyType.SIMULATION,
        MethodologyType.INSTRUMENTAL_VARIABLES,
        MethodologyType.DIFFERENCE_IN_DIFFERENCES,
    ],
    "empirical": [
        MethodologyType.REGRESSION_ANALYSIS,
        MethodologyType.PANEL_DATA,
        MethodologyType.EVENT_STUDY,
        MethodologyType.CROSS_SECTIONAL,
    ],
    "theoretical": [
        MethodologyType.CONCEPTUAL_FRAMEWORK,
        MethodologyType.ANALYTICAL_MODEL,
        MethodologyType.GROUNDED_THEORY,
    ],
    "contextual": [
        MethodologyType.CASE_STUDY,
        MethodologyType.CROSS_SECTIONAL,
        MethodologyType.PANEL_DATA,
    ],
    "temporal": [
        MethodologyType.TIME_SERIES,
        MethodologyType.PANEL_DATA,
        MethodologyType.EVENT_STUDY,
    ],
    "conflicting": [
        MethodologyType.META_ANALYSIS,
        MethodologyType.SYSTEMATIC_REVIEW,
        MethodologyType.REPLICATION,
    ],
}

# Map methodologies to appropriate analysis approaches
METHODOLOGY_ANALYSES: dict[MethodologyType, list[AnalysisApproach]] = {
    MethodologyType.REGRESSION_ANALYSIS: [
        AnalysisApproach.OLS_REGRESSION,
        AnalysisApproach.FIXED_EFFECTS,
        AnalysisApproach.RANDOM_EFFECTS,
    ],
    MethodologyType.PANEL_DATA: [
        AnalysisApproach.FIXED_EFFECTS,
        AnalysisApproach.RANDOM_EFFECTS,
        AnalysisApproach.GMM,
    ],
    MethodologyType.EVENT_STUDY: [
        AnalysisApproach.ASSET_PRICING_TESTS,
        AnalysisApproach.OLS_REGRESSION,
        AnalysisApproach.MULTIVARIATE_ANALYSIS,
    ],
    MethodologyType.DIFFERENCE_IN_DIFFERENCES: [
        AnalysisApproach.FIXED_EFFECTS,
        AnalysisApproach.OLS_REGRESSION,
    ],
    MethodologyType.INSTRUMENTAL_VARIABLES: [
        AnalysisApproach.TWO_STAGE_LEAST_SQUARES,
        AnalysisApproach.GMM,
    ],
    MethodologyType.TIME_SERIES: [
        AnalysisApproach.MULTIVARIATE_ANALYSIS,
        AnalysisApproach.INFERENTIAL_STATISTICS,
    ],
    MethodologyType.ANALYTICAL_MODEL: [
        AnalysisApproach.OPTION_PRICING_MODELS,
        AnalysisApproach.ASSET_PRICING_TESTS,
    ],
    MethodologyType.CASE_STUDY: [
        AnalysisApproach.CODING_ANALYSIS,
        AnalysisApproach.NARRATIVE_ANALYSIS,
    ],
    MethodologyType.META_ANALYSIS: [
        AnalysisApproach.MULTIVARIATE_ANALYSIS,
        AnalysisApproach.DESCRIPTIVE_STATISTICS,
    ],
}


# =============================================================================
# Methodology Selection Functions
# =============================================================================


def select_methodology(
    research_type: str,
    gap_type: str,
    has_data: bool = False,
    precedents: list[str] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Select an appropriate methodology based on research context.
    
    Considers:
    - Research type (empirical, theoretical, mixed, etc.)
    - Gap type being addressed (methodological, empirical, theoretical)
    - Whether data is available
    - Methodology precedents from literature
    
    Args:
        research_type: Type of research (empirical, theoretical, etc.)
        gap_type: Type of gap being addressed
        has_data: Whether user has data available
        precedents: Methodology precedents from literature
        model_name: Model to use for selection reasoning
        
    Returns:
        Dictionary with selected methodology and justification.
    """
    # Get candidate methodologies based on research type
    research_candidates = RESEARCH_TYPE_METHODOLOGIES.get(
        research_type.lower(), 
        list(MethodologyType)
    )
    
    # Get preferred methodologies based on gap type
    gap_candidates = GAP_TYPE_METHODOLOGIES.get(
        gap_type.lower(),
        []
    )
    
    # Find intersection or use research type candidates
    if gap_candidates:
        common = [m for m in research_candidates if m in gap_candidates]
        candidates = common if common else research_candidates
    else:
        candidates = research_candidates
    
    # Filter based on data availability
    if not has_data:
        # Remove data-intensive methodologies
        data_intensive = [
            MethodologyType.REGRESSION_ANALYSIS,
            MethodologyType.PANEL_DATA,
            MethodologyType.EVENT_STUDY,
            MethodologyType.TIME_SERIES,
            MethodologyType.CROSS_SECTIONAL,
        ]
        candidates = [m for m in candidates if m not in data_intensive] or candidates
    
    # Use LLM to select best methodology with reasoning
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2000,
        api_key=settings.anthropic_api_key,
    )
    
    system_prompt = """You are an expert academic research methodology advisor.
Select the most appropriate research methodology and provide justification.

Consider:
1. The research type and gap type
2. Whether data is available
3. Precedents from existing literature
4. Methodological rigor and feasibility

Respond in this exact format:
METHODOLOGY: [name of methodology]
JUSTIFICATION: [2-3 sentences explaining why this is the best choice]
ANALYSIS_APPROACH: [recommended analysis approach]
KEY_CONSIDERATIONS: [bullet points of important considerations]"""

    human_prompt = f"""Select the best methodology for this research:

Research Type: {research_type}
Gap Type: {gap_type}
Data Available: {has_data}
Literature Precedents: {', '.join(precedents or ['None noted'])}

Candidate Methodologies:
{chr(10).join(f'- {m.value}' for m in candidates[:8])}

Select the most appropriate methodology and explain your choice."""

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])
    
    # Parse response
    content = response.content
    
    # Extract methodology
    methodology = candidates[0]  # Default
    for candidate in candidates:
        if candidate.value.lower() in content.lower():
            methodology = candidate
            break
    
    # Extract justification
    justification = ""
    if "JUSTIFICATION:" in content:
        parts = content.split("JUSTIFICATION:")
        if len(parts) > 1:
            justification = parts[1].split("ANALYSIS_APPROACH:")[0].strip()
    
    # Get matching analysis approaches
    analysis_approaches = METHODOLOGY_ANALYSES.get(
        methodology, 
        [AnalysisApproach.OTHER]
    )
    
    return {
        "methodology_type": methodology,
        "methodology_name": methodology.value,
        "justification": justification or f"Selected {methodology.value} based on {research_type} research addressing {gap_type} gap.",
        "analysis_approaches": [a.value for a in analysis_approaches],
        "recommended_analysis": analysis_approaches[0].value if analysis_approaches else "other",
        "full_response": content,
    }


def validate_methodology_fit(
    methodology_type: MethodologyType,
    gap_type: str,
    research_type: str,
    has_data: bool,
) -> dict[str, Any]:
    """
    Validate if a methodology fits the research context.
    
    Args:
        methodology_type: The methodology to validate
        gap_type: Type of gap being addressed
        research_type: Type of research
        has_data: Whether data is available
        
    Returns:
        Validation result with score and issues.
    """
    issues = []
    score = 1.0
    
    # Check gap type fit
    gap_methodologies = GAP_TYPE_METHODOLOGIES.get(gap_type.lower(), [])
    if gap_methodologies and methodology_type not in gap_methodologies:
        issues.append(f"Methodology may not be ideal for {gap_type} gaps")
        score -= 0.2
    
    # Check research type fit
    research_methodologies = RESEARCH_TYPE_METHODOLOGIES.get(research_type.lower(), [])
    if research_methodologies and methodology_type not in research_methodologies:
        issues.append(f"Methodology not typical for {research_type} research")
        score -= 0.2
    
    # Check data requirements
    data_required = [
        MethodologyType.REGRESSION_ANALYSIS,
        MethodologyType.PANEL_DATA,
        MethodologyType.EVENT_STUDY,
        MethodologyType.TIME_SERIES,
        MethodologyType.CROSS_SECTIONAL,
        MethodologyType.DIFFERENCE_IN_DIFFERENCES,
        MethodologyType.INSTRUMENTAL_VARIABLES,
        MethodologyType.META_ANALYSIS,
    ]
    if methodology_type in data_required and not has_data:
        issues.append("Methodology requires data but none available")
        score -= 0.4
    
    return {
        "is_valid": score >= 0.6,
        "score": max(0, score),
        "issues": issues,
        "methodology": methodology_type.value,
    }


def assess_feasibility(
    methodology_type: MethodologyType,
    data_available: dict[str, Any] | None = None,
    time_constraints: str | None = None,
    resource_constraints: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Assess feasibility of the research plan.
    
    Args:
        methodology_type: The selected methodology
        data_available: Information about available data
        time_constraints: Time constraints (deadline)
        resource_constraints: Resource constraints
        model_name: Model to use for assessment
        
    Returns:
        Feasibility assessment with score and notes.
    """
    model = ChatAnthropic(
        model=model_name or "claude-3-5-haiku-latest",
        temperature=0,
        max_tokens=1500,
        api_key=settings.anthropic_api_key,
    )
    
    system_prompt = """You are an expert research feasibility assessor.
Evaluate the feasibility of the proposed research methodology.

Consider:
1. Data availability and quality
2. Time constraints
3. Resource constraints
4. Methodological complexity
5. Potential challenges

Respond with:
FEASIBILITY_SCORE: [0.0 to 1.0]
ASSESSMENT: [2-3 sentences overall assessment]
CHALLENGES: [bullet list of potential challenges]
RECOMMENDATIONS: [bullet list of recommendations to improve feasibility]"""

    data_info = "No data available"
    if data_available:
        data_info = f"""
- Files: {len(data_available.get('files_analyzed', []))}
- Total rows: {data_available.get('total_rows', 'Unknown')}
- Quality: {data_available.get('quality_level', 'Not assessed')}
"""

    human_prompt = f"""Assess feasibility of this research:

Methodology: {methodology_type.value}

Data Available:
{data_info}

Time Constraints: {time_constraints or 'None specified'}
Resource Constraints: {resource_constraints or 'None specified'}

Provide a feasibility assessment."""

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])
    
    content = response.content
    
    # Extract score
    score = 0.7  # Default moderate feasibility
    if "FEASIBILITY_SCORE:" in content:
        try:
            score_str = content.split("FEASIBILITY_SCORE:")[1].split()[0]
            score = float(score_str.strip())
            score = max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            # Parsing failed; use default score (0.7)
            pass
    
    # Extract assessment
    assessment = ""
    if "ASSESSMENT:" in content:
        assessment = content.split("ASSESSMENT:")[1].split("CHALLENGES:")[0].strip()
    
    return {
        "feasibility_score": score,
        "is_feasible": score >= 0.5,
        "assessment": assessment or "Feasibility assessment completed.",
        "full_response": content,
        "methodology": methodology_type.value,
    }


def explain_methodology_choice(
    methodology_type: MethodologyType,
    gap_analysis: dict[str, Any],
    literature_synthesis: dict[str, Any],
    model_name: str | None = None,
) -> str:
    """
    Generate a detailed explanation of methodology choice.
    
    Justifies the selection with citations to literature precedents.
    
    Args:
        methodology_type: The selected methodology
        gap_analysis: Gap analysis results
        literature_synthesis: Literature synthesis results
        model_name: Model to use for explanation
        
    Returns:
        Detailed methodology justification.
    """
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2000,
        api_key=settings.anthropic_api_key,
    )
    
    # Extract methodology precedents from synthesis
    precedents = literature_synthesis.get("methodological_approaches", [])
    key_papers = literature_synthesis.get("key_findings", [])[:5]
    
    system_prompt = """You are an expert academic methodology advisor.
Write a concise but thorough justification for the methodology choice.

The justification should:
1. Reference relevant literature precedents
2. Explain why this method addresses the identified gap
3. Acknowledge limitations and how they will be addressed
4. Be written in formal academic prose

Keep the response to 2-3 paragraphs."""

    human_prompt = f"""Justify this methodology choice:

Selected Methodology: {methodology_type.value}

Research Gap:
- Type: {gap_analysis.get('primary_gap', {}).get('gap_type', 'Unknown')}
- Description: {gap_analysis.get('primary_gap', {}).get('description', 'Not specified')[:500]}

Methodology Precedents from Literature:
{chr(10).join(f'- {p}' for p in precedents[:5]) or 'No specific precedents noted'}

Key Related Papers:
{chr(10).join(f'- {p}' for p in key_papers[:5]) or 'No papers noted'}

Write a methodology justification suitable for an academic paper."""

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])
    
    return response.content


# =============================================================================
# Tool Definitions for LangGraph
# =============================================================================


@tool
def select_research_methodology(
    research_type: str,
    gap_type: str,
    has_data: bool = False,
) -> dict[str, Any]:
    """
    Select an appropriate research methodology.
    
    Args:
        research_type: Type of research (empirical, theoretical, mixed, etc.)
        gap_type: Type of gap being addressed (methodological, empirical, theoretical)
        has_data: Whether the researcher has data available
        
    Returns:
        Selected methodology with justification.
    """
    return select_methodology(
        research_type=research_type,
        gap_type=gap_type,
        has_data=has_data,
    )


@tool
def validate_methodology(
    methodology: str,
    gap_type: str,
    research_type: str,
    has_data: bool,
) -> dict[str, Any]:
    """
    Validate if a methodology fits the research context.
    
    Args:
        methodology: The methodology name to validate
        gap_type: Type of gap being addressed
        research_type: Type of research
        has_data: Whether data is available
        
    Returns:
        Validation result with score and issues.
    """
    try:
        method_type = MethodologyType(methodology)
    except ValueError:
        return {
            "is_valid": False,
            "score": 0.0,
            "issues": [f"Unknown methodology: {methodology}"],
            "methodology": methodology,
        }
    
    return validate_methodology_fit(
        methodology_type=method_type,
        gap_type=gap_type,
        research_type=research_type,
        has_data=has_data,
    )


@tool
def assess_research_feasibility(
    methodology: str,
    has_data: bool = False,
    deadline: str | None = None,
) -> dict[str, Any]:
    """
    Assess feasibility of the research methodology.
    
    Args:
        methodology: The methodology to assess
        has_data: Whether data is available
        deadline: Project deadline if any
        
    Returns:
        Feasibility assessment.
    """
    try:
        method_type = MethodologyType(methodology)
    except ValueError:
        method_type = MethodologyType.OTHER
    
    data_info = {"data_available": has_data} if has_data else None
    
    return assess_feasibility(
        methodology_type=method_type,
        data_available=data_info,
        time_constraints=deadline,
    )
