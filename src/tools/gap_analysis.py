"""Gap analysis tools for identifying research gaps in literature.

This module provides tools for:
1. Comparing literature coverage against research questions
2. Identifying methodological, empirical, and theoretical gaps
3. Assessing gap significance and addressability
4. Generating gap rankings and recommendations
"""

from datetime import datetime, timezone
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from src.config import settings
from src.state.models import (
    GapAnalysis,
    LiteratureSynthesis,
    ResearchGap,
    SearchResult,
)


# =============================================================================
# Coverage Analysis
# =============================================================================


def compare_coverage(
    original_question: str,
    literature_synthesis: LiteratureSynthesis | dict[str, Any],
    search_results: list[SearchResult] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Compare what the literature covers vs. what the research question asks.
    
    Analyzes the alignment between the user's research question and
    what existing literature addresses. Identifies areas that are
    well-covered, partially covered, and not covered at all.
    
    Args:
        original_question: The user's original research question.
        literature_synthesis: Synthesis from LITERATURE_REVIEWER node.
        search_results: Optional list of search results for additional context.
        model_name: Model to use (defaults to settings.default_model).
        
    Returns:
        Dictionary containing:
        - coverage_summary: Summary of coverage
        - coverage_percentage: Estimated percentage covered
        - well_covered: Aspects well covered by literature
        - partially_covered: Aspects partially covered
        - not_covered: Aspects not covered (potential gaps)
    """
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2048,
        api_key=settings.anthropic_api_key,
    )
    
    # Handle both dict and model forms
    if isinstance(literature_synthesis, dict):
        synthesis_text = _format_synthesis_dict(literature_synthesis)
    else:
        synthesis_text = _format_synthesis_model(literature_synthesis)
    
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    prompt = f"""Current date: {current_date}

Analyze the coverage of the following research question by existing literature.

RESEARCH QUESTION:
{original_question}

LITERATURE SYNTHESIS:
{synthesis_text}

Provide your analysis in the following format:

COVERAGE_SUMMARY: <2-3 sentence summary of how well literature covers the question>

COVERAGE_PERCENTAGE: <estimated percentage 0-100>

WELL_COVERED:
- <aspect 1>
- <aspect 2>
...

PARTIALLY_COVERED:
- <aspect 1>: <what's missing>
- <aspect 2>: <what's missing>
...

NOT_COVERED:
- <aspect 1>: <why this is a gap>
- <aspect 2>: <why this is a gap>
...

Be specific and cite evidence from the literature synthesis where possible."""

    response = model.invoke([HumanMessage(content=prompt)])
    
    # Parse response
    result = _parse_coverage_response(response.content)
    return result


def _format_synthesis_dict(synthesis: dict[str, Any]) -> str:
    """Format a synthesis dictionary as text."""
    parts = []
    
    if synthesis.get("summary"):
        parts.append(f"Summary: {synthesis['summary']}")
    
    if synthesis.get("state_of_field"):
        parts.append(f"State of Field: {synthesis['state_of_field']}")
    
    if synthesis.get("key_findings"):
        findings = "\n".join(f"- {f}" for f in synthesis["key_findings"][:10])
        parts.append(f"Key Findings:\n{findings}")
    
    if synthesis.get("theoretical_frameworks"):
        frameworks = "\n".join(f"- {f}" for f in synthesis["theoretical_frameworks"][:5])
        parts.append(f"Theoretical Frameworks:\n{frameworks}")
    
    if synthesis.get("methodological_approaches"):
        methods = "\n".join(f"- {m}" for m in synthesis["methodological_approaches"][:5])
        parts.append(f"Methodological Approaches:\n{methods}")
    
    if synthesis.get("contribution_opportunities"):
        opps = "\n".join(f"- {o}" for o in synthesis["contribution_opportunities"][:5])
        parts.append(f"Contribution Opportunities:\n{opps}")
    
    return "\n\n".join(parts)


def _format_synthesis_model(synthesis: LiteratureSynthesis) -> str:
    """Format a LiteratureSynthesis model as text."""
    return _format_synthesis_dict(synthesis.model_dump())


def _parse_coverage_response(content: str) -> dict[str, Any]:
    """Parse the coverage analysis response."""
    result = {
        "coverage_summary": "",
        "coverage_percentage": 50.0,
        "well_covered": [],
        "partially_covered": [],
        "not_covered": [],
    }
    
    lines = content.split("\n")
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith("COVERAGE_SUMMARY:"):
            result["coverage_summary"] = line.replace("COVERAGE_SUMMARY:", "").strip()
            current_section = None
        elif line.startswith("COVERAGE_PERCENTAGE:"):
            try:
                pct_text = line.replace("COVERAGE_PERCENTAGE:", "").strip()
                pct_text = pct_text.replace("%", "").strip()
                result["coverage_percentage"] = float(pct_text)
            except ValueError:
                pass  # Keep default
            current_section = None
        elif line.startswith("WELL_COVERED:"):
            current_section = "well_covered"
        elif line.startswith("PARTIALLY_COVERED:"):
            current_section = "partially_covered"
        elif line.startswith("NOT_COVERED:"):
            current_section = "not_covered"
        elif line.startswith("- ") and current_section:
            item = line[2:].strip()
            if item:
                result[current_section].append(item)
    
    return result


# =============================================================================
# Gap Identification Tools
# =============================================================================


def identify_methodological_gaps(
    original_question: str,
    literature_synthesis: LiteratureSynthesis | dict[str, Any],
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Identify methodological gaps in the literature.
    
    Finds gaps where new methods, approaches, or techniques are needed
    to address the research question.
    
    Args:
        original_question: The user's original research question.
        literature_synthesis: Synthesis from LITERATURE_REVIEWER node.
        model_name: Model to use (defaults to settings.default_model).
        
    Returns:
        List of methodological gaps with descriptions and significance.
    """
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2048,
        api_key=settings.anthropic_api_key,
    )
    
    if isinstance(literature_synthesis, dict):
        synthesis_text = _format_synthesis_dict(literature_synthesis)
    else:
        synthesis_text = _format_synthesis_model(literature_synthesis)
    
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    prompt = f"""Current date: {current_date}

Identify METHODOLOGICAL GAPS in the literature for this research question.

Methodological gaps are areas where:
- Existing methods are inadequate or outdated
- New analytical approaches are needed
- Current techniques have limitations for this context
- Data collection methods need improvement
- Statistical/computational methods are lacking

RESEARCH QUESTION:
{original_question}

LITERATURE SYNTHESIS:
{synthesis_text}

For each gap identified, provide:

GAP: <short title>
DESCRIPTION: <detailed description of the methodological gap>
CURRENT_METHODS: <what methods are currently used>
LIMITATION: <why current methods are insufficient>
SIGNIFICANCE: <high/medium/low>
ADDRESSABLE: <yes/no>
HOW_TO_ADDRESS: <how this gap could be filled>

Identify up to 5 methodological gaps. If fewer exist, that's fine."""

    response = model.invoke([HumanMessage(content=prompt)])
    
    return _parse_gap_response(response.content, "methodological")


def identify_empirical_gaps(
    original_question: str,
    literature_synthesis: LiteratureSynthesis | dict[str, Any],
    data_context: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Identify empirical gaps in the literature.
    
    Finds gaps where evidence is lacking for certain contexts,
    populations, time periods, or settings.
    
    Args:
        original_question: The user's original research question.
        literature_synthesis: Synthesis from LITERATURE_REVIEWER node.
        data_context: Optional context about user's available data.
        model_name: Model to use (defaults to settings.default_model).
        
    Returns:
        List of empirical gaps with descriptions and significance.
    """
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2048,
        api_key=settings.anthropic_api_key,
    )
    
    if isinstance(literature_synthesis, dict):
        synthesis_text = _format_synthesis_dict(literature_synthesis)
    else:
        synthesis_text = _format_synthesis_model(literature_synthesis)
    
    data_info = ""
    if data_context:
        data_info = f"\n\nUSER'S DATA CONTEXT:\n{data_context}"
    
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    prompt = f"""Current date: {current_date}

Identify EMPIRICAL GAPS in the literature for this research question.

Empirical gaps are areas where:
- Evidence has not been collected for certain contexts
- Populations or groups have not been studied
- Geographic regions lack research
- Time periods have not been examined
- Settings or conditions have not been tested
- Sample sizes have been too small
- Data quality has been poor
{data_info}

RESEARCH QUESTION:
{original_question}

LITERATURE SYNTHESIS:
{synthesis_text}

For each gap identified, provide:

GAP: <short title>
DESCRIPTION: <detailed description of the empirical gap>
EXISTING_EVIDENCE: <what evidence exists>
MISSING_EVIDENCE: <what's missing>
CONTEXT: <specific context/population/setting lacking evidence>
SIGNIFICANCE: <high/medium/low>
ADDRESSABLE: <yes/no>
DATA_NEEDED: <what data would address this gap>

Identify up to 5 empirical gaps. If fewer exist, that's fine."""

    response = model.invoke([HumanMessage(content=prompt)])
    
    return _parse_gap_response(response.content, "empirical")


def identify_theoretical_gaps(
    original_question: str,
    literature_synthesis: LiteratureSynthesis | dict[str, Any],
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Identify theoretical gaps in the literature.
    
    Finds gaps where theoretical frameworks, explanations, or
    conceptualizations are missing or incomplete.
    
    Args:
        original_question: The user's original research question.
        literature_synthesis: Synthesis from LITERATURE_REVIEWER node.
        model_name: Model to use (defaults to settings.default_model).
        
    Returns:
        List of theoretical gaps with descriptions and significance.
    """
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2048,
        api_key=settings.anthropic_api_key,
    )
    
    if isinstance(literature_synthesis, dict):
        synthesis_text = _format_synthesis_dict(literature_synthesis)
    else:
        synthesis_text = _format_synthesis_model(literature_synthesis)
    
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    prompt = f"""Current date: {current_date}

Identify THEORETICAL GAPS in the literature for this research question.

Theoretical gaps are areas where:
- Phenomena lack adequate explanation
- Existing theories don't fully account for observations
- Conceptual frameworks are incomplete
- Mechanisms are not well understood
- Theoretical predictions need testing
- Integration of multiple theories is needed
- Boundary conditions are unclear

RESEARCH QUESTION:
{original_question}

LITERATURE SYNTHESIS:
{synthesis_text}

For each gap identified, provide:

GAP: <short title>
DESCRIPTION: <detailed description of the theoretical gap>
EXISTING_THEORY: <relevant existing theoretical frameworks>
LIMITATION: <why existing theory is insufficient>
UNEXPLAINED: <what phenomena remain unexplained>
SIGNIFICANCE: <high/medium/low>
ADDRESSABLE: <yes/no>
THEORY_NEEDED: <what kind of theoretical development would help>

Identify up to 5 theoretical gaps. If fewer exist, that's fine."""

    response = model.invoke([HumanMessage(content=prompt)])
    
    return _parse_gap_response(response.content, "theoretical")


def _parse_gap_response(content: str, gap_type: str) -> list[dict[str, Any]]:
    """Parse a gap identification response into structured data."""
    gaps = []
    current_gap = {}
    
    lines = content.split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith("GAP:"):
            # Save previous gap if exists
            if current_gap.get("title"):
                current_gap["gap_type"] = gap_type
                gaps.append(current_gap)
            current_gap = {"title": line.replace("GAP:", "").strip()}
        elif line.startswith("DESCRIPTION:"):
            current_gap["description"] = line.replace("DESCRIPTION:", "").strip()
        elif line.startswith("SIGNIFICANCE:"):
            sig = line.replace("SIGNIFICANCE:", "").strip().lower()
            current_gap["significance"] = sig if sig in ["high", "medium", "low"] else "medium"
        elif line.startswith("ADDRESSABLE:"):
            addr = line.replace("ADDRESSABLE:", "").strip().lower()
            current_gap["addressable"] = addr in ["yes", "true", "y"]
        elif line.startswith("CURRENT_METHODS:"):
            current_gap["current_methods"] = line.replace("CURRENT_METHODS:", "").strip()
        elif line.startswith("LIMITATION:"):
            current_gap["limitation"] = line.replace("LIMITATION:", "").strip()
        elif line.startswith("HOW_TO_ADDRESS:"):
            current_gap["how_to_address"] = line.replace("HOW_TO_ADDRESS:", "").strip()
        elif line.startswith("EXISTING_EVIDENCE:"):
            current_gap["existing_evidence"] = line.replace("EXISTING_EVIDENCE:", "").strip()
        elif line.startswith("MISSING_EVIDENCE:"):
            current_gap["missing_evidence"] = line.replace("MISSING_EVIDENCE:", "").strip()
        elif line.startswith("CONTEXT:"):
            current_gap["context"] = line.replace("CONTEXT:", "").strip()
        elif line.startswith("DATA_NEEDED:"):
            current_gap["data_needed"] = line.replace("DATA_NEEDED:", "").strip()
        elif line.startswith("EXISTING_THEORY:"):
            current_gap["existing_theory"] = line.replace("EXISTING_THEORY:", "").strip()
        elif line.startswith("UNEXPLAINED:"):
            current_gap["unexplained"] = line.replace("UNEXPLAINED:", "").strip()
        elif line.startswith("THEORY_NEEDED:"):
            current_gap["theory_needed"] = line.replace("THEORY_NEEDED:", "").strip()
    
    # Don't forget the last gap
    if current_gap.get("title"):
        current_gap["gap_type"] = gap_type
        gaps.append(current_gap)
    
    return gaps


# =============================================================================
# Gap Significance Assessment
# =============================================================================


def assess_gap_significance(
    gaps: list[dict[str, Any]],
    original_question: str,
    user_contribution: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Assess and rank the significance of identified gaps.
    
    Evaluates each gap based on:
    - Academic impact potential
    - Practical relevance
    - Feasibility of addressing
    - Alignment with research question
    
    Args:
        gaps: List of identified gaps from gap identification tools.
        original_question: The user's original research question.
        user_contribution: Optional user's expected contribution.
        model_name: Model to use (defaults to settings.default_model).
        
    Returns:
        Gaps with updated significance scores and rankings.
    """
    if not gaps:
        return []
    
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2048,
        api_key=settings.anthropic_api_key,
    )
    
    # Format gaps for analysis
    gaps_text = []
    for i, gap in enumerate(gaps, 1):
        gap_info = f"{i}. [{gap.get('gap_type', 'unknown').upper()}] {gap.get('title', 'Untitled')}"
        if gap.get("description"):
            gap_info += f"\n   Description: {gap['description']}"
        if gap.get("significance"):
            gap_info += f"\n   Initial Significance: {gap['significance']}"
        gaps_text.append(gap_info)
    
    gaps_formatted = "\n\n".join(gaps_text)
    
    contribution_info = ""
    if user_contribution:
        contribution_info = f"\n\nUSER'S EXPECTED CONTRIBUTION:\n{user_contribution}"
    
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    prompt = f"""Current date: {current_date}

Assess and rank the significance of these research gaps.

RESEARCH QUESTION:
{original_question}
{contribution_info}

IDENTIFIED GAPS:
{gaps_formatted}

For each gap, assess:
1. ACADEMIC_IMPACT: How much would addressing this gap advance knowledge? (1-10)
2. PRACTICAL_RELEVANCE: How useful would this be for practitioners? (1-10)
3. FEASIBILITY: How feasible is it to address this gap? (1-10)
4. ALIGNMENT: How well does this align with the research question? (1-10)
5. OVERALL_SCORE: Weighted average (academic 0.3, practical 0.2, feasibility 0.2, alignment 0.3)

Output format:
GAP_NUMBER: <1-N>
ACADEMIC_IMPACT: <score>
PRACTICAL_RELEVANCE: <score>
FEASIBILITY: <score>
ALIGNMENT: <score>
OVERALL_SCORE: <calculated score>
REVISED_SIGNIFICANCE: <high/medium/low based on overall score>
JUSTIFICATION: <one sentence justification>

After scoring all gaps, provide:
RANKING: <comma-separated gap numbers from most to least significant>
PRIMARY_GAP: <number of the most significant addressable gap>
PRIMARY_GAP_REASON: <why this is the best gap to address>"""

    response = model.invoke([HumanMessage(content=prompt)])
    
    # Parse and update gaps
    return _parse_significance_response(response.content, gaps)


def _parse_significance_response(
    content: str, 
    gaps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Parse significance assessment response and update gaps."""
    # Create a copy of gaps to avoid modifying original
    updated_gaps = [dict(gap) for gap in gaps]
    
    lines = content.split("\n")
    current_gap_num = None
    ranking = []
    primary_gap = None
    primary_reason = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith("GAP_NUMBER:"):
            try:
                current_gap_num = int(line.replace("GAP_NUMBER:", "").strip()) - 1
            except ValueError:
                current_gap_num = None
        elif current_gap_num is not None and 0 <= current_gap_num < len(updated_gaps):
            if line.startswith("ACADEMIC_IMPACT:"):
                try:
                    updated_gaps[current_gap_num]["academic_impact"] = int(
                        line.replace("ACADEMIC_IMPACT:", "").strip()
                    )
                except ValueError:
                    # Skip malformed score; model output may be non-numeric
                    pass
            elif line.startswith("PRACTICAL_RELEVANCE:"):
                try:
                    updated_gaps[current_gap_num]["practical_relevance"] = int(
                        line.replace("PRACTICAL_RELEVANCE:", "").strip()
                    )
                except ValueError:
                    # Skip malformed score; model output may be non-numeric
                    pass
            elif line.startswith("FEASIBILITY:"):
                try:
                    updated_gaps[current_gap_num]["feasibility_score"] = int(
                        line.replace("FEASIBILITY:", "").strip()
                    )
                except ValueError:
                    # Skip malformed score; model output may be non-numeric
                    pass
            elif line.startswith("ALIGNMENT:"):
                try:
                    updated_gaps[current_gap_num]["alignment_score"] = int(
                        line.replace("ALIGNMENT:", "").strip()
                    )
                except ValueError:
                    # Skip malformed score; model output may be non-numeric
                    pass
            elif line.startswith("OVERALL_SCORE:"):
                try:
                    updated_gaps[current_gap_num]["overall_score"] = float(
                        line.replace("OVERALL_SCORE:", "").strip()
                    )
                except ValueError:
                    # Skip malformed score; model output may be non-numeric
                    pass
            elif line.startswith("REVISED_SIGNIFICANCE:"):
                sig = line.replace("REVISED_SIGNIFICANCE:", "").strip().lower()
                if sig in ["high", "medium", "low"]:
                    updated_gaps[current_gap_num]["significance"] = sig
            elif line.startswith("JUSTIFICATION:"):
                updated_gaps[current_gap_num]["significance_justification"] = (
                    line.replace("JUSTIFICATION:", "").strip()
                )
        
        if line.startswith("RANKING:"):
            try:
                ranking_text = line.replace("RANKING:", "").strip()
                ranking = [int(x.strip()) for x in ranking_text.split(",")]
            except ValueError:
                # Skip malformed ranking; fall back to default ordering
                pass
        elif line.startswith("PRIMARY_GAP:"):
            try:
                primary_gap = int(line.replace("PRIMARY_GAP:", "").strip()) - 1
            except ValueError:
                # Skip malformed primary gap number; use default selection
                pass
        elif line.startswith("PRIMARY_GAP_REASON:"):
            primary_reason = line.replace("PRIMARY_GAP_REASON:", "").strip()
    
    # Add ranking info
    for i, gap in enumerate(updated_gaps):
        if ranking:
            try:
                gap["rank"] = ranking.index(i + 1) + 1
            except ValueError:
                gap["rank"] = len(updated_gaps)
        gap["is_primary"] = (i == primary_gap)
        if i == primary_gap:
            gap["primary_reason"] = primary_reason
    
    # Sort by rank if available
    if ranking:
        updated_gaps.sort(key=lambda g: g.get("rank", 999))
    
    return updated_gaps


# =============================================================================
# Combined Gap Analysis
# =============================================================================


def perform_gap_analysis(
    original_question: str,
    literature_synthesis: LiteratureSynthesis | dict[str, Any],
    data_context: str | None = None,
    user_contribution: str | None = None,
    model_name: str | None = None,
) -> GapAnalysis:
    """
    Perform comprehensive gap analysis combining all gap types.
    
    This is the main entry point for gap analysis that:
    1. Compares coverage
    2. Identifies all gap types
    3. Assesses significance
    4. Selects primary gap
    
    Args:
        original_question: The user's original research question.
        literature_synthesis: Synthesis from LITERATURE_REVIEWER node.
        data_context: Optional context about user's available data.
        user_contribution: Optional user's expected contribution.
        model_name: Model to use (defaults to settings.default_model).
        
    Returns:
        Complete GapAnalysis object.
    """
    # Step 1: Coverage analysis
    coverage = compare_coverage(
        original_question=original_question,
        literature_synthesis=literature_synthesis,
        model_name=model_name,
    )
    
    # Step 2: Identify all gap types
    methodological_gaps = identify_methodological_gaps(
        original_question=original_question,
        literature_synthesis=literature_synthesis,
        model_name=model_name,
    )
    
    empirical_gaps = identify_empirical_gaps(
        original_question=original_question,
        literature_synthesis=literature_synthesis,
        data_context=data_context,
        model_name=model_name,
    )
    
    theoretical_gaps = identify_theoretical_gaps(
        original_question=original_question,
        literature_synthesis=literature_synthesis,
        model_name=model_name,
    )
    
    # Combine all gaps
    all_gaps = methodological_gaps + empirical_gaps + theoretical_gaps
    
    # Step 3: Assess significance
    if all_gaps:
        assessed_gaps = assess_gap_significance(
            gaps=all_gaps,
            original_question=original_question,
            user_contribution=user_contribution,
            model_name=model_name,
        )
    else:
        assessed_gaps = []
    
    # Step 4: Convert to ResearchGap models
    research_gaps = []
    primary_gap = None
    
    for gap in assessed_gaps:
        research_gap = ResearchGap(
            gap_type=gap.get("gap_type", "unknown"),
            title=gap.get("title", "Untitled Gap"),
            description=gap.get("description", ""),
            significance=gap.get("significance", "medium"),
            significance_justification=gap.get("significance_justification", ""),
            supporting_evidence=[],  # Could be populated from coverage analysis
            addressable=gap.get("addressable", True),
            addressability_notes=gap.get("how_to_address", ""),
        )
        research_gaps.append(research_gap)
        
        if gap.get("is_primary"):
            primary_gap = research_gap
    
    # Build GapAnalysis
    analysis = GapAnalysis(
        original_question=original_question,
        literature_coverage_summary=coverage.get("coverage_summary", ""),
        gaps=research_gaps,
        primary_gap=primary_gap,
        coverage_comparison=coverage.get("coverage_summary", ""),
        coverage_percentage=coverage.get("coverage_percentage", 50.0),
        methodological_gaps=[g.get("title", "") for g in methodological_gaps],
        empirical_gaps=[g.get("title", "") for g in empirical_gaps],
        theoretical_gaps=[g.get("title", "") for g in theoretical_gaps],
        gap_significance_ranking=[
            g.gap_id for g in sorted(
                research_gaps, 
                key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x.significance, 0),
                reverse=True
            )
        ],
    )
    
    return analysis


# =============================================================================
# Tool Versions (for LangGraph ToolNode)
# =============================================================================


@tool
def compare_coverage_tool(
    original_question: str,
    literature_synthesis_json: str,
) -> str:
    """
    Compare what literature covers vs. what the research question asks.
    
    Args:
        original_question: The research question to analyze.
        literature_synthesis_json: JSON string of literature synthesis.
        
    Returns:
        Coverage analysis as formatted text.
    """
    import json
    synthesis = json.loads(literature_synthesis_json)
    result = compare_coverage(original_question, synthesis)
    
    output = f"Coverage Summary: {result['coverage_summary']}\n"
    output += f"Coverage Percentage: {result['coverage_percentage']}%\n\n"
    output += "Well Covered:\n"
    for item in result["well_covered"]:
        output += f"  - {item}\n"
    output += "\nPartially Covered:\n"
    for item in result["partially_covered"]:
        output += f"  - {item}\n"
    output += "\nNot Covered (Gaps):\n"
    for item in result["not_covered"]:
        output += f"  - {item}\n"
    
    return output


@tool
def identify_gaps_tool(
    original_question: str,
    literature_synthesis_json: str,
    gap_type: str = "all",
) -> str:
    """
    Identify research gaps in the literature.
    
    Args:
        original_question: The research question to analyze.
        literature_synthesis_json: JSON string of literature synthesis.
        gap_type: Type of gaps to identify (methodological, empirical, theoretical, or all).
        
    Returns:
        Identified gaps as formatted text.
    """
    import json
    synthesis = json.loads(literature_synthesis_json)
    
    all_gaps = []
    
    if gap_type in ["methodological", "all"]:
        all_gaps.extend(identify_methodological_gaps(original_question, synthesis))
    
    if gap_type in ["empirical", "all"]:
        all_gaps.extend(identify_empirical_gaps(original_question, synthesis))
    
    if gap_type in ["theoretical", "all"]:
        all_gaps.extend(identify_theoretical_gaps(original_question, synthesis))
    
    output = f"Identified {len(all_gaps)} gaps:\n\n"
    for i, gap in enumerate(all_gaps, 1):
        output += f"{i}. [{gap.get('gap_type', 'unknown').upper()}] {gap.get('title', 'Untitled')}\n"
        output += f"   Description: {gap.get('description', 'N/A')}\n"
        output += f"   Significance: {gap.get('significance', 'medium')}\n"
        output += f"   Addressable: {'Yes' if gap.get('addressable', True) else 'No'}\n\n"
    
    return output
