"""Contribution generation tools for academic research.

This module provides tools for:
1. Generating contribution statements from gap analysis
2. Positioning research within existing literature
3. Differentiating from prior work
4. Articulating novelty and impact
"""

from datetime import datetime, timezone
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from src.config import settings
from src.state.models import (
    ContributionStatement,
    GapAnalysis,
    LiteratureSynthesis,
    ResearchGap,
)


# =============================================================================
# Contribution Statement Generation
# =============================================================================


def generate_contribution_statement(
    refined_question: str,
    primary_gap: ResearchGap | dict[str, Any],
    literature_synthesis: LiteratureSynthesis | dict[str, Any],
    user_expected_contribution: str | None = None,
    model_name: str | None = None,
) -> ContributionStatement:
    """
    Generate a clear contribution statement based on gap analysis.
    
    Creates a compelling statement of what the research contributes
    to the field, grounded in the identified gap and literature context.
    
    Args:
        refined_question: The refined research question targeting a gap.
        primary_gap: The primary gap being addressed.
        literature_synthesis: Synthesis from LITERATURE_REVIEWER node.
        user_expected_contribution: Optional user's expected contribution.
        model_name: Model to use (defaults to settings.default_model).
        
    Returns:
        ContributionStatement with main statement and supporting info.
    """
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2048,
        api_key=settings.anthropic_api_key,
    )
    
    # Format inputs
    if isinstance(primary_gap, dict):
        gap_text = _format_gap_dict(primary_gap)
        gap_type = primary_gap.get("gap_type", "empirical")
    else:
        gap_text = _format_gap_model(primary_gap)
        gap_type = primary_gap.gap_type
    
    if isinstance(literature_synthesis, dict):
        synthesis_text = _format_synthesis_brief(literature_synthesis)
    else:
        synthesis_text = _format_synthesis_brief(literature_synthesis.model_dump())
    
    user_context = ""
    if user_expected_contribution:
        user_context = f"\n\nUSER'S EXPECTED CONTRIBUTION:\n{user_expected_contribution}"
    
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    prompt = f"""Current date: {current_date}

Generate a compelling contribution statement for this research.

REFINED RESEARCH QUESTION:
{refined_question}

GAP BEING ADDRESSED:
{gap_text}

LITERATURE CONTEXT:
{synthesis_text}
{user_context}

Create a contribution statement following academic conventions. The statement should:
1. Be specific about what is new or different
2. Be grounded in the identified gap
3. Articulate the value to the field
4. Be appropriately modest (avoid overclaiming)
5. Follow the style of top finance/economics journals

Provide your response in this format:

MAIN_STATEMENT: <A clear, one-paragraph contribution statement suitable for an introduction section>

CONTRIBUTION_TYPE: <methodological/empirical/theoretical>

NOVELTY_EXPLANATION: <What specifically is new about this contribution>

GAP_ADDRESSED: <How this contribution fills the identified gap>

POTENTIAL_IMPACT: <Expected impact on the field>

TARGET_AUDIENCE: <Who will benefit from this contribution>

Do NOT use banned words like: groundbreaking, novel, unique, cutting-edge, revolutionary, etc.
Use precise, academic language instead."""

    response = model.invoke([HumanMessage(content=prompt)])
    
    # Parse response
    parsed = _parse_contribution_response(response.content)
    
    return ContributionStatement(
        main_statement=parsed.get("main_statement", ""),
        contribution_type=parsed.get("contribution_type", gap_type),
        gap_addressed=parsed.get("gap_addressed", ""),
        novelty_explanation=parsed.get("novelty_explanation", ""),
        potential_impact=parsed.get("potential_impact", ""),
        target_audience=parsed.get("target_audience", []),
    )


def _format_gap_dict(gap: dict[str, Any]) -> str:
    """Format a gap dictionary as text."""
    parts = [f"Type: {gap.get('gap_type', 'unknown')}"]
    
    if gap.get("title"):
        parts.append(f"Title: {gap['title']}")
    
    if gap.get("description"):
        parts.append(f"Description: {gap['description']}")
    
    if gap.get("significance"):
        parts.append(f"Significance: {gap['significance']}")
    
    return "\n".join(parts)


def _format_gap_model(gap: ResearchGap) -> str:
    """Format a ResearchGap model as text."""
    return _format_gap_dict(gap.model_dump())


def _format_synthesis_brief(synthesis: dict[str, Any]) -> str:
    """Format a brief version of literature synthesis."""
    parts = []
    
    if synthesis.get("state_of_field"):
        parts.append(f"State of Field: {synthesis['state_of_field'][:500]}...")
    
    if synthesis.get("key_findings"):
        findings = synthesis["key_findings"][:5]
        parts.append("Key Findings: " + "; ".join(findings))
    
    if synthesis.get("methodological_approaches"):
        methods = synthesis["methodological_approaches"][:3]
        parts.append("Common Methods: " + ", ".join(methods))
    
    return "\n".join(parts) if parts else "No synthesis available"


def _parse_contribution_response(content: str) -> dict[str, Any]:
    """Parse the contribution statement response."""
    result = {
        "main_statement": "",
        "contribution_type": "empirical",
        "novelty_explanation": "",
        "gap_addressed": "",
        "potential_impact": "",
        "target_audience": [],
    }
    
    lines = content.split("\n")
    current_key = None
    current_value = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Check for new section
        found_key = None
        for key in ["MAIN_STATEMENT:", "CONTRIBUTION_TYPE:", "NOVELTY_EXPLANATION:",
                    "GAP_ADDRESSED:", "POTENTIAL_IMPACT:", "TARGET_AUDIENCE:"]:
            if line_stripped.startswith(key):
                # Save previous
                if current_key:
                    result[current_key] = " ".join(current_value).strip()
                
                # Start new
                found_key = key.replace(":", "").lower()
                current_key = found_key
                current_value = [line_stripped.replace(key, "").strip()]
                break
        
        if not found_key and current_key:
            current_value.append(line_stripped)
    
    # Save last section
    if current_key:
        result[current_key] = " ".join(current_value).strip()
    
    # Parse target audience as list if it's a string with delimiters
    if isinstance(result["target_audience"], str):
        audience_str = result["target_audience"]
        if "," in audience_str or ";" in audience_str:
            import re
            result["target_audience"] = [
                a.strip() for a in re.split(r"[,;]", audience_str) if a.strip()
            ]
        else:
            result["target_audience"] = [audience_str] if audience_str else []
    
    return result


# =============================================================================
# Literature Positioning
# =============================================================================


def position_in_literature(
    contribution_statement: str,
    literature_synthesis: LiteratureSynthesis | dict[str, Any],
    related_papers: list[str] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Position the research contribution within existing literature.
    
    Articulates how the research fits with, builds upon, and relates
    to existing work in the field.
    
    Args:
        contribution_statement: The main contribution statement.
        literature_synthesis: Synthesis from LITERATURE_REVIEWER node.
        related_papers: Optional list of specific related papers.
        model_name: Model to use (defaults to settings.default_model).
        
    Returns:
        Dictionary with positioning information.
    """
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2048,
        api_key=settings.anthropic_api_key,
    )
    
    if isinstance(literature_synthesis, dict):
        synthesis_text = _format_synthesis_for_positioning(literature_synthesis)
    else:
        synthesis_text = _format_synthesis_for_positioning(literature_synthesis.model_dump())
    
    papers_context = ""
    if related_papers:
        papers_context = "\n\nKEY RELATED PAPERS:\n" + "\n".join(f"- {p}" for p in related_papers[:10])
    
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    prompt = f"""Current date: {current_date}

Position this research contribution within the existing literature.

CONTRIBUTION STATEMENT:
{contribution_statement}

LITERATURE CONTEXT:
{synthesis_text}
{papers_context}

Analyze how this research fits within the literature. Provide:

LITERATURE_STREAM: <Which stream(s) of literature does this contribute to?>

BUILDS_ON: <What existing work does this build upon? Be specific about papers/authors.>

EXTENDS: <What does this extend or expand?>

COMPLEMENTS: <What existing work does this complement?>

CONTRASTS_WITH: <What does this contrast with or challenge?>

FILLS_GAP_IN: <Which specific gap in the literature does this fill?>

RELATIONSHIP_SUMMARY: <A 2-3 sentence summary of how this fits in the literature, suitable for a paper introduction>

Use specific references where possible. Be precise about the relationships."""

    response = model.invoke([HumanMessage(content=prompt)])
    
    return _parse_positioning_response(response.content)


def _format_synthesis_for_positioning(synthesis: dict[str, Any]) -> str:
    """Format synthesis for positioning analysis."""
    parts = []
    
    if synthesis.get("state_of_field"):
        parts.append(f"State of Field:\n{synthesis['state_of_field']}")
    
    if synthesis.get("theoretical_frameworks"):
        frameworks = synthesis["theoretical_frameworks"][:5]
        parts.append("Key Frameworks:\n" + "\n".join(f"- {f}" for f in frameworks))
    
    if synthesis.get("key_findings"):
        findings = synthesis["key_findings"][:7]
        parts.append("Key Findings:\n" + "\n".join(f"- {f}" for f in findings))
    
    if synthesis.get("contribution_opportunities"):
        opps = synthesis["contribution_opportunities"][:3]
        parts.append("Identified Opportunities:\n" + "\n".join(f"- {o}" for o in opps))
    
    return "\n\n".join(parts) if parts else "No synthesis available"


def _parse_positioning_response(content: str) -> dict[str, Any]:
    """Parse the positioning response."""
    result = {
        "literature_stream": "",
        "builds_on": "",
        "extends": "",
        "complements": "",
        "contrasts_with": "",
        "fills_gap_in": "",
        "relationship_summary": "",
    }
    
    lines = content.split("\n")
    current_key = None
    current_value = []
    
    key_mapping = {
        "LITERATURE_STREAM:": "literature_stream",
        "BUILDS_ON:": "builds_on",
        "EXTENDS:": "extends",
        "COMPLEMENTS:": "complements",
        "CONTRASTS_WITH:": "contrasts_with",
        "FILLS_GAP_IN:": "fills_gap_in",
        "RELATIONSHIP_SUMMARY:": "relationship_summary",
    }
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        found_key = None
        for key_str, key_name in key_mapping.items():
            if line_stripped.startswith(key_str):
                if current_key:
                    result[current_key] = " ".join(current_value).strip()
                current_key = key_name
                current_value = [line_stripped.replace(key_str, "").strip()]
                found_key = True
                break
        
        if not found_key and current_key:
            current_value.append(line_stripped)
    
    if current_key:
        result[current_key] = " ".join(current_value).strip()
    
    return result


# =============================================================================
# Differentiation from Prior Work
# =============================================================================


def differentiate_from_prior(
    contribution_statement: str,
    similar_papers: list[dict[str, Any]],
    gap_analysis: GapAnalysis | dict[str, Any] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Differentiate the research from similar prior work.
    
    Articulates what makes this research different from and potentially
    better than existing similar work.
    
    Args:
        contribution_statement: The main contribution statement.
        similar_papers: List of similar papers to differentiate from.
        gap_analysis: Optional gap analysis for context.
        model_name: Model to use (defaults to settings.default_model).
        
    Returns:
        Dictionary with differentiation information.
    """
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2048,
        api_key=settings.anthropic_api_key,
    )
    
    # Format similar papers
    papers_text = []
    for i, paper in enumerate(similar_papers[:10], 1):
        paper_info = f"{i}. {paper.get('title', 'Untitled')}"
        if paper.get("authors"):
            authors = paper["authors"]
            if isinstance(authors, list):
                paper_info += f" ({', '.join(authors[:3])})"
            else:
                paper_info += f" ({authors})"
        if paper.get("year"):
            paper_info += f" [{paper['year']}]"
        if paper.get("summary") or paper.get("snippet"):
            paper_info += f"\n   Summary: {paper.get('summary') or paper.get('snippet', '')[:200]}"
        if paper.get("methodology"):
            paper_info += f"\n   Method: {paper['methodology']}"
        papers_text.append(paper_info)
    
    papers_formatted = "\n\n".join(papers_text) if papers_text else "No similar papers provided"
    
    gap_context = ""
    if gap_analysis:
        if isinstance(gap_analysis, dict):
            if gap_analysis.get("primary_gap"):
                gap_context = f"\n\nPRIMARY GAP BEING ADDRESSED:\n{gap_analysis['primary_gap']}"
        else:
            if gap_analysis.primary_gap:
                gap_context = f"\n\nPRIMARY GAP BEING ADDRESSED:\n{gap_analysis.primary_gap.title}: {gap_analysis.primary_gap.description}"
    
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    prompt = f"""Current date: {current_date}

Differentiate this research contribution from similar prior work.

CONTRIBUTION STATEMENT:
{contribution_statement}
{gap_context}

SIMILAR PRIOR WORK:
{papers_formatted}

For each similar paper (by number), explain how this research differs:

PAPER_1_DIFF: <How does this differ from paper 1? Be specific.>
PAPER_2_DIFF: <How does this differ from paper 2?>
... (continue for all papers)

Then provide overall differentiation:

KEY_DIFFERENCES: <Bullet list of key ways this research differs from the collective prior work>

METHODOLOGICAL_DIFFERENCES: <How does the methodology differ?>

SCOPE_DIFFERENCES: <How does the scope differ?>

DATA_DIFFERENCES: <How does the data/context differ?>

THEORETICAL_DIFFERENCES: <How does the theoretical approach differ?>

DIFFERENTIATION_SUMMARY: <A paragraph suitable for a paper that positions this work relative to prior work, explaining what's different without dismissing prior work>

Be precise and specific. Avoid vague claims of being "better" - focus on what's different."""

    response = model.invoke([HumanMessage(content=prompt)])
    
    return _parse_differentiation_response(response.content, len(similar_papers))


def _parse_differentiation_response(
    content: str, 
    num_papers: int
) -> dict[str, Any]:
    """Parse the differentiation response."""
    result = {
        "paper_differences": {},
        "key_differences": [],
        "methodological_differences": "",
        "scope_differences": "",
        "data_differences": "",
        "theoretical_differences": "",
        "differentiation_summary": "",
    }
    
    lines = content.split("\n")
    current_key = None
    current_value = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Check for paper-specific differences
        for i in range(1, num_papers + 1):
            if line_stripped.startswith(f"PAPER_{i}_DIFF:"):
                if current_key:
                    _save_diff_value(result, current_key, current_value)
                current_key = f"paper_{i}"
                current_value = [line_stripped.replace(f"PAPER_{i}_DIFF:", "").strip()]
                break
        else:
            # Check for general sections
            key_mapping = {
                "KEY_DIFFERENCES:": "key_differences",
                "METHODOLOGICAL_DIFFERENCES:": "methodological_differences",
                "SCOPE_DIFFERENCES:": "scope_differences",
                "DATA_DIFFERENCES:": "data_differences",
                "THEORETICAL_DIFFERENCES:": "theoretical_differences",
                "DIFFERENTIATION_SUMMARY:": "differentiation_summary",
            }
            
            found = False
            for key_str, key_name in key_mapping.items():
                if line_stripped.startswith(key_str):
                    if current_key:
                        _save_diff_value(result, current_key, current_value)
                    current_key = key_name
                    current_value = [line_stripped.replace(key_str, "").strip()]
                    found = True
                    break
            
            if not found and current_key:
                current_value.append(line_stripped)
    
    # Save last value
    if current_key:
        _save_diff_value(result, current_key, current_value)
    
    # Parse key_differences as list
    if isinstance(result["key_differences"], str):
        diff_str = result["key_differences"]
        if "- " in diff_str or "• " in diff_str:
            import re
            diffs = re.split(r"\n\s*[-•]\s*", diff_str)
            result["key_differences"] = [d.strip() for d in diffs if d.strip()]
        else:
            result["key_differences"] = [diff_str] if diff_str else []
    
    return result


def _save_diff_value(result: dict, key: str, value: list[str]) -> None:
    """Save a differentiation value to the result dict."""
    combined = " ".join(value).strip()
    
    if key.startswith("paper_"):
        result["paper_differences"][key] = combined
    else:
        result[key] = combined


# =============================================================================
# Refine Research Question
# =============================================================================


def refine_research_question(
    original_question: str,
    gap_analysis: GapAnalysis | dict[str, Any],
    literature_synthesis: LiteratureSynthesis | dict[str, Any],
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Refine the research question to target a specific gap.
    
    Takes the original question and narrows/focuses it based on
    the gap analysis and literature context.
    
    Args:
        original_question: The user's original research question.
        gap_analysis: Gap analysis results.
        literature_synthesis: Literature synthesis results.
        model_name: Model to use (defaults to settings.default_model).
        
    Returns:
        Dictionary with refined question and refinement details.
    """
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2048,
        api_key=settings.anthropic_api_key,
    )
    
    # Format gap analysis
    if isinstance(gap_analysis, dict):
        primary_gap = gap_analysis.get("primary_gap", {})
        if isinstance(primary_gap, dict):
            gap_text = f"Type: {primary_gap.get('gap_type', 'unknown')}\n"
            gap_text += f"Title: {primary_gap.get('title', 'Unknown')}\n"
            gap_text += f"Description: {primary_gap.get('description', 'No description')}"
        else:
            gap_text = str(primary_gap)
        coverage = gap_analysis.get("coverage_percentage", 50)
        coverage_summary = gap_analysis.get("coverage_comparison", "")
    else:
        if gap_analysis.primary_gap:
            gap_text = f"Type: {gap_analysis.primary_gap.gap_type}\n"
            gap_text += f"Title: {gap_analysis.primary_gap.title}\n"
            gap_text += f"Description: {gap_analysis.primary_gap.description}"
        else:
            gap_text = "No primary gap identified"
        coverage = gap_analysis.coverage_percentage
        coverage_summary = gap_analysis.coverage_comparison
    
    # Format synthesis briefly
    if isinstance(literature_synthesis, dict):
        state = literature_synthesis.get("state_of_field", "")[:300]
    else:
        state = literature_synthesis.state_of_field[:300] if literature_synthesis.state_of_field else ""
    
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    prompt = f"""Current date: {current_date}

Refine this research question to better target the identified gap.

ORIGINAL RESEARCH QUESTION:
{original_question}

PRIMARY GAP TO ADDRESS:
{gap_text}

LITERATURE COVERAGE: {coverage}%
{coverage_summary}

STATE OF THE FIELD:
{state}

Refine the research question to:
1. Be more specific and focused
2. Directly target the identified gap
3. Be answerable with available methods
4. Have clear scope boundaries

Provide your response in this format:

REFINED_QUESTION: <The refined research question>

REFINEMENT_RATIONALE: <Why was the question refined this way?>

SCOPE_CHANGES:
- <Change 1>
- <Change 2>
...

SPECIFICITY_IMPROVEMENT: <How is the refined question more specific?>

GAP_TARGETING: <How does the refined question directly address the gap?>

FEASIBILITY_ASSESSMENT: <How feasible is it to answer this refined question?>

If the original question is already well-focused on the gap, you may keep it largely unchanged but explain why."""

    response = model.invoke([HumanMessage(content=prompt)])
    
    return _parse_refinement_response(response.content, original_question)


def _parse_refinement_response(content: str, original: str) -> dict[str, Any]:
    """Parse the question refinement response."""
    result = {
        "original_question": original,
        "refined_question": original,  # Default to original
        "refinement_rationale": "",
        "scope_changes": [],
        "specificity_improvement": "",
        "gap_targeting": "",
        "feasibility_assessment": "",
    }
    
    lines = content.split("\n")
    current_key = None
    current_value = []
    
    key_mapping = {
        "REFINED_QUESTION:": "refined_question",
        "REFINEMENT_RATIONALE:": "refinement_rationale",
        "SCOPE_CHANGES:": "scope_changes",
        "SPECIFICITY_IMPROVEMENT:": "specificity_improvement",
        "GAP_TARGETING:": "gap_targeting",
        "FEASIBILITY_ASSESSMENT:": "feasibility_assessment",
    }
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        found = False
        for key_str, key_name in key_mapping.items():
            if line_stripped.startswith(key_str):
                if current_key:
                    _save_refinement_value(result, current_key, current_value)
                current_key = key_name
                current_value = [line_stripped.replace(key_str, "").strip()]
                found = True
                break
        
        if not found and current_key:
            # Handle bullet points for scope_changes
            if current_key == "scope_changes" and line_stripped.startswith("- "):
                current_value.append(line_stripped[2:].strip())
            else:
                current_value.append(line_stripped)
    
    if current_key:
        _save_refinement_value(result, current_key, current_value)
    
    return result


def _save_refinement_value(result: dict, key: str, value: list[str]) -> None:
    """Save a refinement value to the result dict."""
    if key == "scope_changes":
        # Keep as list, filtering out empty items
        result[key] = [v for v in value if v and not v.startswith("SCOPE_CHANGES")]
    else:
        result[key] = " ".join(value).strip()


# =============================================================================
# Tool Versions (for LangGraph ToolNode)
# =============================================================================


@tool
def generate_contribution_tool(
    refined_question: str,
    gap_title: str,
    gap_description: str,
    gap_type: str,
) -> str:
    """
    Generate a contribution statement for research.
    
    Args:
        refined_question: The refined research question.
        gap_title: Title of the gap being addressed.
        gap_description: Description of the gap.
        gap_type: Type of gap (methodological, empirical, theoretical).
        
    Returns:
        Contribution statement as formatted text.
    """
    gap = {
        "title": gap_title,
        "description": gap_description,
        "gap_type": gap_type,
    }
    
    # Use minimal synthesis for tool version
    synthesis = {"state_of_field": "See literature review for details."}
    
    statement = generate_contribution_statement(
        refined_question=refined_question,
        primary_gap=gap,
        literature_synthesis=synthesis,
    )
    
    output = f"CONTRIBUTION STATEMENT:\n{statement.main_statement}\n\n"
    output += f"Type: {statement.contribution_type}\n"
    output += f"Gap Addressed: {statement.gap_addressed}\n"
    output += f"Novelty: {statement.novelty_explanation}\n"
    output += f"Impact: {statement.potential_impact}\n"
    
    return output


@tool
def refine_question_tool(
    original_question: str,
    gap_title: str,
    gap_description: str,
    gap_type: str,
) -> str:
    """
    Refine a research question to target a specific gap.
    
    Args:
        original_question: The original research question.
        gap_title: Title of the gap to target.
        gap_description: Description of the gap.
        gap_type: Type of gap (methodological, empirical, theoretical).
        
    Returns:
        Refined question and rationale as formatted text.
    """
    gap_analysis = {
        "primary_gap": {
            "title": gap_title,
            "description": gap_description,
            "gap_type": gap_type,
        },
        "coverage_percentage": 50,
        "coverage_comparison": "",
    }
    
    synthesis = {"state_of_field": ""}
    
    result = refine_research_question(
        original_question=original_question,
        gap_analysis=gap_analysis,
        literature_synthesis=synthesis,
    )
    
    output = f"ORIGINAL: {result['original_question']}\n\n"
    output += f"REFINED: {result['refined_question']}\n\n"
    output += f"RATIONALE: {result['refinement_rationale']}\n\n"
    
    if result["scope_changes"]:
        output += "SCOPE CHANGES:\n"
        for change in result["scope_changes"]:
            output += f"  - {change}\n"
    
    return output
