"""LITERATURE_SYNTHESIZER node for synthesizing literature findings.

This node:
1. Analyzes search results to extract key themes
2. Identifies research gaps in the literature
3. Synthesizes findings into a coherent narrative
4. Generates contribution opportunities
5. Refines the research question based on gaps
"""

import re
from datetime import datetime, timezone
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage

from src.config import settings
from src.state.enums import ResearchStatus
from src.state.models import SearchResult, WorkflowError
from src.state.schema import WorkflowState


# =============================================================================
# Theme Extraction
# =============================================================================


def extract_themes(
    search_results: list[SearchResult],
    model_name: str | None = None,
) -> list[str]:
    """
    Extract key themes from literature search results.
    
    Uses Claude to identify recurring themes, concepts, and patterns
    across the collected literature.
    
    Args:
        search_results: List of processed search results.
        model_name: Model to use (defaults to settings.default_model).
        
    Returns:
        List of theme descriptions.
    """
    if not search_results:
        return []
    
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2048,
        api_key=settings.anthropic_api_key,
    )
    
    # Prepare literature summary
    papers_text = []
    for i, result in enumerate(search_results[:30], 1):  # Limit to 30 papers
        paper_info = f"{i}. {result.title}"
        if result.venue:
            paper_info += f" ({result.venue})"
        if result.snippet:
            paper_info += f"\n   Abstract: {result.snippet[:300]}..."
        if result.citation_count:
            paper_info += f"\n   Citations: {result.citation_count}"
        papers_text.append(paper_info)
    
    papers_summary = "\n\n".join(papers_text)
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    prompt = f"""Current date: {current_date}

Analyze the following literature search results and identify 5-8 key themes:

{papers_summary}

For each theme, provide:
THEME: <concise theme name (3-5 words)>
DESCRIPTION: <one sentence description>
EVIDENCE: <2-3 paper numbers that support this theme>

Focus on:
1. Major theoretical frameworks or perspectives
2. Methodological approaches
3. Key findings or consensus positions
4. Debates or disagreements in the field
5. Emerging trends or new directions"""

    response = model.invoke([HumanMessage(content=prompt)])
    
    # Parse themes from response
    themes = []
    lines = response.content.split("\n")
    
    current_theme = None
    for line in lines:
        line = line.strip()
        if line.startswith("THEME:"):
            if current_theme:
                themes.append(current_theme)
            current_theme = line[6:].strip()
        elif line.startswith("DESCRIPTION:") and current_theme:
            current_theme += f" - {line[12:].strip()}"
    
    if current_theme:
        themes.append(current_theme)
    
    return themes[:8]  # Limit to 8 themes


# =============================================================================
# Gap Identification
# =============================================================================


def identify_gaps(
    research_question: str,
    search_results: list[SearchResult],
    themes: list[str],
    model_name: str | None = None,
) -> list[str]:
    """
    Identify research gaps in the literature.
    
    Analyzes the literature to find what has NOT been studied,
    where methodologies are weak, or where findings are inconsistent.
    
    Args:
        research_question: The original research question.
        search_results: Processed search results.
        themes: Extracted themes from the literature.
        model_name: Model to use.
        
    Returns:
        List of identified research gaps.
    """
    if not search_results:
        return ["Insufficient literature found to identify gaps"]
    
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=2048,
        api_key=settings.anthropic_api_key,
    )
    
    # Prepare context
    themes_text = "\n".join(f"- {theme}" for theme in themes)
    
    papers_summary = []
    for result in search_results[:20]:
        summary = f"- {result.title}"
        if result.snippet:
            summary += f": {result.snippet[:200]}..."
        papers_summary.append(summary)
    papers_text = "\n".join(papers_summary)
    
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    prompt = f"""Current date: {current_date}

Research Question: {research_question}

Identified Themes in Literature:
{themes_text}

Sample Papers Found:
{papers_text}

Identify 4-6 specific research gaps. For each gap:
GAP: <specific gap statement>
TYPE: <theoretical|methodological|empirical|contextual>
OPPORTUNITY: <how this gap could be addressed>

Consider:
1. Topics not covered in the existing literature
2. Methodological weaknesses or limitations
3. Conflicting findings that need resolution
4. Under-studied contexts, populations, or time periods
5. Unanswered questions from prior research
6. Opportunities for novel contributions"""

    response = model.invoke([HumanMessage(content=prompt)])
    
    # Parse gaps from response
    gaps = []
    lines = response.content.split("\n")
    
    current_gap = None
    for line in lines:
        line = line.strip()
        if line.startswith("GAP:"):
            if current_gap:
                gaps.append(current_gap)
            current_gap = line[4:].strip()
        elif line.startswith("TYPE:") and current_gap:
            gap_type = line[5:].strip()
            current_gap = f"[{gap_type.upper()}] {current_gap}"
        elif line.startswith("OPPORTUNITY:") and current_gap:
            current_gap += f" (Opportunity: {line[12:].strip()})"
    
    if current_gap:
        gaps.append(current_gap)
    
    return gaps[:6]  # Limit to 6 gaps


# =============================================================================
# Literature Synthesis
# =============================================================================


def synthesize_literature(
    research_question: str,
    search_results: list[SearchResult],
    themes: list[str],
    gaps: list[str],
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Generate a coherent literature synthesis.
    
    Creates a structured synthesis of the literature including:
    - State of the field summary
    - Key findings and consensus
    - Theoretical frameworks
    - Methodological approaches
    - Research gaps and opportunities
    
    Args:
        research_question: The research question.
        search_results: Processed search results.
        themes: Extracted themes.
        gaps: Identified research gaps.
        model_name: Model to use.
        
    Returns:
        Dictionary with synthesis components.
    """
    if not search_results:
        return {
            "summary": "Insufficient literature available for synthesis.",
            "state_of_field": "",
            "key_findings": [],
            "theoretical_frameworks": [],
            "methodological_approaches": [],
            "contribution_opportunities": [],
        }
    
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=4096,
        api_key=settings.anthropic_api_key,
    )
    
    # Prepare comprehensive context
    themes_text = "\n".join(f"- {theme}" for theme in themes)
    gaps_text = "\n".join(f"- {gap}" for gap in gaps)
    
    # Group papers by recency and impact
    recent_papers = [r for r in search_results if r.published_date and 
                     r.published_date.year >= datetime.now(timezone.utc).year - 3]
    highly_cited = sorted(search_results, 
                         key=lambda x: x.citation_count or 0, 
                         reverse=True)[:10]
    
    recent_text = "\n".join(f"- {r.title}" for r in recent_papers[:10])
    cited_text = "\n".join(
        f"- {r.title} ({r.citation_count} citations)" 
        for r in highly_cited if r.citation_count
    )
    
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    prompt = f"""Current date: {current_date}

Research Question: {research_question}

Themes Identified:
{themes_text}

Research Gaps:
{gaps_text}

Recent Papers (Last 3 Years):
{recent_text if recent_text else "None identified"}

Highly Cited Papers:
{cited_text if cited_text else "None identified"}

Generate a comprehensive literature synthesis with the following sections:

1. STATE OF THE FIELD (2-3 paragraphs):
Describe the current state of research on this topic.

2. KEY FINDINGS (3-5 bullet points):
The most important findings from the literature.

3. THEORETICAL FRAMEWORKS (2-4 bullet points):
Major theories or frameworks used in this area.

4. METHODOLOGICAL APPROACHES (2-4 bullet points):
Common research methods and their strengths/limitations.

5. CONTRIBUTION OPPORTUNITIES (2-4 bullet points):
Specific opportunities for novel research contributions based on the gaps.

Use clear section headers and maintain academic tone.
Avoid using banned words: delve, realm, harness, unlock, groundbreaking, etc."""

    response = model.invoke([HumanMessage(content=prompt)])
    
    # Parse synthesis sections
    content = response.content
    
    synthesis = {
        "summary": content[:500] + "..." if len(content) > 500 else content,
        "state_of_field": extract_section(content, "STATE OF THE FIELD"),
        "key_findings": extract_bullets(content, "KEY FINDINGS"),
        "theoretical_frameworks": extract_bullets(content, "THEORETICAL FRAMEWORKS"),
        "methodological_approaches": extract_bullets(content, "METHODOLOGICAL APPROACHES"),
        "contribution_opportunities": extract_bullets(content, "CONTRIBUTION OPPORTUNITIES"),
        "full_synthesis": content,
    }
    
    return synthesis


def extract_section(text: str, section_name: str) -> str:
    """Extract a section from synthesized text."""
    # Try to find section with various header formats
    patterns = [
        rf"{section_name}[:\s]*\n(.*?)(?=\n\d+\.\s|\n[A-Z][A-Z ]+[:\s]*\n|$)",
        rf"\d+\.\s*{section_name}[:\s]*\n(.*?)(?=\n\d+\.\s|\n[A-Z][A-Z ]+[:\s]*\n|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return ""


def extract_bullets(text: str, section_name: str) -> list[str]:
    """Extract bullet points from a section."""
    section = extract_section(text, section_name)
    
    if not section:
        return []
    
    # Split by bullet points or numbered items
    bullets = re.split(r'\n[-•*]\s*|\n\d+\.\s*', section)
    
    return [b.strip() for b in bullets if b.strip() and len(b.strip()) > 10]


# =============================================================================
# Contribution Statement Generation
# =============================================================================


def generate_contribution_statement(
    research_question: str,
    gaps: list[str],
    synthesis: dict[str, Any],
    model_name: str | None = None,
) -> str:
    """
    Generate a clear contribution statement.
    
    Based on the research question and identified gaps, generates a
    statement of how this research will contribute to the field.
    
    Args:
        research_question: The research question.
        gaps: Identified research gaps.
        synthesis: Literature synthesis dictionary.
        model_name: Model to use.
        
    Returns:
        Contribution statement string.
    """
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=1024,
        api_key=settings.anthropic_api_key,
    )
    
    gaps_text = "\n".join(f"- {gap}" for gap in gaps[:4])
    opportunities = synthesis.get("contribution_opportunities", [])
    opportunities_text = "\n".join(f"- {opp}" for opp in opportunities[:4])
    
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    prompt = f"""Current date: {current_date}

Research Question: {research_question}

Research Gaps Identified:
{gaps_text}

Contribution Opportunities:
{opportunities_text if opportunities_text else "Not yet identified"}

Write a clear, concise contribution statement (2-3 sentences) that:
1. States what gap this research addresses
2. Explains how it advances knowledge
3. Identifies who will benefit from this research

The statement should be suitable for the introduction of an academic paper.
Avoid starting with "This paper" or "This study" - be more specific.
Avoid banned words: groundbreaking, novel, unique, innovative, etc."""

    response = model.invoke([HumanMessage(content=prompt)])
    
    return response.content.strip()


# =============================================================================
# Query Refinement
# =============================================================================


def refine_research_question(
    original_query: str,
    gaps: list[str],
    synthesis: dict[str, Any],
    model_name: str | None = None,
) -> str:
    """
    Refine the research question based on literature review.
    
    May narrow scope, add specificity, or adjust focus based on
    what the literature review revealed.
    
    Args:
        original_query: The original research question.
        gaps: Identified research gaps.
        synthesis: Literature synthesis.
        model_name: Model to use.
        
    Returns:
        Refined research question.
    """
    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=512,
        api_key=settings.anthropic_api_key,
    )
    
    state_of_field = synthesis.get("state_of_field", "")
    gaps_text = "\n".join(f"- {gap}" for gap in gaps[:4])
    
    prompt = f"""Original Research Question: {original_query}

State of the Field:
{state_of_field[:500] if state_of_field else "Not available"}

Research Gaps:
{gaps_text}

Based on the literature review, suggest a refined research question that:
1. Is more specific and addressable
2. Targets an identified gap
3. Is achievable within a typical research project

Provide ONLY the refined question, nothing else. 
If the original question is already well-formulated, return it unchanged."""

    response = model.invoke([HumanMessage(content=prompt)])
    
    refined = response.content.strip()
    
    # Remove any quotes or "Refined question:" prefix
    refined = refined.strip('"\'')
    if refined.lower().startswith("refined question:"):
        refined = refined[17:].strip()
    
    return refined if refined else original_query


# =============================================================================
# Main Node Function
# =============================================================================


def literature_synthesizer_node(state: WorkflowState) -> dict[str, Any]:
    """
    LITERATURE_SYNTHESIZER node: Synthesize literature findings.
    
    Takes the search results from LITERATURE_REVIEWER and:
    - Extracts key themes
    - Identifies research gaps
    - Generates synthesis
    - Creates contribution statement
    - Refines research question
    
    Handles empty search results gracefully by acknowledging the literature
    gap and proceeding with data-driven research.
    
    Args:
        state: Current workflow state with search_results.
        
    Returns:
        State updates with literature_synthesis, literature_themes,
        identified_gaps, contribution_statement, refined_query.
    """
    search_results = state.get("search_results", [])
    original_query = state.get("original_query", "")
    
    if not original_query:
        return {
            "status": ResearchStatus.FAILED,
            "errors": [WorkflowError(
                node="literature_synthesizer",
                category="validation",
                message="No research question provided for synthesis",
                recoverable=False,
            )],
            "messages": [AIMessage(
                content="Cannot synthesize literature without a research question."
            )],
        }
    
    # Handle empty search results (e.g., from rate limiting or novel topic)
    if not search_results:
        synthesis_text = (
            "Literature search returned no results. This may indicate: "
            "(1) API rate limiting during search, (2) a novel research area "
            "with limited existing literature, or (3) highly specific query terms. "
            "The research will proceed with data-driven analysis as the primary approach."
        )
        themes = ["Data-driven research approach"]
        gaps = [
            "Existing literature on this specific topic appears limited or inaccessible. "
            "This presents an opportunity for original contribution through empirical analysis."
        ]
        contribution = (
            "Given the limited accessible literature, this research will make an "
            "empirical contribution by analyzing the available data to generate "
            "insights and findings that can inform future work in this area."
        )
        return {
            "status": ResearchStatus.GAP_IDENTIFICATION_COMPLETE,
            "literature_synthesis": {
                "summary": synthesis_text,
                "state_of_field": "",
                "key_findings": [],
                "theoretical_frameworks": [],
                "methodological_approaches": [],
                "contribution_opportunities": [contribution],
                "full_synthesis": synthesis_text,
                "themes": themes,
                "gaps": gaps,
                "contribution_statement": contribution,
                "limitations": "No literature items were available for synthesis.",
                "proceeding_strategy": "data-driven",
                "papers_analyzed": 0,
                "themes_identified": len(themes),
                "gaps_identified": len(gaps),
            },
            "literature_themes": themes,
            "identified_gaps": gaps,
            "contribution_statement": contribution,
            "refined_query": original_query,  # Keep original query
            "messages": [AIMessage(
                content=(
                    "Literature Synthesis: No search results available\n\n"
                    "The literature search did not return results (possibly due to API "
                    "rate limiting or a novel research area). The workflow will continue "
                    "with a data-driven approach, using the uploaded datasets as the "
                    "primary source of analysis.\n\n"
                    "This can be an advantage: the research may produce original empirical "
                    "findings without being constrained by existing paradigms."
                )
            )],
            "checkpoints": [
                f"{datetime.now(timezone.utc).isoformat()}: Literature synthesis - no results, proceeding data-driven"
            ],
            "updated_at": datetime.now(timezone.utc),
        }
    
    try:
        # Step 1: Extract themes
        themes = extract_themes(search_results)
        
        # Step 2: Identify gaps
        gaps = identify_gaps(original_query, search_results, themes)
        
        # Step 3: Synthesize literature
        synthesis = synthesize_literature(
            original_query, 
            search_results, 
            themes, 
            gaps
        )
        
        # Step 4: Generate contribution statement
        contribution = generate_contribution_statement(
            original_query, 
            gaps, 
            synthesis
        )
        
        # Step 5: Refine research question
        refined_query = refine_research_question(
            original_query, 
            gaps, 
            synthesis
        )
        
        # Generate summary message
        message = generate_synthesis_summary(
            themes=themes,
            gaps=gaps,
            contribution=contribution,
            refined_query=refined_query,
            original_query=original_query,
        )
        
        return {
            "status": ResearchStatus.GAP_IDENTIFICATION_COMPLETE,
            "literature_synthesis": synthesis,
            "literature_themes": themes,
            "identified_gaps": gaps,
            "contribution_statement": contribution,
            "refined_query": refined_query,
            "messages": [AIMessage(content=message)],
            "checkpoints": [
                f"{datetime.now(timezone.utc).isoformat()}: Literature synthesis complete - "
                f"{len(themes)} themes, {len(gaps)} gaps identified"
            ],
            "updated_at": datetime.now(timezone.utc),
        }
        
    except Exception as e:
        return {
            "status": ResearchStatus.FAILED,
            "errors": [WorkflowError(
                node="literature_synthesizer",
                category="synthesis",
                message=f"Literature synthesis failed: {str(e)}",
                recoverable=True,
            )],
            "messages": [AIMessage(
                content=f"Literature synthesis encountered an error: {str(e)}. "
                "The workflow will continue with available information."
            )],
        }


def generate_synthesis_summary(
    themes: list[str],
    gaps: list[str],
    contribution: str,
    refined_query: str,
    original_query: str,
) -> str:
    """Generate a human-readable synthesis summary."""
    query_changed = refined_query != original_query
    
    themes_list = "\n".join(f"  - {theme}" for theme in themes[:5])
    gaps_list = "\n".join(f"  - {gap}" for gap in gaps[:4])
    
    summary = f"""Literature Synthesis Complete

Key Themes Identified ({len(themes)} total):
{themes_list}

Research Gaps ({len(gaps)} total):
{gaps_list}

Contribution Statement:
{contribution}

"""
    
    if query_changed:
        summary += f"""Research Question Refinement:
Original: {original_query}
Refined: {refined_query}

"""
    
    summary += """Next step: Gap analysis and research planning."""
    
    return summary
