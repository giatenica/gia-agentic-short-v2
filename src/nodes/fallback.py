"""Fallback node for graceful degradation.

This node generates partial output when the workflow cannot complete
normally due to errors. It:
- Collects available findings from completed stages
- Documents what could not be completed
- Provides recovery suggestions
- Generates a usable (if incomplete) output
"""

import logging
from datetime import datetime, timezone
from typing import Any

from src.state.schema import WorkflowState
from src.state.enums import ResearchStatus
from src.errors.recovery import (
    get_partial_output,
    create_fallback_content,
    _get_incomplete_stages,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Fallback Node
# =============================================================================


def fallback_node(state: WorkflowState) -> dict[str, Any]:
    """Fallback node for graceful degradation.
    
    This node is triggered when the workflow encounters too many errors
    or an unrecoverable error. It generates the best possible output
    from the available data.
    
    Args:
        state: Current workflow state
        
    Returns:
        State updates with partial output and status
    """
    logger.warning("Fallback node activated - generating partial output")
    
    # Get errors that led to fallback
    errors = state.get("errors", [])
    error_summary = _summarize_errors(errors)
    
    # Get partial output
    partial_output = get_partial_output(state)
    
    # Get incomplete stages
    incomplete_stages = _get_incomplete_stages(state)
    
    # Generate fallback paper sections if writer didn't complete
    fallback_sections = {}
    if not state.get("writer_output"):
        fallback_sections = _generate_fallback_sections(state)
    
    # Build fallback report
    fallback_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "partial_completion",
        "error_summary": error_summary,
        "incomplete_stages": incomplete_stages,
        "recovery_suggestions": _get_recovery_suggestions(errors, incomplete_stages),
        "available_output": partial_output,
        "fallback_sections": fallback_sections,
    }
    
    # Build final paper from available content
    final_paper = _assemble_final_paper(state, fallback_sections)
    
    return {
        "status": ResearchStatus.COMPLETED,  # Mark as completed (with partial output)
        "fallback_report": fallback_report,
        "final_paper": final_paper,
        "human_feedback": None,  # Clear any pending feedback
        "_fallback_activated": True,
    }


# =============================================================================
# Error Summarization
# =============================================================================


def _summarize_errors(errors: list[Any]) -> dict[str, Any]:
    """Summarize errors for the fallback report.
    
    Args:
        errors: List of workflow errors
        
    Returns:
        Error summary dictionary
    """
    if not errors:
        return {"count": 0, "categories": {}}
    
    # Count by category
    categories: dict[str, int] = {}
    nodes_affected: set[str] = set()
    recoverable_count = 0
    
    for error in errors:
        # Get category
        category = getattr(error, 'category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
        
        # Get node
        node = getattr(error, 'node', None)
        if node:
            nodes_affected.add(node)
        
        # Count recoverable
        if getattr(error, 'recoverable', True):
            recoverable_count += 1
    
    return {
        "count": len(errors),
        "categories": categories,
        "nodes_affected": list(nodes_affected),
        "recoverable_count": recoverable_count,
        "unrecoverable_count": len(errors) - recoverable_count,
    }


# =============================================================================
# Recovery Suggestions
# =============================================================================


def _get_recovery_suggestions(
    errors: list[Any],
    incomplete_stages: list[str],
) -> list[str]:
    """Generate recovery suggestions based on errors and incomplete stages.
    
    Args:
        errors: List of workflow errors
        incomplete_stages: List of incomplete stage names
        
    Returns:
        List of recovery suggestion strings
    """
    suggestions = []
    
    # Analyze error types
    error_categories = set()
    for error in errors:
        category = getattr(error, 'category', 'unknown')
        error_categories.add(category)
    
    # Rate limit suggestions
    if "rate_limit" in error_categories:
        suggestions.append(
            "Rate limits were encountered. Consider waiting before retrying "
            "or reducing the scope of literature search."
        )
    
    # Context overflow suggestions
    if "context_overflow" in error_categories:
        suggestions.append(
            "Content exceeded model context limits. Consider breaking the "
            "research into smaller, more focused questions."
        )
    
    # Search error suggestions
    if "search_error" in error_categories:
        suggestions.append(
            "Search services encountered issues. Try alternative search terms "
            "or check service availability."
        )
    
    # Analysis error suggestions
    if "analysis_error" in error_categories:
        suggestions.append(
            "Data analysis failed. Check data quality and ensure datasets "
            "are properly formatted."
        )
    
    # Stage-specific suggestions
    if "literature_review" in incomplete_stages:
        suggestions.append(
            "Literature review is incomplete. Consider manually searching for "
            "key papers or providing seed references."
        )
    
    if "analysis" in incomplete_stages:
        suggestions.append(
            "Analysis was not completed. The methodology section may need "
            "revision before analysis can proceed."
        )
    
    if "writing" in incomplete_stages:
        suggestions.append(
            "Paper sections could not be generated. Use the fallback sections "
            "as templates for manual completion."
        )
    
    # General suggestions
    if not suggestions:
        suggestions.append(
            "The workflow encountered unexpected errors. Review the error "
            "summary and consider rerunning with adjusted parameters."
        )
    
    suggestions.append(
        "You may resume this workflow from the last successful checkpoint "
        "using the workflow's time travel capabilities."
    )
    
    return suggestions


# =============================================================================
# Fallback Content Generation
# =============================================================================


def _generate_fallback_sections(state: WorkflowState) -> dict[str, str]:
    """Generate fallback content for paper sections.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary mapping section names to fallback content
    """
    sections = {}
    
    # Standard sections
    section_names = [
        "abstract",
        "introduction",
        "literature_review",
        "methods",
        "results",
        "discussion",
        "conclusion",
    ]
    
    for section in section_names:
        sections[section] = create_fallback_content(state, section)
    
    return sections


def _assemble_final_paper(
    state: WorkflowState,
    fallback_sections: dict[str, str],
) -> str:
    """Assemble the final paper from available and fallback content.
    
    Args:
        state: Current workflow state
        fallback_sections: Generated fallback sections
        
    Returns:
        Final paper as a single string
    """
    parts = []
    
    # Title
    title = state.get("project_title", "Research Paper")
    parts.append(f"# {title}")
    parts.append("")
    parts.append("**Note: This paper was generated with partial completion due to errors.**")
    parts.append("")
    
    # Get writer output if available
    writer_output = state.get("writer_output", {})
    written_sections = writer_output.get("sections", {}) if isinstance(writer_output, dict) else {}
    
    # Section order
    section_order = [
        ("abstract", "Abstract"),
        ("introduction", "Introduction"),
        ("literature_review", "Literature Review"),
        ("methods", "Methodology"),
        ("results", "Results"),
        ("discussion", "Discussion"),
        ("conclusion", "Conclusion"),
    ]
    
    for section_key, section_title in section_order:
        # Use written content if available, otherwise fallback
        if section_key in written_sections and written_sections[section_key]:
            content = written_sections[section_key]
            if isinstance(content, dict):
                content = content.get("content", str(content))
        else:
            content = fallback_sections.get(section_key, f"*{section_title} not available*")
        
        parts.append(f"## {section_title}")
        parts.append("")
        parts.append(content)
        parts.append("")
    
    # References section
    parts.append("## References")
    parts.append("")
    
    # Get citations if available
    citations = state.get("citations", [])
    if citations:
        for i, citation in enumerate(citations, 1):
            if isinstance(citation, dict):
                parts.append(f"{i}. {citation.get('formatted', str(citation))}")
            else:
                parts.append(f"{i}. {citation}")
    else:
        parts.append("*References not available - manual completion required*")
    
    parts.append("")
    
    # Appendix with error summary
    parts.append("## Appendix: Generation Notes")
    parts.append("")
    parts.append("### Completion Status")
    parts.append("")
    
    errors = state.get("errors", [])
    if errors:
        parts.append(f"This paper was partially generated with {len(errors)} errors encountered.")
        parts.append("")
        parts.append("### Errors Encountered")
        parts.append("")
        for error in errors[:5]:  # Limit to 5 errors
            node = getattr(error, 'node', 'unknown')
            message = getattr(error, 'message', str(error))
            parts.append(f"- **{node}**: {message}")
    else:
        parts.append("Generation completed with fallback due to workflow issues.")
    
    return "\n".join(parts)


# =============================================================================
# Routing Function
# =============================================================================


def should_fallback(state: WorkflowState) -> bool:
    """Determine if the workflow should route to fallback.
    
    Args:
        state: Current workflow state
        
    Returns:
        True if should route to fallback node
    """
    # Check explicit fallback flag
    if state.get("_should_fallback"):
        return True
    
    # Check error count
    errors = state.get("errors", [])
    if len(errors) >= 3:
        return True
    
    # Check for unrecoverable errors
    unrecoverable_count = sum(
        1 for e in errors
        if not getattr(e, 'recoverable', True)
    )
    if unrecoverable_count >= 1:
        return True
    
    # Check failed status
    if state.get("status") == ResearchStatus.FAILED:
        return True
    
    return False


def route_to_fallback_or_continue(
    state: WorkflowState,
) -> str:
    """Routing function for error-prone nodes.
    
    Use this as a conditional edge after nodes that might fail.
    
    Args:
        state: Current workflow state
        
    Returns:
        "fallback" or "continue"
    """
    if should_fallback(state):
        return "fallback"
    return "continue"
