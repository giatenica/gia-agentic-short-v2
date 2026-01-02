"""GAP_IDENTIFIER node for identifying research gaps and refining questions.

This node:
1. Analyzes literature synthesis to identify research gaps
2. Categorizes gaps by type (methodological, empirical, theoretical)
3. Assesses gap significance and selects primary gap
4. Refines the research question to target the gap
5. Generates a contribution statement
6. Uses HITL (interrupt) for human approval of refined question
"""

from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from src.state.enums import ResearchStatus
from src.state.models import (
    ContributionStatement,
    GapAnalysis,
    LiteratureSynthesis,
    RefinedResearchQuestion,
    ResearchGap,
    SearchResult,
    WorkflowError,
)
from src.state.schema import WorkflowState
from src.tools.contribution import (
    generate_contribution_statement,
    refine_research_question,
)
from src.tools.gap_analysis import (
    perform_gap_analysis,
)


# =============================================================================
# Gap Identification Functions
# =============================================================================


def identify_gaps(
    original_question: str,
    literature_synthesis: LiteratureSynthesis | dict[str, Any],
    data_exploration_results: dict[str, Any] | None = None,
    user_contribution: str | None = None,
    model_name: str | None = None,
) -> GapAnalysis:
    """
    Identify all research gaps from literature synthesis.
    
    Performs comprehensive gap analysis including:
    - Methodological gaps
    - Empirical gaps
    - Theoretical gaps
    
    Args:
        original_question: The original research question.
        literature_synthesis: Synthesis from LITERATURE_REVIEWER node.
        data_exploration_results: Optional data exploration results.
        user_contribution: Optional user's expected contribution.
        model_name: Model to use for analysis.
        
    Returns:
        Complete GapAnalysis object.
    """
    # Get data context if available
    data_context = None
    if data_exploration_results:
        if isinstance(data_exploration_results, dict):
            data_context = data_exploration_results.get("feasibility_assessment", "")
        else:
            data_context = getattr(data_exploration_results, "feasibility_assessment", "")
    
    # Use the comprehensive gap analysis function
    analysis = perform_gap_analysis(
        original_question=original_question,
        literature_synthesis=literature_synthesis,
        data_context=data_context,
        user_contribution=user_contribution,
        model_name=model_name,
    )
    
    return analysis


def select_primary_gap(
    gap_analysis: GapAnalysis,
    user_preferences: dict[str, Any] | None = None,
) -> ResearchGap | None:
    """
    Select the primary gap to address.
    
    Considers:
    - Gap significance
    - Addressability
    - Alignment with user preferences
    - Feasibility
    
    Args:
        gap_analysis: The gap analysis results.
        user_preferences: Optional user preferences for gap selection.
        
    Returns:
        The selected primary gap, or None if no suitable gap found.
    """
    if gap_analysis.primary_gap:
        return gap_analysis.primary_gap
    
    if not gap_analysis.gaps:
        return None
    
    # Sort by significance and addressability
    addressable_gaps = [g for g in gap_analysis.gaps if g.addressable]
    
    if not addressable_gaps:
        # Fall back to any gap
        addressable_gaps = gap_analysis.gaps
    
    # Prioritize high significance gaps
    high_sig = [g for g in addressable_gaps if g.significance == "high"]
    if high_sig:
        return high_sig[0]
    
    medium_sig = [g for g in addressable_gaps if g.significance == "medium"]
    if medium_sig:
        return medium_sig[0]
    
    return addressable_gaps[0] if addressable_gaps else None


# =============================================================================
# Question Refinement
# =============================================================================


def create_refined_question(
    original_question: str,
    gap_analysis: GapAnalysis,
    literature_synthesis: LiteratureSynthesis | dict[str, Any],
    model_name: str | None = None,
) -> RefinedResearchQuestion:
    """
    Create a refined research question targeting the primary gap.
    
    Args:
        original_question: The original research question.
        gap_analysis: Gap analysis results.
        literature_synthesis: Literature synthesis results.
        model_name: Model to use for refinement.
        
    Returns:
        RefinedResearchQuestion object.
    """
    result = refine_research_question(
        original_question=original_question,
        gap_analysis=gap_analysis,
        literature_synthesis=literature_synthesis,
        model_name=model_name,
    )
    
    return RefinedResearchQuestion(
        original_question=original_question,
        refined_question=result.get("refined_question", original_question),
        refinement_rationale=result.get("refinement_rationale", ""),
        gap_targeted=gap_analysis.primary_gap.title if gap_analysis.primary_gap else "",
        scope_changes=result.get("scope_changes", []),
        specificity_score=0.7,  # Could be computed from model output
        feasibility_score=0.7,
        novelty_score=0.7,
    )


# =============================================================================
# Contribution Generation
# =============================================================================


def create_contribution_statement(
    refined_question: str,
    gap_analysis: GapAnalysis,
    literature_synthesis: LiteratureSynthesis | dict[str, Any],
    user_contribution: str | None = None,
    model_name: str | None = None,
) -> ContributionStatement:
    """
    Create a contribution statement based on the gap and refined question.
    
    Args:
        refined_question: The refined research question.
        gap_analysis: Gap analysis results.
        literature_synthesis: Literature synthesis results.
        user_contribution: Optional user's expected contribution.
        model_name: Model to use for generation.
        
    Returns:
        ContributionStatement object.
    """
    if not gap_analysis.primary_gap:
        # Create a minimal contribution statement
        return ContributionStatement(
            main_statement="This research addresses an identified gap in the literature.",
            contribution_type="empirical",
            gap_addressed="See gap analysis for details.",
            novelty_explanation="",
            potential_impact="",
            target_audience=[],
        )
    
    statement = generate_contribution_statement(
        refined_question=refined_question,
        primary_gap=gap_analysis.primary_gap,
        literature_synthesis=literature_synthesis,
        user_expected_contribution=user_contribution,
        model_name=model_name,
    )
    
    return statement


# =============================================================================
# HITL Approval
# =============================================================================


def prepare_approval_request(
    original_question: str,
    refined_question: RefinedResearchQuestion,
    gap_analysis: GapAnalysis,
    contribution: ContributionStatement,
) -> dict[str, Any]:
    """
    Prepare the data for human approval request.
    
    Args:
        original_question: The original research question.
        refined_question: The refined question.
        gap_analysis: Gap analysis results.
        contribution: The contribution statement.
        
    Returns:
        Dictionary formatted for interrupt().
    """
    # Format gaps for display
    gaps_summary = []
    for gap in gap_analysis.gaps[:5]:  # Top 5 gaps
        gaps_summary.append({
            "type": gap.gap_type,
            "title": gap.title,
            "significance": gap.significance,
            "addressable": gap.addressable,
        })
    
    # Format primary gap
    primary_gap_info = None
    if gap_analysis.primary_gap:
        primary_gap_info = {
            "type": gap_analysis.primary_gap.gap_type,
            "title": gap_analysis.primary_gap.title,
            "description": gap_analysis.primary_gap.description,
            "significance": gap_analysis.primary_gap.significance,
        }
    
    return {
        "action": "approve_refined_question",
        "message": "Please review the refined research question and contribution statement. "
                   "You can approve as-is, modify the refined question, or reject to try again.",
        "original_question": original_question,
        "refined_question": refined_question.refined_question,
        "refinement_rationale": refined_question.refinement_rationale,
        "scope_changes": refined_question.scope_changes,
        "gap_analysis": {
            "coverage_percentage": gap_analysis.coverage_percentage,
            "total_gaps_found": len(gap_analysis.gaps),
            "primary_gap": primary_gap_info,
            "all_gaps": gaps_summary,
        },
        "contribution_statement": contribution.main_statement,
        "contribution_type": contribution.contribution_type,
        "novelty": contribution.novelty_explanation,
        "allowed_actions": ["approve", "modify", "reject"],
    }


def process_approval_response(
    response: dict[str, Any],
    refined_question: RefinedResearchQuestion,
    contribution: ContributionStatement,
) -> tuple[str, str]:
    """
    Process the human's approval response.
    
    Args:
        response: The response from interrupt().
        refined_question: The original refined question.
        contribution: The original contribution statement.
        
    Returns:
        Tuple of (final_refined_question, final_contribution_statement).
    """
    action = response.get("action", "approve")
    
    if action == "approve":
        return refined_question.refined_question, contribution.main_statement
    
    elif action == "modify":
        # Use human's modifications
        modified_question = response.get("refined_question", refined_question.refined_question)
        modified_contribution = response.get("contribution", contribution.main_statement)
        return modified_question, modified_contribution
    
    elif action == "reject":
        # Return original question and empty contribution
        return refined_question.original_question, ""
    
    else:
        # Unknown action, default to approve
        return refined_question.refined_question, contribution.main_statement


# =============================================================================
# GAP_IDENTIFIER Node
# =============================================================================


def gap_identifier_node(state: WorkflowState) -> dict[str, Any]:
    """
    GAP_IDENTIFIER node for identifying research gaps.
    
    This node:
    1. Analyzes literature synthesis to find gaps
    2. Selects the primary gap to address
    3. Refines the research question to target the gap
    4. Generates a contribution statement
    5. Requests human approval via interrupt()
    
    Args:
        state: Current workflow state.
        
    Returns:
        Updated state with gap analysis, refined query, and contribution.
    """
    # Validate required inputs
    if not state.get("original_query"):
        return {
            "errors": [
                WorkflowError(
                    node="gap_identifier",
                    category="validation",
                    message="No original research question found in state",
                    recoverable=False,
                )
            ],
            "status": ResearchStatus.FAILED,
        }
    
    # Get literature synthesis
    literature_synthesis = state.get("literature_synthesis")
    if not literature_synthesis:
        # No synthesis available; create minimal one from search results
        search_results = state.get("search_results", [])
        literature_synthesis = _create_minimal_synthesis(search_results)
    
    original_question = state["original_query"]
    
    # Step 1: Identify gaps
    gap_analysis = identify_gaps(
        original_question=original_question,
        literature_synthesis=literature_synthesis,
        data_exploration_results=state.get("data_exploration_results"),
        user_contribution=state.get("expected_contribution"),
    )
    
    # Step 2: Select primary gap
    primary_gap = select_primary_gap(gap_analysis)
    if primary_gap:
        gap_analysis.primary_gap = primary_gap
    
    # Step 3: Refine research question
    refined_question = create_refined_question(
        original_question=original_question,
        gap_analysis=gap_analysis,
        literature_synthesis=literature_synthesis,
    )
    
    # Step 4: Generate contribution statement
    contribution = create_contribution_statement(
        refined_question=refined_question.refined_question,
        gap_analysis=gap_analysis,
        literature_synthesis=literature_synthesis,
        user_contribution=state.get("expected_contribution"),
    )
    
    # Step 5: Human approval via interrupt()
    approval_request = prepare_approval_request(
        original_question=original_question,
        refined_question=refined_question,
        gap_analysis=gap_analysis,
        contribution=contribution,
    )
    
    # This will pause execution until human responds
    approved = interrupt(approval_request)
    
    # Step 6: Process approval
    final_question, final_contribution = process_approval_response(
        response=approved,
        refined_question=refined_question,
        contribution=contribution,
    )
    
    # Prepare gaps for state (as list of strings for simpler state)
    identified_gaps = [
        f"[{g.gap_type.upper()}] {g.title}: {g.description[:100]}..."
        for g in gap_analysis.gaps
    ]
    
    return {
        "refined_query": final_question,
        "gap_analysis": gap_analysis.model_dump(),
        "contribution_statement": final_contribution,
        "identified_gaps": identified_gaps,
        "status": ResearchStatus.GAP_IDENTIFICATION_COMPLETE,
        "messages": [
            AIMessage(
                content=f"Gap analysis complete. Identified {len(gap_analysis.gaps)} gaps. "
                        f"Primary gap: {gap_analysis.primary_gap.title if gap_analysis.primary_gap else 'None'}. "
                        f"Refined question approved."
            )
        ],
    }


def _create_minimal_synthesis(search_results: list[SearchResult | dict]) -> dict[str, Any]:
    """Create a minimal synthesis from search results when none exists."""
    if not search_results:
        return {
            "summary": "No literature synthesis available.",
            "state_of_field": "Literature review not yet completed.",
            "key_findings": [],
            "theoretical_frameworks": [],
            "methodological_approaches": [],
            "contribution_opportunities": [],
        }
    
    # Extract basic info from search results
    findings = []
    for result in search_results[:10]:
        if isinstance(result, dict):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
        else:
            title = result.title
            snippet = result.snippet
        
        if title and snippet:
            findings.append(f"{title}: {snippet[:100]}")
    
    return {
        "summary": f"Based on {len(search_results)} search results.",
        "state_of_field": "See search results for current literature.",
        "key_findings": findings[:5],
        "theoretical_frameworks": [],
        "methodological_approaches": [],
        "contribution_opportunities": ["Gap analysis needed based on search results."],
    }


# =============================================================================
# Routing Functions
# =============================================================================


def should_refine_further(state: WorkflowState) -> Literal["refine", "proceed"]:
    """
    Determine if the research question needs further refinement.
    
    Args:
        state: Current workflow state.
        
    Returns:
        "refine" if more refinement needed, "proceed" otherwise.
    """
    # Check if gap analysis found sufficient gaps
    gap_analysis = state.get("gap_analysis")
    if not gap_analysis:
        return "refine"
    
    if isinstance(gap_analysis, dict):
        primary_gap = gap_analysis.get("primary_gap")
    else:
        primary_gap = gap_analysis.primary_gap
    
    if not primary_gap:
        return "refine"
    
    # Check if contribution statement was generated
    if not state.get("contribution_statement"):
        return "refine"
    
    return "proceed"


def route_after_gap_identifier(state: WorkflowState) -> Literal["planner", "error"]:
    """
    Route after gap identification based on state.
    
    Args:
        state: Current workflow state.
        
    Returns:
        Next node to route to.
    """
    if state.get("status") == ResearchStatus.FAILED:
        return "error"
    
    if state.get("refined_query") and state.get("contribution_statement"):
        return "planner"
    
    return "error"


# =============================================================================
# Graph Integration Helpers
# =============================================================================


def create_gap_identifier_subgraph():
    """
    Create a subgraph for gap identification with iterative refinement.
    
    This could be used if multiple rounds of gap analysis are needed.
    
    Returns:
        Compiled subgraph for gap identification.
    """
    from langgraph.graph import StateGraph, START, END
    
    # For now, return None as we use the simple node approach
    # This could be expanded for iterative refinement workflows
    return None


def get_gap_identifier_tools():
    """
    Get the tools available for gap identification.
    
    Returns:
        List of tool functions for gap analysis.
    """
    from src.tools.gap_analysis import (
        compare_coverage_tool,
        identify_gaps_tool,
    )
    from src.tools.contribution import (
        generate_contribution_tool,
        refine_question_tool,
    )
    
    return [
        compare_coverage_tool,
        identify_gaps_tool,
        generate_contribution_tool,
        refine_question_tool,
    ]
