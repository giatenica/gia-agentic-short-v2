"""REVIEWER Node for GIA Agentic v2.

This node critically evaluates the paper produced by the WRITER node,
scores it across quality dimensions, and determines whether to:
- APPROVE: Paper meets quality standards (score >= 7.0)
- REVISE: Paper needs revision (score 4.0-6.9)
- REJECT: Paper has fundamental issues (score < 4.0)

Includes HITL (Human-in-the-Loop) interrupt for final approval.
"""

import logging
from datetime import datetime, timezone

from langgraph.types import interrupt

from src.review.criteria import (
    evaluate_paper,
    EVALUATION_DIMENSIONS,
)
from src.state.models import (
    ReviewCritique,
    RevisionRequest,
    ReviewerOutput,
    WorkflowError,
    calculate_overall_score,
    determine_review_decision,
)
from src.state.schema import WorkflowState
from src.state.enums import ResearchStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

MAX_REVISIONS = 3
APPROVE_THRESHOLD = 7.0
REVISE_MIN_THRESHOLD = 4.0


# =============================================================================
# REVIEWER Node
# =============================================================================


def reviewer_node(state: WorkflowState) -> dict:
    """
    REVIEWER node for paper quality evaluation.
    
    This node:
    1. Evaluates the paper across 5 quality dimensions
    2. Calculates an overall weighted score
    3. Determines the review decision (approve/revise/reject)
    4. Generates specific critique items and revision instructions
    5. Triggers HITL interrupt for human approval
    
    Args:
        state: Current workflow state with writer_output
        
    Returns:
        Updated state with review results
    """
    logger.info("REVIEWER: Starting paper evaluation")
    
    # Get writer output
    writer_output = state.get("writer_output")
    if not writer_output:
        logger.error("REVIEWER: No writer_output in state")
        return {
            "status": ResearchStatus.FAILED,
            "errors": state.get("errors", []) + [
                WorkflowError(
                    node="reviewer",
                    message="No writer output to review",
                    category="validation_error",
                    recoverable=False,
                )
            ],
        }
    
    # Get context from state
    research_question = state.get("original_query", "")
    identified_gaps = state.get("identified_gaps", [])
    research_plan = state.get("research_plan")
    analysis = state.get("analysis")
    target_journal = state.get("target_journal", "generic")
    
    # Convert research_plan to dict if needed
    research_plan_dict = None
    if research_plan:
        if hasattr(research_plan, "model_dump"):
            research_plan_dict = research_plan.model_dump()
        elif isinstance(research_plan, dict):
            research_plan_dict = research_plan
    
    # Convert analysis to dict if needed
    analysis_dict = None
    if analysis:
        if hasattr(analysis, "model_dump"):
            analysis_dict = analysis.model_dump()
        elif isinstance(analysis, dict):
            analysis_dict = analysis
    
    # Get current revision count
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", MAX_REVISIONS)
    
    logger.info(f"REVIEWER: Revision {revision_count + 1} of max {max_revisions}")
    
    # ==========================================================================
    # Evaluate the paper
    # ==========================================================================
    
    try:
        dimension_scores, critique_items = evaluate_paper(
            writer_output=writer_output,
            research_question=research_question,
            identified_gaps=identified_gaps,
            research_plan=research_plan_dict,
            analysis_results=analysis_dict,
            target_journal=target_journal,
        )
    except Exception as e:
        logger.error(f"REVIEWER: Evaluation failed: {e}")
        return {
            "status": ResearchStatus.FAILED,
            "errors": state.get("errors", []) + [
                WorkflowError(
                    node="reviewer",
                    message=f"Paper evaluation failed: {str(e)}",
                    category="evaluation_error",
                    recoverable=True,
                )
            ],
        }
    
    # ==========================================================================
    # Calculate overall score and decision
    # ==========================================================================
    
    overall_score = calculate_overall_score(dimension_scores)
    has_critical = any(item.severity == "critical" for item in critique_items)
    decision = determine_review_decision(overall_score, has_critical)
    
    logger.info(
        f"REVIEWER: Overall score: {overall_score:.2f}, "
        f"Critical issues: {has_critical}, Decision: {decision}"
    )
    
    # Log dimension scores
    for score in dimension_scores:
        logger.info(f"  {score.dimension}: {score.score:.2f}")
    
    # ==========================================================================
    # Create review critique
    # ==========================================================================
    
    # Generate summary
    summary_parts = [
        f"Paper evaluation complete with overall score of {overall_score:.1f}/10.",
        f"Reviewed across {len(dimension_scores)} quality dimensions.",
    ]
    
    if decision == "approve":
        summary_parts.append("The paper meets quality standards and is recommended for approval.")
    elif decision == "revise":
        summary_parts.append(f"The paper requires revision. {len(critique_items)} issues identified.")
    else:
        summary_parts.append("The paper has fundamental issues that require major rework.")
    
    summary = " ".join(summary_parts)
    
    # Generate revision instructions
    revision_instructions = ""
    if decision in ["revise", "reject"]:
        instruction_parts = ["Revision priorities:"]
        
        # Group critique items by severity
        critical_items = [i for i in critique_items if i.severity == "critical"]
        major_items = [i for i in critique_items if i.severity == "major"]
        minor_items = [i for i in critique_items if i.severity == "minor"]
        
        if critical_items:
            instruction_parts.append("\nCRITICAL (must fix):")
            for item in critical_items[:5]:
                instruction_parts.append(f"- [{item.section}] {item.issue}")
        
        if major_items:
            instruction_parts.append("\nMAJOR (should fix):")
            for item in major_items[:5]:
                instruction_parts.append(f"- [{item.section}] {item.issue}")
        
        if minor_items and len(minor_items) <= 3:
            instruction_parts.append("\nMINOR (nice to fix):")
            for item in minor_items:
                instruction_parts.append(f"- [{item.section}] {item.issue}")
        
        revision_instructions = "\n".join(instruction_parts)
    
    # Create the critique object
    review_critique = ReviewCritique(
        overall_score=overall_score,
        decision=decision,
        dimension_scores=dimension_scores,
        critique_items=critique_items,
        summary=summary,
        revision_instructions=revision_instructions,
        iteration=revision_count + 1,
    )
    
    # ==========================================================================
    # Handle revision limit
    # ==========================================================================
    
    if decision == "revise" and revision_count >= max_revisions:
        logger.warning(
            f"REVIEWER: Max revisions ({max_revisions}) reached. "
            "Escalating for human decision."
        )
        decision = "escalate"
        review_critique.summary += (
            f" Maximum revision limit ({max_revisions}) reached. "
            "Escalating to human reviewer for final decision."
        )
    
    # ==========================================================================
    # Create revision request if needed
    # ==========================================================================
    
    revision_request = None
    if decision == "revise":
        # Determine which sections need revision
        sections_to_revise = list(set(item.section for item in critique_items if item.severity in ["critical", "major"]))
        
        # Prioritize by severity
        priority_order = []
        for section in sections_to_revise:
            section_criticals = len([i for i in critique_items if i.section == section and i.severity == "critical"])
            section_majors = len([i for i in critique_items if i.section == section and i.severity == "major"])
            priority_order.append((section, section_criticals * 10 + section_majors))
        priority_order.sort(key=lambda x: x[1], reverse=True)
        
        revision_request = RevisionRequest(
            sections_to_revise=sections_to_revise,
            critique_items=critique_items,
            revision_instructions=revision_instructions,
            iteration_count=revision_count + 1,
            max_iterations=max_revisions,
            priority_order=[s[0] for s in priority_order],
        )
    
    # ==========================================================================
    # HITL Interrupt for human approval
    # ==========================================================================
    
    # Prepare human review summary
    human_review_summary = {
        "decision": decision,
        "overall_score": overall_score,
        "dimension_scores": {s.dimension: s.score for s in dimension_scores},
        "summary": summary,
        "critique_count": len(critique_items),
        "critical_issues": len([i for i in critique_items if i.severity == "critical"]),
        "revision_count": revision_count,
        "max_revisions": max_revisions,
    }
    
    if decision == "approve":
        human_review_summary["message"] = (
            "The paper has been evaluated and meets quality standards. "
            "Please review and confirm approval."
        )
    elif decision == "revise":
        human_review_summary["message"] = (
            f"The paper requires revision. {len(critique_items)} issues identified. "
            "Please review the critique and approve sending for revision."
        )
        human_review_summary["revision_instructions"] = revision_instructions
    elif decision == "escalate":
        human_review_summary["message"] = (
            "Maximum revision limit reached. Please make a final decision: "
            "approve the current draft, request another revision, or reject."
        )
    else:  # reject
        human_review_summary["message"] = (
            "The paper has fundamental issues. Please review and confirm rejection "
            "or override to request revision."
        )
    
    # Trigger HITL interrupt
    logger.info("REVIEWER: Triggering HITL interrupt for human approval")
    human_response = interrupt(human_review_summary)
    
    # ==========================================================================
    # Process human response
    # ==========================================================================
    
    human_approved = False
    human_feedback = None
    human_override_decision = None
    
    if isinstance(human_response, dict):
        human_approved = human_response.get("approved", False)
        human_feedback = human_response.get("feedback")
        human_override_decision = human_response.get("override_decision")
    elif isinstance(human_response, bool):
        human_approved = human_response
    elif isinstance(human_response, str):
        human_approved = human_response.lower() in ["yes", "true", "approve", "approved", "ok"]
        human_feedback = human_response
    
    # Apply human override if provided
    if human_override_decision:
        logger.info(f"REVIEWER: Human override decision: {human_override_decision}")
        decision = human_override_decision
        review_critique.decision = human_override_decision
    
    logger.info(
        f"REVIEWER: Human response - approved: {human_approved}, "
        f"feedback: {human_feedback}, final decision: {decision}"
    )
    
    # ==========================================================================
    # Create output and final paper if approved
    # ==========================================================================
    
    final_paper = None
    if decision == "approve" and human_approved:
        # Generate final paper markdown
        if hasattr(writer_output, "to_markdown"):
            final_paper = writer_output.to_markdown()
        elif isinstance(writer_output, dict) and "to_markdown" in dir(writer_output):
            final_paper = writer_output.to_markdown()
        else:
            # Manual markdown generation
            final_paper = _generate_paper_markdown(writer_output)
    
    reviewer_output = ReviewerOutput(
        critique=review_critique,
        decision=decision,
        revision_request=revision_request if decision == "revise" else None,
        human_approved=human_approved,
        human_feedback=human_feedback,
        final_paper=final_paper,
    )
    
    # ==========================================================================
    # Determine next status
    # ==========================================================================
    
    if decision == "approve" and human_approved:
        new_status = ResearchStatus.COMPLETED
    elif decision == "revise":
        new_status = ResearchStatus.REVISION_NEEDED
    else:
        new_status = ResearchStatus.REVIEWING
    
    return {
        "review_critique": review_critique,
        "review_decision": decision,
        "revision_request": revision_request,
        "reviewer_output": reviewer_output,
        "revision_count": revision_count + 1 if decision == "revise" else revision_count,
        "human_approved": human_approved,
        "human_feedback": human_feedback,
        "status": new_status,
        "updated_at": datetime.now(timezone.utc),
    }


# =============================================================================
# Helper Functions
# =============================================================================


def _generate_paper_markdown(writer_output: dict) -> str:
    """Generate markdown from writer output."""
    lines = []
    
    # Get title
    title = ""
    if isinstance(writer_output, dict):
        title = writer_output.get("title", "Research Paper")
    elif hasattr(writer_output, "title"):
        title = writer_output.title
    
    lines.append(f"# {title}")
    lines.append("")
    
    # Get sections
    sections = []
    if isinstance(writer_output, dict):
        sections = writer_output.get("sections", [])
    elif hasattr(writer_output, "sections"):
        sections = writer_output.sections
    
    # Sort by order if available
    def get_order(s):
        if isinstance(s, dict):
            return s.get("order", 0)
        return getattr(s, "order", 0)
    
    sections = sorted(sections, key=get_order)
    
    # Add each section
    for section in sections:
        if isinstance(section, dict):
            section_type = section.get("section_type", "")
            title = section.get("title", section_type.title())
            content = section.get("content", "")
        else:
            section_type = getattr(section, "section_type", "")
            title = getattr(section, "title", section_type.title())
            content = getattr(section, "content", "")
        
        if section_type != "references":
            lines.append(f"## {title}")
            lines.append("")
            lines.append(content)
            lines.append("")
    
    # Add references if available
    reference_list = None
    if isinstance(writer_output, dict):
        reference_list = writer_output.get("reference_list")
    elif hasattr(writer_output, "reference_list"):
        reference_list = writer_output.reference_list
    
    if reference_list:
        lines.append("## References")
        lines.append("")
        if hasattr(reference_list, "format_reference_list"):
            lines.append(reference_list.format_reference_list())
        elif isinstance(reference_list, dict) and "entries" in reference_list:
            for entry in reference_list["entries"]:
                if isinstance(entry, dict):
                    lines.append(f"- {entry.get('formatted', entry.get('citation_key', ''))}")
                else:
                    lines.append(f"- {entry}")
    
    return "\n".join(lines)


# =============================================================================
# Routing Function
# =============================================================================


def route_after_reviewer(state: WorkflowState) -> str:
    """
    Route after REVIEWER node based on review decision.
    
    Returns:
        - "output" if approved
        - "writer" if revision needed (loops back)
        - "__end__" if rejected or error
    """
    decision = state.get("review_decision")
    human_approved = state.get("human_approved", False)
    
    logger.info(f"REVIEWER routing: decision={decision}, human_approved={human_approved}")
    
    if decision == "approve" and human_approved:
        return "output"
    elif decision == "revise":
        return "writer"
    else:
        # reject or error
        return "__end__"
