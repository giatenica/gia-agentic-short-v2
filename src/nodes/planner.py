"""PLANNER node for designing research methodology.

This node:
1. Takes input from GAP_IDENTIFIER (refined question, gaps, contribution)
2. Selects appropriate methodology based on research type and gap
3. Designs analysis approach matching methodology and available data
4. Determines expected paper sections and outputs
5. Defines success criteria for the research
6. Uses HITL (interrupt) for human approval of research plan
"""

from datetime import datetime, timezone
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt

from src.config import settings
from src.state.enums import (
    ResearchStatus,
    MethodologyType,
    AnalysisApproach,
    PlanApprovalStatus,
)
from src.state.models import (
    GapAnalysis,
    LiteratureSynthesis,
    ContributionStatement,
    RefinedResearchQuestion,
    ResearchPlan,
    WorkflowError,
)
from src.state.schema import WorkflowState
from src.tools.methodology import (
    select_methodology,
    validate_methodology_fit,
    assess_feasibility,
    explain_methodology_choice,
)
from src.tools.analysis_design import (
    design_quantitative_analysis,
    design_qualitative_analysis,
    determine_paper_sections,
    define_success_criteria,
)


# =============================================================================
# Helper Functions
# =============================================================================


def _safe_analysis_approach(value: str | None) -> AnalysisApproach:
    """Safely convert string to AnalysisApproach enum, defaulting to OTHER."""
    if value is None:
        return AnalysisApproach.OTHER
    try:
        return AnalysisApproach(value)
    except ValueError:
        return AnalysisApproach.OTHER


def _extract_gap_info(state: WorkflowState) -> dict[str, Any]:
    """Extract gap information from state."""
    gap_analysis = state.get("gap_analysis")
    
    if isinstance(gap_analysis, dict):
        primary_gap = gap_analysis.get("primary_gap", {})
        if isinstance(primary_gap, dict):
            return {
                "gap_type": primary_gap.get("gap_type", "empirical"),
                "gap_title": primary_gap.get("title", ""),
                "gap_description": primary_gap.get("description", ""),
                "significance": primary_gap.get("significance", "medium"),
            }
    elif hasattr(gap_analysis, "primary_gap") and gap_analysis.primary_gap:
        return {
            "gap_type": gap_analysis.primary_gap.gap_type,
            "gap_title": gap_analysis.primary_gap.title,
            "gap_description": gap_analysis.primary_gap.description,
            "significance": gap_analysis.primary_gap.significance,
        }
    
    return {
        "gap_type": "empirical",
        "gap_title": "Research gap",
        "gap_description": "",
        "significance": "medium",
    }


def _extract_synthesis_info(state: WorkflowState) -> dict[str, Any]:
    """Extract literature synthesis information from state."""
    synthesis = state.get("literature_synthesis")
    
    if isinstance(synthesis, dict):
        return {
            "methodological_approaches": synthesis.get("methodological_approaches", []),
            "key_findings": synthesis.get("key_findings", []),
            "theoretical_frameworks": synthesis.get("theoretical_frameworks", []),
        }
    elif hasattr(synthesis, "methodological_approaches"):
        return {
            "methodological_approaches": synthesis.methodological_approaches,
            "key_findings": synthesis.key_findings,
            "theoretical_frameworks": synthesis.theoretical_frameworks,
        }
    
    return {
        "methodological_approaches": [],
        "key_findings": [],
        "theoretical_frameworks": [],
    }


def _extract_data_info(state: WorkflowState) -> dict[str, Any] | None:
    """Extract data exploration information from state."""
    data_results = state.get("data_exploration_results")
    
    if not data_results:
        return None
    
    if isinstance(data_results, dict):
        return {
            "total_rows": data_results.get("total_rows", 0),
            "total_columns": data_results.get("total_columns", 0),
            "quality_level": data_results.get("quality_level", "not_assessed"),
            "columns": data_results.get("columns", []),
            "feasibility_assessment": data_results.get("feasibility_assessment", ""),
        }
    elif hasattr(data_results, "total_rows"):
        return {
            "total_rows": data_results.total_rows,
            "total_columns": data_results.total_columns,
            "quality_level": data_results.quality_level.value if hasattr(data_results.quality_level, "value") else str(data_results.quality_level),
            "columns": [c.model_dump() if hasattr(c, "model_dump") else c for c in data_results.columns[:20]],
            "feasibility_assessment": data_results.feasibility_assessment,
        }
    
    return None


def _get_research_question(state: WorkflowState) -> str:
    """Get the best research question from state."""
    # Prefer refined query from gap identifier
    if state.get("refined_query"):
        return state["refined_query"]
    
    # Check refined_research_question
    refined_rq = state.get("refined_research_question")
    if refined_rq:
        if isinstance(refined_rq, dict):
            return refined_rq.get("refined_question", state.get("original_query", ""))
        elif hasattr(refined_rq, "refined_question"):
            return refined_rq.refined_question
    
    return state.get("original_query", "")


# =============================================================================
# PLANNER Node Implementation
# =============================================================================


def planner_node(state: WorkflowState) -> dict[str, Any]:
    """
    PLANNER node that designs research methodology.
    
    This node:
    1. Extracts context from prior stages (gap analysis, literature, data)
    2. Selects appropriate methodology
    3. Designs analysis approach
    4. Determines paper structure
    5. Defines success criteria
    6. Requests human approval via interrupt()
    
    Args:
        state: Current workflow state.
        
    Returns:
        Updated state with research plan.
    """
    # Extract context from state
    research_question = _get_research_question(state)
    gap_info = _extract_gap_info(state)
    synthesis_info = _extract_synthesis_info(state)
    data_info = _extract_data_info(state)
    
    research_type = state.get("research_type", "empirical")
    paper_type = state.get("paper_type", "short_article")
    contribution = state.get("contribution_statement", "")
    
    has_data = data_info is not None and data_info.get("total_rows", 0) > 0
    
    # Get methodology precedents from literature
    methodology_precedents = synthesis_info.get("methodological_approaches", [])
    
    try:
        # Step 1: Select methodology
        methodology_result = select_methodology(
            research_type=research_type,
            gap_type=gap_info["gap_type"],
            has_data=has_data,
            precedents=methodology_precedents,
            model_name=settings.default_model,
        )
        
        methodology_type = methodology_result["methodology_type"]
        methodology_justification = methodology_result["justification"]
        
        # Step 2: Validate methodology fit
        validation = validate_methodology_fit(
            methodology_type=methodology_type,
            gap_type=gap_info["gap_type"],
            research_type=research_type,
            has_data=has_data,
        )
        
        # Step 3: Assess feasibility
        feasibility = assess_feasibility(
            methodology_type=methodology_type,
            data_available=data_info,
            time_constraints=str(state.get("deadline", "")),
            model_name="claude-3-5-haiku-latest",
        )
        
        # Step 4: Design analysis approach
        if research_type.lower() in ["theoretical", "conceptual"]:
            analysis_design = design_qualitative_analysis(
                methodology_type=methodology_type,
                research_question=research_question,
                model_name="claude-3-5-haiku-latest",
            )
        else:
            key_vars = state.get("key_variables", [])
            if isinstance(key_vars, str):
                key_vars = [v.strip() for v in key_vars.split(",") if v.strip()]
            
            analysis_design = design_quantitative_analysis(
                methodology_type=methodology_type,
                research_question=research_question,
                key_variables=key_vars,
                data_info=data_info,
                model_name=settings.default_model,
            )
        
        # Step 5: Determine paper sections
        paper_sections = determine_paper_sections(
            paper_type=paper_type,
            methodology_type=methodology_type,
            research_type=research_type,
        )
        
        # Step 6: Define success criteria
        gap_analysis_dict = {
            "primary_gap": gap_info,
        }
        success_criteria = define_success_criteria(
            gap_analysis=gap_analysis_dict,
            methodology_type=methodology_type,
            research_question=research_question,
            model_name=settings.default_model,
        )
        
        # Step 7: Generate methodology explanation
        methodology_explanation = explain_methodology_choice(
            methodology_type=methodology_type,
            gap_analysis=gap_analysis_dict,
            literature_synthesis=synthesis_info,
            model_name=settings.default_model,
        )
        
        # Build the research plan
        research_plan = ResearchPlan(
            original_query=state.get("original_query", research_question),
            refined_query=research_question,
            target_gap=gap_info["gap_title"],
            gap_type=gap_info["gap_type"],
            methodology_type=methodology_type,
            methodology=methodology_type.value,
            methodology_justification=methodology_explanation,
            methodology_precedents=methodology_precedents[:5],
            analysis_approach=_safe_analysis_approach(methodology_result.get("recommended_analysis", "other")),
            analysis_design=analysis_design.get("full_design", ""),
            statistical_tests=analysis_design.get("statistical_tests", []),
            key_variables=analysis_design.get("independent_variables", state.get("key_variables", [])),
            control_variables=analysis_design.get("control_variables", []),
            data_requirements=analysis_design.get("data_requirements", []),
            expected_sections=paper_sections,
            expected_tables=analysis_design.get("expected_tables", []),
            expected_figures=analysis_design.get("expected_figures", []),
            success_criteria=success_criteria,
            contribution_statement=contribution or state.get("expected_contribution", ""),
            feasibility_score=feasibility.get("feasibility_score", 0.7),
            feasibility_notes=feasibility.get("assessment", ""),
            limitations=validation.get("issues", []),
            approval_status=PlanApprovalStatus.PENDING,
        )
        
        # Step 8: Request human approval via interrupt()
        approval_request = {
            "action": "approve_research_plan",
            "plan_summary": {
                "research_question": research_question,
                "methodology": methodology_type.value,
                "gap_addressed": gap_info["gap_title"],
                "feasibility_score": feasibility.get("feasibility_score", 0.7),
            },
            "methodology_justification": methodology_explanation[:1500],
            "expected_sections": paper_sections,
            "success_criteria": success_criteria,
            "analysis_approach": methodology_result.get("recommended_analysis", "other"),
            "validation_issues": validation.get("issues", []),
            "message": "Please review and approve the research plan. Reply with 'approve', 'reject', or provide specific feedback.",
        }
        
        # HITL: Wait for human approval
        approval_response = interrupt(approval_request)
        
        # Process approval response
        approved_plan = _process_plan_approval(research_plan, approval_response)
        
        # Create summary message
        summary_message = AIMessage(
            content=f"""Research Plan Created:

**Research Question:** {research_question}

**Methodology:** {methodology_type.value}
{methodology_justification[:500]}

**Analysis Approach:** {methodology_result.get("recommended_analysis", "To be determined")}

**Expected Sections:** {', '.join(paper_sections[:5])}...

**Success Criteria:**
{chr(10).join(f'- {c}' for c in success_criteria[:4])}

**Feasibility:** {feasibility.get("feasibility_score", 0.7):.0%}

**Status:** {approved_plan.approval_status.value}
"""
        )
        
        return {
            "research_plan": approved_plan,
            "messages": [summary_message],
            "status": ResearchStatus.PLANNING_COMPLETE if approved_plan.approval_status == PlanApprovalStatus.APPROVED else ResearchStatus.PLANNING,
        }
    
    except GraphInterrupt:
        # Re-raise GraphInterrupt - this is expected HITL behavior, not an error
        raise
        
    except Exception as e:
        error = WorkflowError(
            category="planner_error",
            message=f"Error in PLANNER node: {str(e)}",
            node="planner",
            recoverable=True,
        )
        
        return {
            "errors": [error],
            "messages": [AIMessage(content=f"Error creating research plan: {str(e)}")],
            "status": ResearchStatus.FAILED,
        }


def _process_plan_approval(
    plan: ResearchPlan,
    approval_response: dict[str, Any] | str,
) -> ResearchPlan:
    """
    Process the human approval response.
    
    Args:
        plan: The research plan awaiting approval
        approval_response: Human response from interrupt()
        
    Returns:
        Updated research plan with approval status.
    """
    # Handle string responses (from LangGraph Studio)
    if isinstance(approval_response, str):
        response_lower = approval_response.lower().strip()
        
        if response_lower in ["approve", "approved", "yes", "ok", "accept", "lgtm"]:
            plan.approval_status = PlanApprovalStatus.APPROVED
            plan.approval_notes = "Approved by researcher"
        elif response_lower in ["reject", "rejected", "no", "cancel"]:
            plan.approval_status = PlanApprovalStatus.REJECTED
            plan.approval_notes = "Rejected by researcher"
        else:
            # Treat as revision feedback
            plan.approval_status = PlanApprovalStatus.REVISION_REQUESTED
            plan.approval_notes = f"Revision requested: {approval_response}"
            plan.revision_count += 1
        
        return plan
    
    # Handle dictionary responses
    if isinstance(approval_response, dict):
        action = approval_response.get("action", "").lower()
        
        if action in ["approve", "approved"]:
            plan.approval_status = PlanApprovalStatus.APPROVED
            plan.approval_notes = approval_response.get("notes", "Approved by researcher")
            
            # Apply any modifications
            if approval_response.get("methodology"):
                try:
                    plan.methodology_type = MethodologyType(approval_response["methodology"])
                    plan.methodology = approval_response["methodology"]
                except ValueError:
                    # Invalid methodology type; keep original
                    pass
            
            if approval_response.get("additional_criteria"):
                plan.success_criteria.extend(approval_response["additional_criteria"])
                
        elif action in ["reject", "rejected"]:
            plan.approval_status = PlanApprovalStatus.REJECTED
            plan.approval_notes = approval_response.get("reason", "Rejected by researcher")
            
        else:
            # Revision requested
            plan.approval_status = PlanApprovalStatus.REVISION_REQUESTED
            plan.approval_notes = approval_response.get("feedback", str(approval_response))
            plan.revision_count += 1
    
    if hasattr(plan, "revised_at"):
        plan.revised_at = datetime.now(timezone.utc)
    return plan


# =============================================================================
# Routing Functions
# =============================================================================


def route_after_planner(state: WorkflowState) -> Literal["data_analyst", "conceptual_synthesizer", "__end__"]:
    """
    Route after PLANNER node based on research type and approval.
    
    Routes to:
    - data_analyst: For empirical research with data
    - conceptual_synthesizer: For theoretical/conceptual research
    - __end__: If plan rejected or error
    
    Args:
        state: Current workflow state.
        
    Returns:
        Next node name.
    """
    # Check for errors
    if state.get("errors"):
        return "__end__"
    
    # Check plan approval
    research_plan = state.get("research_plan")
    if research_plan:
        if isinstance(research_plan, dict):
            status = research_plan.get("approval_status", "pending")
        else:
            status = research_plan.approval_status.value if hasattr(research_plan.approval_status, "value") else str(research_plan.approval_status)
        
        if status in ["rejected", "revision_requested"]:
            return "__end__"
    
    # Route based on research type
    research_type = state.get("research_type", "empirical").lower()
    
    if research_type in ["theoretical", "conceptual", "literature_review"]:
        return "conceptual_synthesizer"
    else:
        # Check if data is available
        data_results = state.get("data_exploration_results")
        if data_results:
            return "data_analyst"
        else:
            # No data available, fall back to conceptual
            return "conceptual_synthesizer"
