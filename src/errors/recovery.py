"""Recovery strategies for workflow failures.

This module provides strategies for recovering from errors and
generating partial output when the workflow cannot complete normally.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.state.enums import ResearchStatus
from src.errors.exceptions import (
    GIAError,
    RateLimitError,
    ContextOverflowError,
    SearchError,
    AnalysisError,
    WritingError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Recovery Types
# =============================================================================


class RecoveryAction(str, Enum):
    """Actions that can be taken for recovery."""
    
    RETRY = "retry"                    # Retry the failed operation
    SKIP = "skip"                      # Skip and continue to next step
    FALLBACK = "fallback"              # Use fallback node
    REDUCE_CONTENT = "reduce_content"  # Reduce content and retry
    WAIT_AND_RETRY = "wait_and_retry"  # Wait then retry
    ABORT = "abort"                    # Abort the workflow
    PARTIAL_OUTPUT = "partial_output"  # Generate partial output


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from an error.
    
    Attributes:
        action: The recovery action to take
        reason: Why this strategy was chosen
        params: Additional parameters for the action
        fallback_actions: Alternative actions if primary fails
    """
    
    action: RecoveryAction
    reason: str
    params: dict[str, Any] = field(default_factory=dict)
    fallback_actions: list[RecoveryAction] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "reason": self.reason,
            "params": self.params,
            "fallback_actions": [a.value for a in self.fallback_actions],
        }


# =============================================================================
# Strategy Determination
# =============================================================================


def determine_recovery_strategy(
    error: Exception,
    node: str,
    state: dict[str, Any],
    attempt: int = 0,
) -> RecoveryStrategy:
    """Determine the best recovery strategy for an error.
    
    Args:
        error: The exception that occurred
        node: Node where the error occurred
        state: Current workflow state
        attempt: Number of previous recovery attempts
        
    Returns:
        Recovery strategy to use
    """
    # Check total error count
    error_count = len(state.get("errors", []))
    
    # Too many errors - go to fallback
    if error_count >= 3:
        return RecoveryStrategy(
            action=RecoveryAction.FALLBACK,
            reason=f"Error count ({error_count}) exceeds threshold",
            fallback_actions=[RecoveryAction.PARTIAL_OUTPUT],
        )
    
    # Rate limit - wait and retry
    if isinstance(error, RateLimitError):
        wait_time = error.retry_after or 30
        if attempt < 2:
            return RecoveryStrategy(
                action=RecoveryAction.WAIT_AND_RETRY,
                reason="Rate limited, will wait and retry",
                params={"wait_seconds": wait_time},
                fallback_actions=[RecoveryAction.SKIP],
            )
        else:
            return RecoveryStrategy(
                action=RecoveryAction.SKIP,
                reason="Rate limit persists after retries",
            )
    
    # Context overflow - reduce content
    if isinstance(error, ContextOverflowError):
        if attempt < 2:
            return RecoveryStrategy(
                action=RecoveryAction.REDUCE_CONTENT,
                reason="Content exceeds context window",
                params={
                    "reduction_factor": 0.5 if attempt == 0 else 0.3,
                    "token_count": getattr(error, 'token_count', None),
                    "max_tokens": getattr(error, 'max_tokens', None),
                },
                fallback_actions=[RecoveryAction.FALLBACK],
            )
        else:
            return RecoveryStrategy(
                action=RecoveryAction.FALLBACK,
                reason="Content cannot be reduced further",
            )
    
    # Search error - try alternative sources or skip
    if isinstance(error, SearchError):
        source = getattr(error, 'source', 'unknown')
        if attempt < 1:
            return RecoveryStrategy(
                action=RecoveryAction.RETRY,
                reason=f"Search failed on {source}, retrying",
                params={"try_alternative_source": True},
                fallback_actions=[RecoveryAction.SKIP],
            )
        else:
            return RecoveryStrategy(
                action=RecoveryAction.SKIP,
                reason="Search unavailable, continuing with available results",
            )
    
    # Analysis error - usually not retryable
    if isinstance(error, AnalysisError):
        return RecoveryStrategy(
            action=RecoveryAction.SKIP,
            reason="Analysis failed, will use available data",
            fallback_actions=[RecoveryAction.FALLBACK],
        )
    
    # Writing error - retry or fallback
    if isinstance(error, WritingError):
        if attempt < 1:
            return RecoveryStrategy(
                action=RecoveryAction.RETRY,
                reason="Writing failed, retrying with simplified prompt",
                params={"simplify_prompt": True},
                fallback_actions=[RecoveryAction.PARTIAL_OUTPUT],
            )
        else:
            return RecoveryStrategy(
                action=RecoveryAction.PARTIAL_OUTPUT,
                reason="Writing cannot be completed, generating partial output",
            )
    
    # Check if error is unrecoverable
    if isinstance(error, GIAError) and not error.recoverable:
        return RecoveryStrategy(
            action=RecoveryAction.FALLBACK,
            reason="Unrecoverable error",
            fallback_actions=[RecoveryAction.ABORT],
        )
    
    # Default: retry once, then skip
    if attempt < 1:
        return RecoveryStrategy(
            action=RecoveryAction.RETRY,
            reason="Unexpected error, retrying",
            fallback_actions=[RecoveryAction.SKIP, RecoveryAction.FALLBACK],
        )
    else:
        return RecoveryStrategy(
            action=RecoveryAction.SKIP,
            reason="Error persists, skipping step",
            fallback_actions=[RecoveryAction.FALLBACK],
        )


# =============================================================================
# Recovery Execution
# =============================================================================


def execute_recovery(
    strategy: RecoveryStrategy,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Execute a recovery strategy and return state updates.
    
    Args:
        strategy: The recovery strategy to execute
        state: Current workflow state
        
    Returns:
        State updates for recovery
    """
    logger.info(
        f"Executing recovery: {strategy.action.value} - {strategy.reason}"
    )
    
    updates: dict[str, Any] = {
        "_recovery_applied": True,
        "_recovery_action": strategy.action.value,
        "_recovery_reason": strategy.reason,
    }
    
    if strategy.action == RecoveryAction.RETRY:
        updates["_should_retry"] = True
        updates.update(strategy.params)
    
    elif strategy.action == RecoveryAction.WAIT_AND_RETRY:
        wait_seconds = strategy.params.get("wait_seconds", 30)
        updates["_should_retry"] = True
        updates["_retry_delay"] = wait_seconds
    
    elif strategy.action == RecoveryAction.REDUCE_CONTENT:
        factor = strategy.params.get("reduction_factor", 0.5)
        updates["_content_reduction_factor"] = factor
        updates["_should_retry"] = True
    
    elif strategy.action == RecoveryAction.SKIP:
        updates["_should_skip"] = True
    
    elif strategy.action == RecoveryAction.FALLBACK:
        updates["_should_fallback"] = True
    
    elif strategy.action == RecoveryAction.PARTIAL_OUTPUT:
        updates["_generate_partial"] = True
    
    elif strategy.action == RecoveryAction.ABORT:
        updates["status"] = ResearchStatus.FAILED
    
    return updates


def can_continue_workflow(state: dict[str, Any]) -> bool:
    """Determine if the workflow can continue after errors.
    
    Args:
        state: Current workflow state
        
    Returns:
        True if workflow can continue
    """
    # Check if explicitly marked for fallback/abort
    if state.get("_should_fallback"):
        return False
    
    if state.get("status") == ResearchStatus.FAILED:
        return False
    
    # Check error count
    errors = state.get("errors", [])
    if len(errors) >= 5:
        logger.warning("Too many errors, workflow cannot continue")
        return False
    
    # Check for critical unrecoverable errors
    unrecoverable_count = sum(
        1 for e in errors if not getattr(e, 'recoverable', True)
    )
    if unrecoverable_count >= 2:
        logger.warning("Multiple unrecoverable errors, workflow cannot continue")
        return False
    
    return True


# =============================================================================
# Partial Output Generation
# =============================================================================


def get_partial_output(state: dict[str, Any]) -> dict[str, Any]:
    """Extract available output from an incomplete workflow state.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary of available outputs
    """
    output: dict[str, Any] = {
        "partial": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    # Project info
    if state.get("project_title"):
        output["project_title"] = state["project_title"]
    
    # Research question
    if state.get("refined_query"):
        output["research_question"] = state["refined_query"]
    elif state.get("original_query"):
        output["research_question"] = state["original_query"]
    
    # Literature review results
    if state.get("literature_review_results"):
        output["literature_review"] = state["literature_review_results"]
    
    # Gap analysis
    if state.get("gap_analysis"):
        output["gap_analysis"] = state["gap_analysis"]
    
    # Research plan
    if state.get("research_plan"):
        output["research_plan"] = state["research_plan"]
    
    # Data analysis results
    if state.get("data_analyst_output"):
        output["analysis_results"] = state["data_analyst_output"]
    elif state.get("conceptual_synthesis_output"):
        output["analysis_results"] = state["conceptual_synthesis_output"]
    
    # Writer output
    if state.get("writer_output"):
        output["draft_sections"] = state["writer_output"]
    
    # Errors encountered
    errors = state.get("errors", [])
    if errors:
        output["errors_encountered"] = [
            {
                "node": getattr(e, 'node', 'unknown'),
                "message": getattr(e, 'message', str(e)),
            }
            for e in errors
        ]
    
    # What was not completed
    output["incomplete_stages"] = _get_incomplete_stages(state)
    
    return output


def _get_incomplete_stages(state: dict[str, Any]) -> list[str]:
    """Identify stages that were not completed.
    
    Args:
        state: Current workflow state
        
    Returns:
        List of incomplete stage names
    """
    incomplete = []
    
    # Check each stage
    if not state.get("literature_review_results"):
        incomplete.append("literature_review")
    
    if not state.get("gap_analysis"):
        incomplete.append("gap_identification")
    
    if not state.get("research_plan"):
        incomplete.append("research_planning")
    
    if not state.get("data_analyst_output") and not state.get("conceptual_synthesis_output"):
        incomplete.append("analysis")
    
    if not state.get("writer_output"):
        incomplete.append("writing")
    
    if not state.get("reviewer_output"):
        incomplete.append("review")
    
    return incomplete


def create_fallback_content(
    state: dict[str, Any],
    section: str,
) -> str:
    """Create fallback content for a section that could not be generated.
    
    Args:
        state: Current workflow state
        section: Section name
        
    Returns:
        Fallback content string
    """
    # Get project context
    title = state.get("project_title", "Research Project")
    question = state.get("refined_query") or state.get("original_query", "")
    
    # Section-specific fallbacks
    if section == "abstract":
        return (
            f"**Note: This section could not be automatically generated.**\n\n"
            f"Research Question: {question}\n\n"
            f"This abstract requires manual completion based on the available "
            f"research findings."
        )
    
    elif section == "introduction":
        return (
            f"# Introduction\n\n"
            f"**Note: This section requires manual completion.**\n\n"
            f"The introduction should establish the context for: {question}\n\n"
            f"Key elements to address:\n"
            f"- Background and context\n"
            f"- Research gap being addressed\n"
            f"- Purpose and objectives\n"
            f"- Contribution to the field"
        )
    
    elif section == "literature_review":
        lit_results = state.get("literature_review_results", {})
        papers_found = len(lit_results.get("papers", []))
        return (
            f"# Literature Review\n\n"
            f"**Note: This section requires manual completion.**\n\n"
            f"{papers_found} papers were identified during the literature search.\n\n"
            f"The literature review should synthesize these sources to:\n"
            f"- Establish the theoretical framework\n"
            f"- Identify key themes and debates\n"
            f"- Position this research within the existing body of work"
        )
    
    elif section == "methods":
        plan = state.get("research_plan", {})
        methodology = plan.get("methodology_type", "Not specified")
        return (
            f"# Methodology\n\n"
            f"**Note: This section requires manual completion.**\n\n"
            f"Planned methodology: {methodology}\n\n"
            f"The methods section should detail:\n"
            f"- Research design and approach\n"
            f"- Data sources and collection\n"
            f"- Analysis techniques\n"
            f"- Validity and reliability considerations"
        )
    
    elif section == "results":
        has_analysis = bool(
            state.get("data_analyst_output") or 
            state.get("conceptual_synthesis_output")
        )
        if has_analysis:
            return (
                f"# Results\n\n"
                f"**Note: This section requires manual formatting.**\n\n"
                f"Analysis results are available but require narrative synthesis."
            )
        else:
            return (
                f"# Results\n\n"
                f"**Note: Analysis was not completed.**\n\n"
                f"This section requires manual analysis and reporting."
            )
    
    elif section == "discussion":
        return (
            f"# Discussion\n\n"
            f"**Note: This section requires manual completion.**\n\n"
            f"The discussion should:\n"
            f"- Interpret the findings\n"
            f"- Compare with existing literature\n"
            f"- Address limitations\n"
            f"- Discuss implications"
        )
    
    elif section == "conclusion":
        return (
            f"# Conclusion\n\n"
            f"**Note: This section requires manual completion.**\n\n"
            f"The conclusion should summarize key findings and their significance."
        )
    
    else:
        return (
            f"**Note: Section '{section}' could not be automatically generated.**\n\n"
            f"Manual completion required."
        )
