#!/usr/bin/env python3
"""
HITL (Human-in-the-Loop) Workflow Example

Demonstrates workflow execution with human approval checkpoints
at gap identification and planning stages.

Usage:
    uv run python examples/hitl_workflow.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graphs import create_research_workflow, WorkflowConfig
from src.state.enums import ResearchStatus


def simulate_human_review(interrupt_data: dict) -> dict:
    """Simulate human review of workflow output.
    
    In a real application, this would present a UI to the user.
    """
    action = interrupt_data.get("action", "unknown")
    
    print(f"\n{'=' * 60}")
    print(f"HUMAN REVIEW REQUIRED: {action}")
    print("=" * 60)
    
    if action == "approve_refined_question":
        print(f"\nOriginal Question: {interrupt_data.get('original_question', 'N/A')}")
        print(f"Refined Question: {interrupt_data.get('refined_question', 'N/A')}")
        print(f"Contribution: {interrupt_data.get('contribution_statement', 'N/A')}")
        
        # Simulate approval
        return {
            "approved": True,
            "refined_question": interrupt_data.get("refined_question"),
            "contribution": interrupt_data.get("contribution_statement"),
            "feedback": "Approved - the refined question targets a clear gap",
        }
        
    elif action == "approve_research_plan":
        print(f"\nMethodology: {interrupt_data.get('methodology', 'N/A')}")
        print(f"Gap Addressed: {interrupt_data.get('gap_addressed', 'N/A')}")
        
        # Simulate approval with modification
        return {
            "approved": True,
            "feedback": "Approved with suggestion to add robustness checks",
        }
        
    elif action == "final_review":
        print(f"\nOverall Score: {interrupt_data.get('overall_score', 'N/A')}")
        print(f"AI Decision: {interrupt_data.get('ai_decision', 'N/A')}")
        
        # Human can override AI decision
        return {
            "decision": "approve",  # Override to approve
            "feedback": "Accepting with minor revisions noted",
        }
    
    # Default: approve
    return {"approved": True}


async def main():
    """Run workflow with HITL checkpoints."""
    print("=" * 60)
    print("GIA Agentic Research System - HITL Workflow Example")
    print("=" * 60)
    
    # Create workflow with HITL interrupts enabled
    config = WorkflowConfig(
        checkpointer=MemorySaver(),
        interrupt_before=["gap_identifier", "planner"],  # Pause before these nodes
        interrupt_after=["reviewer"],  # Pause after review
        enable_caching=True,
        debug=True,
    )
    
    workflow = create_research_workflow(config)
    
    # Define initial state
    initial_state = {
        "form_data": {
            "title": "ESG Investment Performance",
            "research_question": "Do ESG funds outperform traditional funds?",
            "paper_type": "full_paper",
            "research_type": "empirical",
        },
        "original_query": "Do ESG funds outperform traditional funds?",
        "status": ResearchStatus.PENDING,
    }
    
    thread_config = {"configurable": {"thread_id": "example-hitl-001"}}
    
    print("\n[Starting HITL Workflow]")
    print(f"Research Question: {initial_state['original_query']}")
    print("-" * 60)
    
    try:
        # First invocation - will pause at gap_identifier
        print("\n[Phase 1: Running until first interrupt...]")
        result = workflow.invoke(initial_state, thread_config)
        
        # Check if workflow is paused
        state = workflow.get_state(thread_config)
        
        while state.next:
            # Workflow is paused at an interrupt
            next_node = state.next[0] if state.next else None
            print(f"\n[Paused before: {next_node}]")
            
            # Get interrupt data (if available in state)
            interrupt_data = {}
            if hasattr(state, "values") and state.values:
                # Extract relevant info for human review
                if next_node == "gap_identifier":
                    interrupt_data = {
                        "action": "approve_refined_question",
                        "original_question": state.values.get("original_query"),
                        "refined_question": state.values.get("refined_query", state.values.get("original_query")),
                        "contribution_statement": state.values.get("contribution_statement", "To be determined"),
                    }
                elif next_node == "planner":
                    interrupt_data = {
                        "action": "approve_research_plan",
                        "methodology": "Panel regression with fixed effects",
                        "gap_addressed": "Lack of comprehensive ESG performance analysis",
                    }
                else:
                    interrupt_data = {
                        "action": "final_review",
                        "overall_score": state.values.get("review_critique", {}).get("overall_score", "N/A"),
                        "ai_decision": state.values.get("review_decision", "N/A"),
                    }
            
            # Simulate human review
            human_response = simulate_human_review(interrupt_data)
            
            # Resume workflow with human input
            print(f"\n[Resuming workflow with human feedback...]")
            result = workflow.invoke(
                Command(resume=human_response),
                thread_config,
            )
            
            # Check state again
            state = workflow.get_state(thread_config)
        
        # Workflow complete
        print(f"\n{'=' * 60}")
        print("[Workflow Complete]")
        print(f"Final Status: {result.get('status', 'unknown')}")
        print(f"Review Decision: {result.get('review_decision', 'N/A')}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[Error] Workflow failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
