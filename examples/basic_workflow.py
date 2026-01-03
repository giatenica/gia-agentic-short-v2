#!/usr/bin/env python3
"""
Basic Workflow Example

Demonstrates a simple research query execution from start to finish.
This example uses in-memory persistence and mock data.

Usage:
    uv run python examples/basic_workflow.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.checkpoint.memory import MemorySaver

from src.graphs import create_research_workflow, WorkflowConfig
from src.state.enums import ResearchStatus


async def main():
    """Run a basic research workflow."""
    print("=" * 60)
    print("GIA Agentic Research System - Basic Workflow Example")
    print("=" * 60)
    
    # Create workflow with in-memory persistence
    config = WorkflowConfig(
        checkpointer=MemorySaver(),
        interrupt_before=[],  # No interrupts for this example
        interrupt_after=[],
        enable_caching=True,
        debug=True,
    )
    
    workflow = create_research_workflow(config)
    
    # Define initial state with research question
    initial_state = {
        "form_data": {
            "title": "Cryptocurrency Adoption Research",
            "research_question": "What factors drive cryptocurrency adoption among retail investors?",
            "target_journal": "Journal of Finance",
            "paper_type": "full_paper",
            "research_type": "empirical",
            "hypothesis": "Social influence and perceived ease of use are primary drivers of cryptocurrency adoption",
            "key_variables": ["adoption_rate", "social_influence", "perceived_ease_of_use", "risk_tolerance"],
            "methodology": "Survey-based empirical study with regression analysis",
            "expected_contribution": "First comprehensive analysis of retail crypto adoption drivers",
        },
        "original_query": "What factors drive cryptocurrency adoption among retail investors?",
        "status": ResearchStatus.PENDING,
    }
    
    # Thread ID for this workflow instance
    thread_config = {"configurable": {"thread_id": "example-basic-001"}}
    
    print("\n[Starting Workflow]")
    print(f"Research Question: {initial_state['original_query']}")
    print("-" * 60)
    
    try:
        # Run the workflow
        # Note: In a real scenario, this would make API calls to Anthropic and search services
        result = workflow.invoke(initial_state, thread_config)
        
        # Check final status
        status = result.get("status", "unknown")
        print(f"\n[Workflow Complete]")
        print(f"Final Status: {status}")
        
        # Display results summary
        if result.get("literature_synthesis"):
            print("\n[Literature Synthesis]")
            synthesis = result["literature_synthesis"]
            if hasattr(synthesis, "key_themes"):
                print(f"Key Themes: {len(synthesis.key_themes)}")
            
        if result.get("gap_analysis"):
            print("\n[Gap Analysis]")
            gap = result["gap_analysis"]
            if hasattr(gap, "primary_gap"):
                print(f"Primary Gap: {gap.primary_gap}")
                
        if result.get("research_plan"):
            print("\n[Research Plan]")
            plan = result["research_plan"]
            if hasattr(plan, "methodology"):
                print(f"Methodology: {plan.methodology}")
                
        if result.get("draft"):
            print("\n[Draft Generated]")
            draft = result["draft"]
            if hasattr(draft, "sections"):
                print(f"Sections: {len(draft.sections)}")
                
        if result.get("review_decision"):
            print(f"\n[Review Decision: {result['review_decision']}]")
            
    except Exception as e:
        print(f"\n[Error] Workflow failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
