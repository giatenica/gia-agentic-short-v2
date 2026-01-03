#!/usr/bin/env python3
"""
Data Analysis Workflow Example

Demonstrates empirical research workflow with uploaded data files.
Shows how data exploration and analysis tools are used.

Usage:
    uv run python examples/data_analysis.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.checkpoint.memory import MemorySaver

from src.graphs import create_research_workflow, WorkflowConfig
from src.state.enums import ResearchStatus, ResearchType
from src.state.models import DataFile


def create_sample_data_file() -> DataFile:
    """Create a sample data file reference for the example."""
    return DataFile(
        filename="stock_returns.csv",
        content_type="text/csv",
        size=1024000,  # 1MB
        path="/tmp/stock_returns.csv",
        schema={
            "columns": [
                {"name": "date", "type": "date"},
                {"name": "ticker", "type": "string"},
                {"name": "return", "type": "float"},
                {"name": "volume", "type": "integer"},
                {"name": "esg_score", "type": "float"},
                {"name": "market_cap", "type": "float"},
            ],
            "row_count": 50000,
        }
    )


async def main():
    """Run data analysis workflow."""
    print("=" * 60)
    print("GIA Agentic Research System - Data Analysis Example")
    print("=" * 60)
    
    # Create workflow
    config = WorkflowConfig(
        checkpointer=MemorySaver(),
        interrupt_before=[],  # No interrupts for this example
        enable_caching=True,
    )
    
    workflow = create_research_workflow(config)
    
    # Create sample data file
    data_file = create_sample_data_file()
    
    # Define initial state with data
    initial_state = {
        "form_data": {
            "title": "ESG and Stock Returns Analysis",
            "research_question": "What is the relationship between ESG scores and stock returns?",
            "target_journal": "Review of Financial Studies",
            "paper_type": "full_paper",
            "research_type": "empirical",
            "data_description": "Panel data of US stocks with ESG scores and daily returns from 2015-2023",
            "key_variables": ["return", "esg_score", "market_cap", "volume"],
            "methodology": "Panel regression with firm and time fixed effects",
        },
        "original_query": "What is the relationship between ESG scores and stock returns?",
        "research_type": ResearchType.EMPIRICAL.value,
        "uploaded_data": [data_file.model_dump()],
        "data_context": "Panel data of US stocks with ESG scores and daily returns from 2015-2023",
        "key_variables": ["return", "esg_score", "market_cap", "volume"],
        "status": ResearchStatus.PENDING,
    }
    
    thread_config = {"configurable": {"thread_id": "example-data-001"}}
    
    print("\n[Starting Data Analysis Workflow]")
    print(f"Research Question: {initial_state['original_query']}")
    print(f"Data File: {data_file.filename}")
    print(f"Variables: {', '.join(initial_state['key_variables'])}")
    print("-" * 60)
    
    try:
        # Run workflow
        result = workflow.invoke(initial_state, thread_config)
        
        # Display results
        status = result.get("status", "unknown")
        print(f"\n[Workflow Complete]")
        print(f"Final Status: {status}")
        
        # Data exploration results
        if result.get("data_exploration_results"):
            print("\n[Data Exploration Results]")
            exploration = result["data_exploration_results"]
            if hasattr(exploration, "files_analyzed"):
                print(f"Files Analyzed: {len(exploration.files_analyzed)}")
            if hasattr(exploration, "quality_issues"):
                print(f"Quality Issues: {len(exploration.quality_issues)}")
            if hasattr(exploration, "feasibility_assessment"):
                print(f"Feasibility: {exploration.feasibility_assessment[:100]}...")
                
        # Analysis results
        if result.get("analysis_results"):
            print("\n[Analysis Results]")
            analysis = result["analysis_results"]
            if hasattr(analysis, "key_findings"):
                print("Key Findings:")
                for i, finding in enumerate(analysis.key_findings[:3], 1):
                    print(f"  {i}. {finding[:80]}...")
                    
        # Research plan
        if result.get("research_plan"):
            print("\n[Research Plan]")
            plan = result["research_plan"]
            if hasattr(plan, "methodology"):
                print(f"Methodology: {plan.methodology}")
            if hasattr(plan, "analysis_approach"):
                print(f"Analysis Approach: {plan.analysis_approach}")
                
    except Exception as e:
        print(f"\n[Error] Workflow failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
