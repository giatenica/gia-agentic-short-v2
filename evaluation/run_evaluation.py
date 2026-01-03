#!/usr/bin/env python3
"""Run evaluation suite for GIA research workflow.

This script runs test queries through the workflow and evaluates outputs
against expected results.

Usage:
    python -m evaluation.run_evaluation [OPTIONS]
    
Options:
    --queries FILE     Path to test queries JSON (default: evaluation/test_queries.json)
    --output DIR       Directory for results (default: evaluation/results)
    --query-id ID      Run specific query by ID
    --dry-run          Show what would be run without executing
    --mock             Use mock LLM responses for testing
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import EvaluationResult, evaluate_research_output


def load_test_queries(queries_path: str) -> list[dict[str, Any]]:
    """Load test queries from JSON file.
    
    Args:
        queries_path: Path to test queries JSON file
        
    Returns:
        List of query specifications
    """
    with open(queries_path) as f:
        data = json.load(f)
    return data.get("test_queries", [])


def save_results(results: list[EvaluationResult], output_dir: str) -> str:
    """Save evaluation results to JSON file.
    
    Args:
        results: List of evaluation results
        output_dir: Directory for output files
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "average_score": sum(r.overall_score for r in results) / len(results) if results else 0,
        "results": [r.to_dict() for r in results],
    }
    
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    
    return filepath


def create_mock_state(query_spec: dict[str, Any]) -> dict[str, Any]:
    """Create a mock workflow state for testing.
    
    Args:
        query_spec: Query specification
        
    Returns:
        Mock workflow state
    """
    return {
        "research_query": query_spec.get("query", ""),
        "research_type": query_spec.get("research_type", "empirical"),
        "literature_review_results": {
            "papers_found": 25,
            "seminal_works": ["Smith 2020", "Jones 2019", "Brown 2021"],
            "search_queries": ["test query 1", "test query 2"],
        },
        "literature_synthesis": {
            "themes": query_spec.get("expected_themes", [])[:3],
            "gaps": ["Gap 1", "Gap 2"],
            "key_findings": ["Finding 1", "Finding 2"],
        },
        "gap_analysis": {
            "identified_gaps": ["Gap 1", "Gap 2"],
            "research_opportunity": "Novel contribution opportunity",
            "approved": True,
        },
        "research_plan": {
            "methodology": query_spec.get("expected_methodology", ["regression"])[0],
            "analysis_approach": "Statistical analysis",
            "variables": ["var1", "var2"],
            "success_criteria": ["Criteria 1", "Criteria 2"],
            "approved": True,
        },
        "data_analyst_output": {
            "results": "Analysis results",
            "statistics": {"r_squared": 0.75, "p_value": 0.01},
        },
        "writer_output": {
            "abstract": "This study examines " + " ".join(["word"] * 100),
            "introduction": "Introduction text " + " ".join(["word"] * 200),
            "literature_review": "Literature review text " + " ".join(["word"] * 300),
            "methods": "Methods description " + " ".join(["word"] * 150),
            "results": "Results section " + " ".join(["word"] * 200),
            "discussion": "Discussion text " + " ".join(["word"] * 250),
            "conclusion": "Conclusion text " + " ".join(["word"] * 100),
        },
        "status": "completed",
    }


async def run_workflow_query(
    query_spec: dict[str, Any],
    mock: bool = False,
) -> dict[str, Any]:
    """Run a single query through the workflow.
    
    Args:
        query_spec: Query specification
        mock: Whether to use mock responses
        
    Returns:
        Workflow state after execution
    """
    if mock:
        # Return mock state for testing
        return create_mock_state(query_spec)
    
    # Import workflow components
    try:
        from studio.graphs import create_research_workflow
        from src.state.schema import create_initial_state
    except ImportError as e:
        print(f"Warning: Could not import workflow: {e}")
        print("Using mock state instead")
        return create_mock_state(query_spec)
    
    # Create workflow
    workflow = create_research_workflow()
    
    # Create initial state
    initial_state = create_initial_state(
        research_query=query_spec.get("query", ""),
        research_type=query_spec.get("research_type", "empirical"),
    )
    
    # Run workflow
    config = {
        "configurable": {
            "thread_id": f"eval-{query_spec.get('id', 'unknown')}",
        }
    }
    
    result = await workflow.ainvoke(initial_state, config=config)
    return result


async def evaluate_query(
    query_spec: dict[str, Any],
    mock: bool = False,
) -> EvaluationResult:
    """Evaluate a single test query.
    
    Args:
        query_spec: Query specification with expected values
        mock: Whether to use mock responses
        
    Returns:
        Evaluation result
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {query_spec.get('id', 'unknown')}")
    print(f"Query: {query_spec.get('query', '')[:50]}...")
    print(f"{'='*60}")
    
    # Run workflow
    state = await run_workflow_query(query_spec, mock=mock)
    
    # Evaluate output
    result = evaluate_research_output(state, query_spec)
    
    # Print results
    print(f"\nResults for {query_spec.get('id')}:")
    print(f"  Overall Score: {result.overall_score:.2%}")
    print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"\n  Metrics:")
    for metric in result.metrics:
        status = "✓" if metric.passed else "✗"
        print(f"    {status} {metric.metric.value}: {metric.score:.2%} - {metric.feedback}")
    
    return result


async def run_evaluation(
    queries_path: str,
    output_dir: str,
    query_id: str | None = None,
    dry_run: bool = False,
    mock: bool = False,
) -> None:
    """Run the full evaluation suite.
    
    Args:
        queries_path: Path to test queries JSON
        output_dir: Directory for results
        query_id: Optional specific query to run
        dry_run: If True, show queries without running
        mock: If True, use mock responses
    """
    # Load queries
    queries = load_test_queries(queries_path)
    print(f"Loaded {len(queries)} test queries from {queries_path}")
    
    # Filter if specific query requested
    if query_id:
        queries = [q for q in queries if q.get("id") == query_id]
        if not queries:
            print(f"Error: Query '{query_id}' not found")
            return
    
    if dry_run:
        print("\nDry run - queries that would be executed:")
        for q in queries:
            print(f"  - {q.get('id')}: {q.get('query', '')[:50]}...")
        return
    
    # Run evaluations
    results = []
    for query_spec in queries:
        try:
            result = await evaluate_query(query_spec, mock=mock)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {query_spec.get('id')}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if results:
        filepath = save_results(results, output_dir)
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total queries: {len(results)}")
        print(f"Passed: {sum(1 for r in results if r.passed)}")
        print(f"Failed: {sum(1 for r in results if not r.passed)}")
        avg_score = sum(r.overall_score for r in results) / len(results)
        print(f"Average score: {avg_score:.2%}")
        print(f"\nResults saved to: {filepath}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run GIA evaluation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--queries",
        default="evaluation/test_queries.json",
        help="Path to test queries JSON file",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results",
        help="Directory for evaluation results",
    )
    parser.add_argument(
        "--query-id",
        help="Run specific query by ID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM responses for testing",
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_evaluation(
        queries_path=args.queries,
        output_dir=args.output,
        query_id=args.query_id,
        dry_run=args.dry_run,
        mock=args.mock,
    ))


if __name__ == "__main__":
    main()
