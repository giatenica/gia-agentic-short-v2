"""DATA_EXPLORER node for analyzing uploaded research data.

This node:
1. Loads and parses uploaded data files
2. Detects schema and data types
3. Generates summary statistics
4. Identifies data quality issues
5. Assesses feasibility for research
6. Maps variables to research questions
"""

from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage

from src.state.enums import ResearchStatus
from src.state.models import (
    DataFile,
    DataExplorationResult,
    VariableMapping,
    WorkflowError,
)
from src.state.schema import WorkflowState
from src.tools.data_exploration import analyze_file


def data_explorer_node(state: WorkflowState) -> dict[str, Any]:
    """
    DATA_EXPLORER node: Analyze uploaded data files.
    
    Performs comprehensive data exploration:
    - Schema detection
    - Summary statistics
    - Quality assessment
    - Feasibility analysis
    
    Args:
        state: Current workflow state with uploaded_data.
        
    Returns:
        State updates with data_exploration results.
    """
    uploaded_data: list[DataFile] = state.get("uploaded_data", [])
    key_variables: list[str] = state.get("key_variables", [])
    
    if not uploaded_data:
        return {
            "status": ResearchStatus.DATA_EXPLORED,
            "data_exploration_results": None,
            "messages": [AIMessage(
                content="No data files uploaded. Proceeding without data exploration."
            )],
            "checkpoints": [f"{datetime.utcnow().isoformat()}: Data exploration skipped (no files)"],
        }
    
    # Analyze all uploaded files
    exploration_results: list[DataExplorationResult] = []
    errors: list[WorkflowError] = []
    
    for data_file in uploaded_data:
        try:
            result = analyze_file(data_file)
            exploration_results.append(result)
        except Exception as e:
            errors.append(WorkflowError(
                node="data_explorer",
                category="analysis",
                message=f"Failed to analyze {data_file.filename}: {str(e)}",
                recoverable=True,
                details={"filename": data_file.filename},
            ))
    
    if not exploration_results:
        return {
            "status": ResearchStatus.FAILED,
            "errors": errors,
            "messages": [AIMessage(
                content="Failed to analyze any uploaded data files. Please check file formats."
            )],
        }
    
    # Use the primary (first) result for main state
    primary_result = exploration_results[0]
    
    # Map key variables to columns
    variable_mappings = map_variables_to_columns(
        key_variables, 
        primary_result
    )
    
    # Generate summary message
    message = generate_exploration_summary(exploration_results, variable_mappings)
    
    # Determine overall feasibility (use quality score threshold)
    all_feasible = all(r.quality_score >= 0.4 for r in exploration_results)
    overall_quality = min(r.quality_score for r in exploration_results)
    
    status = ResearchStatus.DATA_EXPLORED if all_feasible else ResearchStatus.DATA_QUALITY_ISSUES
    
    return {
        "status": status,
        "data_exploration_results": primary_result,
        "variable_mappings": variable_mappings,
        "errors": errors if errors else state.get("errors", []),
        "messages": [AIMessage(content=message)],
        "checkpoints": [
            f"{datetime.utcnow().isoformat()}: Data exploration complete - "
            f"Quality: {overall_quality:.0%}, Feasible: {all_feasible}"
        ],
        "updated_at": datetime.utcnow(),
    }


def map_variables_to_columns(
    key_variables: list[str],
    exploration: DataExplorationResult,
) -> list[VariableMapping]:
    """
    Map user-specified key variables to actual data columns.
    
    Uses fuzzy matching to find likely column matches.
    
    Args:
        key_variables: List of variable names from intake form.
        exploration: Data exploration result with column info.
        
    Returns:
        List of VariableMapping objects.
    """
    mappings = []
    
    for var in key_variables:
        var_lower = var.lower()
        
        # Try exact match first
        matched_column = None
        confidence = 0.0
        
        for col in exploration.columns:
            col_lower = col.name.lower()
            
            # Exact match
            if col_lower == var_lower:
                matched_column = col.name
                confidence = 1.0
                break
            
            # Contains match
            if var_lower in col_lower or col_lower in var_lower:
                if confidence < 0.8:
                    matched_column = col.name
                    confidence = 0.8
            
            # Word overlap
            var_words = set(var_lower.split())
            col_words = set(col_lower.replace("_", " ").split())
            overlap = len(var_words & col_words)
            if overlap > 0:
                score = overlap / max(len(var_words), len(col_words))
                if score > confidence:
                    matched_column = col.name
                    confidence = score
        
        mappings.append(VariableMapping(
            user_variable=var,
            matched_column=matched_column,
            confidence=round(confidence, 2),
            match_reason="exact" if confidence >= 1.0 else "fuzzy" if matched_column else None,
        ))
    
    return mappings


def generate_exploration_summary(
    results: list[DataExplorationResult],
    mappings: list[VariableMapping],
) -> str:
    """Generate a human-readable summary of data exploration."""
    lines = ["Data Exploration Summary", "=" * 40, ""]
    
    for i, result in enumerate(results, 1):
        # Get filename from files_analyzed if available
        filename = result.files_analyzed[0].filename if result.files_analyzed else f"File {i}"
        lines.append(f"File {i}: {filename}")
        lines.append(f"  Rows: {result.total_rows:,}")
        lines.append(f"  Columns: {result.total_columns}")
        lines.append(f"  Quality Score: {result.quality_score:.0%} ({result.quality_level.value})")
        lines.append(f"  Feasibility: {result.feasibility_assessment}")
        
        if result.quality_issues:
            lines.append(f"  Issues ({len(result.quality_issues)}):")
            for issue in result.quality_issues[:3]:  # Show top 3
                lines.append(f"    - [{issue.severity.value}] {issue.description}")
        
        lines.append("")
    
    # Variable mappings
    if mappings:
        lines.append("Variable Mappings:")
        lines.append("-" * 20)
        for mapping in mappings:
            status = "Confirmed" if mapping.confidence >= 0.8 else "Needs Review"
            col = mapping.matched_column or "Not Found"
            lines.append(f"  {mapping.user_variable} -> {col} ({status})")
        lines.append("")
    
    return "\n".join(lines)


def route_after_data_explorer(state: WorkflowState) -> str:
    """
    Route after DATA_EXPLORER node based on state.
    
    Routes to:
    - "end" if data quality issues prevent analysis
    - "literature_reviewer" otherwise
    
    Args:
        state: Current workflow state.
        
    Returns:
        Name of next node.
    """
    status = state.get("status")
    
    if status == ResearchStatus.FAILED:
        return "end"
    
    if status == ResearchStatus.DATA_QUALITY_ISSUES:
        # Could route to HITL for confirmation
        # For now, continue but flag the issue
        pass
    
    return "literature_reviewer"
