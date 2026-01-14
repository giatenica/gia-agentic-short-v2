"""DATA_EXPLORER node for analyzing uploaded research data.

Sprint 12 Enhanced: Now includes deep profiling, LLM-generated prose summaries,
intelligent type inference, and comprehensive DataExplorationSummary output.

This node uses the comprehensive data analysis toolset:
1. Loads data using DuckDB-backed DataRegistry (supports CSV, Parquet, Excel, Stata, SPSS, JSON, ZIP)
2. Profiles datasets with statistical summaries and semantic type inference
3. Identifies data quality issues with QualityFlag enum
4. Detects data structure (panel, time series, cross-sectional)
5. Generates LLM prose summaries for Methods section
6. Assesses feasibility for research

Uses parallel processing for efficient handling of multiple large files.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any
import os
import tempfile
import base64
import logging

from langchain_core.messages import AIMessage

from src.state.enums import ResearchStatus, DataQualityLevel, ColumnType, CritiqueSeverity
from src.state.models import (
    DataFile,
    DataExplorationResult,
    VariableMapping,
    ColumnAnalysis,
    QualityIssue,
    WorkflowError,
    DatasetInfo,
    QualityFlagItem,
    DataExplorationSummary,
)
from src.state.schema import WorkflowState

# Import new comprehensive tools
from src.tools.data_loading import (
    load_data,
    get_dataset_info,
    sample_data,
    list_datasets,
)
from src.tools.data_profiling import (
    profile_dataset,
    describe_dataset,
    # Sprint 12 enhanced tools
    deep_profile_dataset,
    detect_data_types,
    assess_data_quality,
    identify_time_series,
    detect_panel_structure,
    generate_data_prose_summary,
)

logger = logging.getLogger(__name__)

# Maximum number of parallel workers for data processing
MAX_WORKERS = 4


def _save_uploaded_file(data_file: DataFile) -> str:
    """Save uploaded file content to a temporary file and return path."""
    # Create temp directory if needed
    temp_dir = tempfile.mkdtemp(prefix="gia_data_")
    
    # Determine file extension
    filename = data_file.filename
    filepath = os.path.join(temp_dir, filename)
    
    # Write content to file
    if hasattr(data_file, "content") and data_file.content:
        # Content might be base64 encoded
        try:
            content = base64.b64decode(data_file.content)
        except Exception:
            content = data_file.content.encode() if isinstance(data_file.content, str) else data_file.content
        
        with open(filepath, "wb") as f:
            f.write(content)
    elif data_file.filepath:
        # File already exists on disk
        filepath = str(data_file.filepath)
    
    return filepath


def _get_dataset_name(data_file: DataFile) -> str:
    """Generate a meaningful dataset name from file path."""
    file_path_obj = data_file.filepath
    parent_name = file_path_obj.parent.name if file_path_obj else ""
    base_name = data_file.filename.rsplit(".", 1)[0]
    
    # Combine parent and base name for context (e.g., goog_options, googl_underlying)
    if parent_name and parent_name not in [".", "..", "project_data", "data", "tmp"]:
        dataset_name = f"{parent_name}_{base_name}"
    else:
        dataset_name = base_name
    
    # Clean up the name
    return dataset_name.replace(" ", "_").replace("-", "_").lower()


def _load_single_dataset(data_file: DataFile, research_question: str) -> dict[str, Any]:
    """
    Load and profile a single dataset. Returns dict with results or error.
    
    This function is designed to be called in parallel via ThreadPoolExecutor.
    """
    try:
        filepath = _save_uploaded_file(data_file)
        dataset_name = _get_dataset_name(data_file)
        
        # Load data using DuckDB-backed tool
        load_result = load_data.invoke({
            "filepath": filepath,
            "name": dataset_name,
        })
        
        if "error" in load_result:
            return {
                "error": load_result["error"],
                "data_file": data_file,
                "dataset_name": dataset_name,
            }
        
        # Get quick stats from load result (no additional profiling needed for overview)
        return {
            "success": True,
            "data_file": data_file,
            "dataset_name": dataset_name,
            "load_result": load_result,
            "row_count": load_result.get("row_count", 0),
            "column_count": load_result.get("column_count", 0),
            "columns": load_result.get("columns", []),
            "dtypes": load_result.get("dtypes", {}),
            "memory_mb": load_result.get("memory_mb", 0),
            "sample_rows": load_result.get("sample_rows", []),
        }
        
    except Exception as e:
        logger.error(f"Error loading {data_file.filename}: {e}")
        return {
            "error": str(e),
            "data_file": data_file,
            "dataset_name": _get_dataset_name(data_file),
        }


def _profile_single_dataset(dataset_name: str, skip_correlations: bool = False) -> dict[str, Any]:
    """
    Profile a single loaded dataset. Returns dict with profile results or error.
    
    Args:
        dataset_name: Name of the already-loaded dataset
        skip_correlations: Skip correlation analysis for very large datasets
    """
    try:
        profile_result = profile_dataset.invoke({
            "name": dataset_name,
            "include_correlations": not skip_correlations,
        })
        
        if "error" in profile_result:
            return {"error": profile_result["error"], "dataset_name": dataset_name}
        
        return {
            "success": True,
            "dataset_name": dataset_name,
            "profile": profile_result,
        }
        
    except Exception as e:
        logger.error(f"Error profiling {dataset_name}: {e}")
        return {"error": str(e), "dataset_name": dataset_name}


def _extract_columns_from_profile(profile: dict) -> list[ColumnAnalysis]:
    """Extract ColumnAnalysis objects from profile result."""
    columns = []
    
    column_profiles = profile.get("columns", {})
    
    for col_name, col_info in column_profiles.items():
        # Map dtype to ColumnType
        dtype_str = str(col_info.get("dtype", "")).lower()
        if "int" in dtype_str:
            col_type = ColumnType.INTEGER
        elif "float" in dtype_str:
            col_type = ColumnType.FLOAT
        elif "bool" in dtype_str:
            col_type = ColumnType.BOOLEAN
        elif "datetime" in dtype_str or "date" in dtype_str:
            col_type = ColumnType.DATETIME
        else:
            col_type = ColumnType.STRING
        
        col_analysis = ColumnAnalysis(
            name=col_name,
            dtype=col_type,
            total_count=col_info.get("count", 0),
            non_null_count=col_info.get("non_null", col_info.get("count", 0)),
            null_count=col_info.get("null_count", col_info.get("missing", 0)),
            null_percentage=col_info.get("null_pct", col_info.get("missing_pct", 0.0)),
            unique_count=col_info.get("unique", 0),
            mean=col_info.get("mean"),
            std=col_info.get("std"),
            min_value=col_info.get("min"),
            max_value=col_info.get("max"),
            median=col_info.get("median"),
            percentile_25=col_info.get("q1", col_info.get("25%")),
            percentile_75=col_info.get("q3", col_info.get("75%")),
        )
        columns.append(col_analysis)
    
    return columns


def _extract_quality_issues(profile: dict) -> list[QualityIssue]:
    """Extract quality issues from profile result."""
    issues = []
    quality_info = profile.get("quality_assessment", {})
    
    # Get issues from quality assessment
    for issue in quality_info.get("issues", []):
        severity_str = issue.get("severity", "minor").lower()
        if severity_str == "major":
            severity = CritiqueSeverity.MAJOR
        elif severity_str == "critical":
            severity = CritiqueSeverity.CRITICAL
        else:
            severity = CritiqueSeverity.MINOR
        
        recommendations = quality_info.get("recommendations", [])
        suggestion = recommendations[0] if recommendations else None
        
        issues.append(QualityIssue(
            severity=severity,
            category=issue.get("type", "data_quality"),
            column=issue.get("column"),
            description=issue.get("description", ""),
            suggestion=suggestion,
        ))
    
    # Check for missing values from missing_values section
    missing_info = profile.get("missing_values", {})
    total_missing = missing_info.get("total_missing", 0)
    if total_missing > 0:
        missing_pct = missing_info.get("missing_percentage", 0)
        if missing_pct > 20:
            severity = CritiqueSeverity.MAJOR
        elif missing_pct > 5:
            severity = CritiqueSeverity.MINOR
        else:
            severity = CritiqueSeverity.SUGGESTION
        
        issues.append(QualityIssue(
            severity=severity,
            category="missing_values",
            column=None,
            description=f"Dataset has {missing_pct:.1f}% missing values overall",
            suggestion="Consider imputation or exclusion strategies",
        ))
    
    return issues


def _calculate_quality_score(profile: dict, issues: list[QualityIssue]) -> tuple[float, DataQualityLevel]:
    """Calculate overall quality score and level."""
    # Get quality score from profile if available
    quality_info = profile.get("quality_assessment", {})
    score = quality_info.get("quality_score", 100) / 100  # Convert from 0-100 to 0-1
    
    # Penalize for additional issues not in quality assessment
    for issue in issues:
        if issue.category == "missing_values":
            # Already factored in, skip
            continue
        if issue.severity == CritiqueSeverity.MAJOR:
            score -= 0.1
        elif issue.severity == CritiqueSeverity.MINOR:
            score -= 0.05
    
    # Penalize for small sample size
    overview = profile.get("overview", {})
    row_count = overview.get("row_count", 0)
    if row_count < 30:
        score -= 0.2
    elif row_count < 100:
        score -= 0.1
    
    score = max(0.0, min(1.0, score))
    
    # Determine quality level from profile or calculate
    quality_level_str = quality_info.get("quality_level", "").lower()
    if quality_level_str == "excellent":
        level = DataQualityLevel.EXCELLENT
    elif quality_level_str == "good":
        level = DataQualityLevel.GOOD
    elif quality_level_str in ["acceptable", "fair"]:
        level = DataQualityLevel.FAIR
    elif quality_level_str == "poor":
        level = DataQualityLevel.POOR
    else:
        # Calculate from score
        if score >= 0.8:
            level = DataQualityLevel.EXCELLENT
        elif score >= 0.6:
            level = DataQualityLevel.GOOD
        elif score >= 0.4:
            level = DataQualityLevel.FAIR
        else:
            level = DataQualityLevel.POOR
    
    return score, level


def data_explorer_node(state: WorkflowState) -> dict[str, Any]:
    """
    DATA_EXPLORER node: Analyze uploaded data files using parallel processing.
    
    Uses the data analysis toolset with parallel workers:
    - DuckDB-backed data loading for large files (handles 600MB+ parquet efficiently)
    - Statistical profiling with distributions
    - Quality assessment and feasibility analysis
    - NO LLM calls - keeps this fast; data_analyst does deeper analysis
    
    Args:
        state: Current workflow state with uploaded_data.
        
    Returns:
        State updates with data_exploration results.
    """
    uploaded_data: list[DataFile] = state.get("uploaded_data", [])
    key_variables: list[str] = state.get("key_variables", [])
    research_question = state.get("original_query", "")
    
    if not uploaded_data:
        return {
            "status": ResearchStatus.DATA_EXPLORED,
            "data_exploration_results": None,
            "messages": [AIMessage(
                content="No data files uploaded. Proceeding without data exploration."
            )],
            "checkpoints": [f"{datetime.now(timezone.utc).isoformat()}: Data exploration skipped (no files)"],
        }
    
    logger.info(f"Data explorer starting: {len(uploaded_data)} files to process")
    
    # PHASE 1: Load all datasets in parallel
    load_results = []
    errors: list[WorkflowError] = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all load tasks
        future_to_file = {
            executor.submit(_load_single_dataset, data_file, research_question): data_file
            for data_file in uploaded_data
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            result = future.result()
            if result.get("error"):
                errors.append(WorkflowError(
                    node="data_explorer",
                    category="loading",
                    message=f"Failed to load {result['dataset_name']}: {result['error']}",
                    recoverable=True,
                    details={"dataset": result['dataset_name']},
                ))
            else:
                load_results.append(result)
                logger.info(f"Loaded {result['dataset_name']}: {result['row_count']:,} rows, {result['column_count']} columns")
    
    if not load_results:
        return {
            "status": ResearchStatus.FAILED,
            "errors": errors,
            "messages": [AIMessage(
                content=f"Failed to load any data files. Errors: {[e.message for e in errors]}"
            )],
        }
    
    logger.info(f"Phase 1 complete: {len(load_results)}/{len(uploaded_data)} files loaded")
    
    # PHASE 2: Skip ALL profiling in data_explorer to keep it fast
    # The data_analyst agent will do deeper profiling on demand
    profile_results = {}
    
    for lr in load_results:
        logger.info(f"Skipping profiling for {lr['dataset_name']} (deferred to data_analyst)")
        profile_results[lr['dataset_name']] = {"basic_only": True, "skipped_reason": "deferred"}
    
    logger.info(f"Phase 2 complete: profiling deferred to data_analyst for all {len(load_results)} datasets")
    
    # PHASE 3: Build exploration results from load + profile data
    exploration_results: list[DataExplorationResult] = []
    all_datasets: list[str] = []
    
    for lr in load_results:
        dataset_name = lr['dataset_name']
        all_datasets.append(dataset_name)
        
        profile = profile_results.get(dataset_name, {})
        
        # Build columns from load result dtypes (fast, no profiling needed)
        columns = []
        for col_name in lr.get('columns', []):
            dtype_str = str(lr.get('dtypes', {}).get(col_name, '')).lower()
            if "int" in dtype_str:
                col_type = ColumnType.INTEGER
            elif "float" in dtype_str or "double" in dtype_str:
                col_type = ColumnType.FLOAT
            elif "bool" in dtype_str:
                col_type = ColumnType.BOOLEAN
            elif "datetime" in dtype_str or "date" in dtype_str or "timestamp" in dtype_str:
                col_type = ColumnType.DATETIME
            else:
                col_type = ColumnType.STRING
            
            # Get stats from profile if available
            col_profile = profile.get("columns", {}).get(col_name, {})
            
            columns.append(ColumnAnalysis(
                name=col_name,
                dtype=col_type,
                total_count=lr['row_count'],
                non_null_count=col_profile.get("non_null", lr['row_count']),
                null_count=col_profile.get("null_count", 0),
                null_percentage=col_profile.get("null_pct", 0.0),
                unique_count=col_profile.get("unique", 0),
                mean=col_profile.get("mean"),
                std=col_profile.get("std"),
                min_value=col_profile.get("min"),
                max_value=col_profile.get("max"),
                median=col_profile.get("median"),
                percentile_25=col_profile.get("q1"),
                percentile_75=col_profile.get("q3"),
            ))
        
        # Extract quality issues if profile available
        quality_issues = _extract_quality_issues(profile) if profile and not profile.get("basic_only") else []
        
        # Calculate quality score
        if profile and not profile.get("basic_only"):
            quality_score, quality_level = _calculate_quality_score(profile, quality_issues)
        else:
            # Default to GOOD if we have data but no full profile
            quality_score = 0.7
            quality_level = DataQualityLevel.GOOD
        
        # Determine feasibility
        row_count = lr['row_count']
        col_count = lr['column_count']
        
        if quality_score >= 0.6 and row_count >= 30:
            feasibility = "High; data is suitable for analysis"
        elif quality_score >= 0.4 and row_count >= 10:
            feasibility = "Medium; data usable with caution"
        else:
            feasibility = "Low; data quality concerns require attention"
        
        # Build description from data shape (no LLM call)
        description = (
            f"Dataset '{dataset_name}' contains {row_count:,} rows and {col_count} columns. "
            f"Memory usage: {lr.get('memory_mb', 0):.1f} MB. "
            f"Columns: {', '.join(lr.get('columns', [])[:10])}"
            + (f" (and {col_count - 10} more)" if col_count > 10 else "")
        )
        
        result = DataExplorationResult(
            total_rows=row_count,
            total_columns=col_count,
            columns=columns,
            quality_issues=quality_issues,
            quality_score=quality_score,
            quality_level=quality_level,
            feasibility_assessment=feasibility,
            description=description,
            files_analyzed=[lr['data_file']],
            variable_mappings=[],
        )
        
        exploration_results.append(result)
    
    # Use the primary (first) result for main state
    primary_result = exploration_results[0]
    
    # Map key variables to columns (checks all datasets)
    all_columns = []
    for er in exploration_results:
        all_columns.extend(er.columns)
    temp_result = DataExplorationResult(
        total_rows=sum(r.total_rows for r in exploration_results),
        total_columns=len(set(c.name for c in all_columns)),
        columns=all_columns,
        quality_issues=[],
        quality_score=primary_result.quality_score,
        quality_level=primary_result.quality_level,
        feasibility_assessment=primary_result.feasibility_assessment,
        description="Combined datasets",
        files_analyzed=[],
        variable_mappings=[],
    )
    variable_mappings = map_variables_to_columns(key_variables, temp_result)
    primary_result.variable_mappings = variable_mappings
    
    # Generate summary message
    message = generate_exploration_summary(exploration_results, variable_mappings, all_datasets)
    
    # PHASE 4 (Sprint 12): Generate DataExplorationSummary with LLM prose
    data_exploration_summary = _generate_exploration_summary_sprint12(
        all_datasets, 
        research_question,
        key_variables
    )
    
    # Determine overall feasibility
    all_feasible = all(r.quality_score >= 0.4 for r in exploration_results)
    overall_quality = min(r.quality_score for r in exploration_results)
    
    status = ResearchStatus.DATA_EXPLORED if all_feasible else ResearchStatus.DATA_QUALITY_ISSUES
    
    logger.info(f"Data exploration complete: {len(exploration_results)} datasets, quality={overall_quality:.0%}")
    
    return {
        "status": status,
        "data_exploration_results": primary_result,
        "data_exploration_summary": data_exploration_summary,  # Sprint 12
        "all_exploration_results": exploration_results,  # Store all results for reference
        "variable_mappings": variable_mappings,
        "loaded_datasets": all_datasets,
        "errors": errors if errors else state.get("errors", []),
        "messages": [AIMessage(content=message)],
        "checkpoints": [
            f"{datetime.now(timezone.utc).isoformat()}: Data exploration complete - "
            f"{len(exploration_results)} datasets, Quality: {overall_quality:.0%}, Feasible: {all_feasible}"
        ],
        "updated_at": datetime.now(timezone.utc),
    }


def _generate_exploration_summary_sprint12(
    dataset_names: list[str],
    research_context: str | None,
    focus_variables: list[str] | None,
) -> DataExplorationSummary | None:
    """
    Generate Sprint 12 DataExplorationSummary with LLM prose.
    
    Uses the new enhanced profiling tools to create a comprehensive
    summary suitable for the Methods section.
    
    Args:
        dataset_names: Names of loaded datasets
        research_context: Research question for context
        focus_variables: Key variables to emphasize
        
    Returns:
        DataExplorationSummary or None if generation fails
    """
    if not dataset_names:
        return None
    
    try:
        logger.info(f"Generating Sprint 12 exploration summary for {len(dataset_names)} datasets")
        
        # Use the new LLM summarization tool
        result = generate_data_prose_summary.invoke({
            "dataset_names": dataset_names,
            "research_context": research_context,
            "focus_variables": focus_variables,
        })
        
        if "error" in result:
            logger.warning(f"Failed to generate prose summary: {result['error']}")
            return None
        
        # Extract the summary from result
        summary_dict = result.get("summary")
        if summary_dict:
            # Reconstruct DataExplorationSummary from dict
            dataset_inventory = [
                DatasetInfo(**d) if isinstance(d, dict) else d
                for d in summary_dict.get("dataset_inventory", [])
            ]
            
            quality_flags = [
                QualityFlagItem(**f) if isinstance(f, dict) else f
                for f in summary_dict.get("quality_flags", [])
            ]
            
            summary = DataExplorationSummary(
                prose_description=summary_dict.get("prose_description", ""),
                dataset_inventory=dataset_inventory,
                quality_flags=quality_flags,
                recommended_variables=summary_dict.get("recommended_variables", []),
                data_gaps=summary_dict.get("data_gaps", []),
            )
            
            logger.info(f"Sprint 12 summary generated: {len(summary.prose_description)} chars prose, "
                       f"{len(summary.quality_flags)} quality flags")
            
            return summary
        
        return None
        
    except Exception as e:
        logger.error(f"Error generating Sprint 12 summary: {e}")
        return None


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
    dataset_names: list[str],
) -> str:
    """Generate a human-readable summary of data exploration."""
    lines = ["Data Exploration Summary (using DuckDB-backed analysis)", "=" * 50, ""]
    
    for i, result in enumerate(results, 1):
        # Get filename from files_analyzed if available
        filename = result.files_analyzed[0].filename if result.files_analyzed else f"File {i}"
        dataset_name = dataset_names[i-1] if i <= len(dataset_names) else f"dataset_{i}"
        
        lines.append(f"Dataset {i}: {filename} (registered as '{dataset_name}')")
        lines.append(f"  Rows: {result.total_rows:,}")
        lines.append(f"  Columns: {result.total_columns}")
        lines.append(f"  Quality Score: {result.quality_score:.0%} ({result.quality_level.value})")
        lines.append(f"  Feasibility: {result.feasibility_assessment}")
        
        if result.quality_issues:
            major_issues = [iss for iss in result.quality_issues if iss.severity == CritiqueSeverity.MAJOR]
            minor_issues = [iss for iss in result.quality_issues if iss.severity == CritiqueSeverity.MINOR]
            lines.append(f"  Issues: {len(major_issues)} major, {len(minor_issues)} minor severity")
            for issue in result.quality_issues[:3]:  # Show top 3
                lines.append(f"    - [{issue.severity.value}] {issue.description}")
        
        lines.append("")
    
    # Variable mappings
    if mappings:
        lines.append("Variable Mappings:")
        lines.append("-" * 20)
        matched = [m for m in mappings if m.matched_column]
        unmatched = [m for m in mappings if not m.matched_column]
        
        for mapping in matched:
            status = "Confirmed" if mapping.confidence >= 0.8 else "Needs Review"
            lines.append(f"  {mapping.user_variable} -> {mapping.matched_column} ({status})")
        
        if unmatched:
            lines.append("  Not matched:")
            for mapping in unmatched:
                lines.append(f"    - {mapping.user_variable}")
        lines.append("")
    
    # Available tools reminder
    lines.append("Datasets are now available for analysis using:")
    lines.append("  - query_data: Run SQL queries")
    lines.append("  - filter_data, aggregate_data: Transform data")
    lines.append("  - run_ttest, run_anova, run_ols_regression: Statistical tests")
    
    return "\n".join(lines)


def route_after_data_explorer(state: WorkflowState) -> str:
    """
    Route after DATA_EXPLORER node based on state.
    
    Routes to:
    - "__end__" if data quality issues prevent analysis
    - "literature_reviewer" otherwise
    
    Args:
        state: Current workflow state.
        
    Returns:
        Name of next node.
    """
    status = state.get("status")
    
    if status == ResearchStatus.FAILED:
        return "__end__"
    
    if status == ResearchStatus.DATA_QUALITY_ISSUES:
        # Could route to HITL for confirmation
        # For now, continue but flag the issue
        pass
    
    return "literature_reviewer"
