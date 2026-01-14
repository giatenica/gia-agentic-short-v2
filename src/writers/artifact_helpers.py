"""Helper functions for formatting table and figure references in academic writing.

Sprint 16: Integration of visualization artifacts into paper sections.
"""

from typing import Any


def format_table_reference(
    table: dict[str, Any] | Any,
    table_number: int,
    style: str = "chicago"
) -> str:
    """
    Format a reference to a table for use in prose.
    
    Args:
        table: TableArtifact or dict with table metadata
        table_number: Sequential number for the table
        style: Citation style (chicago, apa, etc.)
        
    Returns:
        Formatted reference string
    """
    # Handle both dict and object
    if hasattr(table, "title"):
        title = table.title
        table_id = getattr(table, "table_id", f"tab:{table_number}")
    else:
        title = table.get("title", f"Table {table_number}")
        table_id = table.get("table_id", f"tab:{table_number}")
    
    # Generate LaTeX reference
    return f"Table {table_number}"


def format_figure_reference(
    figure: dict[str, Any] | Any,
    figure_number: int,
    style: str = "chicago"
) -> str:
    """
    Format a reference to a figure for use in prose.
    
    Args:
        figure: FigureArtifact or dict with figure metadata
        figure_number: Sequential number for the figure
        style: Citation style
        
    Returns:
        Formatted reference string
    """
    # Handle both dict and object
    if hasattr(figure, "title"):
        title = figure.title
        figure_id = getattr(figure, "figure_id", f"fig:{figure_number}")
    else:
        title = figure.get("title", f"Figure {figure_number}")
        figure_id = figure.get("figure_id", f"fig:{figure_number}")
    
    return f"Figure {figure_number}"


def generate_table_summary(tables: list[dict[str, Any] | Any]) -> str:
    """
    Generate a summary of available tables for the writer prompt.
    
    Args:
        tables: List of TableArtifact objects or dicts
        
    Returns:
        Formatted summary string for inclusion in prompts
    """
    if not tables:
        return ""
    
    lines = ["AVAILABLE TABLES:"]
    for i, table in enumerate(tables, 1):
        if hasattr(table, "title"):
            title = table.title
            notes = getattr(table, "notes", "") or ""
        else:
            title = table.get("title", f"Table {i}")
            notes = table.get("notes", "") or ""
        
        lines.append(f"  - Table {i}: {title}")
        if notes:
            lines.append(f"    Notes: {notes[:100]}...")
    
    lines.append("")
    lines.append("INSTRUCTIONS FOR TABLE REFERENCES:")
    lines.append("  - Reference tables as: \"Table 1 presents...\", \"As shown in Table 2...\"")
    lines.append("  - For LaTeX: use \\ref{tab:summary}, \\ref{tab:regression}, etc.")
    lines.append("  - Describe key findings from each table")
    lines.append("")
    
    return "\n".join(lines)


def generate_figure_summary(figures: list[dict[str, Any] | Any]) -> str:
    """
    Generate a summary of available figures for the writer prompt.
    
    Args:
        figures: List of FigureArtifact objects or dicts
        
    Returns:
        Formatted summary string for inclusion in prompts
    """
    if not figures:
        return ""
    
    lines = ["AVAILABLE FIGURES:"]
    for i, figure in enumerate(figures, 1):
        if hasattr(figure, "title"):
            title = figure.title
            notes = getattr(figure, "notes", "") or ""
        else:
            title = figure.get("title", f"Figure {i}")
            notes = figure.get("notes", "") or ""
        
        lines.append(f"  - Figure {i}: {title}")
        if notes:
            lines.append(f"    Notes: {notes[:100]}...")
    
    lines.append("")
    lines.append("INSTRUCTIONS FOR FIGURE REFERENCES:")
    lines.append("  - Reference figures as: \"Figure 1 illustrates...\", \"As shown in Figure 2...\"")
    lines.append("  - For LaTeX: use \\ref{fig:timeseries}, \\ref{fig:scatter}, etc.")
    lines.append("  - Describe the pattern or trend shown in each figure")
    lines.append("")
    
    return "\n".join(lines)


def format_data_exploration_for_methods(
    prose_description: str,
    dataset_count: int = 0,
    total_observations: int = 0,
) -> str:
    """
    Format data exploration summary for the methods section.
    
    Args:
        prose_description: LLM-generated prose from data explorer
        dataset_count: Number of datasets loaded
        total_observations: Total row count across datasets
        
    Returns:
        Formatted data description for methods section
    """
    if not prose_description:
        if dataset_count > 0:
            return f"The analysis uses {dataset_count} dataset(s) with approximately {total_observations:,} observations."
        return ""
    
    # The prose_description should already be academically written
    return prose_description


def generate_results_artifacts_prompt(
    tables: list[dict[str, Any] | Any],
    figures: list[dict[str, Any] | Any],
) -> str:
    """
    Generate a complete artifacts prompt section for the results writer.
    
    Args:
        tables: List of TableArtifact objects or dicts
        figures: List of FigureArtifact objects or dicts
        
    Returns:
        Combined prompt section for artifacts
    """
    parts = []
    
    table_summary = generate_table_summary(tables)
    if table_summary:
        parts.append(table_summary)
    
    figure_summary = generate_figure_summary(figures)
    if figure_summary:
        parts.append(figure_summary)
    
    if not parts:
        parts.append("NOTE: No tables or figures are available. Present results in prose only.")
    
    return "\n".join(parts)


def get_table_labels(tables: list[dict[str, Any] | Any]) -> dict[int, str]:
    """
    Get a mapping of table numbers to their LaTeX labels.
    
    Args:
        tables: List of TableArtifact objects or dicts
        
    Returns:
        Dict mapping table number to label
    """
    labels = {}
    for i, table in enumerate(tables, 1):
        if hasattr(table, "table_id"):
            labels[i] = table.table_id
        else:
            labels[i] = table.get("table_id", f"tab:{i}")
    return labels


def get_figure_labels(figures: list[dict[str, Any] | Any]) -> dict[int, str]:
    """
    Get a mapping of figure numbers to their LaTeX labels.
    
    Args:
        figures: List of FigureArtifact objects or dicts
        
    Returns:
        Dict mapping figure number to label
    """
    labels = {}
    for i, figure in enumerate(figures, 1):
        if hasattr(figure, "figure_id"):
            labels[i] = figure.figure_id
        else:
            labels[i] = figure.get("figure_id", f"fig:{i}")
    return labels
