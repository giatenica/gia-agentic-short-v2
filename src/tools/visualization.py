"""Visualization tools for publication-ready tables and figures.

This module provides tools to generate academic-quality tables (LaTeX, markdown)
and figures (PNG, PDF, SVG) suitable for academic finance papers.
"""

import base64
import io
from typing import Any, Literal
from uuid import uuid4

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from tabulate import tabulate

from src.state.enums import ArtifactFormat, FigureFormat
from src.state.models import (
    TableArtifact,
    FigureArtifact,
)
from src.tools.data_loading import get_registry

# Configure matplotlib for academic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': False,  # Academic papers typically don't have gridlines
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# =============================================================================
# Significance Stars Helper
# =============================================================================


def _significance_stars(p_value: float) -> str:
    """Return significance stars based on p-value."""
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.1:
        return "*"
    return ""


def _format_number(value: float, decimals: int = 3) -> str:
    """Format number for academic tables."""
    if pd.isna(value):
        return ""
    return f"{value:.{decimals}f}"


def _format_coefficient(coef: float, se: float, p_value: float) -> str:
    """Format coefficient with standard error and significance stars."""
    stars = _significance_stars(p_value)
    return f"{coef:.3f}{stars}\n({se:.3f})"


# =============================================================================
# Table Generation Tools
# =============================================================================


@tool
def create_summary_statistics_table(
    dataset_name: str,
    variables: list[str] | None = None,
    statistics: list[str] | None = None,
    format: Literal["latex", "markdown", "html"] = "latex",
    title: str = "Summary Statistics",
    caption: str = "",
    label: str = "tab:summary",
) -> dict[str, Any]:
    """
    Generate a publication-style summary statistics table.
    
    Produces Table 1 for academic papers with means, standard deviations,
    min, max, quartiles, and observation counts.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        variables: Specific variables to include. If None, uses all numeric.
        statistics: Statistics to compute. Default: ["mean", "std", "min", "p25", "median", "p75", "max", "n"]
        format: Output format - "latex", "markdown", or "html".
        title: Table title for the caption.
        caption: Additional caption text.
        label: LaTeX label for cross-referencing.
    
    Returns:
        Dictionary with table_content, artifact model, and metadata.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return {
            "status": "error",
            "error": f"Dataset '{dataset_name}' not found. Available: {list(registry.datasets.keys())}",
        }
    
    df = registry.get_dataframe(dataset_name)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if variables:
        numeric_cols = [c for c in variables if c in numeric_cols]
    
    if not numeric_cols:
        return {
            "status": "error",
            "error": "No numeric variables found for summary statistics",
        }
    
    # Default statistics
    if statistics is None:
        statistics = ["mean", "std", "min", "p25", "median", "p75", "max", "n"]
    
    # Build statistics DataFrame
    stats_data = []
    for col in numeric_cols:
        data = df[col].dropna()
        row = {"Variable": col}
        
        if "n" in statistics:
            row["N"] = len(data)
        if "mean" in statistics:
            row["Mean"] = data.mean()
        if "std" in statistics:
            row["Std. Dev."] = data.std()
        if "min" in statistics:
            row["Min"] = data.min()
        if "p25" in statistics:
            row["P25"] = data.quantile(0.25)
        if "median" in statistics:
            row["Median"] = data.median()
        if "p75" in statistics:
            row["P75"] = data.quantile(0.75)
        if "max" in statistics:
            row["Max"] = data.max()
        
        stats_data.append(row)
    
    stats_df = pd.DataFrame(stats_data)
    
    # Format numeric columns
    for col in stats_df.columns:
        if col not in ["Variable", "N"]:
            stats_df[col] = stats_df[col].apply(lambda x: _format_number(x, 3) if pd.notna(x) else "")
        elif col == "N":
            stats_df[col] = stats_df[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")
    
    # Generate output in requested format
    if format == "latex":
        table_content = _generate_latex_table(
            stats_df, 
            title=title,
            caption=caption,
            label=label,
            notes="Note: This table reports summary statistics for the sample.",
        )
    elif format == "markdown":
        table_content = tabulate(stats_df, headers="keys", tablefmt="pipe", showindex=False)
    else:  # html
        table_content = stats_df.to_html(index=False, classes="table table-striped")
    
    # Create artifact
    artifact = TableArtifact(
        table_id=f"tab_{str(uuid4())[:8]}",
        title=title,
        caption=caption or f"Summary statistics for {dataset_name}",
        format=ArtifactFormat(format.upper()),
        content=table_content,
        source_data=dataset_name,
        notes=f"Variables: {', '.join(numeric_cols)}",
    )
    
    return {
        "status": "success",
        "table_content": table_content,
        "artifact": artifact.model_dump(),
        "variables_included": numeric_cols,
        "n_observations": len(df),
    }


@tool
def create_regression_table(
    regression_results: list[dict[str, Any]],
    model_names: list[str] | None = None,
    format: Literal["latex", "markdown", "html"] = "latex",
    title: str = "Regression Results",
    caption: str = "",
    label: str = "tab:regression",
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """
    Generate a Stargazer-style regression table with multiple models.
    
    Creates publication-ready regression output with coefficients, standard errors
    in parentheses, and significance stars (*** p<0.01, ** p<0.05, * p<0.1).
    
    Args:
        regression_results: List of regression result dictionaries from run_ols_regression.
        model_names: Names for each model column. Default: ["(1)", "(2)", ...].
        format: Output format - "latex", "markdown", or "html".
        title: Table title.
        caption: Additional caption text.
        label: LaTeX label for cross-referencing.
        include_diagnostics: Include R², Adj. R², N, F-stat at bottom.
    
    Returns:
        Dictionary with table_content, artifact model, and metadata.
    """
    if not regression_results:
        return {
            "status": "error",
            "error": "No regression results provided",
        }
    
    n_models = len(regression_results)
    
    if model_names is None:
        model_names = [f"({i+1})" for i in range(n_models)]
    
    # Collect all unique variables across models
    all_variables = []
    for result in regression_results:
        coeffs = result.get("coefficients", [])
        for coef in coeffs:
            var = coef.get("variable", "")
            if var not in all_variables:
                all_variables.append(var)
    
    # Sort: intercept first, then alphabetically
    intercept_names = ["(Intercept)", "Intercept", "const", "_cons"]
    intercepts = [v for v in all_variables if v in intercept_names]
    others = sorted([v for v in all_variables if v not in intercept_names])
    all_variables = intercepts + others
    
    # Build table rows
    rows = []
    for var in all_variables:
        coef_row = {"Variable": var}
        se_row = {"Variable": ""}
        
        for i, result in enumerate(regression_results):
            model_col = model_names[i]
            coeffs = result.get("coefficients", [])
            
            # Find coefficient for this variable
            var_coef = next((c for c in coeffs if c.get("variable") == var), None)
            
            if var_coef:
                coef = var_coef.get("coefficient", 0)
                se = var_coef.get("std_error", 0)
                p_val = var_coef.get("p_value", 1)
                stars = _significance_stars(p_val)
                
                coef_row[model_col] = f"{coef:.4f}{stars}"
                se_row[model_col] = f"({se:.4f})"
            else:
                coef_row[model_col] = ""
                se_row[model_col] = ""
        
        rows.append(coef_row)
        rows.append(se_row)
    
    # Add diagnostics
    if include_diagnostics:
        # Add separator
        rows.append({col: "" for col in ["Variable"] + model_names})
        
        # Observations
        n_row = {"Variable": "Observations"}
        for i, result in enumerate(regression_results):
            n_row[model_names[i]] = f"{result.get('n_observations', 0):,}"
        rows.append(n_row)
        
        # R-squared
        r2_row = {"Variable": "R²"}
        for i, result in enumerate(regression_results):
            r2_row[model_names[i]] = f"{result.get('r_squared', 0):.4f}"
        rows.append(r2_row)
        
        # Adjusted R-squared
        adj_r2_row = {"Variable": "Adjusted R²"}
        for i, result in enumerate(regression_results):
            adj_r2_row[model_names[i]] = f"{result.get('adj_r_squared', 0):.4f}"
        rows.append(adj_r2_row)
        
        # F-statistic
        f_row = {"Variable": "F-statistic"}
        for i, result in enumerate(regression_results):
            f_stat = result.get("f_statistic", 0)
            f_p = result.get("f_p_value", 1)
            stars = _significance_stars(f_p)
            f_row[model_names[i]] = f"{f_stat:.2f}{stars}"
        rows.append(f_row)
    
    table_df = pd.DataFrame(rows)
    
    # Generate output
    if format == "latex":
        table_content = _generate_regression_latex(
            table_df, 
            model_names=model_names,
            title=title,
            caption=caption,
            label=label,
        )
    elif format == "markdown":
        table_content = tabulate(table_df, headers="keys", tablefmt="pipe", showindex=False)
    else:
        table_content = table_df.to_html(index=False, classes="table table-striped")
    
    # Dependent variable info
    dep_vars = [r.get("dependent_variable", "unknown") for r in regression_results]
    
    artifact = TableArtifact(
        table_id=f"tab_{str(uuid4())[:8]}",
        title=title,
        caption=caption or f"Regression results. Dependent variable: {dep_vars[0]}",
        format=ArtifactFormat(format.upper()),
        content=table_content,
        source_data=", ".join(set(dep_vars)),
        notes="Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.1",
    )
    
    return {
        "status": "success",
        "table_content": table_content,
        "artifact": artifact.model_dump(),
        "n_models": n_models,
        "variables": all_variables,
    }


@tool
def create_correlation_matrix_table(
    dataset_name: str,
    variables: list[str] | None = None,
    format: Literal["latex", "markdown", "html"] = "latex",
    title: str = "Correlation Matrix",
    caption: str = "",
    label: str = "tab:correlation",
    include_significance: bool = True,
) -> dict[str, Any]:
    """
    Generate a correlation matrix table with significance indicators.
    
    Creates a lower-triangular correlation matrix with significance stars,
    suitable for academic papers.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        variables: Variables to include. If None, uses all numeric.
        format: Output format.
        title: Table title.
        caption: Additional caption text.
        label: LaTeX label.
        include_significance: Whether to add significance stars.
    
    Returns:
        Dictionary with table_content, artifact model, and metadata.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return {
            "status": "error",
            "error": f"Dataset '{dataset_name}' not found",
        }
    
    df = registry.get_dataframe(dataset_name)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if variables:
        numeric_cols = [c for c in variables if c in numeric_cols]
    
    if len(numeric_cols) < 2:
        return {
            "status": "error",
            "error": "Need at least 2 numeric variables for correlation matrix",
        }
    
    # Compute correlations and p-values
    from scipy.stats import pearsonr
    
    n_vars = len(numeric_cols)
    corr_matrix = np.zeros((n_vars, n_vars))
    p_matrix = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            data_i = df[numeric_cols[i]].dropna()
            data_j = df[numeric_cols[j]].dropna()
            
            # Align the data
            common_idx = data_i.index.intersection(data_j.index)
            if len(common_idx) > 2:
                corr, p_val = pearsonr(
                    df.loc[common_idx, numeric_cols[i]],
                    df.loc[common_idx, numeric_cols[j]]
                )
                corr_matrix[i, j] = corr
                p_matrix[i, j] = p_val
            else:
                corr_matrix[i, j] = np.nan
                p_matrix[i, j] = np.nan
    
    # Build formatted table (lower triangular)
    rows = []
    for i, var_i in enumerate(numeric_cols):
        row = {"Variable": f"({i+1}) {var_i}"}
        for j, var_j in enumerate(numeric_cols):
            col_name = f"({j+1})"
            if i > j:  # Lower triangle
                corr = corr_matrix[i, j]
                p_val = p_matrix[i, j]
                if pd.notna(corr):
                    stars = _significance_stars(p_val) if include_significance else ""
                    row[col_name] = f"{corr:.3f}{stars}"
                else:
                    row[col_name] = ""
            elif i == j:
                row[col_name] = "1.000"
            else:
                row[col_name] = ""
        rows.append(row)
    
    corr_df = pd.DataFrame(rows)
    
    # Generate output
    if format == "latex":
        table_content = _generate_latex_table(
            corr_df,
            title=title,
            caption=caption,
            label=label,
            notes="*** p<0.01, ** p<0.05, * p<0.1" if include_significance else "",
        )
    elif format == "markdown":
        table_content = tabulate(corr_df, headers="keys", tablefmt="pipe", showindex=False)
    else:
        table_content = corr_df.to_html(index=False, classes="table table-striped")
    
    artifact = TableArtifact(
        table_id=f"tab_{str(uuid4())[:8]}",
        title=title,
        caption=caption or f"Pairwise correlations for {dataset_name}",
        format=ArtifactFormat(format.upper()),
        content=table_content,
        source_data=dataset_name,
        notes=f"Variables: {', '.join(numeric_cols)}",
    )
    
    return {
        "status": "success",
        "table_content": table_content,
        "artifact": artifact.model_dump(),
        "variables": numeric_cols,
    }


@tool
def create_crosstab_table(
    dataset_name: str,
    row_var: str,
    col_var: str,
    values_var: str | None = None,
    aggfunc: Literal["count", "mean", "sum", "median"] = "count",
    format: Literal["latex", "markdown", "html"] = "latex",
    title: str = "Cross-tabulation",
    caption: str = "",
    label: str = "tab:crosstab",
) -> dict[str, Any]:
    """
    Generate a cross-tabulation (pivot) table.
    
    Creates a contingency table or aggregated pivot table for categorical
    analysis in academic papers.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        row_var: Variable for row categories.
        col_var: Variable for column categories.
        values_var: Variable to aggregate. If None with count, counts observations.
        aggfunc: Aggregation function - count, mean, sum, or median.
        format: Output format.
        title: Table title.
        caption: Additional caption text.
        label: LaTeX label.
    
    Returns:
        Dictionary with table_content, artifact model, and metadata.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return {
            "status": "error",
            "error": f"Dataset '{dataset_name}' not found",
        }
    
    df = registry.get_dataframe(dataset_name)
    
    if row_var not in df.columns or col_var not in df.columns:
        return {
            "status": "error",
            "error": f"Variables not found. Available: {df.columns.tolist()}",
        }
    
    # Create crosstab
    try:
        if values_var and values_var in df.columns:
            crosstab = pd.crosstab(
                df[row_var], 
                df[col_var], 
                values=df[values_var],
                aggfunc=aggfunc,
            )
        else:
            crosstab = pd.crosstab(df[row_var], df[col_var])
        
        # Add margins (totals)
        crosstab["Total"] = crosstab.sum(axis=1)
        crosstab.loc["Total"] = crosstab.sum(axis=0)
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to create crosstab: {str(e)}",
        }
    
    # Format numbers
    if aggfunc in ["mean", "median"]:
        crosstab = crosstab.round(3)
    else:
        crosstab = crosstab.astype(int)
    
    # Reset index for formatting
    crosstab = crosstab.reset_index()
    crosstab = crosstab.rename(columns={row_var: f"{row_var} \\ {col_var}"})
    
    # Generate output
    if format == "latex":
        table_content = _generate_latex_table(
            crosstab,
            title=title,
            caption=caption,
            label=label,
        )
    elif format == "markdown":
        table_content = tabulate(crosstab, headers="keys", tablefmt="pipe", showindex=False)
    else:
        table_content = crosstab.to_html(index=False, classes="table table-striped")
    
    artifact = TableArtifact(
        table_id=f"tab_{str(uuid4())[:8]}",
        title=title,
        caption=caption or f"Cross-tabulation of {row_var} by {col_var}",
        format=ArtifactFormat(format.upper()),
        content=table_content,
        source_data=dataset_name,
        notes=f"Aggregation: {aggfunc}",
    )
    
    return {
        "status": "success",
        "table_content": table_content,
        "artifact": artifact.model_dump(),
    }


# =============================================================================
# Figure Generation Tools
# =============================================================================


@tool
def create_time_series_plot(
    dataset_name: str,
    date_column: str,
    value_columns: list[str],
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
    figsize: tuple[int, int] = (10, 6),
    legend_loc: str = "best",
) -> dict[str, Any]:
    """
    Generate a time series line plot.
    
    Creates publication-quality time series visualization with multiple series,
    proper date formatting, and academic styling.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        date_column: Column containing dates.
        value_columns: Columns to plot as lines.
        title: Plot title.
        ylabel: Y-axis label.
        xlabel: X-axis label.
        figsize: Figure size (width, height) in inches.
        legend_loc: Legend location.
    
    Returns:
        Dictionary with base64 image, artifact model, and metadata.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return {
            "status": "error",
            "error": f"Dataset '{dataset_name}' not found",
        }
    
    df = registry.get_dataframe(dataset_name)
    
    if date_column not in df.columns:
        return {
            "status": "error",
            "error": f"Date column '{date_column}' not found",
        }
    
    # Ensure date column is datetime
    try:
        df[date_column] = pd.to_datetime(df[date_column])
    except Exception:
        return {
            "status": "error",
            "error": f"Could not parse '{date_column}' as dates",
        }
    
    # Filter to valid value columns
    valid_cols = [c for c in value_columns if c in df.columns]
    if not valid_cols:
        return {
            "status": "error",
            "error": "No valid value columns found",
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each series
    for col in valid_cols:
        ax.plot(df[date_column], df[col], label=col, linewidth=1.5)
    
    # Styling
    ax.set_xlabel(xlabel or date_column)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    if len(valid_cols) > 1:
        ax.legend(loc=legend_loc, frameon=True, fancybox=False, edgecolor='black')
    
    # Format dates on x-axis
    fig.autofmt_xdate()
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    artifact = FigureArtifact(
        figure_id=f"fig_{str(uuid4())[:8]}",
        title=title or "Time Series Plot",
        caption=f"Time series of {', '.join(valid_cols)}",
        format=FigureFormat.PNG,
        content_base64=img_base64,
        source_data=dataset_name,
        width_inches=figsize[0],
        height_inches=figsize[1],
    )
    
    return {
        "status": "success",
        "image_base64": img_base64,
        "artifact": artifact.model_dump(),
        "variables_plotted": valid_cols,
    }


@tool
def create_scatter_plot(
    dataset_name: str,
    x_column: str,
    y_column: str,
    color_column: str | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    add_regression_line: bool = False,
    figsize: tuple[int, int] = (8, 6),
) -> dict[str, Any]:
    """
    Generate a scatter plot with optional regression line.
    
    Creates publication-quality scatter plot showing relationship between
    two variables, with optional OLS fit line and grouping by color.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        x_column: Variable for x-axis.
        y_column: Variable for y-axis.
        color_column: Optional variable for color coding points.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        add_regression_line: Whether to add OLS regression line.
        figsize: Figure size (width, height) in inches.
    
    Returns:
        Dictionary with base64 image, artifact model, and metadata.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return {
            "status": "error",
            "error": f"Dataset '{dataset_name}' not found",
        }
    
    df = registry.get_dataframe(dataset_name)
    
    if x_column not in df.columns or y_column not in df.columns:
        return {
            "status": "error",
            "error": f"Columns not found. Available: {df.columns.tolist()}",
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot scatter
    if color_column and color_column in df.columns:
        scatter = ax.scatter(
            df[x_column], 
            df[y_column], 
            c=pd.factorize(df[color_column])[0],
            cmap='viridis',
            alpha=0.6,
            s=30,
        )
        # Add legend for color
        handles, labels = scatter.legend_elements()
        unique_labels = df[color_column].unique()
        ax.legend(handles, unique_labels, title=color_column, loc='best')
    else:
        ax.scatter(df[x_column], df[y_column], alpha=0.6, s=30, color='steelblue')
    
    # Add regression line
    if add_regression_line:
        # Remove NaN values
        mask = df[[x_column, y_column]].notna().all(axis=1)
        x_data = df.loc[mask, x_column]
        y_data = df.loc[mask, y_column]
        
        if len(x_data) > 2:
            # Fit OLS
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"OLS fit")
            
            # Calculate R²
            y_pred = p(x_data)
            ss_res = ((y_data - y_pred) ** 2).sum()
            ss_tot = ((y_data - y_data.mean()) ** 2).sum()
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            ax.text(
                0.05, 0.95, 
                f"R² = {r_squared:.3f}",
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=10,
            )
    
    # Styling
    ax.set_xlabel(xlabel or x_column)
    ax.set_ylabel(ylabel or y_column)
    if title:
        ax.set_title(title)
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    artifact = FigureArtifact(
        figure_id=f"fig_{str(uuid4())[:8]}",
        title=title or f"{y_column} vs {x_column}",
        caption=f"Scatter plot of {y_column} against {x_column}",
        format=FigureFormat.PNG,
        content_base64=img_base64,
        source_data=dataset_name,
        width_inches=figsize[0],
        height_inches=figsize[1],
    )
    
    return {
        "status": "success",
        "image_base64": img_base64,
        "artifact": artifact.model_dump(),
    }


@tool
def create_distribution_plot(
    dataset_name: str,
    column: str,
    plot_type: Literal["histogram", "density", "box"] = "histogram",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    bins: int = 30,
    figsize: tuple[int, int] = (8, 5),
) -> dict[str, Any]:
    """
    Generate a distribution visualization.
    
    Creates histogram, density plot, or box plot showing the distribution
    of a variable in academic style.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        column: Variable to visualize.
        plot_type: Type of distribution plot.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        bins: Number of histogram bins.
        figsize: Figure size (width, height) in inches.
    
    Returns:
        Dictionary with base64 image, artifact model, and metadata.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return {
            "status": "error",
            "error": f"Dataset '{dataset_name}' not found",
        }
    
    df = registry.get_dataframe(dataset_name)
    
    if column not in df.columns:
        return {
            "status": "error",
            "error": f"Column '{column}' not found",
        }
    
    data = df[column].dropna()
    
    if len(data) == 0:
        return {
            "status": "error",
            "error": "No valid data to plot",
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type == "histogram":
        ax.hist(data, bins=bins, edgecolor='white', color='steelblue', alpha=0.7)
        ax.set_ylabel(ylabel or "Frequency")
    elif plot_type == "density":
        sns.kdeplot(data, ax=ax, fill=True, color='steelblue', alpha=0.5)
        ax.set_ylabel(ylabel or "Density")
    elif plot_type == "box":
        box_data = ax.boxplot(data, vert=True, patch_artist=True)
        box_data['boxes'][0].set_facecolor('steelblue')
        box_data['boxes'][0].set_alpha(0.7)
        ax.set_ylabel(ylabel or column)
        ax.set_xticklabels([column])
    
    ax.set_xlabel(xlabel or column)
    if title:
        ax.set_title(title)
    
    # Add summary statistics annotation for histogram/density
    if plot_type in ["histogram", "density"]:
        stats_text = f"Mean: {data.mean():.3f}\nStd: {data.std():.3f}\nN: {len(data):,}"
        ax.text(
            0.95, 0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    artifact = FigureArtifact(
        figure_id=f"fig_{str(uuid4())[:8]}",
        title=title or f"Distribution of {column}",
        caption=f"{plot_type.capitalize()} showing the distribution of {column}",
        format=FigureFormat.PNG,
        content_base64=img_base64,
        source_data=dataset_name,
        width_inches=figsize[0],
        height_inches=figsize[1],
    )
    
    return {
        "status": "success",
        "image_base64": img_base64,
        "artifact": artifact.model_dump(),
        "n_observations": len(data),
        "mean": data.mean(),
        "std": data.std(),
    }


@tool
def create_heatmap(
    dataset_name: str,
    columns: list[str] | None = None,
    title: str = "Correlation Heatmap",
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "RdBu_r",
    annot: bool = True,
) -> dict[str, Any]:
    """
    Generate a correlation heatmap.
    
    Creates a heatmap visualization of the correlation matrix,
    suitable for academic papers.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        columns: Variables to include. If None, uses all numeric.
        title: Plot title.
        figsize: Figure size (width, height) in inches.
        cmap: Colormap name.
        annot: Whether to annotate cells with correlation values.
    
    Returns:
        Dictionary with base64 image, artifact model, and metadata.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return {
            "status": "error",
            "error": f"Dataset '{dataset_name}' not found",
        }
    
    df = registry.get_dataframe(dataset_name)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if columns:
        numeric_cols = [c for c in columns if c in numeric_cols]
    
    if len(numeric_cols) < 2:
        return {
            "status": "error",
            "error": "Need at least 2 numeric variables for heatmap",
        }
    
    # Compute correlation
    corr_matrix = df[numeric_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Upper triangle mask
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        annot=annot,
        fmt='.2f',
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    
    ax.set_title(title)
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    artifact = FigureArtifact(
        figure_id=f"fig_{str(uuid4())[:8]}",
        title=title,
        caption=f"Correlation heatmap for {len(numeric_cols)} variables",
        format=FigureFormat.PNG,
        content_base64=img_base64,
        source_data=dataset_name,
        width_inches=figsize[0],
        height_inches=figsize[1],
    )
    
    return {
        "status": "success",
        "image_base64": img_base64,
        "artifact": artifact.model_dump(),
        "variables": numeric_cols,
    }


# =============================================================================
# LaTeX Helpers
# =============================================================================


def _generate_latex_table(
    df: pd.DataFrame,
    title: str,
    caption: str,
    label: str,
    notes: str = "",
) -> str:
    """Generate LaTeX table environment."""
    # Convert DataFrame to LaTeX tabular
    latex_tabular = df.to_latex(index=False, escape=False)
    
    # Wrap in table environment
    latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{title}}}
\\label{{{label}}}
{latex_tabular}
"""
    if notes:
        latex += f"\\footnotesize{{\\textit{{Note:}} {notes}}}\n"
    if caption:
        latex += f"\\footnotesize{{{caption}}}\n"
    
    latex += "\\end{table}"
    
    return latex


def _generate_regression_latex(
    df: pd.DataFrame,
    model_names: list[str],
    title: str,
    caption: str,
    label: str,
) -> str:
    """Generate Stargazer-style LaTeX regression table."""
    n_cols = len(model_names) + 1  # +1 for Variable column
    
    # Column specification
    col_spec = "l" + "c" * len(model_names)
    
    latex_lines = [
        f"\\begin{{table}}[htbp]",
        f"\\centering",
        f"\\caption{{{title}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline\\hline",
    ]
    
    # Header
    header = " & " + " & ".join(model_names) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\hline")
    
    # Content rows
    for _, row in df.iterrows():
        row_values = [str(row.get(col, "")) for col in df.columns]
        latex_lines.append(" & ".join(row_values) + " \\\\")
    
    latex_lines.extend([
        "\\hline\\hline",
        "\\multicolumn{" + str(n_cols) + "}{l}{\\footnotesize{Standard errors in parentheses.}} \\\\",
        "\\multicolumn{" + str(n_cols) + "}{l}{\\footnotesize{*** p<0.01, ** p<0.05, * p<0.1}} \\\\",
        "\\end{tabular}",
    ])
    
    if caption:
        latex_lines.append(f"\\footnotesize{{{caption}}}")
    
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


# =============================================================================
# Export Tools
# =============================================================================


@tool
def export_all_artifacts(
    output_dir: str,
    tables: list[dict[str, Any]] | None = None,
    figures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Export all table and figure artifacts to files.
    
    Writes tables as .tex files and figures as .png files
    to the specified output directory.
    
    Args:
        output_dir: Directory to write files to.
        tables: List of table artifact dictionaries.
        figures: List of figure artifact dictionaries.
    
    Returns:
        Dictionary with list of exported files.
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    exported = []
    
    # Export tables
    if tables:
        for table in tables:
            table_id = table.get("table_id", f"table_{len(exported)}")
            content = table.get("content", "")
            fmt = table.get("format", "LATEX").lower()
            
            ext = ".tex" if fmt == "latex" else ".md" if fmt == "markdown" else ".html"
            filepath = os.path.join(output_dir, f"{table_id}{ext}")
            
            with open(filepath, "w") as f:
                f.write(content)
            exported.append(filepath)
    
    # Export figures
    if figures:
        for figure in figures:
            figure_id = figure.get("figure_id", f"figure_{len(exported)}")
            content_b64 = figure.get("content_base64", "")
            
            if content_b64:
                filepath = os.path.join(output_dir, f"{figure_id}.png")
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(content_b64))
                exported.append(filepath)
    
    return {
        "status": "success",
        "exported_files": exported,
        "output_directory": output_dir,
    }


# =============================================================================
# Tool Collection for Export
# =============================================================================

__all__ = [
    # Table tools
    "create_summary_statistics_table",
    "create_regression_table",
    "create_correlation_matrix_table",
    "create_crosstab_table",
    # Figure tools
    "create_time_series_plot",
    "create_scatter_plot",
    "create_distribution_plot",
    "create_heatmap",
    # Export
    "export_all_artifacts",
]
