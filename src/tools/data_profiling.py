"""Data profiling tools with LLM-integrated descriptions.

Provides comprehensive data profiling that generates:
1. Statistical profiles (distributions, correlations, patterns)
2. LLM-generated narrative descriptions suitable for methods sections
3. Data quality assessments with actionable recommendations
"""

from typing import Any
from datetime import datetime, timezone

from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from src.config import settings
from src.tools.data_loading import get_registry


# =============================================================================
# Profile Generation
# =============================================================================


class ProfileDatasetInput(BaseModel):
    """Input for profile_dataset tool."""
    name: str = Field(description="Name of the dataset to profile")
    include_correlations: bool = Field(default=True, description="Include correlation matrix")
    include_distributions: bool = Field(default=True, description="Include distribution analysis")


@tool(args_schema=ProfileDatasetInput)
def profile_dataset(
    name: str,
    include_correlations: bool = True,
    include_distributions: bool = True,
) -> dict[str, Any]:
    """
    Generate a comprehensive statistical profile of a dataset.
    
    Analyzes:
    - Basic statistics (mean, median, std, quartiles)
    - Distribution characteristics (skewness, kurtosis, normality tests)
    - Correlation matrix for numeric variables
    - Categorical variable frequencies
    - Missing value patterns
    - Potential data quality issues
    
    Args:
        name: Name of the registered dataset
        include_correlations: Whether to compute correlation matrix
        include_distributions: Whether to analyze distributions
        
    Returns:
        Comprehensive profile dictionary
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required for profiling"}
    
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    profile = {
        "dataset_name": name,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "overview": _profile_overview(df),
        "columns": _profile_columns(df),
        "missing_values": _profile_missing(df),
    }
    
    if include_distributions and HAS_SCIPY:
        profile["distributions"] = _profile_distributions(df)
    
    if include_correlations:
        profile["correlations"] = _profile_correlations(df)
    
    profile["quality_assessment"] = _assess_quality(df, profile)
    
    return profile


def _profile_overview(df: "pd.DataFrame") -> dict[str, Any]:
    """Generate dataset overview."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "datetime_columns": len(datetime_cols),
        "duplicate_rows": int(df.duplicated().sum()),
        "complete_rows": int((~df.isna().any(axis=1)).sum()),
        "complete_row_pct": round((~df.isna().any(axis=1)).mean() * 100, 1),
    }


def _profile_columns(df: "pd.DataFrame") -> dict[str, dict[str, Any]]:
    """Generate per-column profiles."""
    profiles = {}
    
    for col in df.columns:
        col_data = df[col]
        profile = {
            "dtype": str(col_data.dtype),
            "non_null_count": int(col_data.notna().sum()),
            "null_count": int(col_data.isna().sum()),
            "null_pct": round(col_data.isna().mean() * 100, 2),
            "unique_count": int(col_data.nunique()),
            "unique_pct": round(col_data.nunique() / len(df) * 100, 2),
        }
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            desc = col_data.describe()
            profile.update({
                "mean": float(desc["mean"]) if pd.notna(desc["mean"]) else None,
                "std": float(desc["std"]) if pd.notna(desc["std"]) else None,
                "min": float(desc["min"]) if pd.notna(desc["min"]) else None,
                "q25": float(desc["25%"]) if pd.notna(desc["25%"]) else None,
                "median": float(desc["50%"]) if pd.notna(desc["50%"]) else None,
                "q75": float(desc["75%"]) if pd.notna(desc["75%"]) else None,
                "max": float(desc["max"]) if pd.notna(desc["max"]) else None,
                "zeros": int((col_data == 0).sum()),
                "negatives": int((col_data < 0).sum()),
            })
            
            # Skewness and kurtosis
            if HAS_SCIPY and col_data.notna().sum() > 2:
                clean = col_data.dropna()
                profile["skewness"] = round(float(stats.skew(clean)), 4)
                profile["kurtosis"] = round(float(stats.kurtosis(clean)), 4)
        
        # Categorical columns
        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
            value_counts = col_data.value_counts()
            profile["top_values"] = value_counts.head(10).to_dict()
            profile["mode"] = str(value_counts.index[0]) if len(value_counts) > 0 else None
            profile["mode_frequency"] = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
        
        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            profile["min_date"] = str(col_data.min())
            profile["max_date"] = str(col_data.max())
            profile["date_range_days"] = (col_data.max() - col_data.min()).days if col_data.notna().any() else None
        
        profiles[col] = profile
    
    return profiles


def _profile_missing(df: "pd.DataFrame") -> dict[str, Any]:
    """Analyze missing value patterns."""
    missing_counts = df.isna().sum()
    missing_pcts = df.isna().mean() * 100
    
    # Find columns with missing values
    cols_with_missing = missing_counts[missing_counts > 0].sort_values(ascending=False)
    
    # Analyze missing patterns (co-occurrence)
    missing_patterns = {}
    if len(cols_with_missing) > 1:
        # Check if missing values tend to occur together
        missing_matrix = df[cols_with_missing.index].isna()
        for col in cols_with_missing.index:
            co_missing = {}
            for other_col in cols_with_missing.index:
                if col != other_col:
                    # Proportion of times both are missing when col is missing
                    col_missing = missing_matrix[col]
                    if col_missing.sum() > 0:
                        co_missing[other_col] = round(
                            (missing_matrix[other_col] & col_missing).sum() / col_missing.sum() * 100, 1
                        )
            if co_missing:
                missing_patterns[col] = co_missing
    
    return {
        "total_missing_cells": int(df.isna().sum().sum()),
        "total_cells": int(df.size),
        "missing_pct": round(df.isna().sum().sum() / df.size * 100, 2),
        "columns_with_missing": {
            col: {
                "count": int(missing_counts[col]),
                "pct": round(float(missing_pcts[col]), 2),
            }
            for col in cols_with_missing.index
        },
        "missing_patterns": missing_patterns,
    }


def _profile_distributions(df: "pd.DataFrame") -> dict[str, dict[str, Any]]:
    """Analyze distributions of numeric columns."""
    distributions = {}
    numeric_cols = df.select_dtypes(include=["number"]).columns
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) < 8:  # Need minimum observations
            continue
        
        dist_profile = {}
        
        # Normality test (Shapiro-Wilk for small samples, D'Agostino for large)
        if len(col_data) <= 5000:
            try:
                if len(col_data) >= 20:
                    stat, p_value = stats.shapiro(col_data.sample(min(5000, len(col_data))))
                    dist_profile["normality_test"] = {
                        "test": "shapiro_wilk",
                        "statistic": round(float(stat), 4),
                        "p_value": round(float(p_value), 6),
                        "is_normal": p_value > 0.05,
                    }
            except Exception:
                pass
        else:
            try:
                stat, p_value = stats.normaltest(col_data)
                dist_profile["normality_test"] = {
                    "test": "dagostino_k2",
                    "statistic": round(float(stat), 4),
                    "p_value": round(float(p_value), 6),
                    "is_normal": p_value > 0.05,
                }
            except Exception:
                pass
        
        # Distribution shape characterization
        skew = stats.skew(col_data)
        kurt = stats.kurtosis(col_data)
        
        if abs(skew) < 0.5:
            shape = "symmetric"
        elif skew > 0:
            shape = "right_skewed"
        else:
            shape = "left_skewed"
        
        if kurt > 1:
            tail = "heavy_tailed"
        elif kurt < -1:
            tail = "light_tailed"
        else:
            tail = "normal_tailed"
        
        dist_profile["shape"] = shape
        dist_profile["tail_behavior"] = tail
        
        # Histogram bins for visualization
        hist, bin_edges = np.histogram(col_data, bins=20)
        dist_profile["histogram"] = {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
        }
        
        distributions[col] = dist_profile
    
    return distributions


def _profile_correlations(df: "pd.DataFrame") -> dict[str, Any]:
    """Compute correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=["number"])
    
    if len(numeric_df.columns) < 2:
        return {"message": "Need at least 2 numeric columns for correlations"}
    
    # Pearson correlation
    corr_matrix = numeric_df.corr()
    
    # Find highly correlated pairs (|r| > 0.7)
    high_correlations = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Upper triangle only
                r = corr_matrix.loc[col1, col2]
                if pd.notna(r) and abs(r) > 0.7:
                    high_correlations.append({
                        "var1": col1,
                        "var2": col2,
                        "correlation": round(float(r), 4),
                        "strength": "strong" if abs(r) > 0.9 else "moderate",
                    })
    
    # Sort by absolute correlation
    high_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    return {
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "high_correlations": high_correlations,
        "variables": list(corr_matrix.columns),
    }


def _assess_quality(df: "pd.DataFrame", profile: dict) -> dict[str, Any]:
    """Assess data quality and generate recommendations."""
    issues = []
    recommendations = []
    
    overview = profile["overview"]
    missing = profile["missing_values"]
    columns = profile["columns"]
    
    # Check for high missing rates
    if missing["missing_pct"] > 20:
        issues.append({
            "type": "high_missing",
            "severity": "major",
            "description": f"{missing['missing_pct']:.1f}% of data is missing",
        })
        recommendations.append("Consider imputation strategies or excluding columns with >50% missing")
    
    # Check for constant columns
    for col, col_profile in columns.items():
        if col_profile["unique_count"] == 1:
            issues.append({
                "type": "constant_column",
                "severity": "minor",
                "column": col,
                "description": f"Column '{col}' has only one unique value",
            })
            recommendations.append(f"Remove constant column '{col}' from analysis")
    
    # Check for high cardinality categorical columns
    for col, col_profile in columns.items():
        if col_profile.get("top_values") and col_profile["unique_pct"] > 50:
            issues.append({
                "type": "high_cardinality",
                "severity": "minor",
                "column": col,
                "description": f"Column '{col}' has high cardinality ({col_profile['unique_count']} unique values)",
            })
    
    # Check for duplicates
    if overview["duplicate_rows"] > 0:
        dup_pct = overview["duplicate_rows"] / overview["row_count"] * 100
        issues.append({
            "type": "duplicates",
            "severity": "minor" if dup_pct < 5 else "major",
            "description": f"{overview['duplicate_rows']} duplicate rows ({dup_pct:.1f}%)",
        })
        if dup_pct > 1:
            recommendations.append("Investigate and potentially remove duplicate rows")
    
    # Check for multicollinearity
    if "correlations" in profile and profile["correlations"].get("high_correlations"):
        for corr in profile["correlations"]["high_correlations"]:
            if corr["strength"] == "strong":
                issues.append({
                    "type": "multicollinearity",
                    "severity": "major",
                    "description": f"High correlation ({corr['correlation']:.2f}) between {corr['var1']} and {corr['var2']}",
                })
        if len(profile["correlations"]["high_correlations"]) > 0:
            recommendations.append("Consider removing or combining highly correlated variables to avoid multicollinearity")
    
    # Quality score (0-100)
    score = 100
    for issue in issues:
        if issue["severity"] == "major":
            score -= 15
        elif issue["severity"] == "minor":
            score -= 5
    score = max(0, score)
    
    return {
        "quality_score": score,
        "quality_level": "excellent" if score >= 80 else "good" if score >= 60 else "acceptable" if score >= 40 else "poor",
        "issues": issues,
        "recommendations": recommendations,
    }


# =============================================================================
# LLM-Integrated Description Generation
# =============================================================================


class DescribeDatasetInput(BaseModel):
    """Input for describe_dataset tool."""
    name: str = Field(description="Name of the dataset to describe")
    context: str | None = Field(default=None, description="Research context for the description")
    style: str = Field(default="methods", description="Writing style: methods, abstract, or informal")


@tool(args_schema=DescribeDatasetInput)
def describe_dataset(
    name: str,
    context: str | None = None,
    style: str = "methods",
) -> dict[str, Any]:
    """
    Generate an LLM-written narrative description of a dataset.
    
    Creates academic prose suitable for a methods section, describing:
    - Data source and structure
    - Key variables and their distributions
    - Sample characteristics
    - Data quality considerations
    
    Args:
        name: Name of the registered dataset
        context: Optional research context to tailor the description
        style: Writing style (methods, abstract, or informal)
        
    Returns:
        Dictionary with generated description and supporting statistics
    """
    # First profile the dataset
    profile_result = profile_dataset.invoke({
        "name": name,
        "include_correlations": True,
        "include_distributions": True,
    })
    
    if "error" in profile_result:
        return profile_result
    
    # Generate description using LLM
    description = _generate_description(profile_result, context, style)
    
    return {
        "status": "success",
        "dataset_name": name,
        "description": description,
        "profile_summary": {
            "row_count": profile_result["overview"]["row_count"],
            "column_count": profile_result["overview"]["column_count"],
            "quality_score": profile_result["quality_assessment"]["quality_score"],
        },
    }


def _generate_description(
    profile: dict[str, Any],
    context: str | None,
    style: str,
) -> str:
    """Generate narrative description using LLM."""
    
    # Build the prompt with profile information
    overview = profile["overview"]
    quality = profile["quality_assessment"]
    columns = profile["columns"]
    
    # Summarize key statistics for prompt
    numeric_summaries = []
    categorical_summaries = []
    
    for col, col_profile in columns.items():
        if col_profile.get("mean") is not None:
            numeric_summaries.append(
                f"- {col}: mean={col_profile['mean']:.2f}, std={col_profile['std']:.2f}, "
                f"range=[{col_profile['min']:.2f}, {col_profile['max']:.2f}]"
            )
        elif col_profile.get("top_values"):
            top_vals = list(col_profile["top_values"].items())[:3]
            categorical_summaries.append(
                f"- {col}: {col_profile['unique_count']} unique values, "
                f"top values: {', '.join(f'{k} (n={v})' for k, v in top_vals)}"
            )
    
    profile_text = f"""
Dataset Overview:
- Observations: {overview['row_count']:,}
- Variables: {overview['column_count']}
- Complete cases: {overview['complete_row_pct']:.1f}%
- Memory: {overview['memory_mb']:.1f} MB

Numeric Variables:
{chr(10).join(numeric_summaries[:10]) if numeric_summaries else 'None'}

Categorical Variables:
{chr(10).join(categorical_summaries[:10]) if categorical_summaries else 'None'}

Data Quality:
- Score: {quality['quality_score']}/100 ({quality['quality_level']})
- Issues: {len(quality['issues'])}
"""
    
    if profile.get("correlations", {}).get("high_correlations"):
        corrs = profile["correlations"]["high_correlations"][:5]
        profile_text += f"\nNotable Correlations:\n"
        for c in corrs:
            profile_text += f"- {c['var1']} ↔ {c['var2']}: r={c['correlation']:.2f}\n"
    
    style_instructions = {
        "methods": "Write in formal academic style suitable for a journal's Methods/Data section. Use passive voice and precise language.",
        "abstract": "Write concisely for an abstract, highlighting key characteristics in 2-3 sentences.",
        "informal": "Write in a clear, accessible style suitable for a research memo or blog post.",
    }
    
    system_prompt = f"""You are a research data analyst writing dataset descriptions.
{style_instructions.get(style, style_instructions['methods'])}

IMPORTANT RULES:
- NEVER make up statistics or numbers not provided
- NEVER use emojis
- NEVER use em dashes; use semicolons, colons, or periods
- Cite the actual statistics provided
- Be precise and factual
- Focus on what the data reveals, not what it might reveal"""
    
    user_prompt = f"""Generate a {style} description of this dataset.

{f'Research Context: {context}' if context else ''}

{profile_text}

Write a {style} section describing this dataset. Include:
1. Data source/structure overview
2. Key variable descriptions with actual statistics
3. Sample characteristics
4. Any data quality considerations relevant to analysis"""
    
    try:
        model = ChatAnthropic(
            model=settings.default_model,
            api_key=settings.anthropic_api_key,
            max_tokens=1500,
            temperature=0.3,
        )
        
        response = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        
        return response.content
        
    except Exception as e:
        # Fallback to template-based description
        return _generate_template_description(profile, context, style)


def _generate_template_description(
    profile: dict[str, Any],
    context: str | None,
    style: str,
) -> str:
    """Generate description using templates (fallback if LLM fails)."""
    overview = profile["overview"]
    quality = profile["quality_assessment"]
    
    desc = f"""The dataset contains {overview['row_count']:,} observations across {overview['column_count']} variables. """
    
    if overview["numeric_columns"] > 0:
        desc += f"The data includes {overview['numeric_columns']} numeric variables "
    if overview["categorical_columns"] > 0:
        desc += f"and {overview['categorical_columns']} categorical variables. "
    
    desc += f"\n\nData completeness is {overview['complete_row_pct']:.1f}%, "
    
    if quality["quality_score"] >= 80:
        desc += "indicating high data quality suitable for statistical analysis."
    elif quality["quality_score"] >= 60:
        desc += "indicating acceptable data quality with minor issues to address."
    else:
        desc += "suggesting data quality concerns that should be addressed before analysis."
    
    if quality["issues"]:
        desc += f"\n\nKey considerations: "
        desc += "; ".join(issue["description"] for issue in quality["issues"][:3])
        desc += "."
    
    return desc


# =============================================================================
# Variable-Level Description
# =============================================================================


class DescribeVariableInput(BaseModel):
    """Input for describe_variable tool."""
    dataset_name: str = Field(description="Name of the dataset")
    variable_name: str = Field(description="Name of the variable to describe")


@tool(args_schema=DescribeVariableInput)
def describe_variable(dataset_name: str, variable_name: str) -> dict[str, Any]:
    """
    Generate a detailed description of a specific variable.
    
    Provides:
    - Distribution characteristics
    - Statistical properties
    - Potential issues
    - Interpretation guidance
    
    Args:
        dataset_name: Name of the registered dataset
        variable_name: Name of the variable to describe
        
    Returns:
        Detailed variable description with statistics
    """
    registry = get_registry()
    df = registry.get(dataset_name)
    
    if df is None:
        return {"error": f"Dataset '{dataset_name}' not found"}
    
    if variable_name not in df.columns:
        return {"error": f"Variable '{variable_name}' not found in dataset"}
    
    col_data = df[variable_name]
    
    # Generate profile for just this column
    profile = _profile_columns(df[[variable_name]])[variable_name]
    
    # Add distribution info if numeric
    distribution = None
    if pd.api.types.is_numeric_dtype(col_data) and HAS_SCIPY:
        clean = col_data.dropna()
        if len(clean) >= 8:
            dist_profile = _profile_distributions(df[[variable_name]])
            distribution = dist_profile.get(variable_name)
    
    # Generate natural language description
    description = _describe_variable_text(variable_name, profile, distribution)
    
    return {
        "status": "success",
        "variable_name": variable_name,
        "profile": profile,
        "distribution": distribution,
        "description": description,
    }


def _describe_variable_text(
    name: str,
    profile: dict[str, Any],
    distribution: dict | None,
) -> str:
    """Generate text description of a variable."""
    desc_parts = [f"Variable '{name}' is a {profile['dtype']} type"]
    
    if profile.get("mean") is not None:
        # Numeric variable
        desc_parts.append(
            f"with a mean of {profile['mean']:.2f} (SD={profile['std']:.2f})"
        )
        desc_parts.append(
            f"ranging from {profile['min']:.2f} to {profile['max']:.2f}"
        )
        
        if distribution:
            shape = distribution.get("shape", "")
            if shape == "right_skewed":
                desc_parts.append("The distribution is right-skewed, suggesting positive outliers")
            elif shape == "left_skewed":
                desc_parts.append("The distribution is left-skewed")
            elif shape == "symmetric":
                desc_parts.append("The distribution appears symmetric")
            
            norm_test = distribution.get("normality_test", {})
            if norm_test.get("is_normal"):
                desc_parts.append("and does not significantly deviate from normality")
            elif norm_test.get("p_value"):
                desc_parts.append(f"and significantly deviates from normality (p={norm_test['p_value']:.4f})")
    
    elif profile.get("top_values"):
        # Categorical variable
        desc_parts.append(
            f"with {profile['unique_count']} unique values"
        )
        top = list(profile["top_values"].items())[0]
        desc_parts.append(
            f"The most frequent value is '{top[0]}' (n={top[1]})"
        )
    
    if profile["null_pct"] > 0:
        desc_parts.append(f"{profile['null_pct']:.1f}% of values are missing")
    
    return ". ".join(desc_parts) + "."


# =============================================================================
# Exports
# =============================================================================

DATA_PROFILING_TOOLS = [
    profile_dataset,
    describe_dataset,
    describe_variable,
]


def get_profiling_tools() -> list:
    """Get list of all data profiling tools."""
    return DATA_PROFILING_TOOLS


__all__ = [
    "profile_dataset",
    "describe_dataset",
    "describe_variable",
    "get_profiling_tools",
    "DATA_PROFILING_TOOLS",
]
