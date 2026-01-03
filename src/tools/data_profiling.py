"""Data profiling tools with LLM-integrated descriptions.

Sprint 12 Enhanced: Now includes deep profiling, intelligent type inference,
comprehensive quality assessment, and LLM-generated prose summaries.

Provides comprehensive data profiling that generates:
1. Statistical profiles (distributions, correlations, patterns)
2. LLM-generated narrative descriptions suitable for methods sections
3. Data quality assessments with actionable recommendations
4. Data structure detection (time series, panel, cross-sectional)
5. DataExplorationSummary objects for downstream nodes
"""

from typing import Any
from datetime import datetime, timezone
from collections import defaultdict

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

import logging

from src.config import settings
from src.tools.data_loading import get_registry
from src.state.enums import QualityFlag, DataStructureType, CritiqueSeverity
from src.state.models import DatasetInfo, QualityFlagItem, DataExplorationSummary

# =============================================================================
# Quality Score Constants
# =============================================================================
# Weights for calculating overall data quality scores.
# Higher penalties for more severe issues.
QUALITY_SCORE_PENALTY_CRITICAL = 25  # Critical issues severely impact usability
QUALITY_SCORE_PENALTY_MAJOR = 10     # Major issues require attention
QUALITY_SCORE_PENALTY_MINOR = 3      # Minor issues for awareness

# Semantic type inference thresholds
SEMANTIC_TYPE_UNIQUE_RATIO_ID = 0.9       # Column is likely an ID if >90% unique
SEMANTIC_TYPE_CARDINALITY_LOW = 0.05      # Categorical if <5% cardinality
SEMANTIC_TYPE_CARDINALITY_MEDIUM = 0.2    # Possibly categorical if <20%
SEMANTIC_TYPE_PANEL_ENTITY_MIN = 0.001    # Min unique ratio for entity column
SEMANTIC_TYPE_PANEL_ENTITY_MAX = 0.5      # Max unique ratio for entity column
SEMANTIC_TYPE_FREE_TEXT_LENGTH = 50       # Min avg length for free text
SEMANTIC_TYPE_DATE_MATCH_THRESHOLD = 0.8  # Min proportion to detect dates


def _calculate_quality_score(quality_flags: list[dict[str, Any]]) -> int:
    """Calculate overall quality score based on issue severities.
    
    Args:
        quality_flags: List of quality flag dicts with 'severity' key
        
    Returns:
        Quality score from 0-100 (higher is better)
    """
    penalty = 0
    for flag in quality_flags:
        severity = flag.get("severity", "minor")
        if severity == "critical":
            penalty += QUALITY_SCORE_PENALTY_CRITICAL
        elif severity == "major":
            penalty += QUALITY_SCORE_PENALTY_MAJOR
        elif severity == "minor":
            penalty += QUALITY_SCORE_PENALTY_MINOR
    return max(0, 100 - penalty)


def _severity_to_enum(severity_str: str) -> CritiqueSeverity:
    """Convert string severity to CritiqueSeverity enum."""
    severity_map = {
        "critical": CritiqueSeverity.CRITICAL,
        "major": CritiqueSeverity.MAJOR,
        "minor": CritiqueSeverity.MINOR,
        "suggestion": CritiqueSeverity.SUGGESTION,
    }
    return severity_map.get(severity_str.lower(), CritiqueSeverity.MINOR)


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
# Sprint 12: Enhanced Data Profiling Tools
# =============================================================================


class DeepProfileInput(BaseModel):
    """Input for deep_profile_dataset tool."""
    name: str = Field(description="Name of the dataset to profile")
    sample_size: int | None = Field(
        default=None,
        description="Sample size for expensive computations (defaults to full data for <100k rows)"
    )


@tool(args_schema=DeepProfileInput)
def deep_profile_dataset(
    name: str,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """
    Generate a comprehensive deep profile of a dataset for Sprint 12.
    
    Goes beyond basic profiling to include:
    - Intelligent type inference (currency, percentage, identifier, etc.)
    - Data structure detection (time series, panel, cross-sectional)
    - Comprehensive quality flags with severity levels
    - Variable relationship mapping
    - Outlier detection with multiple methods
    - Temporal pattern detection
    
    Args:
        name: Name of the registered dataset
        sample_size: Sample size for expensive computations
        
    Returns:
        Deep profile dictionary with all analysis results
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required for profiling"}
    
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    # Determine sample size
    n_rows = len(df)
    if sample_size is None:
        sample_size = min(n_rows, 100000)
    
    # Sample for expensive operations if needed
    if n_rows > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    # Run all analyses
    profile = {
        "dataset_name": name,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "row_count": n_rows,
        "sampled_rows": len(df_sample),
        "was_sampled": n_rows > sample_size,
    }
    
    # Basic profile
    profile["overview"] = _profile_overview(df)
    profile["columns"] = _profile_columns(df)
    profile["missing_values"] = _profile_missing(df)
    
    # Enhanced type inference
    profile["inferred_types"] = _infer_semantic_types(df)
    
    # Data structure detection
    profile["data_structure"] = _detect_data_structure(df)
    
    # Quality assessment with flags
    profile["quality_flags"] = _generate_quality_flags(df, name)
    
    # Distribution analysis (on sample)
    if HAS_SCIPY:
        profile["distributions"] = _profile_distributions(df_sample)
    
    # Correlation analysis (on sample)
    profile["correlations"] = _profile_correlations(df_sample)
    
    # Outlier detection
    profile["outliers"] = _detect_outliers(df_sample)
    
    # Time series analysis if temporal
    if profile["data_structure"]["type"] in ["time_series", "panel"]:
        profile["temporal_analysis"] = _analyze_temporal_patterns(df)
    
    # Overall quality score (using consolidated scoring function)
    profile["quality_score"] = _calculate_quality_score(profile["quality_flags"])
    
    return profile


def _infer_semantic_types(df: "pd.DataFrame") -> dict[str, dict[str, Any]]:
    """Infer semantic types beyond pandas dtypes."""
    inferred = {}
    
    for col in df.columns:
        col_data = df[col]
        inference = {
            "pandas_dtype": str(col_data.dtype),
            "semantic_type": "unknown",
            "confidence": 0.0,
            "patterns": [],
        }
        
        # Sample for pattern detection
        sample = col_data.dropna().head(1000)
        if len(sample) == 0:
            inference["semantic_type"] = "empty"
            inference["confidence"] = 1.0
            inferred[col] = inference
            continue
        
        # Check for common patterns
        if pd.api.types.is_numeric_dtype(col_data):
            inference.update(_infer_numeric_type(col, sample))
        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(col_data):
            inference.update(_infer_string_type(col, sample))
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            inference["semantic_type"] = "datetime"
            inference["confidence"] = 1.0
        
        inferred[col] = inference
    
    return inferred


def _infer_numeric_type(col: str, sample: "pd.Series") -> dict[str, Any]:
    """Infer semantic type for numeric column."""
    col_lower = col.lower()
    values = sample.values
    
    # Check for identifier patterns
    if sample.nunique() == len(sample):
        if col_lower in ["id", "index", "key"] or "_id" in col_lower:
            return {"semantic_type": "identifier", "confidence": 0.9}
        # Check if sequential integers
        if pd.api.types.is_integer_dtype(sample):
            sorted_vals = np.sort(values)
            if np.all(np.diff(sorted_vals) == 1):
                return {"semantic_type": "identifier", "confidence": 0.8}
    
    # Check for year
    if pd.api.types.is_integer_dtype(sample):
        if col_lower in ["year", "yr"] or "year" in col_lower:
            if values.min() >= 1900 and values.max() <= 2100:
                return {"semantic_type": "year", "confidence": 0.95}
    
    # Check for percentage
    if values.min() >= 0 and values.max() <= 100:
        if "pct" in col_lower or "percent" in col_lower or "rate" in col_lower:
            return {"semantic_type": "percentage", "confidence": 0.9}
        if "%" in col:
            return {"semantic_type": "percentage", "confidence": 0.95}
    
    # Check for currency/money
    money_keywords = ["price", "cost", "amount", "revenue", "income", "salary", "wage", "fee"]
    if any(kw in col_lower for kw in money_keywords):
        return {"semantic_type": "currency", "confidence": 0.85}
    
    # Check for count/integer measure
    if pd.api.types.is_integer_dtype(sample):
        count_keywords = ["count", "num", "number", "total", "quantity", "n_"]
        if any(kw in col_lower for kw in count_keywords):
            return {"semantic_type": "count", "confidence": 0.85}
    
    # Check for proportion (0-1)
    if values.min() >= 0 and values.max() <= 1:
        if "ratio" in col_lower or "share" in col_lower or "prop" in col_lower:
            return {"semantic_type": "proportion", "confidence": 0.85}
    
    # Default to continuous
    return {"semantic_type": "continuous", "confidence": 0.7}


def _infer_string_type(col: str, sample: "pd.Series") -> dict[str, Any]:
    """Infer semantic type for string column."""
    col_lower = col.lower()
    sample_str = sample.astype(str)
    
    # Check for date/time strings
    try:
        parsed = pd.to_datetime(sample_str.head(100), format="mixed", errors="coerce")
        if parsed.notna().mean() > SEMANTIC_TYPE_DATE_MATCH_THRESHOLD:
            return {"semantic_type": "datetime_string", "confidence": 0.85}
    except Exception:
        # Date parsing can fail for many reasons (exotic locales, ambiguous formats).
        # We silently skip since we'll fall back to other type detection methods.
        pass
    
    # Check for email
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    if sample_str.str.match(email_pattern).mean() > 0.8:
        return {"semantic_type": "email", "confidence": 0.95}
    
    # Check for URL
    url_pattern = r"^https?://"
    if sample_str.str.match(url_pattern).mean() > 0.8:
        return {"semantic_type": "url", "confidence": 0.95}
    
    # Check for identifier
    if "id" in col_lower or "code" in col_lower or "key" in col_lower:
        if sample.nunique() > len(sample) * 0.9:
            return {"semantic_type": "identifier", "confidence": 0.85}
    
    # Check for categorical with low cardinality
    cardinality = sample.nunique() / len(sample)
    if cardinality < 0.05:
        return {"semantic_type": "categorical", "confidence": 0.9}
    elif cardinality < 0.2:
        return {"semantic_type": "categorical", "confidence": 0.7}
    
    # Check for free text (high cardinality, long strings)
    avg_len = sample_str.str.len().mean()
    if avg_len > 50 and cardinality > 0.5:
        return {"semantic_type": "free_text", "confidence": 0.8}
    
    # Check for name
    name_keywords = ["name", "title", "label"]
    if any(kw in col_lower for kw in name_keywords):
        return {"semantic_type": "name", "confidence": 0.75}
    
    return {"semantic_type": "string", "confidence": 0.5}


def _detect_data_structure(df: "pd.DataFrame") -> dict[str, Any]:
    """Detect the data structure type."""
    n_rows = len(df)
    
    result = {
        "type": DataStructureType.CROSS_SECTIONAL.value,
        "confidence": 0.5,
        "time_column": None,
        "entity_column": None,
        "panel_info": None,
    }
    
    # Look for time columns
    time_cols = []
    for col in df.columns:
        col_lower = col.lower()
        # Check datetime type
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_cols.append((col, "datetime", 1.0))
        # Check for year column
        elif pd.api.types.is_integer_dtype(df[col]):
            if "year" in col_lower or col_lower == "yr":
                vals = df[col].dropna()
                if len(vals) > 0 and vals.min() >= 1900 and vals.max() <= 2100:
                    time_cols.append((col, "year", 0.9))
        # Check for date strings
        elif pd.api.types.is_object_dtype(df[col]):
            if "date" in col_lower or "time" in col_lower:
                time_cols.append((col, "date_string", 0.7))
    
    # Look for entity/ID columns
    entity_cols = []
    for col in df.columns:
        col_lower = col.lower()
        id_keywords = ["id", "code", "entity", "firm", "company", "country", "state", "region"]
        if any(kw in col_lower for kw in id_keywords):
            unique_ratio = df[col].nunique() / n_rows
            if 0.001 < unique_ratio < 0.5:  # Not too few, not unique per row
                entity_cols.append((col, unique_ratio))
    
    # Determine structure
    if time_cols:
        best_time = max(time_cols, key=lambda x: x[2])
        result["time_column"] = best_time[0]
        
        if entity_cols:
            # Panel data: entity + time
            best_entity = min(entity_cols, key=lambda x: x[1])  # Lower ratio = more repeated
            result["type"] = DataStructureType.PANEL.value
            result["entity_column"] = best_entity[0]
            result["confidence"] = min(best_time[2], 0.85)
            
            # Analyze panel balance
            n_entities = df[best_entity[0]].nunique()
            n_periods = df[best_time[0]].nunique()
            expected_obs = n_entities * n_periods
            actual_obs = len(df)
            
            result["panel_info"] = {
                "n_entities": n_entities,
                "n_periods": n_periods,
                "expected_observations": expected_obs,
                "actual_observations": actual_obs,
                "balance_ratio": round(actual_obs / expected_obs, 3) if expected_obs > 0 else 0,
                "is_balanced": actual_obs == expected_obs,
            }
        else:
            # Time series: time only, unique observations
            result["type"] = DataStructureType.TIME_SERIES.value
            result["confidence"] = best_time[2]
    elif entity_cols:
        # Cross-sectional with entity identifier
        result["type"] = DataStructureType.CROSS_SECTIONAL.value
        result["entity_column"] = entity_cols[0][0]
        result["confidence"] = 0.7
    
    return result


def _generate_quality_flags(df: "pd.DataFrame", dataset_name: str) -> list[dict[str, Any]]:
    """Generate comprehensive quality flags with QualityFlag enum."""
    flags = []
    n_rows = len(df)
    
    # Check each column for issues
    for col in df.columns:
        col_data = df[col]
        
        # Missing values
        missing_pct = col_data.isna().mean() * 100
        if missing_pct > 50:
            flags.append({
                "flag": QualityFlag.MISSING_VALUES.value,
                "severity": "critical",
                "dataset_name": dataset_name,
                "column_name": col,
                "description": f"{missing_pct:.1f}% missing values",
                "suggestion": "Consider imputation or exclusion",
            })
        elif missing_pct > 20:
            flags.append({
                "flag": QualityFlag.MISSING_VALUES.value,
                "severity": "major",
                "dataset_name": dataset_name,
                "column_name": col,
                "description": f"{missing_pct:.1f}% missing values",
                "suggestion": "Investigate missingness pattern",
            })
        elif missing_pct > 5:
            flags.append({
                "flag": QualityFlag.MISSING_VALUES.value,
                "severity": "minor",
                "dataset_name": dataset_name,
                "column_name": col,
                "description": f"{missing_pct:.1f}% missing values",
                "suggestion": "Document in limitations",
            })
        
        # Constant column
        if col_data.nunique() == 1:
            flags.append({
                "flag": QualityFlag.CONSTANT_COLUMN.value,
                "severity": "major",
                "dataset_name": dataset_name,
                "column_name": col,
                "description": "Column has only one unique value",
                "suggestion": "Remove from analysis",
            })
        
        # High cardinality for categoricals
        if pd.api.types.is_object_dtype(col_data):
            unique_ratio = col_data.nunique() / n_rows
            if unique_ratio > 0.5:
                flags.append({
                    "flag": QualityFlag.HIGH_CARDINALITY.value,
                    "severity": "minor",
                    "dataset_name": dataset_name,
                    "column_name": col,
                    "description": f"High cardinality: {col_data.nunique()} unique values",
                    "suggestion": "Consider grouping or treating as identifier",
                })
        
        # Numeric-specific checks
        if pd.api.types.is_numeric_dtype(col_data):
            clean = col_data.dropna()
            if len(clean) > 0:
                # Outliers (IQR method)
                q1, q3 = clean.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                outlier_pct = ((clean < lower) | (clean > upper)).mean() * 100
                
                if outlier_pct > 5:
                    flags.append({
                        "flag": QualityFlag.OUTLIERS_DETECTED.value,
                        "severity": "major",
                        "dataset_name": dataset_name,
                        "column_name": col,
                        "description": f"{outlier_pct:.1f}% extreme outliers",
                        "suggestion": "Investigate and consider winsorization",
                    })
                elif outlier_pct > 1:
                    flags.append({
                        "flag": QualityFlag.OUTLIERS_DETECTED.value,
                        "severity": "minor",
                        "dataset_name": dataset_name,
                        "column_name": col,
                        "description": f"{outlier_pct:.1f}% potential outliers",
                        "suggestion": "Review outlier treatment",
                    })
                
                # Non-normality / high skew (for regression candidates)
                if HAS_SCIPY and len(clean) >= 20:
                    try:
                        _, p_val = stats.shapiro(clean.sample(min(5000, len(clean))))
                        if p_val < 0.001:
                            flags.append({
                                "flag": QualityFlag.HIGHLY_SKEWED.value,
                                "severity": "minor",
                                "dataset_name": dataset_name,
                                "column_name": col,
                                "description": f"Significant non-normality (p<0.001)",
                                "suggestion": "Consider transformation for parametric tests",
                            })
                    except Exception:
                        # Shapiro-Wilk can fail with extreme values or infinite data.
                        # Safe to skip; non-normality is an informational flag only.
                        pass
    
    # Dataset-level checks
    # Small sample
    if n_rows < 30:
        flags.append({
            "flag": QualityFlag.LOW_SAMPLE_SIZE.value,
            "severity": "critical",
            "dataset_name": dataset_name,
            "column_name": None,
            "description": f"Very small sample size: {n_rows} observations",
            "suggestion": "Limited statistical power; consider non-parametric methods",
        })
    elif n_rows < 100:
        flags.append({
            "flag": QualityFlag.LOW_SAMPLE_SIZE.value,
            "severity": "major",
            "dataset_name": dataset_name,
            "column_name": None,
            "description": f"Small sample size: {n_rows} observations",
            "suggestion": "Document power limitations",
        })
    
    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        dup_pct = dup_count / n_rows * 100
        severity = "critical" if dup_pct > 10 else "major" if dup_pct > 1 else "minor"
        flags.append({
            "flag": QualityFlag.DUPLICATE_ROWS.value,
            "severity": severity,
            "dataset_name": dataset_name,
            "column_name": None,
            "description": f"{dup_count} duplicate rows ({dup_pct:.1f}%)",
            "suggestion": "Investigate and remove if appropriate",
        })
    
    # Multicollinearity check
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        for i, c1 in enumerate(corr.columns):
            for c2 in corr.columns[i+1:]:
                r = corr.loc[c1, c2]
                if pd.notna(r) and abs(r) > 0.9:
                    flags.append({
                        "flag": QualityFlag.MULTICOLLINEARITY.value,
                        "severity": "major",
                        "dataset_name": dataset_name,
                        "column_name": f"{c1}, {c2}",
                        "description": f"High correlation: r={r:.3f}",
                        "suggestion": "Consider removing one variable or using PCA",
                    })
    
    return flags


def _detect_outliers(df: "pd.DataFrame") -> dict[str, Any]:
    """Detect outliers using multiple methods."""
    outliers = {}
    numeric_cols = df.select_dtypes(include=["number"]).columns
    
    for col in numeric_cols:
        clean = df[col].dropna()
        if len(clean) < 10:
            continue
        
        col_outliers = {
            "column": col,
            "n_observations": len(clean),
            "methods": {},
        }
        
        # IQR method
        q1, q3 = clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        iqr_outliers = ((clean < lower_iqr) | (clean > upper_iqr)).sum()
        col_outliers["methods"]["iqr"] = {
            "lower_bound": float(lower_iqr),
            "upper_bound": float(upper_iqr),
            "n_outliers": int(iqr_outliers),
            "pct_outliers": round(iqr_outliers / len(clean) * 100, 2),
        }
        
        # Z-score method
        if HAS_SCIPY:
            z_scores = np.abs(stats.zscore(clean))
            z_outliers = (z_scores > 3).sum()
            col_outliers["methods"]["zscore"] = {
                "threshold": 3,
                "n_outliers": int(z_outliers),
                "pct_outliers": round(z_outliers / len(clean) * 100, 2),
            }
        
        # MAD method (robust)
        median = clean.median()
        mad = np.abs(clean - median).median()
        if mad > 0:
            modified_z = 0.6745 * (clean - median) / mad
            mad_outliers = (np.abs(modified_z) > 3.5).sum()
            col_outliers["methods"]["mad"] = {
                "threshold": 3.5,
                "n_outliers": int(mad_outliers),
                "pct_outliers": round(mad_outliers / len(clean) * 100, 2),
            }
        
        outliers[col] = col_outliers
    
    return outliers


def _analyze_temporal_patterns(df: "pd.DataFrame") -> dict[str, Any]:
    """Analyze temporal patterns in time series or panel data."""
    result = {
        "has_temporal_patterns": False,
        "details": {},
    }
    
    # Find datetime column
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns
    if len(datetime_cols) == 0:
        # Try to find year column
        for col in df.columns:
            if "year" in col.lower():
                if pd.api.types.is_integer_dtype(df[col]):
                    result["time_column"] = col
                    result["time_type"] = "year"
                    
                    years = df[col].dropna().sort_values()
                    result["details"] = {
                        "min_period": int(years.min()),
                        "max_period": int(years.max()),
                        "n_periods": int(years.nunique()),
                        "span_years": int(years.max() - years.min()),
                    }
                    result["has_temporal_patterns"] = True
                    break
    else:
        time_col = datetime_cols[0]
        result["time_column"] = time_col
        result["time_type"] = "datetime"
        
        times = df[time_col].dropna().sort_values()
        result["details"] = {
            "min_date": str(times.min()),
            "max_date": str(times.max()),
            "n_unique_dates": int(times.nunique()),
            "span_days": (times.max() - times.min()).days,
        }
        
        # Detect frequency
        if len(times) > 1:
            diffs = times.diff().dropna()
            median_diff = diffs.median()
            if median_diff.days == 1:
                result["details"]["frequency"] = "daily"
            elif 6 <= median_diff.days <= 8:
                result["details"]["frequency"] = "weekly"
            elif 28 <= median_diff.days <= 31:
                result["details"]["frequency"] = "monthly"
            elif 89 <= median_diff.days <= 92:
                result["details"]["frequency"] = "quarterly"
            elif 364 <= median_diff.days <= 366:
                result["details"]["frequency"] = "annual"
            else:
                result["details"]["frequency"] = "irregular"
        
        result["has_temporal_patterns"] = True
    
    return result


class DetectDataTypesInput(BaseModel):
    """Input for detect_data_types tool."""
    name: str = Field(description="Name of the dataset")


@tool(args_schema=DetectDataTypesInput)
def detect_data_types(name: str) -> dict[str, Any]:
    """
    Intelligently detect semantic data types beyond pandas dtypes.
    
    Identifies:
    - Identifiers (IDs, codes, keys)
    - Temporal types (dates, years, timestamps)
    - Monetary values (prices, costs, revenues)
    - Percentages and proportions
    - Categorical vs free text
    - Email, URL, and other structured formats
    
    Args:
        name: Name of the registered dataset
        
    Returns:
        Dictionary mapping columns to inferred semantic types
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required"}
    
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    inferred = _infer_semantic_types(df)
    
    # Summarize by type
    type_summary = defaultdict(list)
    for col, info in inferred.items():
        type_summary[info["semantic_type"]].append(col)
    
    return {
        "status": "success",
        "dataset_name": name,
        "column_types": inferred,
        "type_summary": dict(type_summary),
        "n_columns": len(df.columns),
    }


class AssessDataQualityInput(BaseModel):
    """Input for assess_data_quality tool."""
    name: str = Field(description="Name of the dataset")


@tool(args_schema=AssessDataQualityInput)
def assess_data_quality(name: str) -> dict[str, Any]:
    """
    Comprehensive data quality assessment with QualityFlag enum.
    
    Checks for:
    - Missing values (with pattern analysis)
    - Duplicate rows
    - Outliers (multiple detection methods)
    - Multicollinearity
    - Constant columns
    - High cardinality
    - Sample size adequacy
    - Distribution issues
    
    Returns QualityFlagItem objects for each issue found.
    
    Args:
        name: Name of the registered dataset
        
    Returns:
        Quality assessment with flags, score, and recommendations
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required"}
    
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    flags = _generate_quality_flags(df, name)
    
    # Calculate quality score using consolidated scoring function
    score = _calculate_quality_score(flags)
    
    # Count issues by severity for summary
    critical_count = sum(1 for f in flags if f["severity"] == "critical")
    major_count = sum(1 for f in flags if f["severity"] == "major")
    minor_count = sum(1 for f in flags if f["severity"] == "minor")
    
    # Determine quality level
    if score >= 80:
        level = "excellent"
    elif score >= 60:
        level = "good"
    elif score >= 40:
        level = "acceptable"
    else:
        level = "poor"
    
    # Group flags by type
    flags_by_type = defaultdict(list)
    for f in flags:
        flags_by_type[f["flag"]].append(f)
    
    return {
        "status": "success",
        "dataset_name": name,
        "quality_score": score,
        "quality_level": level,
        "summary": {
            "critical_issues": critical_count,
            "major_issues": major_count,
            "minor_issues": minor_count,
            "total_issues": len(flags),
        },
        "flags": flags,
        "flags_by_type": dict(flags_by_type),
        "recommendations": list(set(f["suggestion"] for f in flags)),
    }


class IdentifyTimeSeriesInput(BaseModel):
    """Input for identify_time_series tool."""
    name: str = Field(description="Name of the dataset")


@tool(args_schema=IdentifyTimeSeriesInput)
def identify_time_series(name: str) -> dict[str, Any]:
    """
    Detect if data has time series structure and analyze temporal patterns.
    
    Identifies:
    - Time column (datetime or year)
    - Frequency (daily, weekly, monthly, quarterly, annual)
    - Time span and coverage
    - Missing periods
    - Temporal trends
    
    Args:
        name: Name of the registered dataset
        
    Returns:
        Time series analysis results
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required"}
    
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    result = _analyze_temporal_patterns(df)
    result["dataset_name"] = name
    result["status"] = "success" if result["has_temporal_patterns"] else "no_temporal_data"
    
    return result


class DetectPanelStructureInput(BaseModel):
    """Input for detect_panel_structure tool."""
    name: str = Field(description="Name of the dataset")


@tool(args_schema=DetectPanelStructureInput)
def detect_panel_structure(name: str) -> dict[str, Any]:
    """
    Detect if data has panel (longitudinal) structure.
    
    Identifies:
    - Entity identifier column
    - Time identifier column
    - Panel balance (balanced vs unbalanced)
    - Number of entities and periods
    - Within vs between variation
    
    Args:
        name: Name of the registered dataset
        
    Returns:
        Panel structure analysis results
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required"}
    
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    structure = _detect_data_structure(df)
    
    result = {
        "status": "success",
        "dataset_name": name,
        "is_panel": structure["type"] == DataStructureType.PANEL.value,
        "structure_type": structure["type"],
        "confidence": structure["confidence"],
        "entity_column": structure["entity_column"],
        "time_column": structure["time_column"],
    }
    
    if structure["panel_info"]:
        result["panel_details"] = structure["panel_info"]
        
        # Calculate within/between variation for numeric columns
        if structure["entity_column"]:
            entity_col = structure["entity_column"]
            numeric_cols = df.select_dtypes(include=["number"]).columns
            variation = {}
            
            for col in numeric_cols[:10]:  # Limit to first 10
                if col != entity_col:
                    try:
                        overall_var = df[col].var()
                        between_var = df.groupby(entity_col)[col].mean().var()
                        within_var = df.groupby(entity_col)[col].var().mean()
                        
                        if overall_var > 0:
                            variation[col] = {
                                "between_pct": round(between_var / overall_var * 100, 1),
                                "within_pct": round(within_var / overall_var * 100, 1),
                            }
                    except Exception:
                        # Variation decomposition can fail with constant columns or
                        # single-entity groups. Safe to skip per-column; researchers
                        # should review raw data for edge cases.
                        pass
            
            if variation:
                result["variation_decomposition"] = variation
    
    return result


# =============================================================================
# Sprint 12: LLM Summarization for Methods Section
# =============================================================================


class GenerateDataProseSummaryInput(BaseModel):
    """Input for generate_data_prose_summary tool."""
    dataset_names: list[str] = Field(description="List of dataset names to summarize")
    research_context: str | None = Field(default=None, description="Research context for tailoring")
    focus_variables: list[str] | None = Field(default=None, description="Key variables to emphasize")


@tool(args_schema=GenerateDataProseSummaryInput)
def generate_data_prose_summary(
    dataset_names: list[str],
    research_context: str | None = None,
    focus_variables: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate LLM-written prose summary suitable for Methods section.
    
    Creates comprehensive academic prose describing:
    - Data sources and collection
    - Sample characteristics
    - Key variables and their operationalization
    - Data quality and limitations
    - Panel/time series structure if applicable
    
    Returns a dict containing a DataExplorationSummary (model_dump format) for use by downstream nodes.
    
    Args:
        dataset_names: Names of datasets to summarize
        research_context: Optional research context
        focus_variables: Key variables to emphasize
        
    Returns:
        Dict with 'summary' key containing DataExplorationSummary.model_dump(),
        plus additional metadata (status, prose_description, n_datasets, etc.)
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required"}
    
    registry = get_registry()
    
    # Gather profiles for all datasets
    profiles = []
    dataset_inventory = []
    all_quality_flags = []
    
    for name in dataset_names:
        df = registry.get(name)
        if df is None:
            continue
        
        # Run deep profile
        profile = deep_profile_dataset.invoke({"name": name})
        if "error" not in profile:
            profiles.append(profile)
            
            # Build DatasetInfo
            overview = profile["overview"]
            structure = profile["data_structure"]
            
            # Extract date range info if temporal data
            date_range_start = None
            date_range_end = None
            if structure.get("type") in ["time_series", "panel"]:
                temporal = profile.get("temporal_analysis", {})
                details = temporal.get("details", {})
                # Try to extract dates (year-based or datetime-based)
                if "min_period" in details:
                    try:
                        from datetime import date
                        date_range_start = date(int(details['min_period']), 1, 1)
                        date_range_end = date(int(details['max_period']), 12, 31)
                    except (ValueError, TypeError):
                        # Year-based parsing can fail if period is not a valid integer.
                        # Date range is optional; safe to skip.
                        pass
                elif "min_date" in details:
                    try:
                        from datetime import date
                        min_str = details['min_date'][:10]
                        max_str = details['max_date'][:10]
                        date_range_start = date.fromisoformat(min_str)
                        date_range_end = date.fromisoformat(max_str)
                    except (ValueError, TypeError):
                        # ISO date parsing can fail with non-standard formats.
                        # Date range is optional; safe to skip.
                        pass
            
            dataset_info = DatasetInfo(
                name=name,
                row_count=overview["row_count"],
                column_count=overview["column_count"],
                memory_mb=overview["memory_mb"],
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                structure_type=DataStructureType(structure["type"]),
            )
            dataset_inventory.append(dataset_info)
            
            # Collect quality flags
            for flag_dict in profile.get("quality_flags", []):
                flag_item = QualityFlagItem(
                    flag=QualityFlag(flag_dict["flag"]),
                    severity=_severity_to_enum(flag_dict["severity"]),
                    dataset_name=flag_dict["dataset_name"],
                    column_name=flag_dict.get("column_name"),
                    description=flag_dict["description"],
                    suggestion=flag_dict["suggestion"],
                )
                all_quality_flags.append(flag_item)
    
    if not profiles:
        return {"error": "No datasets found"}
    
    # Generate prose using LLM
    prose = _generate_methods_prose(profiles, research_context, focus_variables)
    
    # Identify recommended variables
    recommended_vars = _identify_recommended_variables(profiles, focus_variables)
    
    # Identify data gaps
    data_gaps = _identify_data_gaps(profiles)
    
    # Build the summary
    summary = DataExplorationSummary(
        prose_description=prose,
        dataset_inventory=dataset_inventory,
        quality_flags=all_quality_flags,
        recommended_variables=recommended_vars,
        data_gaps=data_gaps,
    )
    
    return {
        "status": "success",
        "summary": summary.model_dump(),
        "prose_description": prose,
        "n_datasets": len(profiles),
        "total_observations": sum(p["row_count"] for p in profiles),
        "total_quality_issues": len(all_quality_flags),
    }


def _generate_methods_prose(
    profiles: list[dict],
    research_context: str | None,
    focus_variables: list[str] | None,
) -> str:
    """Generate Methods section prose using LLM."""
    
    # Build comprehensive profile summary
    summary_parts = []
    
    for profile in profiles:
        name = profile["dataset_name"]
        overview = profile["overview"]
        structure = profile["data_structure"]
        
        part = f"""
Dataset: {name}
- Observations: {overview['row_count']:,}
- Variables: {overview['column_count']}
- Numeric: {overview['numeric_columns']}, Categorical: {overview['categorical_columns']}
- Complete cases: {overview['complete_row_pct']:.1f}%
- Structure: {structure['type']}
"""
        if structure.get("panel_info"):
            pi = structure["panel_info"]
            part += f"- Panel: {pi['n_entities']} entities × {pi['n_periods']} periods"
            part += f" ({'balanced' if pi['is_balanced'] else 'unbalanced'})\n"
        
        # Add key variable stats
        columns = profile.get("columns", {})
        numeric_stats = []
        for col, stats in list(columns.items())[:5]:
            if stats.get("mean") is not None:
                numeric_stats.append(
                    f"  - {col}: M={stats['mean']:.2f}, SD={stats['std']:.2f}"
                )
        if numeric_stats:
            part += "Key numeric variables:\n" + "\n".join(numeric_stats) + "\n"
        
        # Quality issues
        quality_flags = profile.get("quality_flags", [])
        if quality_flags:
            issues = [f["description"] for f in quality_flags[:3]]
            part += f"Quality notes: {'; '.join(issues)}\n"
        
        summary_parts.append(part)
    
    profile_text = "\n".join(summary_parts)
    
    system_prompt = """You are an academic researcher writing the Data section of a methods chapter.

STRICT RULES:
- NEVER make up statistics, numbers, or facts not provided
- NEVER use emojis
- NEVER use em dashes (use semicolons, colons, or periods instead)
- Use passive voice and formal academic tone
- Cite exact statistics from the data provided
- Be precise and concise
- Include sample sizes, time periods, and key variable distributions
- Note any data quality limitations that could affect analysis"""

    user_prompt = f"""Write a Methods/Data section describing the following dataset(s).

{f'Research Context: {research_context}' if research_context else ''}
{f'Focus Variables: {", ".join(focus_variables)}' if focus_variables else ''}

Dataset Profiles:
{profile_text}

Write 2-4 paragraphs covering:
1. Data source and sample description
2. Key variables and their measurement
3. Data structure (cross-sectional/panel/time series)
4. Any data quality considerations or limitations

Use formal academic prose suitable for a peer-reviewed journal."""

    try:
        model = ChatAnthropic(
            model=settings.default_model,
            api_key=settings.anthropic_api_key,
            max_tokens=2000,
            temperature=0.3,
        )
        
        response = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        
        return response.content
        
    except Exception as e:
        # Log the error for debugging API issues, rate limits, or config problems
        logging.warning(f"LLM prose generation failed, using template fallback: {e}")
        return _generate_template_methods_prose(profiles)


def _generate_template_methods_prose(profiles: list[dict]) -> str:
    """Generate template-based prose as fallback."""
    if len(profiles) == 1:
        p = profiles[0]
        overview = p["overview"]
        structure = p["data_structure"]
        
        text = f"The analysis utilizes a dataset containing {overview['row_count']:,} observations "
        text += f"across {overview['column_count']} variables. "
        
        if structure["type"] == "panel":
            pi = structure["panel_info"]
            text += f"The data has a panel structure with {pi['n_entities']} entities "
            text += f"observed over {pi['n_periods']} time periods. "
        elif structure["type"] == "time_series":
            text += "The data has a time series structure. "
        else:
            text += "The data represents a cross-sectional sample. "
        
        text += f"Data completeness is {overview['complete_row_pct']:.1f}%."
        
        return text
    else:
        total_obs = sum(p["overview"]["row_count"] for p in profiles)
        text = f"The analysis draws on {len(profiles)} datasets with a combined "
        text += f"{total_obs:,} observations. "
        return text


def _identify_recommended_variables(
    profiles: list[dict],
    focus_variables: list[str] | None,
) -> list[str]:
    """Identify variables recommended for analysis."""
    recommended = []
    
    # Add focus variables first
    if focus_variables:
        recommended.extend(focus_variables)
    
    # Add well-behaved numeric variables
    for profile in profiles:
        columns = profile.get("columns", {})
        quality_flags = profile.get("quality_flags", [])
        flagged_cols = {f["column_name"] for f in quality_flags if f.get("column_name")}
        
        for col, stats in columns.items():
            if col in flagged_cols:
                continue
            if stats.get("mean") is not None:
                # Numeric with reasonable distribution
                if stats["null_pct"] < 20 and stats["unique_count"] > 10:
                    if col not in recommended:
                        recommended.append(col)
    
    return recommended[:20]  # Limit


def _identify_data_gaps(profiles: list[dict]) -> list[str]:
    """Identify gaps in the data."""
    gaps = []
    
    for profile in profiles:
        name = profile["dataset_name"]
        overview = profile["overview"]
        quality_flags = profile.get("quality_flags", [])
        
        # High missing data
        if overview["complete_row_pct"] < 70:
            gaps.append(f"Incomplete data in {name}: only {overview['complete_row_pct']:.0f}% complete cases")
        
        # Critical quality issues
        critical_flags = [f for f in quality_flags if f["severity"] == "critical"]
        for flag in critical_flags:
            gaps.append(f"{name}: {flag['description']}")
        
        # Check for short time series
        structure = profile.get("data_structure", {})
        if structure.get("type") in ["time_series", "panel"]:
            temporal = profile.get("temporal_analysis", {})
            details = temporal.get("details", {})
            n_periods = details.get("n_periods", 0)
            if 0 < n_periods < 10:
                gaps.append(f"Short time series in {name}: only {n_periods} periods")
    
    return gaps


# =============================================================================
# Exports
# =============================================================================

# Original tools
DATA_PROFILING_TOOLS = [
    profile_dataset,
    describe_dataset,
    describe_variable,
]

# Sprint 12 enhanced tools
SPRINT_12_PROFILING_TOOLS = [
    deep_profile_dataset,
    detect_data_types,
    assess_data_quality,
    identify_time_series,
    detect_panel_structure,
    generate_data_prose_summary,
]

# Combined
ALL_PROFILING_TOOLS = DATA_PROFILING_TOOLS + SPRINT_12_PROFILING_TOOLS


def get_profiling_tools() -> list:
    """Get list of all data profiling tools (original)."""
    return DATA_PROFILING_TOOLS


def get_enhanced_profiling_tools() -> list:
    """Get list of Sprint 12 enhanced profiling tools."""
    return SPRINT_12_PROFILING_TOOLS


def get_all_profiling_tools() -> list:
    """Get list of all profiling tools (original + enhanced)."""
    return ALL_PROFILING_TOOLS


__all__ = [
    # Original
    "profile_dataset",
    "describe_dataset",
    "describe_variable",
    "get_profiling_tools",
    "DATA_PROFILING_TOOLS",
    # Sprint 12
    "deep_profile_dataset",
    "detect_data_types",
    "assess_data_quality",
    "identify_time_series",
    "detect_panel_structure",
    "generate_data_prose_summary",
    "get_enhanced_profiling_tools",
    "get_all_profiling_tools",
    "SPRINT_12_PROFILING_TOOLS",
    "ALL_PROFILING_TOOLS",
]
