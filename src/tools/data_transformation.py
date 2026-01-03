"""Data transformation tools for data preparation and cleaning.

Provides tools for:
- Filtering and subsetting data
- Aggregation and grouping
- Merging multiple datasets
- Creating derived variables
- Handling missing values
- Encoding categorical variables
"""

import re
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.tools.data_loading import get_registry


# Security: Blocked patterns in eval expressions (code injection prevention)
_BLOCKED_EVAL_PATTERNS = [
    r'__\w+__',          # dunder methods
    r'\bimport\b',       # import statements
    r'\bexec\b',         # exec calls
    r'\beval\b',         # nested eval
    r'\bopen\b',         # file operations
    r'\bos\.',           # os module
    r'\bsys\.',          # sys module
    r'\bsubprocess\.',   # subprocess module
    r'@',                # decorators/matmul that might be misused
    r'\bgetattr\b',      # attribute access
    r'\bsetattr\b',      # attribute setting
    r'\bdelattr\b',      # attribute deletion
    r'\bcompile\b',      # dynamic code compilation
    r'\bglobals\b',      # global namespace access
    r'\blocals\b',       # local namespace access
]


def _validate_eval_expression(expression: str) -> tuple[bool, str | None]:
    """
    Validate that an eval expression doesn't contain dangerous patterns.
    
    Args:
        expression: The expression to validate
        
    Returns:
        Tuple of (is_valid, error_message or None)
    """
    for pattern in _BLOCKED_EVAL_PATTERNS:
        if re.search(pattern, expression, re.IGNORECASE):
            return False, f"Expression contains blocked pattern: {pattern}"
    return True, None


# =============================================================================
# Filtering Tools
# =============================================================================


class FilterDataInput(BaseModel):
    """Input for filter_data tool."""
    name: str = Field(description="Name of the dataset to filter")
    conditions: str = Field(description="Filter conditions as pandas query string (e.g., 'age > 18 and status == \"active\"')")
    output_name: str | None = Field(default=None, description="Name for the filtered dataset (defaults to name_filtered)")


@tool(args_schema=FilterDataInput)
def filter_data(name: str, conditions: str, output_name: str | None = None) -> dict[str, Any]:
    """
    Filter a dataset based on conditions.
    
    Uses pandas query syntax. Supports:
    - Comparisons: ==, !=, <, >, <=, >=
    - Boolean: and, or, not
    - String methods: .str.contains(), .str.startswith()
    - Null checks: .isna(), .notna()
    
    Examples:
        filter_data("sales", "revenue > 1000 and region == 'East'")
        filter_data("users", "age.between(18, 65)")
    
    Args:
        name: Name of the source dataset
        conditions: Pandas query string
        output_name: Name for the result (default: {name}_filtered)
        
    Returns:
        Info about the filtered dataset
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required"}
    
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    try:
        filtered = df.query(conditions)
    except Exception as e:
        return {"error": f"Filter failed: {str(e)}. Use pandas query syntax."}
    
    if output_name is None:
        output_name = f"{name}_filtered"
    
    info = registry.register(filtered, output_name, metadata={"source": name, "filter": conditions})
    
    return {
        "status": "success",
        "dataset_name": output_name,
        "original_rows": len(df),
        "filtered_rows": len(filtered),
        "rows_removed": len(df) - len(filtered),
        "retention_pct": round(len(filtered) / len(df) * 100, 1),
        **info,
    }


class SelectColumnsInput(BaseModel):
    """Input for select_columns tool."""
    name: str = Field(description="Name of the dataset")
    columns: list[str] = Field(description="List of column names to select")
    output_name: str | None = Field(default=None, description="Name for the result")


@tool(args_schema=SelectColumnsInput)
def select_columns(name: str, columns: list[str], output_name: str | None = None) -> dict[str, Any]:
    """
    Select specific columns from a dataset.
    
    Args:
        name: Name of the source dataset
        columns: List of column names to keep
        output_name: Name for the result
        
    Returns:
        Info about the resulting dataset
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required"}
    
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    # Check for missing columns
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return {"error": f"Columns not found: {missing}. Available: {list(df.columns)}"}
    
    selected = df[columns]
    
    if output_name is None:
        output_name = f"{name}_selected"
    
    info = registry.register(selected, output_name, metadata={"source": name, "selected_columns": columns})
    
    return {
        "status": "success",
        "dataset_name": output_name,
        "original_columns": len(df.columns),
        "selected_columns": len(columns),
        **info,
    }


# =============================================================================
# Aggregation Tools
# =============================================================================


class AggregateDataInput(BaseModel):
    """Input for aggregate_data tool."""
    name: str = Field(description="Name of the dataset")
    group_by: list[str] = Field(description="Columns to group by")
    aggregations: dict[str, str | list[str]] = Field(
        description="Aggregation specs: {column: function} or {column: [functions]}. Functions: sum, mean, median, min, max, count, std, var, first, last"
    )
    output_name: str | None = Field(default=None, description="Name for the result")


@tool(args_schema=AggregateDataInput)
def aggregate_data(
    name: str,
    group_by: list[str],
    aggregations: dict[str, str | list[str]],
    output_name: str | None = None,
) -> dict[str, Any]:
    """
    Aggregate data by groups.
    
    Examples:
        aggregate_data("sales", ["region"], {"revenue": "sum", "transactions": "count"})
        aggregate_data("prices", ["ticker", "year"], {"price": ["mean", "std", "min", "max"]})
    
    Args:
        name: Name of the source dataset
        group_by: Columns to group by
        aggregations: {column: function(s)} mapping
        output_name: Name for the result
        
    Returns:
        Info about the aggregated dataset
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required"}
    
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    # Validate columns
    missing_group = [c for c in group_by if c not in df.columns]
    if missing_group:
        return {"error": f"Group columns not found: {missing_group}"}
    
    missing_agg = [c for c in aggregations.keys() if c not in df.columns]
    if missing_agg:
        return {"error": f"Aggregation columns not found: {missing_agg}"}
    
    try:
        aggregated = df.groupby(group_by, as_index=False).agg(aggregations)
        
        # Flatten multi-level column names if needed
        if isinstance(aggregated.columns, pd.MultiIndex):
            aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
    except Exception as e:
        return {"error": f"Aggregation failed: {str(e)}"}
    
    if output_name is None:
        output_name = f"{name}_aggregated"
    
    info = registry.register(aggregated, output_name, metadata={
        "source": name,
        "group_by": group_by,
        "aggregations": aggregations,
    })
    
    return {
        "status": "success",
        "dataset_name": output_name,
        "original_rows": len(df),
        "aggregated_rows": len(aggregated),
        "groups": len(aggregated),
        **info,
    }


# =============================================================================
# Merging Tools
# =============================================================================


class MergeDatasetsInput(BaseModel):
    """Input for merge_datasets tool."""
    left_name: str = Field(description="Name of the left dataset")
    right_name: str = Field(description="Name of the right dataset")
    on: str | list[str] | None = Field(default=None, description="Column(s) to join on")
    left_on: str | list[str] | None = Field(default=None, description="Left join column(s)")
    right_on: str | list[str] | None = Field(default=None, description="Right join column(s)")
    how: str = Field(default="inner", description="Join type: inner, left, right, outer")
    output_name: str | None = Field(default=None, description="Name for the result")


@tool(args_schema=MergeDatasetsInput)
def merge_datasets(
    left_name: str,
    right_name: str,
    on: str | list[str] | None = None,
    left_on: str | list[str] | None = None,
    right_on: str | list[str] | None = None,
    how: str = "inner",
    output_name: str | None = None,
) -> dict[str, Any]:
    """
    Merge two datasets.
    
    Examples:
        merge_datasets("prices", "volume", on="date")
        merge_datasets("users", "orders", left_on="user_id", right_on="customer_id", how="left")
    
    Args:
        left_name: Name of left dataset
        right_name: Name of right dataset
        on: Column(s) to join on (if same name in both)
        left_on: Left join column(s)
        right_on: Right join column(s)
        how: Join type (inner, left, right, outer)
        output_name: Name for the result
        
    Returns:
        Info about the merged dataset
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required"}
    
    registry = get_registry()
    left_df = registry.get(left_name)
    right_df = registry.get(right_name)
    
    if left_df is None:
        return {"error": f"Dataset '{left_name}' not found"}
    if right_df is None:
        return {"error": f"Dataset '{right_name}' not found"}
    
    if how not in ("inner", "left", "right", "outer"):
        return {"error": f"Invalid join type: {how}. Use: inner, left, right, outer"}
    
    try:
        merged = pd.merge(
            left_df, right_df,
            on=on, left_on=left_on, right_on=right_on,
            how=how,
            suffixes=('_left', '_right'),
        )
    except Exception as e:
        return {"error": f"Merge failed: {str(e)}"}
    
    if output_name is None:
        output_name = f"{left_name}_{right_name}_merged"
    
    info = registry.register(merged, output_name, metadata={
        "left_source": left_name,
        "right_source": right_name,
        "join_type": how,
        "join_keys": on or {"left": left_on, "right": right_on},
    })
    
    return {
        "status": "success",
        "dataset_name": output_name,
        "left_rows": len(left_df),
        "right_rows": len(right_df),
        "merged_rows": len(merged),
        "join_type": how,
        **info,
    }


# =============================================================================
# Variable Creation Tools
# =============================================================================


class CreateVariableInput(BaseModel):
    """Input for create_variable tool."""
    name: str = Field(description="Name of the dataset")
    new_column: str = Field(description="Name for the new column")
    expression: str = Field(description="Pandas expression to compute the new column")
    output_name: str | None = Field(default=None, description="Name for the result (default: modifies in place)")


@tool(args_schema=CreateVariableInput)
def create_variable(
    name: str,
    new_column: str,
    expression: str,
    output_name: str | None = None,
) -> dict[str, Any]:
    """
    Create a new variable based on an expression.
    
    Uses pandas eval syntax for safe expression evaluation. Reference columns
    directly by name. Complex transformations requiring pd/np functions should
    be broken into multiple simpler steps.
    
    Examples:
        create_variable("sales", "profit", "revenue - cost")
        create_variable("data", "ratio", "column_a / column_b")
        create_variable("df", "scaled", "value * 100")
    
    For complex transformations (pd.cut, np.log, etc.), use the apply_function
    tool or break into multiple steps.
    
    Args:
        name: Name of the source dataset
        new_column: Name for the new column
        expression: Pandas eval expression for the new value (column math only)
        output_name: Name for result (if None, creates new dataset)
        
    Returns:
        Info about the updated dataset
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required"}
    
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Validate column name to prevent injection
    if not new_column.isidentifier():
        return {"error": f"Invalid column name: '{new_column}'. Must be a valid Python identifier."}
    
    # Validate expression for security (prevent code injection)
    is_valid, error_msg = _validate_eval_expression(expression)
    if not is_valid:
        return {"error": f"Invalid expression: {error_msg}"}
    
    try:
        # Use pandas eval only (safe, no exec/eval fallback)
        # This supports column arithmetic: "a + b", "a * 2", "a / b", etc.
        df[new_column] = df.eval(expression)
    except Exception as e:
        return {
            "error": f"Expression failed: {str(e)}. "
            "Use simple column arithmetic (e.g., 'col_a + col_b'). "
            "For complex transformations, use apply_function tool."
        }
    
    if output_name is None:
        output_name = name  # Overwrite
    
    info = registry.register(df, output_name, metadata={
        "source": name,
        "created_column": new_column,
        "expression": expression,
    })
    
    # Get stats on new column
    new_col_info = {}
    if pd.api.types.is_numeric_dtype(df[new_column]):
        new_col_info = {
            "mean": float(df[new_column].mean()) if df[new_column].notna().any() else None,
            "std": float(df[new_column].std()) if df[new_column].notna().any() else None,
            "null_count": int(df[new_column].isna().sum()),
        }
    else:
        new_col_info = {
            "unique_count": int(df[new_column].nunique()),
            "null_count": int(df[new_column].isna().sum()),
        }
    
    return {
        "status": "success",
        "dataset_name": output_name,
        "new_column": new_column,
        "new_column_stats": new_col_info,
        **info,
    }


# =============================================================================
# Missing Value Handling
# =============================================================================


class HandleMissingInput(BaseModel):
    """Input for handle_missing tool."""
    name: str = Field(description="Name of the dataset")
    strategy: str = Field(description="Strategy: drop_rows, drop_cols, fill_mean, fill_median, fill_mode, fill_value, fill_forward, fill_backward")
    columns: list[str] | None = Field(default=None, description="Columns to apply strategy to (default: all)")
    fill_value: Any = Field(default=None, description="Value to fill with (for fill_value strategy)")
    threshold: float | None = Field(default=None, description="For drop strategies: max proportion of missing allowed")
    output_name: str | None = Field(default=None, description="Name for the result")


@tool(args_schema=HandleMissingInput)
def handle_missing(
    name: str,
    strategy: str,
    columns: list[str] | None = None,
    fill_value: Any = None,
    threshold: float | None = None,
    output_name: str | None = None,
) -> dict[str, Any]:
    """
    Handle missing values in a dataset.
    
    Strategies:
    - drop_rows: Remove rows with missing values
    - drop_cols: Remove columns with missing values
    - fill_mean: Fill numeric columns with mean
    - fill_median: Fill numeric columns with median
    - fill_mode: Fill with most frequent value
    - fill_value: Fill with specified value
    - fill_forward: Forward fill (for time series)
    - fill_backward: Backward fill
    
    Args:
        name: Name of the dataset
        strategy: Missing value handling strategy
        columns: Columns to apply to (default: all)
        fill_value: Value for fill_value strategy
        threshold: Max missing proportion for drop strategies
        output_name: Name for the result
        
    Returns:
        Info about missing value handling
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required"}
    
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    df = df.copy()
    original_missing = int(df.isna().sum().sum())
    original_rows = len(df)
    original_cols = len(df.columns)
    
    # Determine target columns
    target_cols = columns if columns else df.columns.tolist()
    invalid_cols = [c for c in target_cols if c not in df.columns]
    if invalid_cols:
        return {"error": f"Columns not found: {invalid_cols}"}
    
    try:
        if strategy == "drop_rows":
            if threshold is not None:
                # Drop rows exceeding threshold
                row_missing_pct = df[target_cols].isna().mean(axis=1)
                df = df[row_missing_pct <= threshold]
            else:
                df = df.dropna(subset=target_cols)
        
        elif strategy == "drop_cols":
            if threshold is not None:
                col_missing_pct = df[target_cols].isna().mean()
                cols_to_drop = col_missing_pct[col_missing_pct > threshold].index.tolist()
                df = df.drop(columns=cols_to_drop)
            else:
                df = df.dropna(axis=1, subset=target_cols if target_cols != df.columns.tolist() else None)
        
        elif strategy == "fill_mean":
            numeric_cols = df[target_cols].select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        elif strategy == "fill_median":
            numeric_cols = df[target_cols].select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        elif strategy == "fill_mode":
            for col in target_cols:
                if df[col].isna().any():
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val.iloc[0])
        
        elif strategy == "fill_value":
            if fill_value is None:
                return {"error": "fill_value required for fill_value strategy"}
            df[target_cols] = df[target_cols].fillna(fill_value)
        
        elif strategy == "fill_forward":
            df[target_cols] = df[target_cols].ffill()
        
        elif strategy == "fill_backward":
            df[target_cols] = df[target_cols].bfill()
        
        else:
            return {"error": f"Unknown strategy: {strategy}"}
            
    except Exception as e:
        return {"error": f"Strategy failed: {str(e)}"}
    
    if output_name is None:
        output_name = f"{name}_cleaned"
    
    info = registry.register(df, output_name, metadata={
        "source": name,
        "missing_strategy": strategy,
        "target_columns": target_cols,
    })
    
    final_missing = int(df.isna().sum().sum())
    
    return {
        "status": "success",
        "dataset_name": output_name,
        "strategy": strategy,
        "original_missing": original_missing,
        "final_missing": final_missing,
        "missing_resolved": original_missing - final_missing,
        "rows_before": original_rows,
        "rows_after": len(df),
        "cols_before": original_cols,
        "cols_after": len(df.columns),
        **info,
    }


# =============================================================================
# Encoding Tools
# =============================================================================


class EncodeCategoricalInput(BaseModel):
    """Input for encode_categorical tool."""
    name: str = Field(description="Name of the dataset")
    columns: list[str] = Field(description="Categorical columns to encode")
    method: str = Field(default="onehot", description="Encoding method: onehot, label, ordinal")
    drop_first: bool = Field(default=True, description="Drop first category for onehot (avoid multicollinearity)")
    output_name: str | None = Field(default=None, description="Name for the result")


@tool(args_schema=EncodeCategoricalInput)
def encode_categorical(
    name: str,
    columns: list[str],
    method: str = "onehot",
    drop_first: bool = True,
    output_name: str | None = None,
) -> dict[str, Any]:
    """
    Encode categorical variables for analysis.
    
    Methods:
    - onehot: Create dummy variables (0/1 for each category)
    - label: Integer encoding (0, 1, 2, ...)
    - ordinal: Same as label but preserves ordering info
    
    Args:
        name: Name of the dataset
        columns: Columns to encode
        method: Encoding method
        drop_first: For onehot, drop first category
        output_name: Name for the result
        
    Returns:
        Info about encoding results
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required"}
    
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    invalid_cols = [c for c in columns if c not in df.columns]
    if invalid_cols:
        return {"error": f"Columns not found: {invalid_cols}"}
    
    df = df.copy()
    encoding_info = {}
    
    try:
        if method == "onehot":
            df = pd.get_dummies(df, columns=columns, drop_first=drop_first)
            for col in columns:
                encoding_info[col] = {
                    "method": "onehot",
                    "new_columns": [c for c in df.columns if c.startswith(f"{col}_")],
                }
        
        elif method in ("label", "ordinal"):
            for col in columns:
                categories = df[col].astype("category")
                df[f"{col}_encoded"] = categories.cat.codes
                encoding_info[col] = {
                    "method": method,
                    "new_column": f"{col}_encoded",
                    "mapping": dict(enumerate(categories.cat.categories)),
                }
        
        else:
            return {"error": f"Unknown method: {method}. Use: onehot, label, ordinal"}
            
    except Exception as e:
        return {"error": f"Encoding failed: {str(e)}"}
    
    if output_name is None:
        output_name = f"{name}_encoded"
    
    info = registry.register(df, output_name, metadata={
        "source": name,
        "encoding_method": method,
        "encoded_columns": columns,
    })
    
    return {
        "status": "success",
        "dataset_name": output_name,
        "method": method,
        "columns_encoded": columns,
        "encoding_details": encoding_info,
        **info,
    }


# =============================================================================
# Reshaping Tools
# =============================================================================


class PivotDataInput(BaseModel):
    """Input for pivot_data tool."""
    name: str = Field(description="Name of the dataset")
    index: str | list[str] = Field(description="Column(s) to use as row index")
    columns: str = Field(description="Column to pivot into new columns")
    values: str | list[str] = Field(description="Column(s) to aggregate")
    aggfunc: str = Field(default="mean", description="Aggregation function")
    output_name: str | None = Field(default=None, description="Name for the result")


@tool(args_schema=PivotDataInput)
def pivot_data(
    name: str,
    index: str | list[str],
    columns: str,
    values: str | list[str],
    aggfunc: str = "mean",
    output_name: str | None = None,
) -> dict[str, Any]:
    """
    Pivot (reshape) data from long to wide format.
    
    Example:
        pivot_data("sales", index="date", columns="product", values="revenue", aggfunc="sum")
    
    Args:
        name: Name of the dataset
        index: Row identifier column(s)
        columns: Column to spread into new columns
        values: Value column(s) to aggregate
        aggfunc: Aggregation function
        output_name: Name for the result
        
    Returns:
        Info about the pivoted dataset
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required"}
    
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    try:
        pivoted = df.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
        ).reset_index()
        
        # Flatten column names if MultiIndex
        if isinstance(pivoted.columns, pd.MultiIndex):
            pivoted.columns = ['_'.join(str(c) for c in col).strip('_') for col in pivoted.columns.values]
            
    except Exception as e:
        return {"error": f"Pivot failed: {str(e)}"}
    
    if output_name is None:
        output_name = f"{name}_pivoted"
    
    info = registry.register(pivoted, output_name, metadata={
        "source": name,
        "pivot_index": index,
        "pivot_columns": columns,
        "pivot_values": values,
    })
    
    return {
        "status": "success",
        "dataset_name": output_name,
        "original_shape": f"{len(df)} rows x {len(df.columns)} cols",
        "pivoted_shape": f"{len(pivoted)} rows x {len(pivoted.columns)} cols",
        **info,
    }


class MeltDataInput(BaseModel):
    """Input for melt_data tool."""
    name: str = Field(description="Name of the dataset")
    id_vars: list[str] = Field(description="Columns to keep as identifiers")
    value_vars: list[str] | None = Field(default=None, description="Columns to unpivot (default: all others)")
    var_name: str = Field(default="variable", description="Name for the variable column")
    value_name: str = Field(default="value", description="Name for the value column")
    output_name: str | None = Field(default=None, description="Name for the result")


@tool(args_schema=MeltDataInput)
def melt_data(
    name: str,
    id_vars: list[str],
    value_vars: list[str] | None = None,
    var_name: str = "variable",
    value_name: str = "value",
    output_name: str | None = None,
) -> dict[str, Any]:
    """
    Melt (unpivot) data from wide to long format.
    
    Example:
        melt_data("wide_data", id_vars=["date", "ticker"], value_vars=["open", "close", "volume"])
    
    Args:
        name: Name of the dataset
        id_vars: Columns to keep as identifiers
        value_vars: Columns to unpivot (default: all except id_vars)
        var_name: Name for variable column
        value_name: Name for value column
        output_name: Name for the result
        
    Returns:
        Info about the melted dataset
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required"}
    
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    try:
        melted = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name,
        )
    except Exception as e:
        return {"error": f"Melt failed: {str(e)}"}
    
    if output_name is None:
        output_name = f"{name}_melted"
    
    info = registry.register(melted, output_name, metadata={
        "source": name,
        "id_vars": id_vars,
        "value_vars": value_vars,
    })
    
    return {
        "status": "success",
        "dataset_name": output_name,
        "original_shape": f"{len(df)} rows x {len(df.columns)} cols",
        "melted_shape": f"{len(melted)} rows x {len(melted.columns)} cols",
        **info,
    }


# =============================================================================
# Exports
# =============================================================================

DATA_TRANSFORMATION_TOOLS = [
    filter_data,
    select_columns,
    aggregate_data,
    merge_datasets,
    create_variable,
    handle_missing,
    encode_categorical,
    pivot_data,
    melt_data,
]


def get_transformation_tools() -> list:
    """Get list of all data transformation tools."""
    return DATA_TRANSFORMATION_TOOLS


__all__ = [
    "filter_data",
    "select_columns",
    "aggregate_data",
    "merge_datasets",
    "create_variable",
    "handle_missing",
    "encode_categorical",
    "pivot_data",
    "melt_data",
    "get_transformation_tools",
    "DATA_TRANSFORMATION_TOOLS",
]
