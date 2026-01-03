"""Data exploration tools for analyzing uploaded datasets.

These tools are used by the DATA_EXPLORER node to:
1. Parse and load data files (CSV, Excel, etc.)
2. Detect schema and data types
3. Generate summary statistics
4. Detect data quality issues
5. Assess overall data feasibility
"""

from pathlib import Path
from typing import Any

from langchain_core.tools import tool


try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.state.enums import ColumnType, DataQualityLevel
from src.state.models import (
    ColumnAnalysis,
    QualityIssue,
    DataExplorationResult,
    DataFile,
)


def get_column_type(dtype: Any) -> ColumnType:
    """Map pandas dtype to ColumnType enum."""
    dtype_str = str(dtype)
    
    if "int" in dtype_str:
        return ColumnType.INTEGER
    if "float" in dtype_str:
        return ColumnType.FLOAT
    if "bool" in dtype_str:
        return ColumnType.BOOLEAN
    if "datetime" in dtype_str or "date" in dtype_str:
        return ColumnType.DATETIME
    if "timedelta" in dtype_str:
        return ColumnType.TIMEDELTA
    if "category" in dtype_str:
        return ColumnType.CATEGORICAL
    if "object" in dtype_str:
        return ColumnType.STRING
    
    return ColumnType.UNKNOWN


@tool
def parse_csv_file(filepath: str) -> dict[str, Any]:
    """
    Parse a CSV file and return basic information about its structure.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        Dictionary with file info, columns, row count, and sample data.
    """
    if not HAS_PANDAS:
        return {"error": "pandas is not installed"}
    
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}
    
    if path.suffix.lower() != ".csv":
        return {"error": f"Not a CSV file: {filepath}"}
    
    try:
        # Read with pandas
        df = pd.read_csv(filepath, nrows=1000)  # Limit for preview
        full_df = pd.read_csv(filepath)  # Full count
        
        return {
            "filename": path.name,
            "filepath": str(path),
            "row_count": len(full_df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_rows": df.head(5).to_dict(orient="records"),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }
    except Exception as e:
        return {"error": f"Failed to parse CSV: {str(e)}"}


@tool
def parse_excel_file(filepath: str, sheet_name: str | None = None) -> dict[str, Any]:
    """
    Parse an Excel file and return basic information about its structure.
    
    Args:
        filepath: Path to the Excel file.
        sheet_name: Specific sheet to parse (optional, defaults to first sheet).
        
    Returns:
        Dictionary with file info, sheets, columns, row count, and sample data.
    """
    if not HAS_PANDAS:
        return {"error": "pandas is not installed"}
    
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}
    
    if path.suffix.lower() not in (".xlsx", ".xls"):
        return {"error": f"Not an Excel file: {filepath}"}
    
    try:
        # Get sheet names
        excel_file = pd.ExcelFile(filepath)
        sheets = excel_file.sheet_names
        
        # Parse specified sheet or first sheet
        target_sheet = sheet_name if sheet_name in sheets else sheets[0]
        df = pd.read_excel(filepath, sheet_name=target_sheet, nrows=1000)
        full_df = pd.read_excel(filepath, sheet_name=target_sheet)
        
        return {
            "filename": path.name,
            "filepath": str(path),
            "sheets": sheets,
            "active_sheet": target_sheet,
            "row_count": len(full_df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_rows": df.head(5).to_dict(orient="records"),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }
    except Exception as e:
        return {"error": f"Failed to parse Excel file: {str(e)}"}


@tool
def detect_schema(filepath: str) -> dict[str, Any]:
    """
    Detect the schema (column types and constraints) of a data file.
    
    Args:
        filepath: Path to the data file (CSV or Excel).
        
    Returns:
        Schema information including column types, nullable flags, and unique counts.
    """
    if not HAS_PANDAS:
        return {"error": "pandas is not installed"}
    
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}
    
    try:
        # Load based on file type
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(filepath)
        elif path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            return {"error": f"Unsupported file type: {path.suffix}"}
        
        schema = {}
        for col in df.columns:
            col_data = df[col]
            schema[col] = {
                "dtype": str(col_data.dtype),
                "column_type": get_column_type(col_data.dtype).value,
                "nullable": col_data.isna().any(),
                "null_count": int(col_data.isna().sum()),
                "null_percentage": round(col_data.isna().sum() / len(df) * 100, 2),
                "unique_count": int(col_data.nunique()),
                "unique_percentage": round(col_data.nunique() / len(df) * 100, 2),
            }
        
        return {
            "filepath": str(path),
            "row_count": len(df),
            "column_count": len(df.columns),
            "schema": schema,
        }
    except Exception as e:
        return {"error": f"Failed to detect schema: {str(e)}"}


@tool
def generate_summary_stats(filepath: str) -> dict[str, Any]:
    """
    Generate summary statistics for numerical columns in a data file.
    
    Args:
        filepath: Path to the data file.
        
    Returns:
        Summary statistics including mean, std, min, max, quartiles.
    """
    if not HAS_PANDAS:
        return {"error": "pandas is not installed"}
    
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}
    
    try:
        # Load based on file type
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(filepath)
        elif path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            return {"error": f"Unsupported file type: {path.suffix}"}
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        
        if not numeric_cols:
            return {
                "filepath": str(path),
                "message": "No numeric columns found",
                "stats": {},
            }
        
        # Generate statistics
        stats = {}
        for col in numeric_cols:
            col_stats = df[col].describe()
            stats[col] = {
                "count": int(col_stats["count"]),
                "mean": round(float(col_stats["mean"]), 4) if pd.notna(col_stats["mean"]) else None,
                "std": round(float(col_stats["std"]), 4) if pd.notna(col_stats["std"]) else None,
                "min": float(col_stats["min"]) if pd.notna(col_stats["min"]) else None,
                "q25": float(col_stats["25%"]) if pd.notna(col_stats["25%"]) else None,
                "median": float(col_stats["50%"]) if pd.notna(col_stats["50%"]) else None,
                "q75": float(col_stats["75%"]) if pd.notna(col_stats["75%"]) else None,
                "max": float(col_stats["max"]) if pd.notna(col_stats["max"]) else None,
            }
            
            # Add skewness and kurtosis
            skew = df[col].skew()
            kurtosis = df[col].kurtosis()
            stats[col]["skewness"] = round(float(skew), 4) if pd.notna(skew) else None
            stats[col]["kurtosis"] = round(float(kurtosis), 4) if pd.notna(kurtosis) else None
        
        return {
            "filepath": str(path),
            "numeric_columns": numeric_cols,
            "stats": stats,
        }
    except Exception as e:
        return {"error": f"Failed to generate statistics: {str(e)}"}


@tool
def detect_missing_values(filepath: str) -> dict[str, Any]:
    """
    Detect and analyze missing values in a data file.
    
    Args:
        filepath: Path to the data file.
        
    Returns:
        Missing value analysis including counts, patterns, and recommendations.
    """
    if not HAS_PANDAS:
        return {"error": "pandas is not installed"}
    
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}
    
    try:
        # Load based on file type
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(filepath)
        elif path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            return {"error": f"Unsupported file type: {path.suffix}"}
        
        total_cells = df.size
        total_missing = df.isna().sum().sum()
        
        # Per-column analysis
        column_missing = {}
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            if missing_count > 0:
                column_missing[col] = {
                    "missing_count": missing_count,
                    "missing_percentage": round(missing_count / len(df) * 100, 2),
                    "dtype": str(df[col].dtype),
                }
        
        # Rows with any missing
        rows_with_missing = int(df.isna().any(axis=1).sum())
        
        # Complete rows
        complete_rows = len(df) - rows_with_missing
        
        return {
            "filepath": str(path),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "total_cells": total_cells,
            "total_missing": int(total_missing),
            "overall_missing_percentage": round(total_missing / total_cells * 100, 2),
            "rows_with_missing": rows_with_missing,
            "complete_rows": complete_rows,
            "complete_row_percentage": round(complete_rows / len(df) * 100, 2),
            "columns_with_missing": column_missing,
        }
    except Exception as e:
        return {"error": f"Failed to detect missing values: {str(e)}"}


@tool
def detect_outliers(filepath: str, method: str = "iqr") -> dict[str, Any]:
    """
    Detect outliers in numerical columns using IQR or z-score method.
    
    Args:
        filepath: Path to the data file.
        method: Detection method - "iqr" (default) or "zscore".
        
    Returns:
        Outlier analysis including counts and indices for each numeric column.
    """
    if not HAS_PANDAS:
        return {"error": "pandas is not installed"}
    
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}
    
    try:
        # Load based on file type
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(filepath)
        elif path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            return {"error": f"Unsupported file type: {path.suffix}"}
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        
        if not numeric_cols:
            return {
                "filepath": str(path),
                "message": "No numeric columns found",
                "outliers": {},
            }
        
        outliers = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            
            if method == "iqr":
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                
            elif method == "zscore":
                mean = col_data.mean()
                std = col_data.std()
                z_scores = (col_data - mean) / std if std > 0 else pd.Series([0] * len(col_data))
                outlier_mask = abs(z_scores) > 3
                
            else:
                return {"error": f"Unknown method: {method}. Use 'iqr' or 'zscore'."}
            
            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                outliers[col] = {
                    "outlier_count": outlier_count,
                    "outlier_percentage": round(outlier_count / len(col_data) * 100, 2),
                    "method": method,
                }
                if method == "iqr":
                    outliers[col]["lower_bound"] = round(float(lower_bound), 4)
                    outliers[col]["upper_bound"] = round(float(upper_bound), 4)
        
        return {
            "filepath": str(path),
            "method": method,
            "numeric_columns_analyzed": len(numeric_cols),
            "columns_with_outliers": len(outliers),
            "outliers": outliers,
        }
    except Exception as e:
        return {"error": f"Failed to detect outliers: {str(e)}"}


@tool
def assess_data_quality(filepath: str) -> dict[str, Any]:
    """
    Perform comprehensive data quality assessment on a data file.
    
    Checks for:
    - Missing values
    - Duplicate rows
    - Constant columns
    - High cardinality
    - Potential ID columns
    - Data type inconsistencies
    
    Args:
        filepath: Path to the data file.
        
    Returns:
        Quality assessment with score and identified issues.
    """
    if not HAS_PANDAS:
        return {"error": "pandas is not installed"}
    
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}
    
    try:
        # Load based on file type
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(filepath)
        elif path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            return {"error": f"Unsupported file type: {path.suffix}"}
        
        issues = []
        quality_score = 100.0
        
        # Check 1: Missing values
        missing_pct = df.isna().sum().sum() / df.size * 100
        if missing_pct > 0:
            severity = "high" if missing_pct > 20 else "medium" if missing_pct > 5 else "low"
            issues.append({
                "type": "missing_values",
                "severity": severity,
                "description": f"{missing_pct:.1f}% of data is missing",
                "affected_columns": [
                    col for col in df.columns if df[col].isna().any()
                ][:10],  # Limit to 10
            })
            quality_score -= min(missing_pct, 30)
        
        # Check 2: Duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            dup_pct = duplicate_count / len(df) * 100
            severity = "high" if dup_pct > 10 else "medium" if dup_pct > 1 else "low"
            issues.append({
                "type": "duplicate_rows",
                "severity": severity,
                "description": f"{duplicate_count} duplicate rows ({dup_pct:.1f}%)",
            })
            quality_score -= min(dup_pct * 2, 20)
        
        # Check 3: Constant columns (zero variance)
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() == 1:
                constant_cols.append(col)
        if constant_cols:
            issues.append({
                "type": "constant_columns",
                "severity": "medium",
                "description": f"{len(constant_cols)} column(s) with constant values",
                "affected_columns": constant_cols,
            })
            quality_score -= len(constant_cols) * 2
        
        # Check 4: High cardinality string columns
        high_cardinality = []
        for col in df.select_dtypes(include=["object"]).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9 and df[col].nunique() > 100:
                high_cardinality.append(col)
        if high_cardinality:
            issues.append({
                "type": "high_cardinality",
                "severity": "low",
                "description": f"{len(high_cardinality)} column(s) with very high cardinality (possible ID columns)",
                "affected_columns": high_cardinality,
            })
        
        # Check 5: Sample size adequacy
        if len(df) < 30:
            issues.append({
                "type": "small_sample",
                "severity": "high",
                "description": f"Only {len(df)} rows; may be insufficient for statistical analysis",
            })
            quality_score -= 20
        elif len(df) < 100:
            issues.append({
                "type": "small_sample",
                "severity": "medium",
                "description": f"Only {len(df)} rows; consider if sample size is adequate",
            })
            quality_score -= 10
        
        # Determine overall quality level
        if quality_score >= 90:
            quality_level = DataQualityLevel.EXCELLENT.value
        elif quality_score >= 75:
            quality_level = DataQualityLevel.GOOD.value
        elif quality_score >= 60:
            quality_level = DataQualityLevel.FAIR.value
        elif quality_score >= 40:
            quality_level = DataQualityLevel.POOR.value
        else:
            quality_level = DataQualityLevel.UNUSABLE.value
        
        return {
            "filepath": str(path),
            "row_count": len(df),
            "column_count": len(df.columns),
            "quality_score": round(max(0, quality_score), 1),
            "quality_level": quality_level,
            "issues_count": len(issues),
            "issues": issues,
            "recommendations": generate_recommendations(issues),
        }
    except Exception as e:
        return {"error": f"Failed to assess data quality: {str(e)}"}


def generate_recommendations(issues: list[dict]) -> list[str]:
    """Generate recommendations based on identified issues."""
    recommendations = []
    
    for issue in issues:
        issue_type = issue["type"]
        severity = issue["severity"]
        
        if issue_type == "missing_values":
            if severity == "high":
                recommendations.append(
                    "Consider imputation strategies or removing columns with >50% missing data"
                )
            else:
                recommendations.append(
                    "Review missing value patterns; consider mean/median imputation or listwise deletion"
                )
                
        elif issue_type == "duplicate_rows":
            recommendations.append(
                "Review and remove duplicate rows before analysis"
            )
            
        elif issue_type == "constant_columns":
            recommendations.append(
                "Remove constant columns as they provide no analytical value"
            )
            
        elif issue_type == "high_cardinality":
            recommendations.append(
                "High cardinality columns may be identifiers; exclude from analysis or use for grouping only"
            )
            
        elif issue_type == "small_sample":
            if severity == "high":
                recommendations.append(
                    "Sample size is very small; consider collecting more data or using appropriate small-sample methods"
                )
            else:
                recommendations.append(
                    "Verify sample size is adequate for planned statistical tests"
                )
    
    return recommendations


def _analyze_zip_file(data_file: DataFile, filepath: str) -> DataExplorationResult:
    """
    Extract and analyze contents of a ZIP file.
    
    Extracts to a temporary directory, finds all analyzable data files,
    analyzes each, and aggregates the results.
    
    Args:
        data_file: Original DataFile object for the ZIP.
        filepath: Path to the ZIP file.
        
    Returns:
        Aggregated DataExplorationResult from all contained files.
    """
    import zipfile
    import tempfile
    import shutil
    from src.state.enums import CritiqueSeverity
    
    # Security limits for ZIP extraction
    MAX_ZIP_SIZE = 500 * 1024 * 1024  # 500 MB max extracted size
    MAX_ZIP_FILES = 100  # Max files in a ZIP
    
    # Supported data file extensions
    DATA_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet", ".json"}
    
    if not zipfile.is_zipfile(filepath):
        return DataExplorationResult(
            files_analyzed=[data_file],
            total_rows=0,
            total_columns=0,
            columns=[],
            quality_score=0.0,
            quality_level=DataQualityLevel.NOT_ASSESSED,
            quality_issues=[QualityIssue(
                severity=CritiqueSeverity.CRITICAL,
                category="invalid_format",
                description="File is not a valid ZIP archive",
            )],
            feasibility_assessment="Invalid ZIP file",
        )
    
    # Create temp directory for extraction
    temp_dir = tempfile.mkdtemp(prefix="gia_data_")
    extracted_path = Path(temp_dir)
    
    try:
        # Safely extract ZIP contents with protection against zip bombs and path traversal
        total_size = 0
        with zipfile.ZipFile(filepath, 'r') as zf:
            # Check number of files
            if len(zf.namelist()) > MAX_ZIP_FILES:
                return DataExplorationResult(
                    files_analyzed=[data_file],
                    total_rows=0,
                    total_columns=0,
                    columns=[],
                    quality_score=0.0,
                    quality_level=DataQualityLevel.NOT_ASSESSED,
                    quality_issues=[QualityIssue(
                        severity=CritiqueSeverity.CRITICAL,
                        category="security",
                        description=f"ZIP contains too many files (max {MAX_ZIP_FILES})",
                    )],
                    feasibility_assessment="ZIP file contains too many files",
                )
            
            for info in zf.infolist():
                # Check for path traversal
                target_path = extracted_path / info.filename
                try:
                    target_path.resolve().relative_to(extracted_path.resolve())
                except ValueError:
                    return DataExplorationResult(
                        files_analyzed=[data_file],
                        total_rows=0,
                        total_columns=0,
                        columns=[],
                        quality_score=0.0,
                        quality_level=DataQualityLevel.NOT_ASSESSED,
                        quality_issues=[QualityIssue(
                            severity=CritiqueSeverity.CRITICAL,
                            category="security",
                            description="ZIP contains path traversal attempt",
                        )],
                        feasibility_assessment="Malicious ZIP file detected",
                    )
                
                # Check for zip bomb (cumulative size)
                total_size += info.file_size
                if total_size > MAX_ZIP_SIZE:
                    return DataExplorationResult(
                        files_analyzed=[data_file],
                        total_rows=0,
                        total_columns=0,
                        columns=[],
                        quality_score=0.0,
                        quality_level=DataQualityLevel.NOT_ASSESSED,
                        quality_issues=[QualityIssue(
                            severity=CritiqueSeverity.CRITICAL,
                            category="security",
                            description=f"ZIP extraction would exceed size limit ({MAX_ZIP_SIZE / 1024 / 1024:.0f} MB)",
                        )],
                        feasibility_assessment="ZIP file is too large to extract safely",
                    )
                
                # Extract single file
                zf.extract(info, temp_dir)
        
        # Find all data files in the extracted contents
        data_files_found: list[Path] = []
        
        for ext in DATA_EXTENSIONS:
            data_files_found.extend(extracted_path.rglob(f"*{ext}"))
        
        if not data_files_found:
            return DataExplorationResult(
                files_analyzed=[data_file],
                total_rows=0,
                total_columns=0,
                columns=[],
                quality_score=0.0,
                quality_level=DataQualityLevel.NOT_ASSESSED,
                quality_issues=[QualityIssue(
                    severity=CritiqueSeverity.MAJOR,
                    category="no_data",
                    description=f"ZIP contains no analyzable data files. Supported formats: {', '.join(DATA_EXTENSIONS)}",
                )],
                feasibility_assessment=f"No data files found in ZIP. Archive may contain other file types.",
            )
        
        # Analyze each file and aggregate results
        all_results: list[DataExplorationResult] = []
        all_files_analyzed: list[DataFile] = [data_file]  # Include the original ZIP
        all_columns: list[ColumnAnalysis] = []
        all_quality_issues: list[QualityIssue] = []
        total_rows = 0
        total_columns = 0
        quality_scores: list[float] = []
        
        for file_path in data_files_found:
            # Create DataFile for the extracted file
            content_type_map = {
                ".csv": "text/csv",
                ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".xls": "application/vnd.ms-excel",
                ".parquet": "application/octet-stream",
                ".json": "application/json",
            }
            
            extracted_file = DataFile(
                filename=file_path.name,
                filepath=file_path,
                content_type=content_type_map.get(file_path.suffix.lower(), "application/octet-stream"),
                size_bytes=file_path.stat().st_size,
            )
            
            # Recursively analyze (but this won't hit ZIP handling again for normal files)
            result = analyze_file(extracted_file)
            
            if result.total_rows > 0:
                all_results.append(result)
                all_files_analyzed.extend(result.files_analyzed)
                all_columns.extend(result.columns)
                all_quality_issues.extend(result.quality_issues)
                total_rows += result.total_rows
                total_columns += result.total_columns
                if result.quality_score > 0:
                    quality_scores.append(result.quality_score)
        
        # Aggregate quality score (weighted average by row count)
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
        else:
            avg_quality = 0.0
        
        # Determine overall quality level
        if avg_quality >= 0.8:
            quality_level = DataQualityLevel.EXCELLENT
        elif avg_quality >= 0.6:
            quality_level = DataQualityLevel.GOOD
        elif avg_quality >= 0.4:
            quality_level = DataQualityLevel.ACCEPTABLE
        elif avg_quality > 0:
            quality_level = DataQualityLevel.POOR
        else:
            quality_level = DataQualityLevel.NOT_ASSESSED
        
        # Build feasibility assessment
        file_count = len(data_files_found)
        analyzed_count = len(all_results)
        
        if analyzed_count == 0:
            feasibility = "No files could be analyzed from the ZIP archive"
        elif avg_quality >= 0.4:
            feasibility = (
                f"Successfully analyzed {analyzed_count} of {file_count} data files. "
                f"Total: {total_rows:,} rows across {total_columns} columns. "
                f"Data appears suitable for analysis."
            )
        else:
            feasibility = (
                f"Analyzed {analyzed_count} files but overall quality is low ({avg_quality:.0%}). "
                f"Review quality issues before proceeding."
            )
        
        return DataExplorationResult(
            files_analyzed=all_files_analyzed,
            total_rows=total_rows,
            total_columns=total_columns,
            columns=all_columns,
            quality_score=avg_quality,
            quality_level=quality_level,
            quality_issues=all_quality_issues,
            feasibility_assessment=feasibility,
        )
        
    except zipfile.BadZipFile as e:
        return DataExplorationResult(
            files_analyzed=[data_file],
            total_rows=0,
            total_columns=0,
            columns=[],
            quality_score=0.0,
            quality_level=DataQualityLevel.NOT_ASSESSED,
            quality_issues=[QualityIssue(
                severity=CritiqueSeverity.CRITICAL,
                category="corrupt_archive",
                description=f"ZIP file is corrupted: {str(e)}",
            )],
            feasibility_assessment="Cannot extract ZIP file; archive may be corrupted",
        )
    except Exception as e:
        return DataExplorationResult(
            files_analyzed=[data_file],
            total_rows=0,
            total_columns=0,
            columns=[],
            quality_score=0.0,
            quality_level=DataQualityLevel.NOT_ASSESSED,
            quality_issues=[QualityIssue(
                severity=CritiqueSeverity.CRITICAL,
                category="extraction_error",
                description=f"Failed to process ZIP: {str(e)}",
            )],
            feasibility_assessment=f"Error processing ZIP archive: {str(e)}",
        )
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def analyze_file(data_file: DataFile) -> DataExplorationResult:
    """
    Perform complete analysis on a data file.
    
    This is the main function used by the DATA_EXPLORER node.
    Supports CSV, Excel, and ZIP files (extracts and analyzes contents).
    
    Args:
        data_file: DataFile object with file path and metadata.
        
    Returns:
        DataExplorationResult with complete analysis.
    """
    from src.state.enums import CritiqueSeverity
    import zipfile
    import tempfile
    import shutil
    
    if not HAS_PANDAS:
        return DataExplorationResult(
            files_analyzed=[data_file],
            total_rows=0,
            total_columns=0,
            columns=[],
            quality_score=0.0,
            quality_level=DataQualityLevel.NOT_ASSESSED,
            quality_issues=[],
            feasibility_assessment="Cannot analyze file: pandas not installed",
        )
    
    filepath = str(data_file.filepath)
    
    try:
        # Handle ZIP files by extracting and analyzing contents
        if data_file.content_type == "application/zip" or filepath.lower().endswith(".zip"):
            return _analyze_zip_file(data_file, filepath)
        
        # Load data
        if data_file.content_type == "text/csv" or filepath.lower().endswith(".csv"):
            df = pd.read_csv(filepath)
        elif data_file.content_type in (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel"
        ) or filepath.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath)
        elif filepath.lower().endswith(".parquet"):
            df = pd.read_parquet(filepath)
        elif filepath.lower().endswith(".json"):
            df = pd.read_json(filepath)
        else:
            return DataExplorationResult(
                files_analyzed=[data_file],
                total_rows=0,
                total_columns=0,
                columns=[],
                quality_score=0.0,
                quality_level=DataQualityLevel.NOT_ASSESSED,
                quality_issues=[],
                feasibility_assessment=f"Cannot analyze file type: {data_file.content_type}",
            )
        
        # Analyze columns
        columns = []
        for col in df.columns:
            col_data = df[col]
            non_null = int(len(col_data) - col_data.isna().sum())
            col_analysis = ColumnAnalysis(
                name=str(col),
                dtype=get_column_type(col_data.dtype),
                non_null_count=non_null,
                null_count=int(col_data.isna().sum()),
                null_percentage=round(col_data.isna().sum() / len(df) * 100, 2),
                unique_count=int(col_data.nunique()),
            )
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                stats = col_data.describe()
                col_analysis.min_value = float(stats["min"]) if pd.notna(stats["min"]) else None
                col_analysis.max_value = float(stats["max"]) if pd.notna(stats["max"]) else None
                col_analysis.mean = float(stats["mean"]) if pd.notna(stats["mean"]) else None
                col_analysis.std = float(stats["std"]) if pd.notna(stats["std"]) else None
            
            columns.append(col_analysis)
        
        # Quality assessment
        quality_result = assess_data_quality.invoke({"filepath": filepath})
        
        quality_issues = []
        if "issues" in quality_result:
            for issue_dict in quality_result["issues"]:
                # Map severity string to enum
                severity_map = {
                    "critical": CritiqueSeverity.CRITICAL,
                    "major": CritiqueSeverity.MAJOR,
                    "minor": CritiqueSeverity.MINOR,
                    "high": CritiqueSeverity.CRITICAL,
                    "medium": CritiqueSeverity.MAJOR,
                    "low": CritiqueSeverity.MINOR,
                }
                severity = severity_map.get(
                    issue_dict.get("severity", "minor").lower(), 
                    CritiqueSeverity.MINOR
                )
                quality_issues.append(QualityIssue(
                    severity=severity,
                    category=issue_dict.get("type", "unknown"),
                    description=issue_dict["description"],
                    column=issue_dict.get("column"),
                ))
        
        # Determine quality score (convert 0-100 to 0-1)
        raw_score = quality_result.get("quality_score", 50)
        quality_score = raw_score / 100.0 if raw_score > 1 else raw_score
        
        # Determine feasibility
        is_feasible = quality_score >= 0.4 and len(df) >= 10
        
        if not is_feasible:
            if quality_score < 0.4:
                feasibility_assessment = "Data quality is too low for reliable analysis"
            elif len(df) < 10:
                feasibility_assessment = "Insufficient sample size"
            else:
                feasibility_assessment = "Data does not meet minimum requirements"
        else:
            feasibility_assessment = "Data appears suitable for analysis"
        
        # Map quality level
        quality_level_str = quality_result.get("quality_level", "not_assessed")
        try:
            quality_level = DataQualityLevel(quality_level_str)
        except ValueError:
            quality_level = DataQualityLevel.NOT_ASSESSED
        
        return DataExplorationResult(
            files_analyzed=[data_file],
            total_rows=len(df),
            total_columns=len(df.columns),
            columns=columns,
            quality_score=quality_score,
            quality_level=quality_level,
            quality_issues=quality_issues,
            feasibility_assessment=feasibility_assessment,
        )
        
    except Exception as e:
        return DataExplorationResult(
            files_analyzed=[data_file],
            total_rows=0,
            total_columns=0,
            columns=[],
            quality_score=0.0,
            quality_level=DataQualityLevel.NOT_ASSESSED,
            quality_issues=[QualityIssue(
                severity=CritiqueSeverity.CRITICAL,
                category="parse_error",
                description=f"Failed to parse file: {str(e)}",
            )],
            feasibility_assessment=f"Error analyzing file: {str(e)}",
        )


# Export all tools for use in agents
DATA_EXPLORATION_TOOLS = [
    parse_csv_file,
    parse_excel_file,
    detect_schema,
    generate_summary_stats,
    detect_missing_values,
    detect_outliers,
    assess_data_quality,
]
