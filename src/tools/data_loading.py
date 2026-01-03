"""Data loading tools for various file formats.

Sprint 12 Enhanced: Now includes automatic encoding detection and delimiter sniffing
for robust CSV handling.

Provides unified data loading from:
- CSV, Excel, Parquet, JSON (via pandas)
- SQLite, DuckDB databases
- Stata (.dta), SPSS (.sav) files
- ZIP archives (extracts and loads contents)

Uses DuckDB as the unified backend for large file handling
and cross-format querying capabilities.
"""

from pathlib import Path
from typing import Any
import tempfile
import os

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Optional imports with fallbacks
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    import pyreadstat
    HAS_PYREADSTAT = True
except ImportError:
    HAS_PYREADSTAT = False

# Charset detection for encoding
try:
    import charset_normalizer
    HAS_CHARSET = True
except ImportError:
    HAS_CHARSET = False


# =============================================================================
# Data Registry for tracking loaded datasets
# =============================================================================

class DataRegistry:
    """
    Registry for tracking loaded datasets across tool calls.
    
    Uses DuckDB as the unified backend for storage and querying.
    Datasets are registered by name and can be queried via SQL.
    
    Implements a proper singleton pattern to ensure data persists
    across all tool calls within the same process.
    """
    
    _instance: "DataRegistry | None" = None
    
    def __new__(cls) -> "DataRegistry":
        if cls._instance is None:
            instance = super().__new__(cls)
            # Initialize instance attributes (not class attributes)
            instance._db_path: str | None = None
            instance._conn: Any = None
            instance._datasets: dict[str, dict[str, Any]] = {}
            cls._instance = instance
        return cls._instance
    
    def _get_connection(self) -> Any:
        """Get or create DuckDB connection."""
        if not HAS_DUCKDB:
            raise ImportError("duckdb is required for data registry")
        
        if self._conn is None:
            # Use in-memory database for session
            self._conn = duckdb.connect(":memory:")
        return self._conn
    
    def register(
        self, 
        name: str, 
        df: "pd.DataFrame",
        source_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Register a DataFrame in the registry.
        
        Args:
            name: Unique name for the dataset
            df: Pandas DataFrame to register
            source_path: Original file path
            metadata: Additional metadata
            
        Returns:
            Registration info including row/column counts
        """
        conn = self._get_connection()
        
        # Register DataFrame as a DuckDB table
        conn.register(name, df)
        
        # Store metadata
        self._datasets[name] = {
            "name": name,
            "source_path": source_path,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "metadata": metadata or {},
        }
        
        return self._datasets[name]
    
    def get(self, name: str) -> "pd.DataFrame | None":
        """Get a registered DataFrame by name."""
        if name not in self._datasets:
            return None
        
        conn = self._get_connection()
        return conn.execute(f'SELECT * FROM "{name}"').fetchdf()
    
    def get_dataframe(self, name: str) -> "pd.DataFrame | None":
        """Alias for get() - returns registered DataFrame by name."""
        return self.get(name)
    
    def query(self, sql: str) -> "pd.DataFrame":
        """Execute SQL query against registered datasets."""
        conn = self._get_connection()
        return conn.execute(sql).fetchdf()
    
    def list_datasets(self) -> list[dict[str, Any]]:
        """List all registered datasets."""
        return list(self._datasets.values())
    
    def get_info(self, name: str) -> dict[str, Any] | None:
        """Get metadata for a registered dataset."""
        return self._datasets.get(name)
    
    @property
    def datasets(self) -> dict[str, dict[str, Any]]:
        """Public accessor for registered datasets."""
        return self._datasets
    
    def clear(self) -> None:
        """Clear all registered datasets."""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._datasets.clear()


# Global registry instance
_registry = DataRegistry()


def get_registry() -> DataRegistry:
    """Get the global data registry."""
    return _registry


# =============================================================================
# File Loading Tools
# =============================================================================


class LoadDataInput(BaseModel):
    """Input for load_data tool."""
    filepath: str = Field(description="Path to the data file")
    name: str | None = Field(default=None, description="Name for the dataset in registry")
    options: dict[str, Any] | None = Field(default=None, description="Format-specific options")


@tool(args_schema=LoadDataInput)
def load_data(filepath: str, name: str | None = None, options: dict | None = None) -> dict[str, Any]:
    """
    Load data from any supported format and register in the data registry.
    
    Automatically detects format from file extension and loads appropriately.
    For large files (>100MB), uses DuckDB for out-of-core processing.
    
    Supported formats:
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - Parquet (.parquet)
    - JSON (.json)
    - SQLite (.db, .sqlite, .sqlite3)
    - DuckDB (.duckdb)
    - Stata (.dta)
    - SPSS (.sav)
    - ZIP (.zip) - extracts and loads data files inside
    
    Args:
        filepath: Path to the data file
        name: Optional name for the dataset (defaults to filename)
        options: Format-specific loading options
        
    Returns:
        Dictionary with dataset info and registration status
    """
    if not HAS_PANDAS:
        return {"error": "pandas is required for data loading"}
    
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}
    
    # Default name from filename
    if name is None:
        name = path.stem.replace(" ", "_").replace("-", "_")
    
    options = options or {}
    suffix = path.suffix.lower()
    
    try:
        # Route to appropriate loader based on extension
        if suffix == ".csv":
            df = _load_csv(filepath, options)
        elif suffix in (".xlsx", ".xls"):
            df = _load_excel(filepath, options)
        elif suffix == ".parquet":
            df = _load_parquet(filepath, options)
        elif suffix == ".json":
            df = _load_json(filepath, options)
        elif suffix in (".db", ".sqlite", ".sqlite3"):
            df = _load_sqlite(filepath, options)
        elif suffix == ".duckdb":
            df = _load_duckdb_file(filepath, options)
        elif suffix == ".dta":
            df = _load_stata(filepath, options)
        elif suffix == ".sav":
            df = _load_spss(filepath, options)
        elif suffix == ".zip":
            return _load_zip(filepath, name, options)
        else:
            return {"error": f"Unsupported file format: {suffix}"}
        
        # Register in data registry
        registry = get_registry()
        info = registry.register(name, df, source_path=filepath)
        
        return {
            "status": "success",
            "dataset_name": name,
            **info,
            "sample_rows": df.head(5).to_dict(orient="records"),
        }
        
    except Exception as e:
        return {"error": f"Failed to load {filepath}: {str(e)}"}


def _load_csv(filepath: str, options: dict) -> "pd.DataFrame":
    """
    Load CSV file with automatic encoding detection and delimiter sniffing.
    
    Sprint 12 Enhancement: Automatically detects:
    - File encoding (UTF-8, Latin-1, Windows-1252, etc.)
    - Delimiter (comma, semicolon, tab, pipe)
    
    Falls back to DuckDB for very large files (>100MB).
    """
    import csv
    
    # Check file size for large file handling
    size_mb = Path(filepath).stat().st_size / 1024 / 1024
    
    if size_mb > 100 and HAS_DUCKDB:
        # Use DuckDB for large files (it auto-detects encoding/delimiter)
        conn = duckdb.connect(":memory:")
        return conn.execute(f"SELECT * FROM read_csv_auto('{filepath}')").fetchdf()
    
    # Auto-detect encoding if not specified
    if "encoding" not in options:
        detected_encoding = _detect_encoding(filepath)
        if detected_encoding:
            options["encoding"] = detected_encoding
    
    # Auto-detect delimiter if not specified
    if "sep" not in options and "delimiter" not in options:
        detected_delimiter = _detect_delimiter(filepath, options.get("encoding", "utf-8"))
        if detected_delimiter:
            options["sep"] = detected_delimiter
    
    try:
        return pd.read_csv(filepath, **options)
    except UnicodeDecodeError:
        # Fallback to latin-1 if encoding detection failed
        options["encoding"] = "latin-1"
        return pd.read_csv(filepath, **options)


def _detect_encoding(filepath: str) -> str | None:
    """
    Detect file encoding using charset_normalizer or fallback heuristics.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Detected encoding name or None
    """
    if HAS_CHARSET:
        try:
            with open(filepath, "rb") as f:
                # Read first 100KB for detection
                sample = f.read(100 * 1024)
            
            result = charset_normalizer.from_bytes(sample).best()
            if result:
                return result.encoding
        except Exception:
            pass
    
    # Fallback: Try common encodings
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    
    for encoding in encodings_to_try:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                # Try to read first few KB
                f.read(10 * 1024)
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    return "utf-8"  # Default fallback


def _detect_delimiter(filepath: str, encoding: str = "utf-8") -> str | None:
    """
    Detect CSV delimiter using csv.Sniffer or heuristics.
    
    Args:
        filepath: Path to the CSV file
        encoding: File encoding to use
        
    Returns:
        Detected delimiter or None
    """
    import csv
    
    try:
        with open(filepath, "r", encoding=encoding, errors="replace") as f:
            # Read first 64KB for detection
            sample = f.read(64 * 1024)
        
        # Use csv.Sniffer to detect dialect
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except csv.Error:
            pass
        
        # Fallback: Count delimiters in first line
        first_line = sample.split("\n")[0] if "\n" in sample else sample
        
        delimiters = {
            ",": first_line.count(","),
            ";": first_line.count(";"),
            "\t": first_line.count("\t"),
            "|": first_line.count("|"),
        }
        
        # Return delimiter with most occurrences (if > 0)
        max_count = max(delimiters.values())
        if max_count > 0:
            return max(delimiters, key=delimiters.get)
        
    except Exception:
        pass
    
    return None  # Let pandas use default


def _load_excel(filepath: str, options: dict) -> "pd.DataFrame":
    """Load Excel file."""
    sheet_name = options.pop("sheet_name", 0)
    return pd.read_excel(filepath, sheet_name=sheet_name, **options)


def _load_parquet(filepath: str, options: dict) -> "pd.DataFrame":
    """Load Parquet file (including partitioned datasets)."""
    path = Path(filepath)
    
    # Check if it's a directory (partitioned dataset)
    if path.is_dir():
        if HAS_DUCKDB:
            conn = duckdb.connect(":memory:")
            return conn.execute(f"SELECT * FROM parquet_scan('{filepath}/**/*.parquet')").fetchdf()
        else:
            # Fallback: load all parquet files in directory
            dfs = []
            for pq_file in path.rglob("*.parquet"):
                dfs.append(pd.read_parquet(pq_file, **options))
            return pd.concat(dfs, ignore_index=True)
    
    return pd.read_parquet(filepath, **options)


def _load_json(filepath: str, options: dict) -> "pd.DataFrame":
    """Load JSON file."""
    return pd.read_json(filepath, **options)


def _load_sqlite(filepath: str, options: dict) -> "pd.DataFrame":
    """Load SQLite database."""
    import sqlite3
    
    table = options.get("table")
    query = options.get("query")
    
    conn = sqlite3.connect(filepath)
    
    if query:
        df = pd.read_sql_query(query, conn)
    elif table:
        df = pd.read_sql_table(table, conn)
    else:
        # Get first table if not specified
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )
        if tables.empty:
            raise ValueError("No tables found in SQLite database")
        table = tables.iloc[0]["name"]
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    
    conn.close()
    return df


def _load_duckdb_file(filepath: str, options: dict) -> "pd.DataFrame":
    """Load DuckDB database file."""
    if not HAS_DUCKDB:
        raise ImportError("duckdb is required for .duckdb files")
    
    table = options.get("table")
    query = options.get("query")
    
    conn = duckdb.connect(filepath, read_only=True)
    
    if query:
        df = conn.execute(query).fetchdf()
    elif table:
        df = conn.execute(f'SELECT * FROM "{table}"').fetchdf()
    else:
        # Get first table
        tables = conn.execute("SHOW TABLES").fetchdf()
        if tables.empty:
            raise ValueError("No tables found in DuckDB database")
        table = tables.iloc[0]["name"]
        df = conn.execute(f'SELECT * FROM "{table}"').fetchdf()
    
    conn.close()
    return df


def _load_stata(filepath: str, options: dict) -> "pd.DataFrame":
    """Load Stata .dta file."""
    if not HAS_PYREADSTAT:
        raise ImportError("pyreadstat is required for Stata files")
    
    df, meta = pyreadstat.read_dta(filepath)
    return df


def _load_spss(filepath: str, options: dict) -> "pd.DataFrame":
    """Load SPSS .sav file."""
    if not HAS_PYREADSTAT:
        raise ImportError("pyreadstat is required for SPSS files")
    
    df, meta = pyreadstat.read_sav(filepath)
    return df


# Security limits for ZIP extraction
MAX_ZIP_SIZE = 500 * 1024 * 1024  # 500 MB max extracted size
MAX_ZIP_FILES = 100  # Max files in a ZIP


def _safe_extract_zip(zip_path: str, extract_dir: Path) -> list[Path]:
    """
    Safely extract ZIP file with protection against zip bombs and path traversal.
    
    Args:
        zip_path: Path to the ZIP file
        extract_dir: Directory to extract to
        
    Returns:
        List of extracted file paths
        
    Raises:
        ValueError: If ZIP is malicious (zip bomb, path traversal, etc.)
    """
    import zipfile
    
    extracted_files = []
    total_size = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Check number of files
        if len(zf.namelist()) > MAX_ZIP_FILES:
            raise ValueError(f"ZIP contains too many files (max {MAX_ZIP_FILES})")
        
        for info in zf.infolist():
            # Check for path traversal
            target_path = extract_dir / info.filename
            try:
                target_path.resolve().relative_to(extract_dir.resolve())
            except ValueError:
                raise ValueError(f"ZIP contains path traversal attempt: {info.filename}")
            
            # Check for zip bomb (cumulative size)
            total_size += info.file_size
            if total_size > MAX_ZIP_SIZE:
                raise ValueError(f"ZIP extraction would exceed size limit ({MAX_ZIP_SIZE / 1024 / 1024:.0f} MB)")
            
            # Extract single file
            zf.extract(info, extract_dir)
            if not info.is_dir():
                extracted_files.append(target_path)
    
    return extracted_files


def _load_zip(filepath: str, base_name: str, options: dict) -> dict[str, Any]:
    """Extract and load data files from ZIP archive."""
    import shutil
    
    DATA_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".dta", ".sav"}
    
    temp_dir = Path(tempfile.mkdtemp(prefix="gia_load_"))
    
    try:
        # Use safe extraction with zip bomb protection
        try:
            _safe_extract_zip(filepath, temp_dir)
        except ValueError as e:
            return {"error": f"ZIP extraction failed: {str(e)}"}
        
        # Find all data files
        data_files = []
        for ext in DATA_EXTENSIONS:
            data_files.extend(temp_dir.rglob(f"*{ext}"))
        
        if not data_files:
            return {"error": f"No data files found in ZIP. Supported: {DATA_EXTENSIONS}"}
        
        # Load each file
        loaded = []
        for i, data_path in enumerate(data_files):
            dataset_name = f"{base_name}_{data_path.stem}" if len(data_files) > 1 else base_name
            result = load_data.invoke({
                "filepath": str(data_path),
                "name": dataset_name,
                "options": options,
            })
            if "error" not in result:
                loaded.append(result)
        
        return {
            "status": "success",
            "files_loaded": len(loaded),
            "datasets": loaded,
        }
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Query Tools
# =============================================================================


class QueryDataInput(BaseModel):
    """Input for query_data tool."""
    sql: str = Field(description="SQL query to execute against registered datasets")


@tool(args_schema=QueryDataInput)
def query_data(sql: str) -> dict[str, Any]:
    """
    Execute SQL query against registered datasets.
    
    Use standard SQL syntax. Table names are the dataset names
    used when loading data.
    
    Example:
        query_data("SELECT ticker, AVG(price) FROM prices GROUP BY ticker")
    
    Args:
        sql: SQL query string
        
    Returns:
        Query results as dictionary with rows and metadata
    """
    registry = get_registry()
    
    try:
        df = registry.query(sql)
        
        return {
            "status": "success",
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records")[:100],  # Limit for response size
            "truncated": len(df) > 100,
        }
    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}


class ListDatasetsInput(BaseModel):
    """Input for list_datasets tool."""
    pass


@tool
def list_datasets() -> dict[str, Any]:
    """
    List all datasets currently loaded in the data registry.
    
    Returns:
        Dictionary with list of dataset names and their metadata
    """
    registry = get_registry()
    datasets = registry.list_datasets()
    
    return {
        "status": "success",
        "dataset_count": len(datasets),
        "datasets": datasets,
    }


class GetDatasetInfoInput(BaseModel):
    """Input for get_dataset_info tool."""
    name: str = Field(description="Name of the dataset to get info for")


@tool(args_schema=GetDatasetInfoInput)
def get_dataset_info(name: str) -> dict[str, Any]:
    """
    Get detailed information about a registered dataset.
    
    Args:
        name: Name of the dataset
        
    Returns:
        Dataset metadata including columns, dtypes, row count
    """
    registry = get_registry()
    info = registry.get_info(name)
    
    if info is None:
        return {"error": f"Dataset '{name}' not found"}
    
    return {
        "status": "success",
        **info,
    }


class SampleDataInput(BaseModel):
    """Input for sample_data tool."""
    name: str = Field(description="Name of the dataset to sample")
    n: int = Field(default=1000, description="Number of rows to sample")
    strategy: str = Field(default="random", description="Sampling strategy: random, head, tail, stratified")
    stratify_column: str | None = Field(default=None, description="Column for stratified sampling")


@tool(args_schema=SampleDataInput)
def sample_data(
    name: str, 
    n: int = 1000, 
    strategy: str = "random",
    stratify_column: str | None = None,
) -> dict[str, Any]:
    """
    Get a sample from a registered dataset.
    
    Useful for previewing large datasets without loading everything.
    
    Args:
        name: Dataset name
        n: Number of rows to sample
        strategy: Sampling strategy (random, head, tail, stratified)
        stratify_column: Column to use for stratified sampling
        
    Returns:
        Sample rows and sampling metadata
    """
    registry = get_registry()
    df = registry.get(name)
    
    if df is None:
        return {"error": f"Dataset '{name}' not found"}
    
    n = min(n, len(df))
    
    if strategy == "head":
        sample = df.head(n)
    elif strategy == "tail":
        sample = df.tail(n)
    elif strategy == "stratified" and stratify_column:
        if stratify_column not in df.columns:
            return {"error": f"Column '{stratify_column}' not found"}
        # Stratified sample
        sample = df.groupby(stratify_column, group_keys=False).apply(
            lambda x: x.sample(min(len(x), n // df[stratify_column].nunique()))
        )
    else:  # random
        sample = df.sample(n=n)
    
    return {
        "status": "success",
        "sample_size": len(sample),
        "total_rows": len(df),
        "strategy": strategy,
        "rows": sample.to_dict(orient="records"),
    }


# =============================================================================
# Exports
# =============================================================================

DATA_LOADING_TOOLS = [
    load_data,
    query_data,
    list_datasets,
    get_dataset_info,
    sample_data,
]


def get_data_loading_tools() -> list:
    """Get list of all data loading tools."""
    return DATA_LOADING_TOOLS


__all__ = [
    "load_data",
    "query_data",
    "list_datasets",
    "get_dataset_info",
    "sample_data",
    "get_registry",
    "get_data_loading_tools",
    "DataRegistry",
    "DATA_LOADING_TOOLS",
]
