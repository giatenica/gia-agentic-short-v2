# Sprint 12: Enhanced Data Explorer

## Overview

Sprint 12 enhances the DATA_EXPLORER node with:
- Deep statistical profiling with semantic type inference
- Data structure detection (panel, time series, cross-sectional)
- Comprehensive quality flags using the QualityFlag enum
- LLM-generated prose summaries for Methods section writing
- Automatic encoding detection and delimiter sniffing for CSV files

## New State Models

### DataExplorationSummary

High-level summary of data exploration with LLM-generated prose suitable for academic writing.

```python
from src.state.models import DataExplorationSummary

summary = DataExplorationSummary(
    prose_description="The dataset contains 1000 observations...",
    dataset_inventory=[DatasetInfo(...)],
    quality_flags=[QualityFlagItem(...)],
    recommended_variables=["var1", "var2"],
    data_gaps=["Missing recent data"],
)
```

### DatasetInfo

Concise information about a single dataset for inventory purposes.

```python
from src.state.models import DatasetInfo
from src.state.enums import DataStructureType

info = DatasetInfo(
    name="firm_panel",
    row_count=10000,
    column_count=25,
    memory_mb=15.5,
    date_range_start=date(2018, 1, 1),
    date_range_end=date(2023, 12, 31),
    structure_type=DataStructureType.PANEL,
)
```

### QualityFlagItem

A single quality flag with context.

```python
from src.state.models import QualityFlagItem
from src.state.enums import QualityFlag, CritiqueSeverity

flag = QualityFlagItem(
    flag=QualityFlag.MISSING_VALUES,
    severity=CritiqueSeverity.MAJOR,
    dataset_name="firm_panel",
    column_name="revenue",
    description="25% missing values",
    suggestion="Consider imputation",
)
```

## New Enums

### QualityFlag

23 quality flag types for data issues:
- Missing data: `MISSING_VALUES`, `HIGH_MISSING_RATE`
- File issues: `UNREADABLE_FILE`, `ENCODING_ERROR`, `MALFORMED_FILE`
- Schema issues: `SCHEMA_MISMATCH`, `DUPLICATE_COLUMNS`
- Date issues: `DATE_PARSING_FAILED`, `INCONSISTENT_DATES`, `FUTURE_DATES`
- Data quality: `LOW_SAMPLE_SIZE`, `HIGH_CARDINALITY`, `CONSTANT_COLUMN`, `DUPLICATE_ROWS`
- Statistical: `OUTLIERS_DETECTED`, `HIGHLY_SKEWED`, `MULTICOLLINEARITY`
- Panel/Time series: `UNBALANCED_PANEL`, `GAPS_IN_TIME_SERIES`

### DataStructureType

6 data structure types:
- `CROSS_SECTIONAL`: Single point in time, multiple entities
- `TIME_SERIES`: Single entity over time
- `PANEL`: Multiple entities over time (longitudinal)
- `POOLED`: Pooled cross-sections
- `HIERARCHICAL`: Nested/multilevel structure
- `UNKNOWN`: Structure could not be determined

## New Tools

### deep_profile_dataset

Comprehensive deep profiling with semantic type inference, data structure detection,
and quality flags.

```python
from src.tools.data_profiling import deep_profile_dataset

result = deep_profile_dataset.invoke({"name": "my_dataset"})
# Returns: overview, columns, inferred_types, data_structure, quality_flags, etc.
```

### detect_data_types

Intelligent semantic type detection beyond pandas dtypes.

```python
from src.tools.data_profiling import detect_data_types

result = detect_data_types.invoke({"name": "my_dataset"})
# Identifies: identifiers, years, percentages, currency, categorical, etc.
```

### assess_data_quality

Comprehensive quality assessment with QualityFlag enum.

```python
from src.tools.data_profiling import assess_data_quality

result = assess_data_quality.invoke({"name": "my_dataset"})
# Returns: quality_score, quality_level, flags, recommendations
```

### identify_time_series

Detect time series structure and analyze temporal patterns.

```python
from src.tools.data_profiling import identify_time_series

result = identify_time_series.invoke({"name": "my_dataset"})
# Returns: has_temporal_patterns, time_column, frequency, span
```

### detect_panel_structure

Detect panel (longitudinal) data structure.

```python
from src.tools.data_profiling import detect_panel_structure

result = detect_panel_structure.invoke({"name": "my_dataset"})
# Returns: is_panel, entity_column, time_column, panel_details
```

### generate_data_prose_summary

Generate LLM-written prose summary for Methods section.

```python
from src.tools.data_profiling import generate_data_prose_summary

result = generate_data_prose_summary.invoke({
    "dataset_names": ["dataset1", "dataset2"],
    "research_context": "Analyzing firm performance",
    "focus_variables": ["revenue", "employees"],
})
# Returns: DataExplorationSummary with prose_description
```

## Format Handling Improvements

### Automatic Encoding Detection

CSV files are now automatically detected for encoding:
- Uses `charset_normalizer` library when available
- Falls back to heuristic detection
- Supports: UTF-8, Latin-1, CP1252, ISO-8859-1, etc.

### Automatic Delimiter Sniffing

CSV delimiters are automatically detected:
- Uses Python's `csv.Sniffer`
- Falls back to character counting
- Supports: comma, semicolon, tab, pipe

## State Schema Updates

The WorkflowState now includes:

```python
# Sprint 12: Enhanced data exploration summary with LLM-generated prose
data_exploration_summary: DataExplorationSummary | None
```

This field is populated by the DATA_EXPLORER node and used by downstream nodes
(especially WRITER) to generate the Data/Methods section.

## Usage in DATA_EXPLORER Node

The node now:
1. Loads all datasets in parallel (Phase 1)
2. Builds basic exploration results (Phase 2)
3. Maps variables to columns (Phase 3)
4. Generates DataExplorationSummary with LLM prose (Phase 4 - NEW)

The summary is stored in `state["data_exploration_summary"]` and available to all downstream nodes.

## Tests

25 new unit tests in `tests/unit/test_sprint12_data_profiling.py` covering:
- QualityFlag and DataStructureType enums
- DatasetInfo, QualityFlagItem, DataExplorationSummary models
- All new profiling tools
- Encoding detection and delimiter sniffing
- DATA_EXPLORER node integration

## Migration Notes

No breaking changes. All existing code continues to work.

The new `data_exploration_summary` field is optional and defaults to `None`
if the LLM summarization fails or is not needed.
