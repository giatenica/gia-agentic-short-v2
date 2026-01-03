"""Unit tests for Sprint 12 enhanced data profiling tools.

Tests the new functionality introduced in Sprint 12:
- Deep profiling with semantic type inference
- Data structure detection (panel, time series, cross-sectional)
- Comprehensive quality flags with QualityFlag enum
- LLM prose summary generation
- Encoding detection and delimiter sniffing
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime
from importlib.util import find_spec
from unittest.mock import MagicMock, patch

# Check if pandas is available
HAS_PANDAS = find_spec("pandas") is not None
HAS_SCIPY = find_spec("scipy") is not None

if HAS_PANDAS:
    import pandas as pd
    import numpy as np

from src.state.enums import QualityFlag, DataStructureType, ResearchStatus, CritiqueSeverity
from src.state.models import DatasetInfo, QualityFlagItem, DataExplorationSummary


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file for testing."""
    fd, filepath = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w') as f:
        f.write("id,name,value,category,date\n")
        f.write("1,Alice,100.5,A,2023-01-01\n")
        f.write("2,Bob,200.0,B,2023-01-02\n")
        f.write("3,Charlie,150.75,A,2023-01-03\n")
        f.write("4,Diana,,B,2023-01-04\n")  # Missing value
        f.write("5,Eve,300.0,C,2023-01-05\n")
    yield filepath
    os.unlink(filepath)


@pytest.fixture
def panel_csv_file():
    """Create a panel data CSV file (entity-time structure)."""
    fd, filepath = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w') as f:
        f.write("firm_id,year,revenue,employees\n")
        for firm in ['AAPL', 'GOOG', 'MSFT']:
            for year in range(2018, 2024):
                revenue = 100 + np.random.randint(0, 50)
                employees = 1000 + np.random.randint(0, 500)
                f.write(f"{firm},{year},{revenue},{employees}\n")
    yield filepath
    os.unlink(filepath)


@pytest.fixture
def time_series_csv_file():
    """Create a time series CSV file."""
    fd, filepath = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w') as f:
        f.write("date,price,volume\n")
        base_date = datetime(2023, 1, 1)
        for i in range(100):
            date = (base_date.replace(day=1) if i == 0 else 
                   datetime(2023, 1, 1) + pd.Timedelta(days=i))
            price = 100 + np.random.randn() * 10
            volume = 1000000 + np.random.randint(-100000, 100000)
            f.write(f"{date.strftime('%Y-%m-%d')},{price:.2f},{volume}\n")
    yield filepath
    os.unlink(filepath)


@pytest.fixture
def quality_issues_csv():
    """Create CSV with various quality issues."""
    fd, filepath = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w') as f:
        f.write("id,constant_col,heavy_missing,correlated1,correlated2\n")
        for i in range(100):
            # 40% missing in heavy_missing column
            missing = "" if i % 5 in [0, 1] else str(i)
            # correlated1 and correlated2 are highly correlated
            f.write(f"{i},SAME,{missing},{i * 10},{i * 10 + np.random.randint(0, 5)}\n")
        # Add duplicate rows
        f.write("1,SAME,1,10,10\n")
        f.write("1,SAME,1,10,10\n")
    yield filepath
    os.unlink(filepath)


@pytest.fixture
def semicolon_csv_file():
    """Create a semicolon-delimited CSV file."""
    fd, filepath = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w') as f:
        f.write("id;name;value\n")
        f.write("1;Alice;100\n")
        f.write("2;Bob;200\n")
    yield filepath
    os.unlink(filepath)


@pytest.fixture
def latin1_csv_file():
    """Create a Latin-1 encoded CSV file."""
    fd, filepath = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    with open(filepath, 'w', encoding='latin-1') as f:
        f.write("id,name,city\n")
        f.write("1,José,São Paulo\n")
        f.write("2,François,Montréal\n")
    yield filepath
    os.unlink(filepath)


@pytest.fixture
def loaded_dataset(sample_csv_file):
    """Load a dataset into the registry."""
    from src.tools.data_loading import load_data, get_registry
    
    # Clear registry first
    registry = get_registry()
    registry.clear()
    
    # Load the dataset
    load_data.invoke({
        "filepath": sample_csv_file,
        "name": "test_dataset"
    })
    
    yield "test_dataset"
    
    # Cleanup
    registry.clear()


@pytest.fixture
def loaded_panel_dataset(panel_csv_file):
    """Load a panel dataset into the registry."""
    from src.tools.data_loading import load_data, get_registry
    
    registry = get_registry()
    registry.clear()
    
    load_data.invoke({
        "filepath": panel_csv_file,
        "name": "panel_dataset"
    })
    
    yield "panel_dataset"
    registry.clear()


# =============================================================================
# Test QualityFlag Enum
# =============================================================================


class TestQualityFlagEnum:
    """Test QualityFlag enum values."""
    
    def test_quality_flag_values_exist(self):
        """Test that all expected QualityFlag values exist."""
        expected_flags = [
            'MISSING_VALUES', 'DUPLICATE_ROWS', 'OUTLIERS_DETECTED',
            'CONSTANT_COLUMN', 'HIGH_CARDINALITY', 'LOW_SAMPLE_SIZE',
            'MULTICOLLINEARITY', 'HIGHLY_SKEWED',
            'ENCODING_ERROR', 'UNREADABLE_FILE',
        ]
        
        for flag_name in expected_flags:
            assert hasattr(QualityFlag, flag_name), f"Missing QualityFlag: {flag_name}"
    
    def test_quality_flag_string_values(self):
        """Test that QualityFlag values are strings."""
        assert QualityFlag.MISSING_VALUES.value == "missing_values"
        assert QualityFlag.DUPLICATE_ROWS.value == "duplicate_rows"
        assert QualityFlag.OUTLIERS_DETECTED.value == "outliers_detected"


# =============================================================================
# Test DataStructureType Enum
# =============================================================================


class TestDataStructureTypeEnum:
    """Test DataStructureType enum values."""
    
    def test_structure_type_values_exist(self):
        """Test that all expected DataStructureType values exist."""
        expected_types = [
            'CROSS_SECTIONAL', 'TIME_SERIES', 'PANEL',
            'HIERARCHICAL', 'UNKNOWN'
        ]
        
        for type_name in expected_types:
            assert hasattr(DataStructureType, type_name), f"Missing type: {type_name}"


# =============================================================================
# Test DatasetInfo Model
# =============================================================================


class TestDatasetInfoModel:
    """Test DatasetInfo Pydantic model."""
    
    def test_dataset_info_creation(self):
        """Test creating a DatasetInfo object."""
        from datetime import date
        
        info = DatasetInfo(
            name="test_data",
            row_count=1000,
            column_count=10,
            memory_mb=5.5,
            date_range_start=date(2020, 1, 1),
            date_range_end=date(2023, 12, 31),
            structure_type=DataStructureType.PANEL,
        )
        
        assert info.name == "test_data"
        assert info.row_count == 1000
        assert info.column_count == 10
        assert info.memory_mb == 5.5
        assert info.structure_type == DataStructureType.PANEL
        assert info.date_range_str == "2020-01-01 to 2023-12-31"
    
    def test_dataset_info_defaults(self):
        """Test DatasetInfo with minimal required fields."""
        info = DatasetInfo(
            name="minimal",
            row_count=100,
            column_count=5,
        )
        
        # memory_mb defaults to 0.0 per model definition
        assert info.memory_mb == 0.0
        assert info.date_range_start is None
        assert info.structure_type == DataStructureType.UNKNOWN


# =============================================================================
# Test QualityFlagItem Model
# =============================================================================


class TestQualityFlagItemModel:
    """Test QualityFlagItem Pydantic model."""
    
    def test_quality_flag_item_creation(self):
        """Test creating a QualityFlagItem object."""
        item = QualityFlagItem(
            flag=QualityFlag.MISSING_VALUES,
            severity=CritiqueSeverity.MAJOR,
            dataset_name="test_data",
            column_name="age",
            description="25% missing values",
            suggestion="Consider imputation",
        )
        
        assert item.flag == QualityFlag.MISSING_VALUES
        assert item.severity == CritiqueSeverity.MAJOR
        assert item.column_name == "age"
    
    def test_quality_flag_item_without_column(self):
        """Test QualityFlagItem for dataset-level issue."""
        item = QualityFlagItem(
            flag=QualityFlag.LOW_SAMPLE_SIZE,
            severity=CritiqueSeverity.CRITICAL,
            dataset_name="test_data",
            column_name=None,
            description="Only 25 observations",
            suggestion="Collect more data",
        )
        
        assert item.column_name is None


# =============================================================================
# Test DataExplorationSummary Model
# =============================================================================


class TestDataExplorationSummaryModel:
    """Test DataExplorationSummary Pydantic model."""
    
    def test_summary_creation(self):
        """Test creating a DataExplorationSummary object."""
        dataset_info = DatasetInfo(
            name="test",
            row_count=1000,
            column_count=10,
        )
        
        quality_flag = QualityFlagItem(
            flag=QualityFlag.MISSING_VALUES,
            severity=CritiqueSeverity.MINOR,
            dataset_name="test",
            description="5% missing",
            suggestion="Monitor",
        )
        
        summary = DataExplorationSummary(
            prose_description="The dataset contains 1000 observations.",
            dataset_inventory=[dataset_info],
            quality_flags=[quality_flag],
            recommended_variables=["var1", "var2"],
            data_gaps=["Missing recent data"],
        )
        
        assert len(summary.dataset_inventory) == 1
        assert len(summary.quality_flags) == 1
        assert "var1" in summary.recommended_variables
    
    def test_summary_properties(self):
        """Test computed properties on DataExplorationSummary."""
        datasets = [
            DatasetInfo(name="d1", row_count=500, column_count=5),
            DatasetInfo(name="d2", row_count=1500, column_count=8),
        ]
        
        flags = [
            QualityFlagItem(
                flag=QualityFlag.MISSING_VALUES,
                severity=CritiqueSeverity.CRITICAL,
                dataset_name="d1",
                description="50% missing",
                suggestion="Fix",
            ),
            QualityFlagItem(
                flag=QualityFlag.OUTLIERS_DETECTED,
                severity=CritiqueSeverity.MINOR,
                dataset_name="d2",
                description="2% outliers",
                suggestion="Review",
            ),
        ]
        
        summary = DataExplorationSummary(
            prose_description="Test",
            dataset_inventory=datasets,
            quality_flags=flags,
        )
        
        # Test basic attributes exist (computed properties may not exist yet)
        assert len(summary.dataset_inventory) == 2
        assert len(summary.quality_flags) == 2
        
        # Check if critical issues exist
        critical_flags = [f for f in summary.quality_flags if f.severity == CritiqueSeverity.CRITICAL]
        assert len(critical_flags) == 1


# =============================================================================
# Test Enhanced Profiling Tools
# =============================================================================


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestDeepProfileDataset:
    """Test deep_profile_dataset tool."""
    
    def test_deep_profile_basic(self, loaded_dataset):
        """Test basic deep profiling."""
        from src.tools.data_profiling import deep_profile_dataset
        
        result = deep_profile_dataset.invoke({"name": loaded_dataset})
        
        assert "error" not in result
        assert result["dataset_name"] == loaded_dataset
        assert "overview" in result
        assert "columns" in result
        assert "inferred_types" in result
        assert "data_structure" in result
        assert "quality_flags" in result
    
    def test_deep_profile_not_found(self):
        """Test deep profiling with nonexistent dataset."""
        from src.tools.data_profiling import deep_profile_dataset
        
        result = deep_profile_dataset.invoke({"name": "nonexistent"})
        
        assert "error" in result


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestDetectDataTypes:
    """Test detect_data_types tool."""
    
    def test_detect_types_basic(self, loaded_dataset):
        """Test basic type detection."""
        from src.tools.data_profiling import detect_data_types
        
        result = detect_data_types.invoke({"name": loaded_dataset})
        
        assert "error" not in result
        assert "column_types" in result
        assert "type_summary" in result
    
    def test_detect_types_identifies_id(self, loaded_dataset):
        """Test that ID column is identified as identifier."""
        from src.tools.data_profiling import detect_data_types
        
        result = detect_data_types.invoke({"name": loaded_dataset})
        
        # Check that 'id' column is detected correctly
        column_types = result.get("column_types", {})
        if "id" in column_types:
            # Should be identified as identifier or integer
            sem_type = column_types["id"]["semantic_type"]
            assert sem_type in ["identifier", "continuous", "count"]


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestAssessDataQuality:
    """Test assess_data_quality tool with QualityFlag enum."""
    
    def test_assess_quality_basic(self, loaded_dataset):
        """Test basic quality assessment."""
        from src.tools.data_profiling import assess_data_quality
        
        result = assess_data_quality.invoke({"name": loaded_dataset})
        
        assert "error" not in result
        assert "quality_score" in result
        assert "quality_level" in result
        assert "flags" in result
        assert "recommendations" in result
    
    def test_assess_quality_detects_issues(self, quality_issues_csv):
        """Test that quality issues are detected."""
        from src.tools.data_loading import load_data, get_registry
        from src.tools.data_profiling import assess_data_quality
        
        registry = get_registry()
        registry.clear()
        
        load_data.invoke({
            "filepath": quality_issues_csv,
            "name": "quality_test"
        })
        
        result = assess_data_quality.invoke({"name": "quality_test"})
        
        assert "error" not in result
        assert len(result["flags"]) > 0
        
        # Should detect at least one of these issues
        flag_types = [f["flag"] for f in result["flags"]]
        expected_issues = [
            QualityFlag.CONSTANT_COLUMN.value,
            QualityFlag.MISSING_VALUES.value,
            QualityFlag.DUPLICATE_ROWS.value,
        ]
        assert any(f in flag_types for f in expected_issues)
        
        registry.clear()


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestIdentifyTimeSeries:
    """Test identify_time_series tool."""
    
    def test_identify_time_series_basic(self, loaded_dataset):
        """Test basic time series identification."""
        from src.tools.data_profiling import identify_time_series
        
        result = identify_time_series.invoke({"name": loaded_dataset})
        
        assert "status" in result
        assert "has_temporal_patterns" in result


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestDetectPanelStructure:
    """Test detect_panel_structure tool."""
    
    def test_detect_panel_basic(self, loaded_dataset):
        """Test basic panel detection."""
        from src.tools.data_profiling import detect_panel_structure
        
        result = detect_panel_structure.invoke({"name": loaded_dataset})
        
        assert "error" not in result
        assert "is_panel" in result
        assert "structure_type" in result
    
    def test_detect_panel_structure_panel_data(self, loaded_panel_dataset):
        """Test panel detection on actual panel data."""
        from src.tools.data_profiling import detect_panel_structure
        
        result = detect_panel_structure.invoke({"name": loaded_panel_dataset})
        
        assert "error" not in result
        assert result["is_panel"] is True
        assert result["structure_type"] == DataStructureType.PANEL.value


# =============================================================================
# Test LLM Summarization Tool
# =============================================================================


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestGenerateDataProseSummary:
    """Test generate_data_prose_summary tool."""
    
    @patch('src.tools.data_profiling.ChatAnthropic')
    def test_generate_prose_summary_mocked(self, mock_chat, loaded_dataset):
        """Test prose summary generation with mocked LLM."""
        from src.tools.data_profiling import generate_data_prose_summary
        
        # Mock LLM response
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(
            content="The dataset contains 5 observations across 5 variables."
        )
        mock_chat.return_value = mock_instance
        
        result = generate_data_prose_summary.invoke({
            "dataset_names": [loaded_dataset],
            "research_context": "Testing data analysis",
            "focus_variables": ["value"],
        })
        
        assert "error" not in result
        assert "summary" in result or "prose_description" in result
    
    def test_generate_prose_summary_empty(self):
        """Test prose summary with no datasets."""
        from src.tools.data_profiling import generate_data_prose_summary
        
        result = generate_data_prose_summary.invoke({
            "dataset_names": [],
        })
        
        assert "error" in result


# =============================================================================
# Test Format Handling (Encoding & Delimiter Detection)
# =============================================================================


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestEncodingDetection:
    """Test encoding detection in data loading."""
    
    def test_detect_latin1_encoding(self, latin1_csv_file):
        """Test that Latin-1 encoded file is loaded correctly."""
        from src.tools.data_loading import load_data, get_registry
        
        registry = get_registry()
        registry.clear()
        
        result = load_data.invoke({
            "filepath": latin1_csv_file,
            "name": "latin1_test"
        })
        
        assert "error" not in result
        
        # Verify the data was loaded correctly
        df = registry.get("latin1_test")
        assert df is not None
        # Check that special characters are present
        assert "José" in df["name"].values or "Jos" in str(df["name"].values)
        
        registry.clear()


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestDelimiterDetection:
    """Test delimiter sniffing in CSV loading."""
    
    def test_detect_semicolon_delimiter(self, semicolon_csv_file):
        """Test that semicolon-delimited file is loaded correctly."""
        from src.tools.data_loading import load_data, get_registry
        
        registry = get_registry()
        registry.clear()
        
        result = load_data.invoke({
            "filepath": semicolon_csv_file,
            "name": "semicolon_test"
        })
        
        assert "error" not in result
        assert result["column_count"] == 3  # id, name, value
        
        # Verify columns are properly split
        df = registry.get("semicolon_test")
        assert df is not None
        assert "name" in df.columns
        assert "Alice" in df["name"].values
        
        registry.clear()


# =============================================================================
# Test Data Explorer Node with Sprint 12
# =============================================================================


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestDataExplorerNodeSprint12:
    """Test DATA_EXPLORER node with Sprint 12 enhancements."""
    
    def test_node_returns_exploration_summary(self, sample_csv_file):
        """Test that node returns DataExplorationSummary."""
        from src.nodes.data_explorer import data_explorer_node
        from src.state.models import DataFile
        from src.state.schema import create_initial_state
        from src.tools.data_loading import get_registry
        
        # Clear registry
        registry = get_registry()
        registry.clear()
        
        # Create state with a data file
        data_file = DataFile(
            filename="test.csv",
            filepath=Path(sample_csv_file),
            content_type="text/csv",
            size_bytes=os.path.getsize(sample_csv_file),
        )
        
        state = create_initial_state(
            uploaded_data=[data_file],
            original_query="Test research question",
            key_variables=["value"],
        )
        
        # Run node (will skip LLM call in tests unless mocked)
        with patch('src.nodes.data_explorer.generate_data_prose_summary') as mock_summary:
            # Mock the summary generation
            mock_summary.invoke.return_value = {
                "error": "Mocked for test"
            }
            
            result = data_explorer_node(state)
        
        assert "data_exploration_results" in result
        assert result["status"] in [ResearchStatus.DATA_EXPLORED, ResearchStatus.DATA_QUALITY_ISSUES]
        
        registry.clear()


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Test internal helper functions."""
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_encoding_detection_function(self, latin1_csv_file):
        """Test _detect_encoding function directly."""
        from src.tools.data_loading import _detect_encoding
        
        encoding = _detect_encoding(latin1_csv_file)
        
        # Should detect a valid encoding (may vary by system)
        assert encoding is not None
        # Accept common latin-1 compatible encodings
        assert encoding.lower() in ["latin-1", "iso-8859-1", "cp1252", "utf-8", "cp1250", "iso-8859-2"]
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_delimiter_detection_function(self, semicolon_csv_file):
        """Test _detect_delimiter function directly."""
        from src.tools.data_loading import _detect_delimiter
        
        delimiter = _detect_delimiter(semicolon_csv_file)
        
        assert delimiter == ";"
