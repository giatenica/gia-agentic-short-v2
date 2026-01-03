"""Unit tests for DATA_EXPLORER node and data exploration tools."""

import pytest
import tempfile
import os
from pathlib import Path
from importlib.util import find_spec

# Check if pandas is available
HAS_PANDAS = find_spec("pandas") is not None

from src.state.enums import ResearchStatus, DataQualityLevel, ColumnType
from src.state.models import DataFile, DataExplorationResult, ColumnAnalysis
from src.state.schema import create_initial_state


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file for testing."""
    fd, filepath = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w') as f:
        f.write("id,name,value,category\n")
        f.write("1,Alice,100,A\n")
        f.write("2,Bob,200,B\n")
        f.write("3,Charlie,150,A\n")
        f.write("4,Diana,,B\n")  # Missing value
        f.write("5,Eve,300,C\n")
    yield filepath
    os.unlink(filepath)


@pytest.fixture
def sample_csv_with_issues():
    """Create a CSV file with various quality issues."""
    fd, filepath = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w') as f:
        f.write("id,constant,missing_heavy,numeric\n")
        for i in range(100):
            missing = "" if i % 3 == 0 else str(i)  # 33% missing
            f.write(f"{i},same_value,{missing},{i * 10}\n")
        # Add duplicates
        f.write("1,same_value,1,10\n")
        f.write("1,same_value,1,10\n")
    yield filepath
    os.unlink(filepath)


@pytest.fixture
def sample_data_file(sample_csv_file):
    """Create a DataFile object from sample CSV."""
    return DataFile(
        filename="test.csv",
        filepath=Path(sample_csv_file),
        content_type="text/csv",
        size_bytes=os.path.getsize(sample_csv_file),
    )


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestDataExplorationTools:
    """Test data exploration tools."""
    
    def test_parse_csv_file(self, sample_csv_file):
        """Test parsing CSV file."""
        from src.tools.data_exploration import parse_csv_file
        
        result = parse_csv_file.invoke({"filepath": sample_csv_file})
        
        assert "error" not in result
        assert result["row_count"] == 5
        assert result["column_count"] == 4
        assert "id" in result["columns"]
        assert len(result["sample_rows"]) == 5
    
    def test_parse_csv_file_not_found(self):
        """Test parsing nonexistent CSV file."""
        from src.tools.data_exploration import parse_csv_file
        
        result = parse_csv_file.invoke({"filepath": "/nonexistent/file.csv"})
        
        assert "error" in result
        assert "not found" in result["error"].lower()
    
    def test_detect_schema(self, sample_csv_file):
        """Test schema detection."""
        from src.tools.data_exploration import detect_schema
        
        result = detect_schema.invoke({"filepath": sample_csv_file})
        
        assert "error" not in result
        assert "schema" in result
        assert "id" in result["schema"]
        # Use == instead of 'is' to handle numpy bool types
        assert result["schema"]["id"]["nullable"] == False
    
    def test_generate_summary_stats(self, sample_csv_file):
        """Test summary statistics generation."""
        from src.tools.data_exploration import generate_summary_stats
        
        result = generate_summary_stats.invoke({"filepath": sample_csv_file})
        
        assert "error" not in result
        assert "stats" in result
        assert "id" in result["stats"]
        assert "mean" in result["stats"]["id"]
    
    def test_detect_missing_values(self, sample_csv_file):
        """Test missing value detection."""
        from src.tools.data_exploration import detect_missing_values
        
        result = detect_missing_values.invoke({"filepath": sample_csv_file})
        
        assert "error" not in result
        assert result["total_rows"] == 5
        assert result["total_missing"] >= 1  # We have one missing value
        assert "value" in result["columns_with_missing"]
    
    def test_detect_outliers_iqr(self, sample_csv_file):
        """Test outlier detection using IQR method."""
        from src.tools.data_exploration import detect_outliers
        
        result = detect_outliers.invoke({
            "filepath": sample_csv_file,
            "method": "iqr",
        })
        
        assert "error" not in result
        assert result["method"] == "iqr"
        assert "outliers" in result
    
    def test_detect_outliers_zscore(self, sample_csv_file):
        """Test outlier detection using z-score method."""
        from src.tools.data_exploration import detect_outliers
        
        result = detect_outliers.invoke({
            "filepath": sample_csv_file,
            "method": "zscore",
        })
        
        assert "error" not in result
        assert result["method"] == "zscore"
    
    def test_assess_data_quality(self, sample_csv_file):
        """Test data quality assessment."""
        from src.tools.data_exploration import assess_data_quality
        
        result = assess_data_quality.invoke({"filepath": sample_csv_file})
        
        assert "error" not in result
        assert "quality_score" in result
        assert "quality_level" in result
        assert "issues" in result
        assert "recommendations" in result
    
    def test_assess_data_quality_with_issues(self, sample_csv_with_issues):
        """Test quality assessment on problematic data."""
        from src.tools.data_exploration import assess_data_quality
        
        result = assess_data_quality.invoke({"filepath": sample_csv_with_issues})
        
        assert "error" not in result
        # Should detect issues
        assert len(result["issues"]) > 0
        # Should have recommendations
        assert len(result["recommendations"]) > 0


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestAnalyzeFile:
    """Test the analyze_file function."""
    
    def test_analyze_csv_file(self, sample_data_file):
        """Test analyzing a CSV file."""
        from src.tools.data_exploration import analyze_file
        
        result = analyze_file(sample_data_file)
        
        assert isinstance(result, DataExplorationResult)
        assert len(result.files_analyzed) == 1
        assert result.files_analyzed[0].filename == "test.csv"
        assert result.total_rows == 5
        assert result.total_columns == 4
        assert len(result.columns) == 4
    
    def test_analyze_file_quality_score(self, sample_data_file):
        """Test quality score calculation."""
        from src.tools.data_exploration import analyze_file
        
        result = analyze_file(sample_data_file)
        
        assert result.quality_score >= 0.0
        assert result.quality_score <= 1.0
        assert result.quality_level in DataQualityLevel
    
    def test_analyze_file_column_analysis(self, sample_data_file):
        """Test column analysis in result."""
        from src.tools.data_exploration import analyze_file
        
        result = analyze_file(sample_data_file)
        
        # Find the 'value' column
        value_col = next(c for c in result.columns if c.name == "value")
        
        assert value_col.null_count == 1
        assert value_col.dtype in [ColumnType.FLOAT, ColumnType.INTEGER]


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestDataExplorerNode:
    """Test the DATA_EXPLORER node."""
    
    def test_node_with_no_data(self):
        """Test node when no data files uploaded."""
        from src.nodes.data_explorer import data_explorer_node
        
        state = create_initial_state()
        state["status"] = ResearchStatus.INTAKE_COMPLETE
        state["uploaded_data"] = []
        
        result = data_explorer_node(state)
        
        assert result["status"] == ResearchStatus.DATA_EXPLORED
        assert result["data_exploration_results"] is None
    
    def test_node_with_data(self, sample_csv_file):
        """Test node with uploaded data."""
        from src.nodes.data_explorer import data_explorer_node
        
        state = create_initial_state()
        state["status"] = ResearchStatus.INTAKE_COMPLETE
        state["uploaded_data"] = [
            DataFile(
                filename="test.csv",
                filepath=Path(sample_csv_file),
                content_type="text/csv",
                size_bytes=os.path.getsize(sample_csv_file),
            )
        ]
        state["key_variables"] = ["id", "value", "category"]
        
        result = data_explorer_node(state)
        
        assert result["status"] in [ResearchStatus.DATA_EXPLORED, ResearchStatus.DATA_QUALITY_ISSUES]
        assert result["data_exploration_results"] is not None
        assert len(result["variable_mappings"]) == 3
    
    def test_variable_mapping(self, sample_csv_file):
        """Test variable to column mapping."""
        from src.nodes.data_explorer import data_explorer_node
        
        state = create_initial_state()
        state["status"] = ResearchStatus.INTAKE_COMPLETE
        state["uploaded_data"] = [
            DataFile(
                filename="test.csv",
                filepath=Path(sample_csv_file),
                content_type="text/csv",
                size_bytes=os.path.getsize(sample_csv_file),
            )
        ]
        state["key_variables"] = ["id", "name", "unknown_var"]
        
        result = data_explorer_node(state)
        
        mappings = result["variable_mappings"]
        
        # id should match exactly
        id_mapping = next(m for m in mappings if m.user_variable == "id")
        assert id_mapping.matched_column == "id"
        assert id_mapping.confidence == 1.0
        
        # unknown_var should have low confidence
        unknown_mapping = next(m for m in mappings if m.user_variable == "unknown_var")
        assert unknown_mapping.confidence < 0.8


class TestMapVariablesToColumns:
    """Test variable mapping logic."""
    
    def test_exact_match(self):
        """Test exact variable name match."""
        from src.nodes.data_explorer import map_variables_to_columns
        
        exploration = DataExplorationResult(
            total_rows=100,
            total_columns=3,
            columns=[
                ColumnAnalysis(name="revenue", dtype=ColumnType.FLOAT,
                             non_null_count=100, null_count=0, 
                             null_percentage=0.0, unique_count=100),
                ColumnAnalysis(name="cost", dtype=ColumnType.FLOAT,
                             non_null_count=100, null_count=0,
                             null_percentage=0.0, unique_count=100),
            ],
            quality_score=0.9,
            quality_level=DataQualityLevel.EXCELLENT,
            quality_issues=[],
        )
        
        mappings = map_variables_to_columns(["revenue"], exploration)
        
        assert len(mappings) == 1
        assert mappings[0].matched_column == "revenue"
        assert mappings[0].confidence == 1.0
    
    def test_partial_match(self):
        """Test partial variable name match."""
        from src.nodes.data_explorer import map_variables_to_columns
        
        exploration = DataExplorationResult(
            total_rows=100,
            total_columns=2,
            columns=[
                ColumnAnalysis(name="total_revenue_usd", dtype=ColumnType.FLOAT,
                             non_null_count=100, null_count=0,
                             null_percentage=0.0, unique_count=100),
            ],
            quality_score=0.9,
            quality_level=DataQualityLevel.EXCELLENT,
            quality_issues=[],
        )
        
        mappings = map_variables_to_columns(["revenue"], exploration)
        
        assert len(mappings) == 1
        assert mappings[0].matched_column == "total_revenue_usd"
        assert mappings[0].confidence > 0.5


class TestRouteAfterDataExplorer:
    """Test routing after DATA_EXPLORER node."""
    
    def test_route_to_end_on_failure(self):
        """Test routing to end on failure."""
        from src.nodes.data_explorer import route_after_data_explorer
        
        state = create_initial_state()
        state["status"] = ResearchStatus.FAILED
        
        route = route_after_data_explorer(state)
        assert route == "end"
    
    def test_route_to_literature_on_success(self):
        """Test routing to literature reviewer on success."""
        from src.nodes.data_explorer import route_after_data_explorer
        
        state = create_initial_state()
        state["status"] = ResearchStatus.DATA_EXPLORED
        
        route = route_after_data_explorer(state)
        assert route == "literature_reviewer"


class TestColumnTypeMapping:
    """Test column type inference."""
    
    def test_get_column_type_integer(self):
        """Test integer type detection."""
        from src.tools.data_exploration import get_column_type
        import numpy as np
        
        assert get_column_type(np.dtype('int64')) == ColumnType.INTEGER
        assert get_column_type(np.dtype('int32')) == ColumnType.INTEGER
    
    def test_get_column_type_float(self):
        """Test float type detection."""
        from src.tools.data_exploration import get_column_type
        import numpy as np
        
        assert get_column_type(np.dtype('float64')) == ColumnType.FLOAT
        assert get_column_type(np.dtype('float32')) == ColumnType.FLOAT
    
    def test_get_column_type_object(self):
        """Test object/string type detection."""
        from src.tools.data_exploration import get_column_type
        import numpy as np
        
        assert get_column_type(np.dtype('object')) == ColumnType.STRING


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestZipFileHandling:
    """Test ZIP file extraction and analysis."""
    
    @pytest.fixture
    def zip_with_csv(self):
        """Create a ZIP file containing CSV data."""
        import zipfile
        
        # Create temp dir and CSV file
        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "data.csv")
        
        with open(csv_path, 'w') as f:
            f.write("id,name,value\n")
            f.write("1,Alice,100\n")
            f.write("2,Bob,200\n")
            f.write("3,Charlie,150\n")
        
        # Create ZIP file
        zip_path = os.path.join(temp_dir, "data.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(csv_path, "data.csv")
        
        yield zip_path
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def zip_with_multiple_files(self):
        """Create a ZIP file containing multiple CSV files."""
        import zipfile
        
        temp_dir = tempfile.mkdtemp()
        
        # Create first CSV
        csv1_path = os.path.join(temp_dir, "prices.csv")
        with open(csv1_path, 'w') as f:
            f.write("date,ticker,price\n")
            f.write("2024-01-01,GOOG,150\n")
            f.write("2024-01-02,GOOG,152\n")
        
        # Create second CSV
        csv2_path = os.path.join(temp_dir, "volume.csv")
        with open(csv2_path, 'w') as f:
            f.write("date,ticker,volume\n")
            f.write("2024-01-01,GOOG,1000000\n")
            f.write("2024-01-02,GOOG,1200000\n")
        
        # Create ZIP file
        zip_path = os.path.join(temp_dir, "market_data.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(csv1_path, "prices.csv")
            zf.write(csv2_path, "volume.csv")
        
        yield zip_path
        
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def zip_with_no_data_files(self):
        """Create a ZIP file with no analyzable data files."""
        import zipfile
        
        temp_dir = tempfile.mkdtemp()
        
        # Create a text file (not analyzable)
        txt_path = os.path.join(temp_dir, "readme.txt")
        with open(txt_path, 'w') as f:
            f.write("This is just a readme file.")
        
        # Create ZIP file
        zip_path = os.path.join(temp_dir, "no_data.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(txt_path, "readme.txt")
        
        yield zip_path
        
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_analyze_zip_with_csv(self, zip_with_csv):
        """Test analyzing a ZIP file containing a CSV."""
        from src.tools.data_exploration import analyze_file
        
        data_file = DataFile(
            filename="data.zip",
            filepath=Path(zip_with_csv),
            content_type="application/zip",
            size_bytes=os.path.getsize(zip_with_csv),
        )
        
        result = analyze_file(data_file)
        
        assert result.total_rows == 3
        assert result.total_columns == 3
        assert len(result.columns) == 3
        assert result.quality_score > 0
        assert "suitable for analysis" in result.feasibility_assessment.lower() or "successfully" in result.feasibility_assessment.lower()
    
    def test_analyze_zip_with_multiple_files(self, zip_with_multiple_files):
        """Test analyzing a ZIP file containing multiple CSVs."""
        from src.tools.data_exploration import analyze_file
        
        data_file = DataFile(
            filename="market_data.zip",
            filepath=Path(zip_with_multiple_files),
            content_type="application/zip",
            size_bytes=os.path.getsize(zip_with_multiple_files),
        )
        
        result = analyze_file(data_file)
        
        # Should have aggregated results from both files
        assert result.total_rows == 4  # 2 rows from each file
        assert result.total_columns == 6  # 3 columns from each file
        assert len(result.files_analyzed) >= 2  # Original ZIP + extracted files
        assert "2 of 2" in result.feasibility_assessment or result.total_rows > 0
    
    def test_analyze_zip_with_no_data_files(self, zip_with_no_data_files):
        """Test analyzing a ZIP file with no analyzable data."""
        from src.tools.data_exploration import analyze_file
        
        data_file = DataFile(
            filename="no_data.zip",
            filepath=Path(zip_with_no_data_files),
            content_type="application/zip",
            size_bytes=os.path.getsize(zip_with_no_data_files),
        )
        
        result = analyze_file(data_file)
        
        assert result.total_rows == 0
        assert result.total_columns == 0
        assert len(result.quality_issues) > 0
        assert any("no" in issue.description.lower() and "data" in issue.description.lower() 
                   for issue in result.quality_issues)
    
    def test_analyze_invalid_zip(self):
        """Test analyzing a file that's not a valid ZIP."""
        from src.tools.data_exploration import analyze_file
        
        # Create a fake "zip" file that's actually just text
        fd, filepath = tempfile.mkstemp(suffix='.zip')
        with os.fdopen(fd, 'w') as f:
            f.write("This is not a ZIP file")
        
        try:
            data_file = DataFile(
                filename="fake.zip",
                filepath=Path(filepath),
                content_type="application/zip",
                size_bytes=os.path.getsize(filepath),
            )
            
            result = analyze_file(data_file)
            
            assert result.total_rows == 0
            assert len(result.quality_issues) > 0
            assert any("invalid" in issue.description.lower() or "not a valid" in issue.description.lower()
                       for issue in result.quality_issues)
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
