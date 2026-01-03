"""Unit tests for visualization tools (Sprint 15).

Tests table generation, figure generation, and artifact models.
"""

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.state.enums import ArtifactFormat, FigureFormat
from src.state.models import TableArtifact, FigureArtifact


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "price": np.random.normal(100, 15, n),
        "volume": np.random.randint(1000, 10000, n),
        "returns": np.random.normal(0.001, 0.02, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "binary": np.random.choice([0, 1], n),
    })


@pytest.fixture
def mock_registry(sample_dataframe):
    """Mock the DataRegistry with sample data."""
    with patch("src.tools.visualization.get_registry") as mock:
        registry = MagicMock()
        registry.datasets = {"test_data": {"row_count": len(sample_dataframe)}}
        registry.get_dataframe.return_value = sample_dataframe
        mock.return_value = registry
        yield mock


@pytest.fixture
def sample_regression_results():
    """Create sample regression results for testing."""
    return [
        {
            "dependent_variable": "returns",
            "coefficients": [
                {
                    "variable": "(Intercept)",
                    "coefficient": 0.001,
                    "std_error": 0.002,
                    "t_stat": 0.5,
                    "p_value": 0.617,
                    "significant": False,
                },
                {
                    "variable": "price",
                    "coefficient": 0.0001,
                    "std_error": 0.00005,
                    "t_stat": 2.0,
                    "p_value": 0.048,
                    "significant": True,
                },
                {
                    "variable": "volume",
                    "coefficient": -0.000001,
                    "std_error": 0.000002,
                    "t_stat": -0.5,
                    "p_value": 0.617,
                    "significant": False,
                },
            ],
            "r_squared": 0.15,
            "adj_r_squared": 0.13,
            "f_statistic": 8.5,
            "f_p_value": 0.004,
            "n_observations": 100,
        }
    ]


# =============================================================================
# Artifact Model Tests
# =============================================================================


class TestTableArtifact:
    """Tests for TableArtifact model."""
    
    def test_create_table_artifact(self):
        """Test creating a table artifact."""
        artifact = TableArtifact(
            title="Summary Statistics",
            content="\\begin{table}...",
            source_data="test_dataset",
        )
        
        assert artifact.title == "Summary Statistics"
        assert artifact.format == ArtifactFormat.LATEX  # Default
        assert artifact.table_id.startswith("tab_")
        assert artifact.created_at is not None
    
    def test_table_artifact_all_fields(self):
        """Test table artifact with all fields."""
        artifact = TableArtifact(
            table_id="tab_custom_id",
            title="Regression Results",
            caption="OLS regression with robust standard errors",
            format=ArtifactFormat.MARKDOWN,
            content="| Variable | Coef |\n|---|---|",
            source_data="main_data",
            notes="*** p<0.01, ** p<0.05, * p<0.1",
        )
        
        assert artifact.table_id == "tab_custom_id"
        assert artifact.format == ArtifactFormat.MARKDOWN
        assert "p<0.01" in artifact.notes


class TestFigureArtifact:
    """Tests for FigureArtifact model."""
    
    def test_create_figure_artifact(self):
        """Test creating a figure artifact."""
        artifact = FigureArtifact(
            title="Time Series",
            content_base64="iVBORw0KGgo...",  # Partial base64
            source_data="test_dataset",
        )
        
        assert artifact.title == "Time Series"
        assert artifact.format == FigureFormat.PNG  # Default
        assert artifact.figure_id.startswith("fig_")
        assert artifact.width_inches == 10.0  # Default
        assert artifact.height_inches == 6.0  # Default
    
    def test_figure_artifact_all_fields(self):
        """Test figure artifact with all fields."""
        artifact = FigureArtifact(
            figure_id="fig_custom",
            title="Scatter Plot",
            caption="Relationship between X and Y",
            format=FigureFormat.PDF,
            content_base64="JVBERi0...",
            source_data="analysis_data",
            width_inches=8.0,
            height_inches=5.0,
            notes="Points colored by category",
        )
        
        assert artifact.figure_id == "fig_custom"
        assert artifact.format == FigureFormat.PDF
        assert artifact.width_inches == 8.0


# =============================================================================
# Table Generation Tests
# =============================================================================


class TestSummaryStatisticsTable:
    """Tests for create_summary_statistics_table tool."""
    
    def test_summary_stats_latex(self, mock_registry, sample_dataframe):
        """Test generating summary statistics in LaTeX format."""
        from src.tools.visualization import create_summary_statistics_table
        
        result = create_summary_statistics_table.invoke({
            "dataset_name": "test_data",
            "variables": ["price", "volume", "returns"],
            "format": "latex",
            "title": "Summary Statistics",
        })
        
        assert result["status"] == "success"
        assert "table_content" in result
        assert "artifact" in result
        assert "\\begin{table}" in result["table_content"]
        assert "Summary Statistics" in result["table_content"]
    
    def test_summary_stats_markdown(self, mock_registry, sample_dataframe):
        """Test generating summary statistics in markdown format."""
        from src.tools.visualization import create_summary_statistics_table
        
        result = create_summary_statistics_table.invoke({
            "dataset_name": "test_data",
            "format": "markdown",
        })
        
        assert result["status"] == "success"
        assert "|" in result["table_content"]  # Markdown table uses pipes
    
    def test_summary_stats_missing_dataset(self, mock_registry):
        """Test error handling for missing dataset."""
        from src.tools.visualization import create_summary_statistics_table
        
        mock_registry.return_value.datasets = {}
        
        result = create_summary_statistics_table.invoke({
            "dataset_name": "nonexistent",
        })
        
        assert result["status"] == "error"
        assert "not found" in result["error"]
    
    def test_summary_stats_custom_statistics(self, mock_registry, sample_dataframe):
        """Test with custom statistics selection."""
        from src.tools.visualization import create_summary_statistics_table
        
        result = create_summary_statistics_table.invoke({
            "dataset_name": "test_data",
            "statistics": ["mean", "std", "n"],
        })
        
        assert result["status"] == "success"


class TestRegressionTable:
    """Tests for create_regression_table tool."""
    
    def test_regression_table_single_model(self, sample_regression_results):
        """Test generating regression table with single model."""
        from src.tools.visualization import create_regression_table
        
        result = create_regression_table.invoke({
            "regression_results": sample_regression_results,
            "format": "latex",
        })
        
        assert result["status"] == "success"
        assert "table_content" in result
        assert "(1)" in result["table_content"]  # Model column header
        assert "***" in result["table_content"] or "**" in result["table_content"]  # Significance stars
    
    def test_regression_table_multiple_models(self, sample_regression_results):
        """Test generating regression table with multiple models."""
        from src.tools.visualization import create_regression_table
        
        # Add a second model
        model2 = sample_regression_results[0].copy()
        model2["r_squared"] = 0.25
        
        result = create_regression_table.invoke({
            "regression_results": sample_regression_results + [model2],
            "model_names": ["OLS", "OLS+Controls"],
        })
        
        assert result["status"] == "success"
        assert result["n_models"] == 2
        assert "OLS" in result["table_content"]
    
    def test_regression_table_no_diagnostics(self, sample_regression_results):
        """Test regression table without diagnostics."""
        from src.tools.visualization import create_regression_table
        
        result = create_regression_table.invoke({
            "regression_results": sample_regression_results,
            "include_diagnostics": False,
        })
        
        assert result["status"] == "success"
    
    def test_regression_table_empty_results(self):
        """Test error handling for empty results."""
        from src.tools.visualization import create_regression_table
        
        result = create_regression_table.invoke({
            "regression_results": [],
        })
        
        assert result["status"] == "error"


class TestCorrelationTable:
    """Tests for create_correlation_matrix_table tool."""
    
    def test_correlation_matrix(self, mock_registry, sample_dataframe):
        """Test generating correlation matrix."""
        from src.tools.visualization import create_correlation_matrix_table
        
        result = create_correlation_matrix_table.invoke({
            "dataset_name": "test_data",
            "variables": ["price", "volume", "returns"],
        })
        
        assert result["status"] == "success"
        assert "1.000" in result["table_content"]  # Diagonal
    
    def test_correlation_with_significance(self, mock_registry, sample_dataframe):
        """Test correlation matrix with significance stars."""
        from src.tools.visualization import create_correlation_matrix_table
        
        result = create_correlation_matrix_table.invoke({
            "dataset_name": "test_data",
            "include_significance": True,
        })
        
        assert result["status"] == "success"
    
    def test_correlation_insufficient_variables(self, mock_registry):
        """Test error with too few variables."""
        from src.tools.visualization import create_correlation_matrix_table
        
        # Create DataFrame with only one numeric column
        mock_registry.return_value.get_dataframe.return_value = pd.DataFrame({
            "x": [1, 2, 3],
            "category": ["a", "b", "c"],
        })
        
        result = create_correlation_matrix_table.invoke({
            "dataset_name": "test_data",
            "variables": ["x"],
        })
        
        assert result["status"] == "error"
        assert "at least 2" in result["error"]


class TestCrosstabTable:
    """Tests for create_crosstab_table tool."""
    
    def test_crosstab_count(self, mock_registry, sample_dataframe):
        """Test creating crosstab with counts."""
        from src.tools.visualization import create_crosstab_table
        
        result = create_crosstab_table.invoke({
            "dataset_name": "test_data",
            "row_var": "category",
            "col_var": "binary",
        })
        
        assert result["status"] == "success"
        assert "Total" in result["table_content"]
    
    def test_crosstab_mean(self, mock_registry, sample_dataframe):
        """Test creating crosstab with mean values."""
        from src.tools.visualization import create_crosstab_table
        
        result = create_crosstab_table.invoke({
            "dataset_name": "test_data",
            "row_var": "category",
            "col_var": "binary",
            "values_var": "price",
            "aggfunc": "mean",
        })
        
        assert result["status"] == "success"


# =============================================================================
# Figure Generation Tests
# =============================================================================


class TestTimeSeriesPlot:
    """Tests for create_time_series_plot tool."""
    
    def test_time_series_basic(self, mock_registry, sample_dataframe):
        """Test basic time series plot."""
        from src.tools.visualization import create_time_series_plot
        
        result = create_time_series_plot.invoke({
            "dataset_name": "test_data",
            "date_column": "date",
            "value_columns": ["price"],
            "title": "Price Over Time",
        })
        
        assert result["status"] == "success"
        assert "image_base64" in result
        assert result["artifact"]["format"] == "PNG"
        
        # Verify valid base64
        try:
            decoded = base64.b64decode(result["image_base64"])
            assert len(decoded) > 0
        except Exception:
            pytest.fail("Invalid base64 encoding")
    
    def test_time_series_multiple_series(self, mock_registry, sample_dataframe):
        """Test time series with multiple series."""
        from src.tools.visualization import create_time_series_plot
        
        result = create_time_series_plot.invoke({
            "dataset_name": "test_data",
            "date_column": "date",
            "value_columns": ["price", "volume"],
        })
        
        assert result["status"] == "success"
        assert len(result["variables_plotted"]) == 2
    
    def test_time_series_invalid_date(self, mock_registry):
        """Test error handling for invalid date column."""
        from src.tools.visualization import create_time_series_plot
        
        mock_registry.return_value.get_dataframe.return_value = pd.DataFrame({
            "not_a_date": ["abc", "def"],
            "value": [1, 2],
        })
        
        result = create_time_series_plot.invoke({
            "dataset_name": "test_data",
            "date_column": "not_a_date",
            "value_columns": ["value"],
        })
        
        assert result["status"] == "error"


class TestScatterPlot:
    """Tests for create_scatter_plot tool."""
    
    def test_scatter_basic(self, mock_registry, sample_dataframe):
        """Test basic scatter plot."""
        from src.tools.visualization import create_scatter_plot
        
        result = create_scatter_plot.invoke({
            "dataset_name": "test_data",
            "x_column": "price",
            "y_column": "returns",
        })
        
        assert result["status"] == "success"
        assert "image_base64" in result
    
    def test_scatter_with_regression(self, mock_registry, sample_dataframe):
        """Test scatter plot with regression line."""
        from src.tools.visualization import create_scatter_plot
        
        result = create_scatter_plot.invoke({
            "dataset_name": "test_data",
            "x_column": "price",
            "y_column": "returns",
            "add_regression_line": True,
        })
        
        assert result["status"] == "success"
    
    def test_scatter_with_color(self, mock_registry, sample_dataframe):
        """Test scatter plot with color coding."""
        from src.tools.visualization import create_scatter_plot
        
        result = create_scatter_plot.invoke({
            "dataset_name": "test_data",
            "x_column": "price",
            "y_column": "returns",
            "color_column": "category",
        })
        
        assert result["status"] == "success"


class TestDistributionPlot:
    """Tests for create_distribution_plot tool."""
    
    def test_histogram(self, mock_registry, sample_dataframe):
        """Test histogram plot."""
        from src.tools.visualization import create_distribution_plot
        
        result = create_distribution_plot.invoke({
            "dataset_name": "test_data",
            "column": "price",
            "plot_type": "histogram",
        })
        
        assert result["status"] == "success"
        assert "mean" in result
        assert "std" in result
    
    def test_density_plot(self, mock_registry, sample_dataframe):
        """Test density plot."""
        from src.tools.visualization import create_distribution_plot
        
        result = create_distribution_plot.invoke({
            "dataset_name": "test_data",
            "column": "returns",
            "plot_type": "density",
        })
        
        assert result["status"] == "success"
    
    def test_box_plot(self, mock_registry, sample_dataframe):
        """Test box plot."""
        from src.tools.visualization import create_distribution_plot
        
        result = create_distribution_plot.invoke({
            "dataset_name": "test_data",
            "column": "volume",
            "plot_type": "box",
        })
        
        assert result["status"] == "success"


class TestHeatmap:
    """Tests for create_heatmap tool."""
    
    def test_heatmap_basic(self, mock_registry, sample_dataframe):
        """Test basic correlation heatmap."""
        from src.tools.visualization import create_heatmap
        
        result = create_heatmap.invoke({
            "dataset_name": "test_data",
            "columns": ["price", "volume", "returns"],
        })
        
        assert result["status"] == "success"
        assert "image_base64" in result
    
    def test_heatmap_all_numeric(self, mock_registry, sample_dataframe):
        """Test heatmap with all numeric columns."""
        from src.tools.visualization import create_heatmap
        
        result = create_heatmap.invoke({
            "dataset_name": "test_data",
        })
        
        assert result["status"] == "success"


# =============================================================================
# Export Tests
# =============================================================================


class TestExportArtifacts:
    """Tests for export_all_artifacts tool."""
    
    def test_export_tables(self, tmp_path):
        """Test exporting table artifacts to files."""
        from src.tools.visualization import export_all_artifacts
        
        tables = [
            {
                "table_id": "tab_summary",
                "content": "\\begin{table}...\\end{table}",
                "format": "LATEX",
            }
        ]
        
        result = export_all_artifacts.invoke({
            "output_dir": str(tmp_path),
            "tables": tables,
        })
        
        assert result["status"] == "success"
        assert len(result["exported_files"]) == 1
        assert (tmp_path / "tab_summary.tex").exists()
    
    def test_export_figures(self, tmp_path):
        """Test exporting figure artifacts to files."""
        from src.tools.visualization import export_all_artifacts
        
        # Create a minimal valid PNG
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1, 1))
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        
        figures = [
            {
                "figure_id": "fig_test",
                "content_base64": img_b64,
            }
        ]
        
        result = export_all_artifacts.invoke({
            "output_dir": str(tmp_path),
            "figures": figures,
        })
        
        assert result["status"] == "success"
        assert (tmp_path / "fig_test.png").exists()


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestSignificanceStars:
    """Tests for significance star helper."""
    
    def test_three_stars(self):
        """Test *** for p < 0.01."""
        from src.tools.visualization import _significance_stars
        
        assert _significance_stars(0.005) == "***"
        assert _significance_stars(0.009) == "***"
    
    def test_two_stars(self):
        """Test ** for 0.01 <= p < 0.05."""
        from src.tools.visualization import _significance_stars
        
        assert _significance_stars(0.01) == "**"
        assert _significance_stars(0.04) == "**"
    
    def test_one_star(self):
        """Test * for 0.05 <= p < 0.1."""
        from src.tools.visualization import _significance_stars
        
        assert _significance_stars(0.05) == "*"
        assert _significance_stars(0.09) == "*"
    
    def test_no_stars(self):
        """Test no stars for p >= 0.1."""
        from src.tools.visualization import _significance_stars
        
        assert _significance_stars(0.1) == ""
        assert _significance_stars(0.5) == ""


# =============================================================================
# Integration Tests
# =============================================================================


class TestVisualizationIntegration:
    """Integration tests for visualization workflow."""
    
    def test_full_table_workflow(self, mock_registry, sample_dataframe, tmp_path):
        """Test complete workflow: generate tables -> export."""
        from src.tools.visualization import (
            create_summary_statistics_table,
            create_correlation_matrix_table,
            export_all_artifacts,
        )
        
        # Generate tables
        summary = create_summary_statistics_table.invoke({
            "dataset_name": "test_data",
            "format": "latex",
        })
        
        corr = create_correlation_matrix_table.invoke({
            "dataset_name": "test_data",
            "format": "latex",
        })
        
        assert summary["status"] == "success"
        assert corr["status"] == "success"
        
        # Export
        result = export_all_artifacts.invoke({
            "output_dir": str(tmp_path),
            "tables": [summary["artifact"], corr["artifact"]],
        })
        
        assert result["status"] == "success"
        assert len(result["exported_files"]) == 2


class TestEnumValues:
    """Tests for enum values."""
    
    def test_artifact_format_values(self):
        """Test ArtifactFormat enum values."""
        assert ArtifactFormat.LATEX.value == "LATEX"
        assert ArtifactFormat.MARKDOWN.value == "MARKDOWN"
        assert ArtifactFormat.HTML.value == "HTML"
    
    def test_figure_format_values(self):
        """Test FigureFormat enum values."""
        assert FigureFormat.PNG.value == "PNG"
        assert FigureFormat.PDF.value == "PDF"
        assert FigureFormat.SVG.value == "SVG"
