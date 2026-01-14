"""Tests for Sprint 16: Integration, Testing & Writer Enhancement.

This module tests:
- Writer enhancement with table/figure references
- Artifact helper functions
- Integration of data exploration prose
- SectionWritingContext updates
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.writers.artifact_helpers import (
    format_table_reference,
    format_figure_reference,
    generate_table_summary,
    generate_figure_summary,
    format_data_exploration_for_methods,
    generate_results_artifacts_prompt,
    get_table_labels,
    get_figure_labels,
)
from src.state.models import (
    SectionWritingContext,
    TableArtifact,
    FigureArtifact,
    PaperSection,
)
from src.state.enums import ArtifactFormat, FigureFormat


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_table_artifact():
    """Create a sample table artifact."""
    return TableArtifact(
        table_id="tab:summary",
        title="Summary Statistics",
        caption="Descriptive statistics for key variables",
        format=ArtifactFormat.LATEX,
        content="\\begin{table}...\\end{table}",
        source_data="main_dataset",
        notes="*** p<0.01, ** p<0.05, * p<0.1",
    )


@pytest.fixture
def sample_figure_artifact():
    """Create a sample figure artifact."""
    return FigureArtifact(
        figure_id="fig:timeseries",
        title="Time Series of Returns",
        caption="Daily returns over the sample period",
        format=FigureFormat.PNG,
        content_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        source_data="returns_data",
        width_inches=10.0,
        height_inches=6.0,
        notes="Sample period: 2010-2024",
    )


@pytest.fixture
def sample_table_dict():
    """Create a sample table as dict."""
    return {
        "table_id": "tab:regression",
        "title": "Regression Results",
        "caption": "OLS regression results",
        "notes": "Standard errors in parentheses",
    }


@pytest.fixture
def sample_figure_dict():
    """Create a sample figure as dict."""
    return {
        "figure_id": "fig:scatter",
        "title": "Scatter Plot of X vs Y",
        "caption": "Relationship between variables",
        "notes": "With fitted regression line",
    }


# =============================================================================
# Test format_table_reference
# =============================================================================


class TestFormatTableReference:
    """Tests for format_table_reference function."""
    
    def test_format_with_artifact(self, sample_table_artifact):
        """Test formatting with TableArtifact object."""
        result = format_table_reference(sample_table_artifact, 1)
        assert result == "Table 1"
    
    def test_format_with_dict(self, sample_table_dict):
        """Test formatting with dict."""
        result = format_table_reference(sample_table_dict, 2)
        assert result == "Table 2"
    
    def test_format_different_numbers(self, sample_table_artifact):
        """Test formatting with different table numbers."""
        assert format_table_reference(sample_table_artifact, 1) == "Table 1"
        assert format_table_reference(sample_table_artifact, 5) == "Table 5"


# =============================================================================
# Test format_figure_reference
# =============================================================================


class TestFormatFigureReference:
    """Tests for format_figure_reference function."""
    
    def test_format_with_artifact(self, sample_figure_artifact):
        """Test formatting with FigureArtifact object."""
        result = format_figure_reference(sample_figure_artifact, 1)
        assert result == "Figure 1"
    
    def test_format_with_dict(self, sample_figure_dict):
        """Test formatting with dict."""
        result = format_figure_reference(sample_figure_dict, 3)
        assert result == "Figure 3"


# =============================================================================
# Test generate_table_summary
# =============================================================================


class TestGenerateTableSummary:
    """Tests for generate_table_summary function."""
    
    def test_empty_tables(self):
        """Test with no tables."""
        result = generate_table_summary([])
        assert result == ""
    
    def test_single_table(self, sample_table_artifact):
        """Test with single table artifact."""
        result = generate_table_summary([sample_table_artifact])
        
        assert "AVAILABLE TABLES:" in result
        assert "Table 1: Summary Statistics" in result
        assert "INSTRUCTIONS FOR TABLE REFERENCES:" in result
    
    def test_multiple_tables(self, sample_table_artifact, sample_table_dict):
        """Test with multiple tables."""
        result = generate_table_summary([sample_table_artifact, sample_table_dict])
        
        assert "Table 1: Summary Statistics" in result
        assert "Table 2: Regression Results" in result
    
    def test_includes_notes(self, sample_table_artifact):
        """Test that notes are included."""
        result = generate_table_summary([sample_table_artifact])
        
        assert "Notes:" in result


# =============================================================================
# Test generate_figure_summary
# =============================================================================


class TestGenerateFigureSummary:
    """Tests for generate_figure_summary function."""
    
    def test_empty_figures(self):
        """Test with no figures."""
        result = generate_figure_summary([])
        assert result == ""
    
    def test_single_figure(self, sample_figure_artifact):
        """Test with single figure artifact."""
        result = generate_figure_summary([sample_figure_artifact])
        
        assert "AVAILABLE FIGURES:" in result
        assert "Figure 1: Time Series of Returns" in result
        assert "INSTRUCTIONS FOR FIGURE REFERENCES:" in result
    
    def test_multiple_figures(self, sample_figure_artifact, sample_figure_dict):
        """Test with multiple figures."""
        result = generate_figure_summary([sample_figure_artifact, sample_figure_dict])
        
        assert "Figure 1: Time Series of Returns" in result
        assert "Figure 2: Scatter Plot of X vs Y" in result


# =============================================================================
# Test format_data_exploration_for_methods
# =============================================================================


class TestFormatDataExplorationForMethods:
    """Tests for format_data_exploration_for_methods function."""
    
    def test_with_prose(self):
        """Test with prose description."""
        prose = "The dataset contains 1.2 million observations from 2010-2024."
        result = format_data_exploration_for_methods(prose)
        
        assert result == prose
    
    def test_empty_prose_with_counts(self):
        """Test with empty prose but counts available."""
        result = format_data_exploration_for_methods(
            "",
            dataset_count=3,
            total_observations=1000000,
        )
        
        assert "3 dataset(s)" in result
        assert "1,000,000 observations" in result
    
    def test_completely_empty(self):
        """Test with no data at all."""
        result = format_data_exploration_for_methods("")
        assert result == ""


# =============================================================================
# Test generate_results_artifacts_prompt
# =============================================================================


class TestGenerateResultsArtifactsPrompt:
    """Tests for generate_results_artifacts_prompt function."""
    
    def test_with_tables_and_figures(self, sample_table_artifact, sample_figure_artifact):
        """Test with both tables and figures."""
        result = generate_results_artifacts_prompt(
            tables=[sample_table_artifact],
            figures=[sample_figure_artifact],
        )
        
        assert "AVAILABLE TABLES:" in result
        assert "AVAILABLE FIGURES:" in result
    
    def test_tables_only(self, sample_table_artifact):
        """Test with tables only."""
        result = generate_results_artifacts_prompt(
            tables=[sample_table_artifact],
            figures=[],
        )
        
        assert "AVAILABLE TABLES:" in result
        assert "AVAILABLE FIGURES:" not in result
    
    def test_figures_only(self, sample_figure_artifact):
        """Test with figures only."""
        result = generate_results_artifacts_prompt(
            tables=[],
            figures=[sample_figure_artifact],
        )
        
        assert "AVAILABLE FIGURES:" in result
        assert "AVAILABLE TABLES:" not in result
    
    def test_no_artifacts(self):
        """Test with no artifacts."""
        result = generate_results_artifacts_prompt([], [])
        
        assert "No tables or figures are available" in result


# =============================================================================
# Test get_table_labels / get_figure_labels
# =============================================================================


class TestGetLabels:
    """Tests for label extraction functions."""
    
    def test_get_table_labels(self, sample_table_artifact, sample_table_dict):
        """Test getting table labels."""
        labels = get_table_labels([sample_table_artifact, sample_table_dict])
        
        assert labels[1] == "tab:summary"
        assert labels[2] == "tab:regression"
    
    def test_get_figure_labels(self, sample_figure_artifact, sample_figure_dict):
        """Test getting figure labels."""
        labels = get_figure_labels([sample_figure_artifact, sample_figure_dict])
        
        assert labels[1] == "fig:timeseries"
        assert labels[2] == "fig:scatter"
    
    def test_empty_labels(self):
        """Test with empty lists."""
        assert get_table_labels([]) == {}
        assert get_figure_labels([]) == {}


# =============================================================================
# Test SectionWritingContext Updates
# =============================================================================


class TestSectionWritingContextUpdates:
    """Tests for updated SectionWritingContext model."""
    
    def test_context_with_tables(self, sample_table_artifact):
        """Test context with tables field."""
        context = SectionWritingContext(
            section_type="results",
            research_question="Test question",
            tables=[sample_table_artifact],
        )
        
        assert len(context.tables) == 1
        assert context.tables[0].title == "Summary Statistics"
    
    def test_context_with_figures(self, sample_figure_artifact):
        """Test context with figures field."""
        context = SectionWritingContext(
            section_type="results",
            research_question="Test question",
            figures=[sample_figure_artifact],
        )
        
        assert len(context.figures) == 1
        assert context.figures[0].title == "Time Series of Returns"
    
    def test_context_with_data_exploration_prose(self):
        """Test context with data exploration prose."""
        prose = "The dataset spans 2010-2024 with 1M observations."
        context = SectionWritingContext(
            section_type="methods",
            research_question="Test question",
            data_exploration_prose=prose,
        )
        
        assert context.data_exploration_prose == prose
    
    def test_context_defaults(self):
        """Test that new fields have proper defaults."""
        context = SectionWritingContext(
            section_type="results",
            research_question="Test question",
        )
        
        assert context.tables == []
        assert context.figures == []
        assert context.data_exploration_prose == ""


# =============================================================================
# Test Writer Integration
# =============================================================================


class TestWriterIntegration:
    """Tests for writer integration with artifacts."""
    
    def test_results_writer_import(self):
        """Test that ResultsWriter can import artifact helpers."""
        from src.writers.results import ResultsWriter
        from src.writers.artifact_helpers import generate_results_artifacts_prompt
        
        assert ResultsWriter is not None
        assert generate_results_artifacts_prompt is not None
    
    def test_methods_writer_import(self):
        """Test that MethodsWriter can import artifact helpers."""
        from src.writers.methods import MethodsWriter
        from src.writers.artifact_helpers import format_data_exploration_for_methods
        
        assert MethodsWriter is not None
        assert format_data_exploration_for_methods is not None
    
    def test_writers_init_exports(self):
        """Test that __init__ exports all helper functions."""
        from src.writers import (
            format_table_reference,
            format_figure_reference,
            generate_table_summary,
            generate_figure_summary,
            format_data_exploration_for_methods,
            generate_results_artifacts_prompt,
            get_table_labels,
            get_figure_labels,
        )
        
        assert format_table_reference is not None
        assert format_figure_reference is not None


# =============================================================================
# Test ResultsWriter Prompt Generation
# =============================================================================


class TestResultsWriterPrompt:
    """Tests for ResultsWriter prompt generation with artifacts."""
    
    def test_results_writer_uses_artifacts(self, sample_table_artifact, sample_figure_artifact):
        """Test that ResultsWriter includes artifact info in prompt."""
        from src.writers.results import ResultsWriter
        
        writer = ResultsWriter()
        context = SectionWritingContext(
            section_type="results",
            research_question="What is the effect of X on Y?",
            findings_summary="X significantly affects Y",
            tables=[sample_table_artifact],
            figures=[sample_figure_artifact],
            has_quantitative_results=True,
        )
        
        prompt = writer.get_user_prompt(context)
        
        assert "AVAILABLE TABLES:" in prompt
        assert "Table 1: Summary Statistics" in prompt
        assert "AVAILABLE FIGURES:" in prompt
        assert "Figure 1: Time Series of Returns" in prompt


# =============================================================================
# Test MethodsWriter Prompt Generation
# =============================================================================


class TestMethodsWriterPrompt:
    """Tests for MethodsWriter prompt generation with data exploration."""
    
    def test_methods_writer_uses_exploration_prose(self):
        """Test that MethodsWriter includes data exploration prose."""
        from src.writers.methods import MethodsWriter
        
        writer = MethodsWriter()
        context = SectionWritingContext(
            section_type="methods",
            research_question="What is the effect of X on Y?",
            methodology_summary="OLS regression with fixed effects",
            data_exploration_prose="The dataset contains 1.2 million options contract records spanning 2010-2024.",
            has_quantitative_results=True,
        )
        
        prompt = writer.get_user_prompt(context)
        
        assert "DATA DESCRIPTION" in prompt
        assert "1.2 million options" in prompt


# =============================================================================
# Test build_section_context Integration
# =============================================================================


class TestBuildSectionContext:
    """Tests for build_section_context with Sprint 16 fields."""
    
    def test_context_includes_tables_from_state(self):
        """Test that tables from state are passed to context."""
        from src.nodes.writer import build_section_context
        from src.writers.argument import ArgumentManager
        from src.state.enums import SectionType
        
        # Create mock state with tables
        mock_table = {
            "table_id": "tab:summary",
            "title": "Summary Statistics",
            "content": "\\begin{table}...\\end{table}",
        }
        
        state = {
            "research_intake": {"research_question": "Test?"},
            "tables": [mock_table],
            "figures": [],
        }
        
        argument_manager = ArgumentManager()
        
        context = build_section_context(
            state=state,
            section_type=SectionType.RESULTS,
            completed_sections=[],
            argument_manager=argument_manager,
        )
        
        assert len(context.tables) == 1
        # Tables can be dicts or TableArtifact objects
        table = context.tables[0]
        if hasattr(table, "title"):
            assert table.title == "Summary Statistics"
        else:
            assert table["title"] == "Summary Statistics"
    
    def test_context_includes_data_exploration_prose(self):
        """Test that data exploration prose is passed to context."""
        from src.nodes.writer import build_section_context
        from src.writers.argument import ArgumentManager
        from src.state.enums import SectionType
        
        state = {
            "research_intake": {"research_question": "Test?"},
            "data_exploration_summary": {
                "prose_description": "The dataset contains 46M observations."
            },
            "tables": [],
            "figures": [],
        }
        
        argument_manager = ArgumentManager()
        
        context = build_section_context(
            state=state,
            section_type=SectionType.METHODS,
            completed_sections=[],
            argument_manager=argument_manager,
        )
        
        assert "46M observations" in context.data_exploration_prose


# =============================================================================
# End-to-End Integration Tests
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests for the writer enhancement."""
    
    def test_full_results_writing_flow(self, sample_table_artifact, sample_figure_artifact):
        """Test complete results writing with artifacts."""
        context = SectionWritingContext(
            section_type="results",
            research_question="Does liquidity affect option pricing?",
            findings_summary="Liquidity significantly impacts option prices with coefficient of 0.023 (p<0.01)",
            methodology_summary="Panel regression with fixed effects",
            tables=[sample_table_artifact],
            figures=[sample_figure_artifact],
            has_quantitative_results=True,
            target_word_count=1000,
        )
        
        # Verify context is properly constructed
        assert len(context.tables) == 1
        assert len(context.figures) == 1
        assert context.has_quantitative_results is True
        
        # Verify prompt generation works
        from src.writers.results import ResultsWriter
        writer = ResultsWriter()
        prompt = writer.get_user_prompt(context)
        
        assert "liquidity" in prompt.lower()
        assert "Table 1" in prompt
        assert "Figure 1" in prompt
    
    def test_full_methods_writing_flow(self):
        """Test complete methods writing with data exploration."""
        context = SectionWritingContext(
            section_type="methods",
            research_question="Does liquidity affect option pricing?",
            methodology_summary="We employ panel regression with entity and time fixed effects.",
            data_exploration_prose="The sample comprises 46.2 million options contracts from OptionMetrics spanning 2004-2024.",
            has_quantitative_results=True,
            target_word_count=700,
        )
        
        # Verify context
        assert "46.2 million" in context.data_exploration_prose
        
        # Verify prompt generation
        from src.writers.methods import MethodsWriter
        writer = MethodsWriter()
        prompt = writer.get_user_prompt(context)
        
        assert "46.2 million" in prompt
        assert "DATA DESCRIPTION" in prompt
