"""Unit tests for state management module."""

import pytest
from datetime import date
from pathlib import Path

from src.state.enums import (
    ResearchStatus,
    CritiqueSeverity,
    EvidenceStrength,
    PaperType,
    ResearchType,
    TargetJournal,
    DataQualityLevel,
    ColumnType,
)
from src.state.models import (
    IntakeFormData,
    DataFile,
    ColumnAnalysis,
    DataExplorationResult,
    ResearchPlan,
    SearchResult,
    CritiqueItem,
    Critique,
    EvidenceItem,
    WorkflowError,
)
from src.state.schema import (
    create_initial_state,
    validate_state_for_node,
)


class TestEnums:
    """Test enum definitions."""
    
    def test_research_status_values(self):
        """Test ResearchStatus enum has expected values."""
        assert ResearchStatus.INITIALIZED.value == "initialized"
        assert ResearchStatus.INTAKE_PENDING.value == "intake_pending"
        assert ResearchStatus.COMPLETED.value == "completed"
        assert ResearchStatus.FAILED.value == "failed"
    
    def test_critique_severity_ordering(self):
        """Test CritiqueSeverity enum values."""
        assert CritiqueSeverity.CRITICAL.value == "critical"
        assert CritiqueSeverity.MAJOR.value == "major"
        assert CritiqueSeverity.MINOR.value == "minor"
        assert CritiqueSeverity.SUGGESTION.value == "suggestion"
    
    def test_evidence_strength_values(self):
        """Test EvidenceStrength enum values."""
        assert EvidenceStrength.STRONG.value == "strong"
        assert EvidenceStrength.MODERATE.value == "moderate"
        assert EvidenceStrength.WEAK.value == "weak"
        assert EvidenceStrength.INSUFFICIENT.value == "insufficient"
    
    def test_paper_type_values(self):
        """Test PaperType enum values."""
        assert PaperType.SHORT_ARTICLE.value == "short_article"
        assert PaperType.FULL_PAPER.value == "full_paper"
        assert PaperType.WORKING_PAPER.value == "working_paper"
    
    def test_research_type_values(self):
        """Test ResearchType enum values."""
        assert ResearchType.EMPIRICAL.value == "empirical"
        assert ResearchType.THEORETICAL.value == "theoretical"
        assert ResearchType.MIXED.value == "mixed"
    
    def test_data_quality_level_values(self):
        """Test DataQualityLevel enum values."""
        assert DataQualityLevel.EXCELLENT.value == "excellent"
        assert DataQualityLevel.GOOD.value == "good"
        assert DataQualityLevel.POOR.value == "poor"
        assert DataQualityLevel.UNUSABLE.value == "unusable"


class TestIntakeFormData:
    """Test IntakeFormData model."""
    
    def test_minimal_valid_intake(self):
        """Test creating intake with minimal required fields."""
        intake = IntakeFormData(
            title="Test Research Project",
            research_question="What is the effect of X on Y in the population?",
            target_journal=TargetJournal.OTHER,
            paper_type=PaperType.FULL_PAPER,
            research_type=ResearchType.EMPIRICAL,
        )
        assert intake.title == "Test Research Project"
        assert intake.research_question == "What is the effect of X on Y in the population?"
        assert intake.target_journal == TargetJournal.OTHER
        assert intake.paper_type == PaperType.FULL_PAPER
    
    def test_full_intake_form(self):
        """Test creating intake with all fields."""
        intake = IntakeFormData(
            title="Complete Research Project",
            research_question="How does intervention A affect outcome B in population C?",
            target_journal=TargetJournal.MANAGEMENT_SCIENCE,
            paper_type=PaperType.SHORT_ARTICLE,
            research_type=ResearchType.EMPIRICAL,
            has_hypothesis=True,
            hypothesis="A positive relationship exists between A and B.",
            has_data=True,
            data_description="Panel data from 2010-2020",
            data_files=["data.csv"],
            data_sources="World Bank, IMF",
            key_variables="GDP, inflation, unemployment",
            methodology="Regression analysis with fixed effects",
            related_literature="Smith (2020), Jones (2021)",
            expected_contribution="New methodology for measuring X",
            deadline=date.today(),
            constraints="Word limit: 5000",
            additional_notes="Priority project",
        )
        assert intake.target_journal == TargetJournal.MANAGEMENT_SCIENCE
        assert intake.has_hypothesis is True
        assert intake.deadline == date.today()
    
    def test_title_validation(self):
        """Test title length validation."""
        with pytest.raises(ValueError):
            IntakeFormData(
                title="",  # Too short
                research_question="Valid question that is long enough?",
                target_journal=TargetJournal.OTHER,
                paper_type=PaperType.FULL_PAPER,
                research_type=ResearchType.EMPIRICAL,
            )
    
    def test_get_seed_literature_list(self):
        """Test seed literature parsing."""
        intake = IntakeFormData(
            title="Test Project Title",
            research_question="What is the relationship between X and Y in the market?",
            target_journal=TargetJournal.OTHER,
            paper_type=PaperType.FULL_PAPER,
            research_type=ResearchType.EMPIRICAL,
            related_literature="Smith (2020); Jones (2021); Brown (2022)",
        )
        seeds = intake.get_seed_literature_list()
        assert len(seeds) == 3
        assert "Smith (2020)" in seeds
    
    def test_get_key_variables_list(self):
        """Test key variables parsing."""
        intake = IntakeFormData(
            title="Test Project Title",
            research_question="What is the relationship between X and Y in the market?",
            target_journal=TargetJournal.OTHER,
            paper_type=PaperType.FULL_PAPER,
            research_type=ResearchType.EMPIRICAL,
            key_variables="GDP, inflation rate; unemployment",
        )
        variables = intake.get_key_variables_list()
        assert len(variables) == 3
        assert "GDP" in variables


class TestDataFile:
    """Test DataFile model."""
    
    def test_data_file_creation(self):
        """Test creating a DataFile."""
        df = DataFile(
            filename="test.csv",
            filepath=Path("/tmp/test.csv"),
            content_type="text/csv",
            size_bytes=1024,
        )
        assert df.filename == "test.csv"
        assert df.size_bytes == 1024
    
    def test_size_human(self):
        """Test human-readable size formatting."""
        df = DataFile(
            filename="test.csv",
            filepath=Path("/tmp/test.csv"),
            content_type="text/csv",
            size_bytes=1024 * 1024 * 5,  # 5 MB
        )
        assert "MB" in df.size_human


class TestColumnAnalysis:
    """Test ColumnAnalysis model."""
    
    def test_column_analysis_creation(self):
        """Test creating a ColumnAnalysis."""
        col = ColumnAnalysis(
            name="test_column",
            dtype=ColumnType.FLOAT,
            non_null_count=90,
            null_count=10,
            null_percentage=10.0,
            unique_count=100,
        )
        assert col.name == "test_column"
        assert col.dtype == ColumnType.FLOAT
        assert col.null_percentage == 10.0


class TestDataExplorationResult:
    """Test DataExplorationResult model."""
    
    def test_exploration_result_creation(self):
        """Test creating a DataExplorationResult."""
        result = DataExplorationResult(
            total_rows=1000,
            total_columns=10,
            columns=[],
            quality_score=0.85,
            quality_level=DataQualityLevel.GOOD,
            quality_issues=[],
            feasibility_assessment="Data suitable for analysis",
            feasibility_score=0.9,
        )
        assert result.total_rows == 1000
        assert result.quality_score == 0.85


class TestWorkflowState:
    """Test WorkflowState schema and utilities."""
    
    def test_create_initial_state(self):
        """Test creating initial workflow state."""
        state = create_initial_state()
        
        assert state["status"] == ResearchStatus.INTAKE_PENDING
        assert state["messages"] == []
        assert state["errors"] == []
        assert state["iteration_count"] == 0
    
    def test_create_initial_state_with_form_data(self):
        """Test creating state with initial form data."""
        form_data = {
            "title": "Test Project",
            "research_question": "Test question?",
        }
        state = create_initial_state(form_data=form_data)
        
        assert state["form_data"] == form_data
    
    def test_validate_state_for_intake(self):
        """Test validation for INTAKE node."""
        # Valid state
        state = create_initial_state(form_data={"title": "Test"})
        valid, errors = validate_state_for_node(state, "intake")
        assert valid is True
        assert len(errors) == 0
        
        # Invalid state (no form_data)
        state_no_form = create_initial_state()
        state_no_form["form_data"] = {}
        valid, errors = validate_state_for_node(state_no_form, "intake")
        assert valid is False
        assert "form_data" in errors[0].lower()
    
    def test_validate_state_for_data_explorer(self):
        """Test validation for DATA_EXPLORER node."""
        state = create_initial_state()
        state["status"] = ResearchStatus.INTAKE_COMPLETE
        # DATA_EXPLORER requires uploaded_data to be non-empty
        state["uploaded_data"] = [
            DataFile(
                filename="test.csv",
                filepath=Path("/tmp/test.csv"),
                content_type="text/csv",
                size_bytes=1024,
            )
        ]
        
        valid, errors = validate_state_for_node(state, "data_explorer")
        assert valid is True
    
    def test_validate_state_for_unknown_node(self):
        """Test validation for unknown node."""
        state = create_initial_state()
        valid, errors = validate_state_for_node(state, "unknown_node")
        assert valid is True  # Unknown nodes pass by default


class TestWorkflowError:
    """Test WorkflowError model."""
    
    def test_error_creation(self):
        """Test creating a WorkflowError."""
        error = WorkflowError(
            node="intake",
            category="validation",
            message="Missing required field",
            recoverable=True,
        )
        assert error.node == "intake"
        assert error.recoverable is True
        assert error.occurred_at is not None
    
    def test_error_with_details(self):
        """Test error with additional details."""
        error = WorkflowError(
            node="data_explorer",
            category="parsing",
            message="Failed to parse CSV",
            recoverable=True,
            details={"filename": "test.csv", "line": 42},
        )
        assert error.details["filename"] == "test.csv"


class TestResearchPlan:
    """Test ResearchPlan model."""
    
    def test_research_plan_creation(self):
        """Test creating a ResearchPlan."""
        plan = ResearchPlan(
            original_query="What is the causal effect of X on Y?",
            refined_query="What is the causal effect of X on Y controlling for Z?",
            sub_questions=["What is X?", "What is Y?", "How are they related?"],
            methodology="Panel regression with fixed effects",
            expected_sections=["Introduction", "Literature", "Methods", "Results"],
            success_criteria=["Clear findings", "Statistical significance"],
        )
        assert len(plan.sub_questions) == 3
        assert plan.methodology == "Panel regression with fixed effects"


class TestSearchResult:
    """Test SearchResult model."""
    
    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            query_id="q123",
            title="A Study of X and Y",
            url="https://example.com/paper",
            snippet="This paper examines the relationship between X and Y...",
            authors=["Smith, J.", "Jones, K."],
            published_date=date(2023, 1, 15),
            venue="Journal of Testing",
            relevance_score=0.95,
        )
        assert result.published_date.year == 2023
        assert result.relevance_score == 0.95


class TestCritique:
    """Test Critique and CritiqueItem models."""
    
    def test_critique_item_creation(self):
        """Test creating a CritiqueItem."""
        item = CritiqueItem(
            severity=CritiqueSeverity.MAJOR,
            category="methodology",
            location="methods",
            description="Sample size too small",
            suggestion="Increase sample or use bootstrapping",
        )
        assert item.severity == CritiqueSeverity.MAJOR
    
    def test_critique_creation(self):
        """Test creating a Critique."""
        critique = Critique(
            draft_version=1,
            items=[
                CritiqueItem(
                    severity=CritiqueSeverity.MINOR,
                    category="style",
                    location="introduction",
                    description="Missing context",
                    suggestion="Add background",
                ),
            ],
            overall_score=7.0,
            summary="Good draft with minor issues",
            recommendation="revise",
        )
        assert critique.draft_version == 1
        assert len(critique.items) == 1


class TestEvidenceItem:
    """Test EvidenceItem model."""
    
    def test_evidence_item_creation(self):
        """Test creating an EvidenceItem."""
        evidence = EvidenceItem(
            claim="X causes Y",
            source_id="src123",
            source_url="https://example.com/paper",
            locator="Page 5, paragraph 2",
            quote="Our findings show that X leads to Y...",
            strength=EvidenceStrength.STRONG,
            confidence=0.9,
        )
        assert evidence.strength == EvidenceStrength.STRONG


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
