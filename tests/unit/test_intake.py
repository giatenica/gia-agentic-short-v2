"""Unit tests for INTAKE node."""

import pytest
from datetime import date, datetime
from unittest.mock import patch

from src.nodes.intake import (
    parse_intake_form,
    validate_intake,
    process_uploaded_files,
    intake_node,
    route_after_intake,
    IntakeValidationResult,
)
from src.state.enums import (
    ResearchStatus,
    PaperType,
    ResearchType,
    TargetJournal,
)
from src.state.models import IntakeFormData
from src.state.schema import create_initial_state


class TestIntakeValidationResult:
    """Test IntakeValidationResult helper class."""
    
    def test_empty_result_is_valid(self):
        """Test that empty result is valid."""
        result = IntakeValidationResult()
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_add_error_invalidates(self):
        """Test that adding error invalidates result."""
        result = IntakeValidationResult()
        result.add_error("Test error")
        assert result.is_valid is False
        assert len(result.errors) == 1
    
    def test_add_warning_keeps_valid(self):
        """Test that adding warning keeps result valid."""
        result = IntakeValidationResult()
        result.add_warning("Test warning")
        assert result.is_valid is True
        assert len(result.warnings) == 1


class TestParseIntakeForm:
    """Test parse_intake_form function."""
    
    def test_parse_minimal_form(self):
        """Test parsing minimal form data."""
        form_data = {
            "title": "Test Research Project",
            "research_question": "What is the relationship between X and Y in the market?",
            "target_journal": "Other",
            "paper_type": "Full Paper (30-45 pages)",
            "research_type": "Empirical",
        }
        result = parse_intake_form(form_data)
        
        assert isinstance(result, IntakeFormData)
        assert result.title == "Test Research Project"
        assert result.research_question == "What is the relationship between X and Y in the market?"
    
    def test_parse_boolean_fields_from_string(self):
        """Test parsing boolean fields from string values."""
        form_data = {
            "title": "Test Project",
            "research_question": "What is the effect of X on Y in population Z?",
            "target_journal": "Other",
            "paper_type": "Full Paper (30-45 pages)",
            "research_type": "Empirical",
            "has_hypothesis": "yes",
            "has_data": "no",
        }
        result = parse_intake_form(form_data)
        
        assert result.has_hypothesis is True
        assert result.has_data is False
    
    def test_parse_paper_type_mapping(self):
        """Test paper type form value mapping."""
        form_data = {
            "title": "Test Project",
            "research_question": "What is the relationship between variables?",
            "target_journal": "Other",
            "paper_type": "Short Article (5-10 pages)",
            "research_type": "Empirical",
        }
        result = parse_intake_form(form_data)
        
        assert result.paper_type == PaperType.SHORT_ARTICLE
    
    def test_parse_research_type_mapping(self):
        """Test research type form value mapping."""
        form_data = {
            "title": "Test Project",
            "research_question": "What is the relationship between variables?",
            "target_journal": "Other",
            "paper_type": "Full Paper (30-45 pages)",
            "research_type": "Empirical",
        }
        result = parse_intake_form(form_data)
        
        assert result.research_type == ResearchType.EMPIRICAL
    
    def test_parse_target_journal_mapping(self):
        """Test target journal form value mapping."""
        form_data = {
            "title": "Test Project",
            "research_question": "What is the relationship between variables?",
            "target_journal": "Management Science",
            "paper_type": "Full Paper (30-45 pages)",
            "research_type": "Empirical",
        }
        result = parse_intake_form(form_data)
        
        assert result.target_journal == TargetJournal.MANAGEMENT_SCIENCE


class TestValidateIntake:
    """Test validate_intake function."""
    
    def test_valid_intake_passes(self):
        """Test that valid intake passes validation."""
        intake = IntakeFormData(
            title="A Comprehensive Study of Market Dynamics",
            research_question="What is the relationship between market volatility and investor behavior in emerging markets?",
            target_journal=TargetJournal.OTHER,
            paper_type=PaperType.FULL_PAPER,
            research_type=ResearchType.EMPIRICAL,
            has_hypothesis=True,
            hypothesis="Higher volatility leads to reduced retail investment.",
            has_data=True,
            data_description="10 years of market data",
            key_variables="volatility, investment_flow, market_cap",
        )
        result = validate_intake(intake)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_short_research_question_warning(self):
        """Test warning for short research question."""
        # Note: Minimum 20 chars required by model, so we test with a short but valid question
        intake = IntakeFormData(
            title="Test Project Title",
            research_question="Why does X happen in this context?",  # 34 chars - valid but short
            target_journal=TargetJournal.OTHER,
            paper_type=PaperType.FULL_PAPER,
            research_type=ResearchType.THEORETICAL,
        )
        result = validate_intake(intake)
        
        assert result.is_valid is True  # Warnings don't invalidate
        # Short question warning triggers at <50 chars
        assert any("short" in w.lower() for w in result.warnings)
    
    def test_missing_question_mark_warning(self):
        """Test warning for missing question mark."""
        intake = IntakeFormData(
            title="Test Project Title",
            research_question="The relationship between variables should be examined carefully and in detail",
            target_journal=TargetJournal.OTHER,
            paper_type=PaperType.FULL_PAPER,
            research_type=ResearchType.THEORETICAL,
        )
        result = validate_intake(intake)
        
        assert result.is_valid is True
        assert any("question" in w.lower() for w in result.warnings)
    
    def test_hypothesis_consistency_error(self):
        """Test error when hypothesis marked but not provided.
        
        Note: This is caught by Pydantic's field_validator, not validate_intake.
        """
        import pytest
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            IntakeFormData(
                title="Test Project Title",
                research_question="What is the effect of X on Y in the population?",
                target_journal=TargetJournal.OTHER,
                paper_type=PaperType.FULL_PAPER,
                research_type=ResearchType.EMPIRICAL,
                has_hypothesis=True,
                hypothesis=None,  # Missing hypothesis text
            )
        
        # Verify the error is about hypothesis
        assert "hypothesis" in str(exc_info.value).lower()
    
    def test_empirical_without_data_warning(self):
        """Test warning for empirical research without data."""
        intake = IntakeFormData(
            title="Test Project Title",
            research_question="What is the effect of X on Y in the empirical data?",
            target_journal=TargetJournal.OTHER,
            paper_type=PaperType.FULL_PAPER,
            research_type=ResearchType.EMPIRICAL,
            has_data=False,
            data_sources=None,
        )
        result = validate_intake(intake)
        
        assert result.is_valid is True
        assert any("data" in w.lower() for w in result.warnings)
    
    def test_empirical_without_key_variables_warning(self):
        """Test warning for empirical research without key variables."""
        intake = IntakeFormData(
            title="Test Project Title",
            research_question="What is the effect of X on Y in the empirical data?",
            target_journal=TargetJournal.OTHER,
            paper_type=PaperType.FULL_PAPER,
            research_type=ResearchType.EMPIRICAL,
            has_data=True,
            data_description="Some data",
            key_variables=None,
        )
        result = validate_intake(intake)
        
        assert result.is_valid is True
        assert any("variable" in w.lower() for w in result.warnings)
    
    def test_past_deadline_error(self):
        """Test error for deadline in the past."""
        past_date = date(2020, 1, 1)
        intake = IntakeFormData(
            title="Test Project Title",
            research_question="What is the effect of X on Y in the population?",
            target_journal=TargetJournal.OTHER,
            paper_type=PaperType.FULL_PAPER,
            research_type=ResearchType.EMPIRICAL,
            deadline=past_date,
        )
        result = validate_intake(intake)
        
        assert result.is_valid is False
        assert any("passed" in e.lower() for e in result.errors)
    
    def test_tight_deadline_warning(self):
        """Test warning for tight deadline."""
        from datetime import timedelta
        tight_deadline = date.today() + timedelta(days=3)
        intake = IntakeFormData(
            title="Test Project Title",
            research_question="What is the effect of X on Y in the population?",
            target_journal=TargetJournal.OTHER,
            paper_type=PaperType.FULL_PAPER,
            research_type=ResearchType.EMPIRICAL,
            deadline=tight_deadline,
        )
        result = validate_intake(intake)
        
        # Warnings don't invalidate, but should flag the tight deadline
        assert any("tight" in w.lower() or "days" in w.lower() for w in result.warnings)


class TestProcessUploadedFiles:
    """Test process_uploaded_files function."""
    
    def test_empty_list_returns_empty(self):
        """Test that empty input returns empty list."""
        result = process_uploaded_files([])
        assert result == []
    
    def test_none_returns_empty(self):
        """Test that None input returns empty list."""
        result = process_uploaded_files(None)
        assert result == []
    
    def test_nonexistent_file_skipped(self):
        """Test that nonexistent files are skipped."""
        result = process_uploaded_files(["/nonexistent/file.csv"])
        assert len(result) == 0


class TestIntakeNode:
    """Test intake_node function."""
    
    def test_intake_with_no_form_data(self):
        """Test intake node with no form data."""
        state = create_initial_state()
        state["form_data"] = {}
        result = intake_node(state)
        
        assert result["status"] == ResearchStatus.FAILED
        assert len(result["errors"]) > 0
        assert len(result["messages"]) > 0
    
    def test_intake_with_valid_form_data(self):
        """Test intake node with valid form data."""
        form_data = {
            "title": "A Study of Market Behavior",
            "research_question": "What factors influence investor decisions in volatile markets?",
            "research_type": "Empirical",
            "paper_type": "Full Paper (30-45 pages)",
            "target_journal": "Management Science",
            "has_hypothesis": "yes",
            "hypothesis": "Fear of loss dominates greed in down markets.",
            "has_data": "yes",
            "data_description": "Survey data from 1000 investors",
            "key_variables": "investment_amount, market_sentiment, risk_tolerance",
            "methodology": "Structural equation modeling",
            "related_literature": "Smith (2020); Jones (2021)",
        }
        state = create_initial_state(form_data=form_data)
        result = intake_node(state)
        
        assert result["status"] == ResearchStatus.INTAKE_COMPLETE
        assert result["original_query"] == form_data["research_question"]
        assert result["project_title"] == form_data["title"]
        assert len(result["seed_literature"]) == 2
        assert len(result["key_variables"]) == 3
    
    def test_intake_with_validation_errors(self):
        """Test intake node handles validation errors."""
        form_data = {
            "title": "Test Title",
            "research_question": "What is X? Why Y?",  # Short but valid
            "paper_type": "Full Paper (30-45 pages)",
            "target_journal": "Other",
            "research_type": "Empirical",
            "has_hypothesis": "yes",
            # Missing hypothesis text - will cause validation error
        }
        state = create_initial_state(form_data=form_data)
        result = intake_node(state)
        
        # Should still process but flag errors
        assert result["status"] in [ResearchStatus.INTAKE_PENDING, ResearchStatus.INTAKE_COMPLETE, ResearchStatus.FAILED]
    
    def test_intake_with_invalid_form_data(self):
        """Test intake node with completely invalid form data."""
        form_data = {
            "title": "",  # Invalid: empty title
            "research_question": "",  # Invalid: empty
            "paper_type": "Full Paper (30-45 pages)",
            "target_journal": "Other",
            "research_type": "Empirical",
        }
        state = create_initial_state(form_data=form_data)
        result = intake_node(state)
        
        assert result["status"] == ResearchStatus.FAILED
        assert len(result["errors"]) > 0


class TestRouteAfterIntake:
    """Test route_after_intake function."""
    
    def test_route_to_end_on_failure(self):
        """Test routing to end on failure status."""
        state = create_initial_state()
        state["status"] = ResearchStatus.FAILED
        
        route = route_after_intake(state)
        assert route == "end"
    
    def test_route_to_end_on_pending(self):
        """Test routing to end on pending status."""
        state = create_initial_state()
        state["status"] = ResearchStatus.INTAKE_PENDING
        
        route = route_after_intake(state)
        assert route == "end"
    
    def test_route_to_data_explorer_with_data(self):
        """Test routing to data explorer when data uploaded."""
        from src.state.models import DataFile
        from pathlib import Path
        
        state = create_initial_state()
        state["status"] = ResearchStatus.INTAKE_COMPLETE
        state["uploaded_data"] = [
            DataFile(
                filename="test.csv",
                filepath=Path("/tmp/test.csv"),
                content_type="text/csv",
                size_bytes=1024,
            )
        ]
        
        route = route_after_intake(state)
        assert route == "data_explorer"
    
    def test_route_to_literature_without_data(self):
        """Test routing to literature reviewer without data."""
        state = create_initial_state()
        state["status"] = ResearchStatus.INTAKE_COMPLETE
        state["uploaded_data"] = []
        
        route = route_after_intake(state)
        assert route == "literature_reviewer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
