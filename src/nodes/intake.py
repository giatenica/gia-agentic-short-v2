"""INTAKE node for processing research intake form submissions.

This node is the entry point for all research projects. It:
1. Parses raw form data into structured IntakeFormData
2. Validates all required fields
3. Extracts key variables and seed literature
4. Prepares the initial workflow state

Uses HITL interrupt if validation errors are found.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from src.state.enums import ResearchStatus, PaperType, ResearchType, TargetJournal
from src.state.models import IntakeFormData, DataFile, WorkflowError
from src.state.schema import WorkflowState


class IntakeValidationResult:
    """Result of intake form validation."""
    
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.is_valid: bool = True
    
    def add_error(self, message: str) -> None:
        """Add a validation error."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)


def parse_intake_form(form_data: dict[str, Any]) -> IntakeFormData:
    """
    Parse raw form data into a validated IntakeFormData model.
    
    Args:
        form_data: Raw form submission dictionary.
        
    Returns:
        Validated IntakeFormData instance.
        
    Raises:
        ValidationError: If required fields are missing or invalid.
    """
    # Map form field names to model field names
    field_mapping = {
        "title": "title",
        "research_question": "research_question",
        "target_journal": "target_journal",
        "paper_type": "paper_type",
        "research_type": "research_type",
        "has_hypothesis": "has_hypothesis",
        "hypothesis": "hypothesis",
        "has_data": "has_data",
        "data_description": "data_description",
        "data_files": "data_files",
        "data_sources": "data_sources",
        "key_variables": "key_variables",
        "methodology": "methodology",
        "related_literature": "related_literature",
        "expected_contribution": "expected_contribution",
        "deadline": "deadline",
        "constraints": "constraints",
        "additional_notes": "additional_notes",
    }
    
    # Transform form data
    parsed_data = {}
    for form_key, model_key in field_mapping.items():
        if form_key in form_data:
            value = form_data[form_key]
            
            # Handle boolean fields
            if model_key in ("has_hypothesis", "has_data"):
                if isinstance(value, str):
                    value = value.lower() == "yes"
                else:
                    value = bool(value)
            
            # Handle enum fields
            if model_key == "target_journal":
                try:
                    value = TargetJournal(value)
                except ValueError:
                    value = TargetJournal.OTHER
                    
            if model_key == "paper_type":
                # Map form values to enum values
                paper_type_map = {
                    "Short Article (5-10 pages)": PaperType.SHORT_ARTICLE,
                    "Full Paper (30-45 pages)": PaperType.FULL_PAPER,
                    "Working Paper": PaperType.WORKING_PAPER,
                }
                value = paper_type_map.get(value, PaperType.FULL_PAPER)
                
            if model_key == "research_type":
                # Map form values to enum values
                research_type_map = {
                    "Empirical": ResearchType.EMPIRICAL,
                    "Theoretical": ResearchType.THEORETICAL,
                    "Mixed": ResearchType.MIXED,
                    "Literature Review": ResearchType.LITERATURE_REVIEW,
                    "Experimental": ResearchType.EXPERIMENTAL,
                }
                value = research_type_map.get(value, ResearchType.EMPIRICAL)
            
            parsed_data[model_key] = value
    
    return IntakeFormData(**parsed_data)


def validate_intake(intake_data: IntakeFormData) -> IntakeValidationResult:
    """
    Validate intake form data beyond basic Pydantic validation.
    
    Performs semantic validation:
    - Research question quality
    - Hypothesis consistency
    - Data availability vs research type
    - Deadline feasibility
    
    Args:
        intake_data: Parsed intake form data.
        
    Returns:
        IntakeValidationResult with any errors/warnings.
    """
    result = IntakeValidationResult()
    
    # Check research question quality
    if len(intake_data.research_question) < 50:
        result.add_warning(
            "Research question seems short. Consider providing more detail."
        )
    
    if "?" not in intake_data.research_question:
        result.add_warning(
            "Research question should typically be phrased as a question."
        )
    
    # Check hypothesis consistency
    if intake_data.has_hypothesis and not intake_data.hypothesis:
        result.add_error(
            "Hypothesis marked as provided but no hypothesis text found."
        )
    
    # Check data availability for empirical research
    if intake_data.research_type == ResearchType.EMPIRICAL:
        if not intake_data.has_data and not intake_data.data_sources:
            result.add_warning(
                "Empirical research selected but no data or data sources specified. "
                "Consider adding data sources or uploading data."
            )
    
    # Check for key variables in empirical research
    if intake_data.research_type == ResearchType.EMPIRICAL:
        if not intake_data.key_variables:
            result.add_warning(
                "No key variables specified for empirical research. "
                "Consider specifying dependent and independent variables."
            )
    
    # Check deadline feasibility
    if intake_data.deadline:
        from datetime import date
        days_until = (intake_data.deadline - date.today()).days
        if days_until < 7:
            result.add_warning(
                f"Deadline is in {days_until} days. This may be tight for thorough research."
            )
        if days_until < 0:
            result.add_error(
                "Deadline has already passed."
            )
    
    return result


def process_uploaded_files(file_paths: list[str] | None) -> list[DataFile]:
    """
    Process uploaded file paths into DataFile objects.
    
    Args:
        file_paths: List of file path strings from form.
        
    Returns:
        List of DataFile objects with metadata.
    """
    if not file_paths:
        return []
    
    # Supported data file extensions
    supported_extensions = {
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".json": "application/json",
        ".parquet": "application/octet-stream",
        ".dta": "application/x-stata",
        ".sav": "application/x-spss-sav",
        ".zip": "application/zip",
    }
    
    data_files = []
    
    def add_file(file_path: Path) -> None:
        """Add a file to data_files if it's a supported type."""
        if file_path.suffix.lower() in supported_extensions:
            content_type = supported_extensions[file_path.suffix.lower()]
            data_files.append(DataFile(
                filename=file_path.name,
                filepath=file_path,
                content_type=content_type,
                size_bytes=file_path.stat().st_size,
            ))
    
    for filepath in file_paths:
        path = Path(filepath)
        if path.exists():
            if path.is_dir():
                # Recursively find all data files in directory
                for root, dirs, files in os.walk(path):
                    # Skip hidden directories
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    for filename in files:
                        if not filename.startswith('.'):
                            file_path = Path(root) / filename
                            add_file(file_path)
            else:
                # Single file
                add_file(path)
    
    return data_files


def intake_node(state: WorkflowState) -> dict[str, Any]:
    """
    INTAKE node: Process intake form submission and validate inputs.
    
    This is the entry point for the research workflow. It:
    1. Parses raw form data into structured IntakeFormData
    2. Validates the input
    3. Processes any uploaded files
    4. Returns the initial workflow state fields
    
    Args:
        state: Current workflow state (must contain form_data).
        
    Returns:
        Dict with state updates including parsed intake fields.
        
    Note:
        Uses HITL interrupt via LangGraph if validation errors are critical.
        For this implementation, errors are collected and stored in state.
    """
    form_data = state.get("form_data", {})
    
    if not form_data:
        # No form data - return error state
        return {
            "status": ResearchStatus.FAILED,
            "errors": [WorkflowError(
                node="intake",
                category="validation",
                message="No form data provided",
                recoverable=False,
            )],
            "messages": [AIMessage(
                content="Error: No intake form data provided. Please submit the research intake form."
            )],
        }
    
    try:
        # Parse form data
        intake_data = parse_intake_form(form_data)
        
        # Validate
        validation = validate_intake(intake_data)
        
        # Process uploaded files
        data_files = process_uploaded_files(intake_data.data_files)
        
        # Build state updates
        updates: dict[str, Any] = {
            "original_query": intake_data.research_question,
            "project_title": intake_data.title,
            "target_journal": intake_data.target_journal.value,
            "paper_type": intake_data.paper_type.value,
            "research_type": intake_data.research_type.value,
            "user_hypothesis": intake_data.hypothesis,
            "proposed_methodology": intake_data.methodology,
            "seed_literature": intake_data.get_seed_literature_list(),
            "expected_contribution": intake_data.expected_contribution,
            "constraints": intake_data.constraints,
            "deadline": intake_data.deadline,
            "uploaded_data": data_files,
            "data_context": intake_data.data_description,
            "key_variables": intake_data.get_key_variables_list(),
            "updated_at": datetime.now(timezone.utc),
        }
        
        # Handle validation results
        errors = []
        if not validation.is_valid:
            for error_msg in validation.errors:
                errors.append(WorkflowError(
                    node="intake",
                    category="validation",
                    message=error_msg,
                    recoverable=True,
                ))
        
        # Determine status
        if validation.is_valid:
            status = ResearchStatus.INTAKE_COMPLETE
            message_content = (
                f"Research project '{intake_data.title}' initialized successfully.\n"
                f"Research question: {intake_data.research_question[:100]}...\n"
                f"Type: {intake_data.research_type.value}\n"
                f"Target journal: {intake_data.target_journal.value}"
            )
            if data_files:
                message_content += f"\n{len(data_files)} data file(s) uploaded."
            if validation.warnings:
                message_content += f"\n\nWarnings:\n" + "\n".join(f"- {w}" for w in validation.warnings)
        else:
            status = ResearchStatus.INTAKE_PENDING
            message_content = (
                "Intake validation found issues:\n" +
                "\n".join(f"- {e}" for e in validation.errors)
            )
            if validation.warnings:
                message_content += "\n\nWarnings:\n" + "\n".join(f"- {w}" for w in validation.warnings)
        
        updates["status"] = status
        updates["errors"] = errors
        updates["messages"] = [AIMessage(content=message_content)]
        updates["checkpoints"] = [f"{datetime.now(timezone.utc).isoformat()}: Intake processing complete"]
        
        return updates
        
    except ValidationError as e:
        # Pydantic validation failed
        error_messages = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
        return {
            "status": ResearchStatus.FAILED,
            "errors": [WorkflowError(
                node="intake",
                category="validation",
                message=f"Form validation failed: {'; '.join(error_messages)}",
                recoverable=True,
                details={"validation_errors": e.errors()},
            )],
            "messages": [AIMessage(
                content=f"Form validation failed:\n" + "\n".join(f"- {m}" for m in error_messages)
            )],
        }
    except Exception as e:
        # Unexpected error
        return {
            "status": ResearchStatus.FAILED,
            "errors": [WorkflowError(
                node="intake",
                category="internal",
                message=f"Unexpected error in intake processing: {str(e)}",
                recoverable=False,
            )],
            "messages": [AIMessage(
                content=f"An unexpected error occurred during intake processing: {str(e)}"
            )],
        }


def route_after_intake(state: WorkflowState) -> str:
    """
    Route after INTAKE node based on state.
    
    Routes to:
    - "data_explorer" if data files were uploaded
    - "literature_reviewer" otherwise
    
    Args:
        state: Current workflow state.
        
    Returns:
        Name of next node.
    """
    # Check for errors first
    if state.get("status") == ResearchStatus.FAILED:
        return "__end__"
    
    if state.get("status") == ResearchStatus.INTAKE_PENDING:
        return "__end__"  # Need to fix validation errors
    
    # Route based on data presence
    if state.get("uploaded_data"):
        return "data_explorer"
    
    return "literature_reviewer"
