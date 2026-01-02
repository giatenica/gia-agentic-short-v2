"""State management for GIA Agentic v2 workflow."""

from src.state.enums import (
    ResearchStatus,
    CritiqueSeverity,
    EvidenceStrength,
    PaperType,
    ResearchType,
)
from src.state.models import (
    IntakeFormData,
    DataFile,
    DataExplorationResult,
    ColumnAnalysis,
    QualityIssue,
    VariableMapping,
    ResearchPlan,
    SearchQuery,
    SearchResult,
    AnalysisResult,
    Finding,
    Theme,
    LiteratureSynthesis,
    DraftSection,
    ResearchDraft,
    CritiqueItem,
    Critique,
    EvidenceItem,
    WorkflowError,
)
from src.state.schema import WorkflowState

__all__ = [
    # Enums
    "ResearchStatus",
    "CritiqueSeverity",
    "EvidenceStrength",
    "PaperType",
    "ResearchType",
    # Models
    "IntakeFormData",
    "DataFile",
    "DataExplorationResult",
    "ColumnAnalysis",
    "QualityIssue",
    "VariableMapping",
    "ResearchPlan",
    "SearchQuery",
    "SearchResult",
    "AnalysisResult",
    "Finding",
    "Theme",
    "LiteratureSynthesis",
    "DraftSection",
    "ResearchDraft",
    "CritiqueItem",
    "Critique",
    "EvidenceItem",
    "WorkflowError",
    # Schema
    "WorkflowState",
]
