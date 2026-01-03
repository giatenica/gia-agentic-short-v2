"""Pydantic models for GIA Agentic v2 workflow state.

These models define the data structures used throughout the research workflow,
from intake form processing through final output generation.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from src.state.enums import (
    CritiqueSeverity,
    EvidenceStrength,
    PaperType,
    ResearchType,
    TargetJournal,
    DataQualityLevel,
    ColumnType,
    MethodologyType,
    AnalysisApproach,
    PlanApprovalStatus,
    FindingType,
    AnalysisStatus,
    StatisticalTestType,
    ConceptType,
    RelationshipType,
    PropositionStatus,
)


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


# =============================================================================
# Intake Form Models
# =============================================================================


class IntakeFormData(BaseModel):
    """Validated intake form submission data.
    
    Maps directly to the HTML form fields in public/research_intake_form.html
    """
    
    # Basic Information (required)
    title: str = Field(
        ..., 
        min_length=5, 
        max_length=500,
        description="Project title (working title for research paper)"
    )
    research_question: str = Field(
        ..., 
        min_length=20, 
        max_length=2000,
        description="The specific research question to investigate"
    )
    target_journal: TargetJournal = Field(
        ...,
        description="Target journal for publication"
    )
    paper_type: PaperType = Field(
        ...,
        description="Type of paper (short article, full paper, etc.)"
    )
    research_type: ResearchType = Field(
        ...,
        description="Research methodology type"
    )
    
    # Hypothesis (optional)
    has_hypothesis: bool = Field(
        default=False,
        description="Whether user has a hypothesis"
    )
    hypothesis: str | None = Field(
        default=None,
        max_length=2000,
        description="User's hypothesis if provided"
    )
    
    # Data (optional)
    has_data: bool = Field(
        default=False,
        description="Whether user has existing data"
    )
    data_description: str | None = Field(
        default=None,
        max_length=2000,
        description="Description of user's data"
    )
    data_files: list[str] | None = Field(
        default=None,
        description="List of uploaded data file paths"
    )
    data_sources: str | None = Field(
        default=None,
        max_length=1000,
        description="Planned or used data sources"
    )
    key_variables: str | None = Field(
        default=None,
        max_length=1000,
        description="Key dependent and independent variables"
    )
    
    # Methodology (optional)
    methodology: str | None = Field(
        default=None,
        max_length=2000,
        description="Proposed methodology"
    )
    related_literature: str | None = Field(
        default=None,
        max_length=2000,
        description="Related literature and key papers"
    )
    
    # Contribution and Timeline (optional)
    expected_contribution: str | None = Field(
        default=None,
        max_length=2000,
        description="Expected contribution to the field"
    )
    deadline: date | None = Field(
        default=None,
        description="Target deadline for completion"
    )
    constraints: str | None = Field(
        default=None,
        max_length=1000,
        description="Page limits, specific requirements, etc."
    )
    additional_notes: str | None = Field(
        default=None,
        max_length=2000,
        description="Any additional information"
    )
    
    @field_validator("hypothesis")
    @classmethod
    def validate_hypothesis(cls, v: str | None, info) -> str | None:
        """Validate hypothesis is provided if has_hypothesis is True."""
        if info.data.get("has_hypothesis") and not v:
            raise ValueError("Hypothesis required when has_hypothesis is True")
        return v
    
    @field_validator("key_variables")
    @classmethod
    def parse_key_variables(cls, v: str | None) -> str | None:
        """Clean up key variables string."""
        if v:
            return v.strip()
        return v

    def get_key_variables_list(self) -> list[str]:
        """Parse key_variables string into a list."""
        if not self.key_variables:
            return []
        # Split on common delimiters
        import re
        variables = re.split(r"[,;\n]+", self.key_variables)
        return [v.strip() for v in variables if v.strip()]
    
    def get_seed_literature_list(self) -> list[str]:
        """Parse related_literature into individual references."""
        if not self.related_literature:
            return []
        # Split on common patterns (periods followed by capital letters, or semicolons)
        import re
        refs = re.split(r"(?<=[.)])\s+(?=[A-Z])|;\s*", self.related_literature)
        return [r.strip() for r in refs if r.strip()]


# =============================================================================
# Data Exploration Models
# =============================================================================


class DataFile(BaseModel):
    """Metadata about an uploaded data file."""
    
    file_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique identifier for the file"
    )
    filename: str = Field(..., description="Original filename")
    filepath: Path = Field(..., description="Path to the file on disk")
    content_type: str = Field(
        default="application/octet-stream",
        description="MIME type of the file"
    )
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    uploaded_at: datetime = Field(
        default_factory=_utc_now,
        description="Upload timestamp"
    )
    
    @property
    def size_human(self) -> str:
        """Human-readable file size."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    @property
    def extension(self) -> str:
        """File extension without dot."""
        return self.filepath.suffix.lstrip(".").lower()


class ColumnAnalysis(BaseModel):
    """Analysis of a single column in a data file."""
    
    name: str = Field(..., description="Column name")
    dtype: ColumnType = Field(..., description="Detected data type")
    non_null_count: int = Field(..., ge=0, description="Number of non-null values")
    null_count: int = Field(..., ge=0, description="Number of null/missing values")
    null_percentage: float = Field(..., ge=0, le=100, description="Percentage missing")
    unique_count: int = Field(..., ge=0, description="Number of unique values")
    
    # Numeric statistics (only for numeric columns)
    mean: float | None = Field(default=None, description="Mean value")
    std: float | None = Field(default=None, description="Standard deviation")
    min_value: float | None = Field(default=None, description="Minimum value")
    max_value: float | None = Field(default=None, description="Maximum value")
    q25: float | None = Field(default=None, description="25th percentile")
    median: float | None = Field(default=None, description="Median (50th percentile)")
    q75: float | None = Field(default=None, description="75th percentile")
    
    # Date statistics (only for date columns)
    date_min: date | None = Field(default=None, description="Earliest date")
    date_max: date | None = Field(default=None, description="Latest date")
    
    # Categorical statistics (only for categorical columns)
    top_values: list[tuple[str, int]] | None = Field(
        default=None, 
        description="Most frequent values with counts"
    )
    
    # Quality indicators
    has_outliers: bool = Field(default=False, description="Whether outliers detected")
    outlier_count: int = Field(default=0, ge=0, description="Number of outliers")


class QualityIssue(BaseModel):
    """A data quality issue detected during exploration."""
    
    issue_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique issue identifier"
    )
    severity: CritiqueSeverity = Field(..., description="Issue severity")
    category: str = Field(..., description="Issue category (missing, outlier, etc.)")
    column: str | None = Field(default=None, description="Affected column if applicable")
    description: str = Field(..., description="Description of the issue")
    suggestion: str | None = Field(default=None, description="Suggested fix")
    affected_rows: int | None = Field(default=None, ge=0, description="Number of affected rows")


class VariableMapping(BaseModel):
    """Mapping between user-specified variables and detected columns."""
    
    user_variable: str = Field(..., description="Variable name from user input")
    matched_column: str | None = Field(
        default=None, 
        description="Matched column name from data"
    )
    confidence: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="Confidence score for the match"
    )
    match_reason: str | None = Field(
        default=None,
        description="Reason for the match (exact, fuzzy, semantic)"
    )


class DataExplorationResult(BaseModel):
    """Complete results from data exploration phase."""
    
    exploration_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique exploration identifier"
    )
    explored_at: datetime = Field(
        default_factory=_utc_now,
        description="Exploration timestamp"
    )
    
    # Files analyzed
    files_analyzed: list[DataFile] = Field(
        default_factory=list,
        description="Files that were analyzed"
    )
    
    # Schema information
    total_rows: int = Field(default=0, ge=0, description="Total rows across all files")
    total_columns: int = Field(default=0, ge=0, description="Total columns across all files")
    columns: list[ColumnAnalysis] = Field(
        default_factory=list,
        description="Analysis of each column"
    )
    
    # Variable mapping
    variable_mappings: list[VariableMapping] = Field(
        default_factory=list,
        description="Mappings from user variables to columns"
    )
    
    # Quality assessment
    quality_level: DataQualityLevel = Field(
        default=DataQualityLevel.NOT_ASSESSED,
        description="Overall data quality level"
    )
    quality_issues: list[QualityIssue] = Field(
        default_factory=list,
        description="Detected quality issues"
    )
    quality_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="Numeric quality score"
    )
    
    # Feasibility assessment
    feasibility_assessment: str = Field(
        default="",
        description="Assessment of whether research question can be answered"
    )
    feasibility_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Feasibility score"
    )
    feasibility_issues: list[str] = Field(
        default_factory=list,
        description="Issues affecting feasibility"
    )
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if there are any critical quality issues."""
        return any(
            issue.severity == CritiqueSeverity.CRITICAL 
            for issue in self.quality_issues
        )
    
    @property
    def issue_count_by_severity(self) -> dict[CritiqueSeverity, int]:
        """Count issues by severity level."""
        counts = {severity: 0 for severity in CritiqueSeverity}
        for issue in self.quality_issues:
            counts[issue.severity] += 1
        return counts


# =============================================================================
# Research Planning Models
# =============================================================================


class SearchQuery(BaseModel):
    """A search query to be executed."""
    
    query_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique query identifier"
    )
    query_text: str = Field(..., min_length=3, description="Search query text")
    source_type: str = Field(
        default="web",
        description="Type of source (web, academic, news, etc.)"
    )
    priority: int = Field(default=1, ge=1, le=5, description="Query priority (1=highest)")
    status: str = Field(default="pending", description="Query status")
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="Creation timestamp"
    )


class ResearchPlan(BaseModel):
    """Structured research plan generated by the PLANNER node.
    
    This model captures the complete research design including methodology,
    analysis approach, expected outputs, and success criteria. It is generated
    after gap identification and requires human approval before execution.
    """
    
    plan_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique plan identifier"
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="Creation timestamp"
    )
    revised_at: datetime | None = Field(
        default=None,
        description="Last revision timestamp"
    )
    
    # Research question (potentially refined from original)
    original_query: str = Field(..., description="Original research question")
    refined_query: str | None = Field(
        default=None,
        description="Refined research question after gap analysis"
    )
    
    # Gap being addressed
    target_gap: str = Field(
        default="",
        description="The primary research gap this plan addresses"
    )
    gap_type: str = Field(
        default="",
        description="Type of gap (methodological, empirical, theoretical)"
    )
    
    # Decomposition
    sub_questions: list[str] = Field(
        default_factory=list,
        description="Decomposed sub-questions to answer"
    )
    
    # Search strategy
    search_queries: list[SearchQuery] = Field(
        default_factory=list,
        description="Search queries to execute"
    )
    
    # Methodology - Enhanced for Sprint 4
    methodology_type: MethodologyType | None = Field(
        default=None,
        description="Type of methodology to use"
    )
    methodology: str = Field(default="", description="Research methodology description")
    methodology_justification: str = Field(
        default="",
        description="Why this methodology was chosen"
    )
    methodology_precedents: list[str] = Field(
        default_factory=list,
        description="Literature precedents supporting methodology choice"
    )
    
    # Analysis Design - New for Sprint 4
    analysis_approach: AnalysisApproach | None = Field(
        default=None,
        description="Primary analysis approach"
    )
    analysis_design: str = Field(
        default="",
        description="Detailed analysis design description"
    )
    statistical_tests: list[str] = Field(
        default_factory=list,
        description="Specific statistical tests to use"
    )
    key_variables: list[str] = Field(
        default_factory=list,
        description="Key dependent and independent variables"
    )
    control_variables: list[str] = Field(
        default_factory=list,
        description="Control variables to include"
    )
    data_requirements: list[str] = Field(
        default_factory=list,
        description="Data requirements for analysis"
    )
    
    # Expected outputs
    expected_sections: list[str] = Field(
        default_factory=list,
        description="Expected sections in final output"
    )
    expected_tables: list[str] = Field(
        default_factory=list,
        description="Expected tables to produce"
    )
    expected_figures: list[str] = Field(
        default_factory=list,
        description="Expected figures to produce"
    )
    
    # Success criteria
    success_criteria: list[str] = Field(
        default_factory=list,
        description="Criteria for successful completion"
    )
    hypothesis: str | None = Field(
        default=None,
        description="Primary hypothesis to test"
    )
    alternative_hypotheses: list[str] = Field(
        default_factory=list,
        description="Alternative hypotheses"
    )
    
    # Contribution statement
    contribution_statement: str = Field(
        default="",
        description="Clear statement of the paper's contribution"
    )
    
    # Approval status - New for Sprint 4
    approval_status: PlanApprovalStatus = Field(
        default=PlanApprovalStatus.PENDING,
        description="Current approval status"
    )
    approval_notes: str = Field(
        default="",
        description="Notes from approval process"
    )
    revision_count: int = Field(
        default=0,
        ge=0,
        description="Number of times plan has been revised"
    )
    
    # Feasibility assessment - New for Sprint 4
    feasibility_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall feasibility score"
    )
    feasibility_notes: str = Field(
        default="",
        description="Notes on feasibility assessment"
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations of the research design"
    )


# =============================================================================
# Search and Analysis Models
# =============================================================================


class SearchResult(BaseModel):
    """A single search result from any source."""
    
    result_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique result identifier"
    )
    query_id: str = Field(..., description="ID of query that produced this result")
    
    # Content
    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Source URL")
    snippet: str = Field(default="", description="Text snippet or abstract")
    full_content: str | None = Field(default=None, description="Full content if retrieved")
    
    # Metadata
    source_type: str = Field(default="web", description="Type of source")
    published_date: date | None = Field(default=None, description="Publication date")
    authors: list[str] = Field(default_factory=list, description="Author names")
    
    # Academic metadata
    citation_count: int | None = Field(default=None, ge=0, description="Citation count")
    venue: str | None = Field(default=None, description="Journal/conference name")
    doi: str | None = Field(default=None, description="DOI if available")
    
    # Quality indicators
    relevance_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="Relevance to query"
    )
    retrieved_at: datetime = Field(
        default_factory=_utc_now,
        description="Retrieval timestamp"
    )


class Finding(BaseModel):
    """A key finding extracted from analysis."""
    
    finding_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique finding identifier"
    )
    statement: str = Field(..., description="The finding statement")
    evidence_strength: EvidenceStrength = Field(
        ...,
        description="Strength of supporting evidence"
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of supporting sources"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the finding"
    )


class Theme(BaseModel):
    """A theme identified across multiple sources."""
    
    theme_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique theme identifier"
    )
    name: str = Field(..., description="Theme name")
    description: str = Field(..., description="Theme description")
    related_findings: list[str] = Field(
        default_factory=list,
        description="IDs of related findings"
    )
    source_count: int = Field(default=0, ge=0, description="Number of supporting sources")


class LiteratureSynthesis(BaseModel):
    """Synthesis of literature review findings."""
    
    synthesis_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique synthesis identifier"
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="Creation timestamp"
    )
    
    # Core synthesis components
    summary: str = Field(
        default="",
        description="Executive summary of the synthesis"
    )
    state_of_field: str = Field(
        default="",
        description="Description of the current state of the field"
    )
    full_synthesis: str = Field(
        default="",
        description="Full synthesis text"
    )
    
    # Extracted insights
    key_findings: list[str] = Field(
        default_factory=list,
        description="Key findings from the literature"
    )
    theoretical_frameworks: list[str] = Field(
        default_factory=list,
        description="Major theoretical frameworks identified"
    )
    methodological_approaches: list[str] = Field(
        default_factory=list,
        description="Common methodological approaches"
    )
    contribution_opportunities: list[str] = Field(
        default_factory=list,
        description="Opportunities for novel contributions"
    )
    
    # Metrics
    papers_analyzed: int = Field(
        default=0,
        ge=0,
        description="Number of papers analyzed"
    )
    themes_identified: int = Field(
        default=0,
        ge=0,
        description="Number of themes identified"
    )
    gaps_identified: int = Field(
        default=0,
        ge=0,
        description="Number of research gaps identified"
    )


# =============================================================================
# Gap Analysis Models
# =============================================================================


class ResearchGap(BaseModel):
    """A single research gap identified in the literature."""
    
    gap_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique gap identifier"
    )
    gap_type: str = Field(
        ...,
        description="Type of gap (methodological, empirical, theoretical, etc.)"
    )
    title: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Short title for the gap"
    )
    description: str = Field(
        ...,
        min_length=20,
        description="Detailed description of the gap"
    )
    significance: str = Field(
        default="medium",
        description="Significance level (high, medium, low)"
    )
    significance_justification: str = Field(
        default="",
        description="Why this gap is significant"
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence from literature supporting this gap"
    )
    related_papers: list[str] = Field(
        default_factory=list,
        description="Paper IDs/titles that relate to this gap"
    )
    addressable: bool = Field(
        default=True,
        description="Whether this gap can be addressed in current research"
    )
    addressability_notes: str = Field(
        default="",
        description="Notes on how/whether gap can be addressed"
    )


class GapAnalysis(BaseModel):
    """Complete gap analysis result from the GAP_IDENTIFIER node."""
    
    analysis_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique gap analysis identifier"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )
    
    # Input context
    original_question: str = Field(
        ...,
        description="Original research question being analyzed"
    )
    literature_coverage_summary: str = Field(
        default="",
        description="Summary of what the literature covers"
    )
    
    # Identified gaps
    gaps: list[ResearchGap] = Field(
        default_factory=list,
        description="All identified research gaps"
    )
    primary_gap: ResearchGap | None = Field(
        default=None,
        description="The most significant addressable gap"
    )
    
    # Coverage analysis
    coverage_comparison: str = Field(
        default="",
        description="Comparison of what's covered vs. what's asked"
    )
    coverage_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Estimated percentage of question covered by literature"
    )
    
    # Gap categorization
    methodological_gaps: list[str] = Field(
        default_factory=list,
        description="Gaps in methods/approaches"
    )
    empirical_gaps: list[str] = Field(
        default_factory=list,
        description="Gaps in empirical evidence"
    )
    theoretical_gaps: list[str] = Field(
        default_factory=list,
        description="Gaps in theory/frameworks"
    )
    
    # Ranking
    gap_significance_ranking: list[str] = Field(
        default_factory=list,
        description="Gap IDs ranked by significance"
    )
    
    @property
    def gap_count(self) -> int:
        """Total number of identified gaps."""
        return len(self.gaps)
    
    @property
    def high_significance_gaps(self) -> list[ResearchGap]:
        """Gaps with high significance."""
        return [g for g in self.gaps if g.significance == "high"]
    
    def get_gap_by_type(self, gap_type: str) -> list[ResearchGap]:
        """Get all gaps of a specific type."""
        return [g for g in self.gaps if g.gap_type == gap_type]


class ContributionStatement(BaseModel):
    """Contribution statement generated from gap analysis."""
    
    statement_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique statement identifier"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    
    # The contribution
    main_statement: str = Field(
        ...,
        min_length=20,
        description="Main contribution statement"
    )
    contribution_type: str = Field(
        default="empirical",
        description="Type of contribution (methodological, empirical, theoretical)"
    )
    
    # Supporting information
    gap_addressed: str = Field(
        default="",
        description="Which gap this contribution addresses"
    )
    novelty_explanation: str = Field(
        default="",
        description="What makes this contribution novel"
    )
    
    # Positioning
    position_in_literature: str = Field(
        default="",
        description="How this fits with existing work"
    )
    differentiation: str = Field(
        default="",
        description="How this differs from prior work"
    )
    prior_work_comparison: list[str] = Field(
        default_factory=list,
        description="Specific comparisons to prior papers"
    )
    
    # Impact
    potential_impact: str = Field(
        default="",
        description="Expected impact of this contribution"
    )
    target_audience: list[str] = Field(
        default_factory=list,
        description="Who would benefit from this contribution"
    )


class RefinedResearchQuestion(BaseModel):
    """A refined research question based on gap analysis."""
    
    question_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique question identifier"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    
    # Questions
    original_question: str = Field(
        ...,
        description="Original research question"
    )
    refined_question: str = Field(
        ...,
        description="Refined research question"
    )
    
    # Refinement details
    refinement_rationale: str = Field(
        default="",
        description="Why the question was refined"
    )
    gap_targeted: str = Field(
        default="",
        description="Which gap the refined question targets"
    )
    scope_changes: list[str] = Field(
        default_factory=list,
        description="How scope was narrowed or adjusted"
    )
    
    # Quality indicators
    specificity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How specific/focused the refined question is"
    )
    feasibility_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How feasible the refined question is to answer"
    )
    novelty_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How novel the refined question is"
    )


class AnalysisResult(BaseModel):
    """Results from the analysis phase."""
    
    analysis_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique analysis identifier"
    )
    analyzed_at: datetime = Field(
        default_factory=_utc_now,
        description="Analysis timestamp"
    )
    
    # Findings
    key_findings: list[Finding] = Field(
        default_factory=list,
        description="Key findings from analysis"
    )
    
    # Themes
    themes: list[Theme] = Field(
        default_factory=list,
        description="Identified themes"
    )
    
    # Issues
    contradictions: list[str] = Field(
        default_factory=list,
        description="Contradictions found in sources"
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Knowledge gaps identified"
    )
    
    # Quality metrics
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall analysis confidence"
    )
    coverage_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How well sub-questions are covered"
    )
    
    # Source tracking
    sources_used: list[str] = Field(
        default_factory=list,
        description="IDs of sources used in analysis"
    )


# =============================================================================
# Sprint 5: Statistical and Data Analysis Models
# =============================================================================


class StatisticalResult(BaseModel):
    """Result of a statistical test."""
    
    result_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique result identifier"
    )
    test_type: StatisticalTestType = Field(
        ...,
        description="Type of statistical test"
    )
    test_name: str = Field(
        ...,
        description="Specific test name (e.g., 'Two-sample t-test')"
    )
    
    # Test statistics
    statistic: float = Field(..., description="Test statistic value")
    p_value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="P-value"
    )
    degrees_of_freedom: int | None = Field(
        default=None,
        description="Degrees of freedom if applicable"
    )
    
    # Confidence interval
    confidence_level: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence level (e.g., 0.95 for 95%)"
    )
    confidence_interval_lower: float | None = Field(
        default=None,
        description="Lower bound of confidence interval"
    )
    confidence_interval_upper: float | None = Field(
        default=None,
        description="Upper bound of confidence interval"
    )
    
    # Effect size
    effect_size: float | None = Field(
        default=None,
        description="Effect size (Cohen's d, eta-squared, etc.)"
    )
    effect_size_type: str | None = Field(
        default=None,
        description="Type of effect size measure"
    )
    
    # Interpretation
    is_significant: bool = Field(
        default=False,
        description="Whether result is statistically significant at alpha=0.05"
    )
    interpretation: str = Field(
        default="",
        description="Plain language interpretation of the result"
    )
    
    # Additional info
    sample_size: int | None = Field(
        default=None,
        ge=1,
        description="Sample size used in test"
    )
    assumptions_met: bool = Field(
        default=True,
        description="Whether test assumptions were met"
    )
    assumption_notes: str = Field(
        default="",
        description="Notes on assumption testing"
    )


class RegressionCoefficient(BaseModel):
    """A single coefficient from a regression model."""
    
    variable: str = Field(..., description="Variable name")
    coefficient: float = Field(..., description="Estimated coefficient")
    std_error: float = Field(..., ge=0.0, description="Standard error")
    t_statistic: float = Field(..., description="T-statistic")
    p_value: float = Field(..., ge=0.0, le=1.0, description="P-value")
    confidence_interval_lower: float = Field(..., description="Lower CI bound")
    confidence_interval_upper: float = Field(..., description="Upper CI bound")
    is_significant: bool = Field(
        default=False,
        description="Significant at alpha=0.05"
    )


class RegressionResult(BaseModel):
    """Result of a regression analysis."""
    
    result_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique result identifier"
    )
    model_type: str = Field(
        ...,
        description="Type of regression (OLS, Fixed Effects, etc.)"
    )
    dependent_variable: str = Field(
        ...,
        description="Dependent variable name"
    )
    
    # Model fit
    r_squared: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="R-squared value"
    )
    adjusted_r_squared: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Adjusted R-squared"
    )
    f_statistic: float | None = Field(
        default=None,
        description="F-statistic for overall model"
    )
    f_p_value: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="P-value for F-statistic"
    )
    
    # Coefficients
    coefficients: list[RegressionCoefficient] = Field(
        default_factory=list,
        description="Regression coefficients"
    )
    
    # Sample info
    n_observations: int = Field(..., ge=1, description="Number of observations")
    n_groups: int | None = Field(
        default=None,
        description="Number of groups (for panel data)"
    )
    
    # Diagnostics
    residual_std_error: float | None = Field(
        default=None,
        ge=0.0,
        description="Residual standard error"
    )
    durbin_watson: float | None = Field(
        default=None,
        description="Durbin-Watson statistic"
    )
    heteroskedasticity_test: str | None = Field(
        default=None,
        description="Heteroskedasticity test result"
    )
    
    # Interpretation
    interpretation: str = Field(
        default="",
        description="Plain language interpretation"
    )


class DataAnalysisFinding(BaseModel):
    """A finding from data analysis with statistical support."""
    
    finding_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique finding identifier"
    )
    finding_type: FindingType = Field(
        ...,
        description="Type of finding"
    )
    
    # The finding
    statement: str = Field(
        ...,
        min_length=20,
        description="The finding statement"
    )
    detailed_description: str = Field(
        default="",
        description="Detailed description of the finding"
    )
    
    # Statistical support
    statistical_results: list[StatisticalResult] = Field(
        default_factory=list,
        description="Statistical tests supporting this finding"
    )
    regression_results: list[RegressionResult] = Field(
        default_factory=list,
        description="Regression analyses supporting this finding"
    )
    
    # Evidence tracking
    evidence: list["EvidenceItem"] = Field(
        default_factory=list,
        description="Evidence items supporting this finding"
    )
    evidence_strength: EvidenceStrength = Field(
        default=EvidenceStrength.MODERATE,
        description="Overall evidence strength"
    )
    
    # Research question linkage
    addresses_research_question: bool = Field(
        default=False,
        description="Whether this finding addresses the main research question"
    )
    addresses_gap: bool = Field(
        default=False,
        description="Whether this finding addresses the identified gap"
    )
    gap_coverage_explanation: str = Field(
        default="",
        description="How this finding addresses the gap"
    )
    
    # Quality indicators
    confidence_level: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this finding"
    )
    robustness_checks_passed: int = Field(
        default=0,
        ge=0,
        description="Number of robustness checks passed"
    )
    
    # Limitations
    limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations of this finding"
    )


class DataAnalysisResult(BaseModel):
    """Complete result from the DATA_ANALYST node."""
    
    result_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique result identifier"
    )
    analysis_status: AnalysisStatus = Field(
        default=AnalysisStatus.COMPLETE,
        description="Status of the analysis"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    
    # Analysis metadata
    methodology_used: str = Field(
        default="",
        description="Methodology used for analysis"
    )
    analysis_approach: AnalysisApproach | None = Field(
        default=None,
        description="Analysis approach used"
    )
    
    # Data summary
    data_summary: str = Field(
        default="",
        description="Summary of data used"
    )
    sample_size: int = Field(
        default=0,
        ge=0,
        description="Total sample size"
    )
    variables_analyzed: list[str] = Field(
        default_factory=list,
        description="Variables included in analysis"
    )
    time_period: str | None = Field(
        default=None,
        description="Time period covered by data"
    )
    
    # Descriptive statistics
    descriptive_stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Descriptive statistics for key variables"
    )
    
    # Findings
    findings: list[DataAnalysisFinding] = Field(
        default_factory=list,
        description="All findings from the analysis"
    )
    main_findings: list[DataAnalysisFinding] = Field(
        default_factory=list,
        description="Main findings (subset of findings)"
    )
    
    # Statistical results
    statistical_tests: list[StatisticalResult] = Field(
        default_factory=list,
        description="All statistical test results"
    )
    regression_analyses: list[RegressionResult] = Field(
        default_factory=list,
        description="All regression analyses"
    )
    
    # Gap assessment
    gap_addressed: bool = Field(
        default=False,
        description="Whether the analysis addresses the identified gap"
    )
    gap_coverage_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How well the gap is addressed (0-1)"
    )
    gap_coverage_explanation: str = Field(
        default="",
        description="Explanation of gap coverage"
    )
    
    # Hypothesis testing
    hypothesis_supported: bool | None = Field(
        default=None,
        description="Whether the main hypothesis is supported"
    )
    hypothesis_test_summary: str = Field(
        default="",
        description="Summary of hypothesis testing"
    )
    
    # Quality assessment
    overall_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the analysis"
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations"
    )
    robustness_summary: str = Field(
        default="",
        description="Summary of robustness checks"
    )
    llm_interpretations: str = Field(
        default="",
        description="LLM-generated academic interpretations of results"
    )
    
    @property
    def finding_count(self) -> int:
        """Total number of findings."""
        return len(self.findings)
    
    @property
    def significant_findings(self) -> list[DataAnalysisFinding]:
        """Findings with significant statistical support."""
        significant = []
        for f in self.findings:
            has_significant = any(
                r.is_significant for r in f.statistical_results
            ) or any(
                r.f_p_value is not None and r.f_p_value < 0.05
                for r in f.regression_results
            )
            if has_significant:
                significant.append(f)
        return significant


# =============================================================================
# Sprint 5: Conceptual Framework Models
# =============================================================================


class Concept(BaseModel):
    """A theoretical concept in a framework."""
    
    concept_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique concept identifier"
    )
    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Concept name"
    )
    concept_type: ConceptType = Field(
        ...,
        description="Type of concept"
    )
    
    # Definition
    definition: str = Field(
        ...,
        min_length=20,
        description="Concept definition"
    )
    operationalization: str | None = Field(
        default=None,
        description="How the concept can be measured/operationalized"
    )
    
    # Sources
    source_literature: list[str] = Field(
        default_factory=list,
        description="Literature sources for this concept"
    )
    derived_from: list[str] = Field(
        default_factory=list,
        description="Other concept IDs this is derived from"
    )
    
    # Properties
    is_observable: bool = Field(
        default=True,
        description="Whether concept is directly observable"
    )
    abstraction_level: str = Field(
        default="medium",
        description="Level of abstraction (low, medium, high)"
    )


class ConceptRelationship(BaseModel):
    """A relationship between two concepts."""
    
    relationship_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique relationship identifier"
    )
    
    # Concepts involved
    source_concept_id: str = Field(..., description="Source concept ID")
    target_concept_id: str = Field(..., description="Target concept ID")
    relationship_type: RelationshipType = Field(
        ...,
        description="Type of relationship"
    )
    
    # Relationship details
    description: str = Field(
        default="",
        description="Description of the relationship"
    )
    strength: str = Field(
        default="moderate",
        description="Expected strength (weak, moderate, strong)"
    )
    direction: str = Field(
        default="positive",
        description="Direction (positive, negative, mixed)"
    )
    
    # Support
    theoretical_basis: str = Field(
        default="",
        description="Theoretical basis for this relationship"
    )
    empirical_support: EvidenceStrength = Field(
        default=EvidenceStrength.MODERATE,
        description="Level of empirical support"
    )
    supporting_literature: list[str] = Field(
        default_factory=list,
        description="Literature supporting this relationship"
    )


class Proposition(BaseModel):
    """A theoretical proposition in a framework."""
    
    proposition_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique proposition identifier"
    )
    
    # The proposition
    statement: str = Field(
        ...,
        min_length=20,
        description="Proposition statement"
    )
    formal_statement: str | None = Field(
        default=None,
        description="Formal/symbolic statement if applicable"
    )
    
    # Derivation
    derived_from_concepts: list[str] = Field(
        default_factory=list,
        description="Concept IDs this proposition is derived from"
    )
    derived_from_relationships: list[str] = Field(
        default_factory=list,
        description="Relationship IDs this proposition is derived from"
    )
    derivation_logic: str = Field(
        default="",
        description="Logic/reasoning for the derivation"
    )
    
    # Properties
    proposition_status: PropositionStatus = Field(
        default=PropositionStatus.PROPOSED,
        description="Current status of the proposition"
    )
    is_testable: bool = Field(
        default=True,
        description="Whether proposition is empirically testable"
    )
    test_approach: str | None = Field(
        default=None,
        description="How this proposition could be tested"
    )
    
    # Support
    empirical_support: EvidenceStrength = Field(
        default=EvidenceStrength.INSUFFICIENT,
        description="Level of empirical support"
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting this proposition"
    )
    contrary_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence against this proposition"
    )
    
    # Boundary conditions
    boundary_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions under which proposition holds"
    )


class ConceptualFramework(BaseModel):
    """A complete conceptual framework from the CONCEPTUAL_SYNTHESIZER node."""
    
    framework_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique framework identifier"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    
    # Framework identity
    title: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Framework title"
    )
    abstract: str = Field(
        default="",
        description="Brief abstract of the framework"
    )
    description: str = Field(
        ...,
        min_length=50,
        description="Full description of the framework"
    )
    
    # Core components
    concepts: list[Concept] = Field(
        default_factory=list,
        description="Concepts in the framework"
    )
    relationships: list[ConceptRelationship] = Field(
        default_factory=list,
        description="Relationships between concepts"
    )
    propositions: list[Proposition] = Field(
        default_factory=list,
        description="Propositions derived from the framework"
    )
    
    # Theoretical grounding
    theoretical_foundations: list[str] = Field(
        default_factory=list,
        description="Existing theories this framework builds on"
    )
    seminal_works: list[str] = Field(
        default_factory=list,
        description="Seminal works informing the framework"
    )
    grounding_explanation: str = Field(
        default="",
        description="How framework connects to existing theory"
    )
    
    # Scope
    domain: str = Field(
        default="",
        description="Domain/field of application"
    )
    scope_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions defining framework scope"
    )
    
    # Quality and contribution
    novelty_assessment: str = Field(
        default="",
        description="Assessment of framework novelty"
    )
    theoretical_contribution: str = Field(
        default="",
        description="Statement of theoretical contribution"
    )
    practical_implications: list[str] = Field(
        default_factory=list,
        description="Practical implications of the framework"
    )
    
    # Limitations
    limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations"
    )
    future_directions: list[str] = Field(
        default_factory=list,
        description="Suggested future research directions"
    )
    
    @property
    def concept_count(self) -> int:
        """Number of concepts in framework."""
        return len(self.concepts)
    
    @property
    def proposition_count(self) -> int:
        """Number of propositions."""
        return len(self.propositions)
    
    @property
    def testable_propositions(self) -> list[Proposition]:
        """Propositions that are empirically testable."""
        return [p for p in self.propositions if p.is_testable]


class ConceptualSynthesisResult(BaseModel):
    """Complete result from the CONCEPTUAL_SYNTHESIZER node."""
    
    result_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique result identifier"
    )
    analysis_status: AnalysisStatus = Field(
        default=AnalysisStatus.COMPLETE,
        description="Status of the synthesis"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    
    # The framework
    framework: ConceptualFramework | None = Field(
        default=None,
        description="The developed conceptual framework"
    )
    
    # Synthesis summary
    synthesis_approach: str = Field(
        default="",
        description="Approach used for synthesis"
    )
    literature_base: str = Field(
        default="",
        description="Description of literature base used"
    )
    papers_synthesized: int = Field(
        default=0,
        ge=0,
        description="Number of papers synthesized"
    )
    
    # Key outputs
    key_concepts_identified: list[str] = Field(
        default_factory=list,
        description="Key concepts identified"
    )
    theoretical_mechanisms: list[str] = Field(
        default_factory=list,
        description="Theoretical mechanisms identified"
    )
    
    # Gap assessment
    gap_addressed: bool = Field(
        default=False,
        description="Whether the synthesis addresses the identified gap"
    )
    gap_coverage_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How well the gap is addressed (0-1)"
    )
    gap_coverage_explanation: str = Field(
        default="",
        description="Explanation of gap coverage"
    )
    
    # Contribution assessment
    contribution_type: str = Field(
        default="theoretical",
        description="Type of contribution (theoretical, integrative, etc.)"
    )
    contribution_statement: str = Field(
        default="",
        description="Statement of the contribution"
    )
    
    # Quality assessment
    overall_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the synthesis"
    )
    coherence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Internal coherence of the framework"
    )
    grounding_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How well grounded in existing theory"
    )
    
    # Limitations
    limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations"
    )


# =============================================================================
# Draft and Review Models
# =============================================================================


class DraftSection(BaseModel):
    """A section of the research draft."""
    
    section_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique section identifier"
    )
    title: str = Field(..., description="Section title")
    content: str = Field(default="", description="Section content")
    order: int = Field(..., ge=0, description="Section order in document")
    
    # Metadata
    word_count: int = Field(default=0, ge=0, description="Word count")
    citations: list[str] = Field(
        default_factory=list,
        description="Citation keys used in section"
    )
    
    @model_validator(mode="after")
    def compute_word_count(self) -> "DraftSection":
        """Compute word count from content."""
        if self.content:
            self.word_count = len(self.content.split())
        return self


class ResearchDraft(BaseModel):
    """Complete research draft from the WRITER node."""
    
    draft_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique draft identifier"
    )
    version: int = Field(default=1, ge=1, description="Draft version number")
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="Creation timestamp"
    )
    
    # Content
    title: str = Field(default="", description="Paper title")
    abstract: str = Field(default="", description="Paper abstract")
    sections: list[DraftSection] = Field(
        default_factory=list,
        description="Paper sections"
    )
    conclusion: str = Field(default="", description="Conclusion section")
    
    # References
    references: list[str] = Field(
        default_factory=list,
        description="Formatted reference list"
    )
    
    @property
    def total_word_count(self) -> int:
        """Total word count across all sections."""
        count = len(self.abstract.split()) if self.abstract else 0
        count += sum(s.word_count for s in self.sections)
        count += len(self.conclusion.split()) if self.conclusion else 0
        return count
    
    @property
    def section_count(self) -> int:
        """Number of sections."""
        return len(self.sections)


class CritiqueItem(BaseModel):
    """A single critique point from the reviewer."""
    
    item_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique item identifier"
    )
    severity: CritiqueSeverity = Field(..., description="Issue severity")
    category: str = Field(
        ...,
        description="Category (accuracy, completeness, coherence, citation, methodology, style)"
    )
    location: str = Field(
        default="general",
        description="Location in document (section name or 'general')"
    )
    description: str = Field(..., description="Description of the issue")
    suggestion: str | None = Field(
        default=None,
        description="Suggested fix"
    )


class Critique(BaseModel):
    """Complete critique from the REVIEWER node."""
    
    critique_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique critique identifier"
    )
    draft_version: int = Field(..., ge=1, description="Version of draft reviewed")
    reviewed_at: datetime = Field(
        default_factory=_utc_now,
        description="Review timestamp"
    )
    
    # Critique items
    items: list[CritiqueItem] = Field(
        default_factory=list,
        description="Individual critique items"
    )
    
    # Scoring
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Overall quality score (0-10)"
    )
    category_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Scores by category"
    )
    
    # Decision
    recommendation: str = Field(
        default="revise",
        description="Recommendation: approve, revise, reject"
    )
    summary: str = Field(
        default="",
        description="Summary of critique"
    )
    
    # Thresholds
    pass_threshold: float = Field(default=7.5, description="Score needed to pass")
    
    @property
    def passes_review(self) -> bool:
        """Check if critique passes the threshold."""
        return (
            self.overall_score >= self.pass_threshold 
            and self.recommendation == "approve"
        )
    
    @property
    def critical_count(self) -> int:
        """Count of critical issues."""
        return sum(1 for item in self.items if item.severity == CritiqueSeverity.CRITICAL)
    
    @property
    def major_count(self) -> int:
        """Count of major issues."""
        return sum(1 for item in self.items if item.severity == CritiqueSeverity.MAJOR)


# =============================================================================
# Evidence Tracking
# =============================================================================


class EvidenceItem(BaseModel):
    """Evidence supporting a claim in the draft."""
    
    evidence_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique evidence identifier"
    )
    claim: str = Field(..., description="The claim being supported")
    source_id: str = Field(..., description="ID of the source")
    source_url: str = Field(..., description="URL of the source")
    locator: str = Field(
        default="",
        description="Specific location in source (page, paragraph, etc.)"
    )
    quote: str | None = Field(
        default=None,
        description="Direct quote from source"
    )
    strength: EvidenceStrength = Field(
        ...,
        description="Evidence strength"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in evidence"
    )
    verified: bool = Field(
        default=False,
        description="Whether evidence has been verified"
    )


# =============================================================================
# Workflow Error Tracking
# =============================================================================


class WorkflowError(BaseModel):
    """An error that occurred during workflow execution."""
    
    error_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique error identifier"
    )
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the error occurred"
    )
    node: str = Field(..., description="Node where error occurred")
    category: str = Field(..., description="Error category")
    message: str = Field(..., description="Error message")
    recoverable: bool = Field(
        default=True,
        description="Whether error is recoverable"
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error details"
    )


# =============================================================================
# Sprint 6: WRITER Node Models
# =============================================================================


class StyleViolation(BaseModel):
    """A style guide violation detected in text."""
    
    violation_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique violation identifier"
    )
    violation_type: str = Field(
        ...,
        description="Type of violation (banned_word, informal_tone, etc.)"
    )
    severity: CritiqueSeverity = Field(
        default=CritiqueSeverity.MINOR,
        description="Severity of the violation"
    )
    location: str = Field(
        default="",
        description="Location in text (e.g., 'paragraph 2, sentence 3')"
    )
    original_text: str = Field(
        ...,
        description="The problematic text"
    )
    suggestion: str = Field(
        default="",
        description="Suggested replacement or fix"
    )
    rule_reference: str = Field(
        default="",
        description="Reference to style guide section"
    )
    auto_fixable: bool = Field(
        default=False,
        description="Whether this violation can be auto-fixed"
    )


class CitationEntry(BaseModel):
    """A citation entry for the reference list."""
    
    key: str = Field(
        ...,
        description="Citation key (e.g., 'Fama1970')"
    )
    authors: list[str] = Field(
        ...,
        description="List of author names"
    )
    year: int = Field(
        ...,
        ge=1900,
        le=2100,
        description="Publication year"
    )
    title: str = Field(
        ...,
        description="Title of the work"
    )
    journal: str | None = Field(
        default=None,
        description="Journal name"
    )
    volume: str | None = Field(
        default=None,
        description="Volume number"
    )
    issue: str | None = Field(
        default=None,
        description="Issue number"
    )
    pages: str | None = Field(
        default=None,
        description="Page range (e.g., '3-56')"
    )
    doi: str | None = Field(
        default=None,
        description="DOI identifier"
    )
    url: str | None = Field(
        default=None,
        description="URL for online sources"
    )
    publisher: str | None = Field(
        default=None,
        description="Publisher name (for books)"
    )
    source_type: str = Field(
        default="journal",
        description="Type: journal, book, chapter, working_paper, etc."
    )
    
    def format_chicago_author_date(self) -> str:
        """Format citation in Chicago Author-Date style for reference list."""
        # Format authors
        if len(self.authors) == 1:
            author_str = self.authors[0]
        elif len(self.authors) == 2:
            author_str = f"{self.authors[0]}, and {self.authors[1]}"
        else:
            author_str = f"{self.authors[0]}, {', '.join(self.authors[1:-1])}, and {self.authors[-1]}"
        
        # Base format
        ref = f'{author_str}. {self.year}. "{self.title}."'
        
        # Add journal info
        if self.journal:
            ref += f" {self.journal}"
            if self.volume:
                ref += f" {self.volume}"
                if self.issue:
                    ref += f" ({self.issue})"
            if self.pages:
                ref += f": {self.pages}"
            ref += "."
        elif self.publisher:
            ref += f" {self.publisher}."
        
        return ref
    
    def format_inline(self, include_page: str | None = None) -> str:
        """Format inline citation in Chicago Author-Date style."""
        # Get last name of first author
        first_author = self.authors[0].split(",")[0] if "," in self.authors[0] else self.authors[0].split()[-1]
        
        if len(self.authors) == 1:
            citation = f"({first_author} {self.year}"
        elif len(self.authors) == 2:
            second_author = self.authors[1].split(",")[0] if "," in self.authors[1] else self.authors[1].split()[-1]
            citation = f"({first_author} and {second_author} {self.year}"
        else:
            citation = f"({first_author} et al. {self.year}"
        
        if include_page:
            citation += f", {include_page}"
        
        citation += ")"
        return citation


class ReferenceList(BaseModel):
    """Complete reference list for a paper."""
    
    entries: list[CitationEntry] = Field(
        default_factory=list,
        description="List of citation entries"
    )
    format_style: str = Field(
        default="chicago_author_date",
        description="Citation format style"
    )
    
    def add_entry(self, entry: CitationEntry) -> None:
        """Add a citation entry if not already present."""
        if not any(e.key == entry.key for e in self.entries):
            self.entries.append(entry)
    
    def get_entry(self, key: str) -> CitationEntry | None:
        """Get a citation entry by key."""
        for entry in self.entries:
            if entry.key == key:
                return entry
        return None
    
    def format_reference_list(self) -> str:
        """Format the complete reference list."""
        # Sort by first author's last name, then by year
        sorted_entries = sorted(
            self.entries,
            key=lambda e: (e.authors[0].split(",")[0] if "," in e.authors[0] else e.authors[0].split()[-1], e.year)
        )
        
        lines = []
        for entry in sorted_entries:
            lines.append(entry.format_chicago_author_date())
        
        return "\n\n".join(lines)
    
    @property
    def entry_count(self) -> int:
        """Number of entries in the reference list."""
        return len(self.entries)


class PaperSection(BaseModel):
    """A section of an academic paper."""
    
    section_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique section identifier"
    )
    section_type: str = Field(
        ...,
        description="Type of section (introduction, methods, results, etc.)"
    )
    title: str = Field(
        ...,
        description="Section title"
    )
    content: str = Field(
        default="",
        description="Section content"
    )
    order: int = Field(
        default=0,
        ge=0,
        description="Order in the paper"
    )
    word_count: int = Field(
        default=0,
        ge=0,
        description="Word count"
    )
    target_word_count: int | None = Field(
        default=None,
        description="Target word count for this section"
    )
    citations_used: list[str] = Field(
        default_factory=list,
        description="Citation keys used in this section"
    )
    status: str = Field(
        default="pending",
        description="Writing status"
    )
    style_violations: list[StyleViolation] = Field(
        default_factory=list,
        description="Style violations in this section"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When section was created"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When section was last updated"
    )
    
    def update_word_count(self) -> int:
        """Update and return word count from content."""
        self.word_count = len(self.content.split()) if self.content else 0
        return self.word_count
    
    def is_within_target(self, tolerance: float = 0.2) -> bool:
        """Check if word count is within tolerance of target."""
        if not self.target_word_count:
            return True
        lower = self.target_word_count * (1 - tolerance)
        upper = self.target_word_count * (1 + tolerance)
        return lower <= self.word_count <= upper


class ArgumentThread(BaseModel):
    """Tracks the logical argument thread across sections."""
    
    thread_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique thread identifier"
    )
    
    # Claims and evidence tracking
    main_thesis: str = Field(
        default="",
        description="The paper's main thesis/argument"
    )
    claims: list[str] = Field(
        default_factory=list,
        description="Key claims made in the paper"
    )
    evidence_map: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Mapping of claims to supporting evidence"
    )
    
    # Coherence tracking
    section_connections: list[dict[str, str]] = Field(
        default_factory=list,
        description="How sections connect (from, to, connection_type)"
    )
    
    # Promise tracking
    promised_contribution: str = Field(
        default="",
        description="Contribution promised in introduction"
    )
    delivered_contribution: str = Field(
        default="",
        description="Contribution delivered in conclusion"
    )
    
    # Scoring
    coherence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall coherence score"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims without adequate evidence"
    )
    
    def add_claim(self, claim: str, evidence: list[str] | None = None) -> None:
        """Add a claim with optional evidence."""
        if claim not in self.claims:
            self.claims.append(claim)
        if evidence:
            self.evidence_map[claim] = evidence
    
    def check_claim_support(self) -> list[str]:
        """Return list of unsupported claims."""
        unsupported = []
        for claim in self.claims:
            if claim not in self.evidence_map or not self.evidence_map[claim]:
                unsupported.append(claim)
        self.unsupported_claims = unsupported
        return unsupported
    
    def contribution_delivered(self) -> bool:
        """Check if promised contribution is delivered."""
        if not self.promised_contribution or not self.delivered_contribution:
            return False
        # Simple check: delivered should address promised
        return len(self.delivered_contribution) > 0


class WriterOutput(BaseModel):
    """Complete output from the WRITER node."""
    
    output_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique output identifier"
    )
    
    # Paper content
    title: str = Field(
        default="",
        description="Paper title"
    )
    sections: list[PaperSection] = Field(
        default_factory=list,
        description="All paper sections"
    )
    reference_list: ReferenceList = Field(
        default_factory=ReferenceList,
        description="Complete reference list"
    )
    
    # Argument tracking
    argument_thread: ArgumentThread = Field(
        default_factory=ArgumentThread,
        description="Argument coherence tracking"
    )
    
    # Quality metrics
    total_word_count: int = Field(
        default=0,
        ge=0,
        description="Total word count"
    )
    target_word_count: int | None = Field(
        default=None,
        description="Target total word count"
    )
    style_violations: list[StyleViolation] = Field(
        default_factory=list,
        description="All style violations across sections"
    )
    
    # Status
    writing_status: str = Field(
        default="pending",
        description="Overall writing status"
    )
    contribution_delivered: bool = Field(
        default=False,
        description="Whether promised contribution is delivered"
    )
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When writing started"
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When writing completed"
    )
    
    def get_section(self, section_type: str) -> PaperSection | None:
        """Get a section by type."""
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None
    
    def update_total_word_count(self) -> int:
        """Update and return total word count."""
        self.total_word_count = sum(s.word_count for s in self.sections)
        return self.total_word_count
    
    def collect_style_violations(self) -> list[StyleViolation]:
        """Collect all style violations from all sections."""
        violations = []
        for section in self.sections:
            violations.extend(section.style_violations)
        self.style_violations = violations
        return violations
    
    def collect_citations(self) -> list[str]:
        """Collect all citation keys used across sections."""
        citations = set()
        for section in self.sections:
            citations.update(section.citations_used)
        return list(citations)
    
    def is_complete(self) -> bool:
        """Check if all sections are finalized."""
        return all(s.status == "finalized" for s in self.sections)
    
    def get_abstract(self) -> str:
        """Get abstract content."""
        abstract = self.get_section("abstract")
        return abstract.content if abstract else ""
    
    def to_markdown(self) -> str:
        """Export paper as markdown."""
        lines = [f"# {self.title}", ""]
        
        # Add sections in order
        for section in sorted(self.sections, key=lambda s: s.order):
            if section.section_type != "references":
                lines.append(f"## {section.title}")
                lines.append("")
                lines.append(section.content)
                lines.append("")
        
        # Add references
        lines.append("## References")
        lines.append("")
        lines.append(self.reference_list.format_reference_list())
        
        return "\n".join(lines)


# =============================================================================
# Section Writing Context Models
# =============================================================================


class SectionWritingContext(BaseModel):
    """Context passed to a section writer."""
    
    section_type: str = Field(
        ...,
        description="Type of section to write"
    )
    target_journal: str = Field(
        default="generic",
        description="Target journal"
    )
    paper_type: str = Field(
        default="short_article",
        description="Type of paper"
    )
    research_type: str = Field(
        default="empirical",
        description="Research methodology type"
    )
    
    # Content from prior nodes
    research_question: str = Field(
        default="",
        description="The research question"
    )
    contribution_statement: str = Field(
        default="",
        description="The contribution statement"
    )
    gap_analysis_summary: str = Field(
        default="",
        description="Summary of identified gaps"
    )
    literature_synthesis_summary: str = Field(
        default="",
        description="Summary of literature synthesis"
    )
    methodology_summary: str = Field(
        default="",
        description="Summary of methodology"
    )
    findings_summary: str = Field(
        default="",
        description="Summary of findings/results"
    )
    
    # Available data for results section
    has_quantitative_results: bool = Field(
        default=False,
        description="Whether quantitative results are available"
    )
    has_qualitative_results: bool = Field(
        default=False,
        description="Whether qualitative results are available"
    )
    
    # Word count targets
    target_word_count: int | None = Field(
        default=None,
        description="Target word count for this section"
    )
    
    # Available citations
    available_citations: list[CitationEntry] = Field(
        default_factory=list,
        description="Citations available for use"
    )
    
    # Prior sections (for coherence)
    prior_sections: list[PaperSection] = Field(
        default_factory=list,
        description="Sections written before this one"
    )


# Word count targets for different paper types and sections
SECTION_WORD_COUNTS = {
    "short_article": {
        "abstract": (50, 75),
        "introduction": (500, 800),
        "literature_review": (0, 0),  # Integrated in introduction for short papers
        "methods": (400, 700),
        "data": (200, 400),
        "results": (800, 1200),
        "discussion": (300, 500),
        "conclusion": (200, 400),
    },
    "full_paper": {
        "abstract": (75, 100),
        "introduction": (1000, 1500),
        "literature_review": (1500, 2500),
        "methods": (1000, 1500),
        "data": (500, 800),
        "results": (2000, 3000),
        "discussion": (1000, 1500),
        "conclusion": (400, 600),
    },
}


def get_section_word_count_target(paper_type: str, section_type: str) -> tuple[int, int] | None:
    """Get word count target range for a section type."""
    if paper_type in SECTION_WORD_COUNTS:
        return SECTION_WORD_COUNTS[paper_type].get(section_type)
    return None


# =============================================================================
# Sprint 7: REVIEWER Node Models
# =============================================================================


class QualityScore(BaseModel):
    """Quality score for a single evaluation dimension."""
    
    dimension: str = Field(
        ...,
        description="Dimension being evaluated (contribution, methodology, etc.)"
    )
    score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Score from 1-10"
    )
    justification: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Justification for the score"
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Identified strengths in this dimension"
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Identified weaknesses in this dimension"
    )


class ReviewCritiqueItem(BaseModel):
    """A specific critique or issue identified in the paper by the REVIEWER node."""
    
    section: str = Field(
        ...,
        description="Section where the issue was found"
    )
    issue: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Description of the issue"
    )
    severity: str = Field(
        ...,
        description="Severity: critical, major, minor, suggestion"
    )
    suggestion: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Suggested fix or improvement"
    )
    line_reference: str | None = Field(
        default=None,
        description="Optional reference to specific location in text"
    )


class ReviewCritique(BaseModel):
    """Complete critique from the REVIEWER node."""
    
    # Overall assessment
    overall_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Overall quality score (weighted average)"
    )
    decision: str = Field(
        ...,
        description="Review decision: approve, revise, or reject"
    )
    
    # Dimension scores
    dimension_scores: list[QualityScore] = Field(
        default_factory=list,
        description="Scores for each quality dimension"
    )
    
    # Detailed critique items
    critique_items: list[ReviewCritiqueItem] = Field(
        default_factory=list,
        description="List of specific issues identified"
    )
    
    # Summary
    summary: str = Field(
        default="",
        max_length=2000,
        description="Executive summary of the review"
    )
    
    # Revision instructions (if revise decision)
    revision_instructions: str = Field(
        default="",
        max_length=3000,
        description="Specific instructions for revision"
    )
    
    # Metadata
    reviewer_id: str = Field(
        default="gia-reviewer",
        description="Identifier for the reviewer"
    )
    reviewed_at: datetime = Field(
        default_factory=_utc_now,
        description="When the review was completed"
    )
    iteration: int = Field(
        default=1,
        ge=1,
        description="Which revision iteration this review is for"
    )
    
    def get_score_by_dimension(self, dimension: str) -> float | None:
        """Get score for a specific dimension."""
        for score in self.dimension_scores:
            if score.dimension == dimension:
                return score.score
        return None
    
    def get_critical_items(self) -> list[ReviewCritiqueItem]:
        """Get all critical severity items."""
        return [item for item in self.critique_items if item.severity == "critical"]
    
    def get_items_by_section(self, section: str) -> list[ReviewCritiqueItem]:
        """Get all critique items for a specific section."""
        return [item for item in self.critique_items if item.section == section]
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return len(self.get_critical_items()) > 0


class RevisionRequest(BaseModel):
    """Request for revision from REVIEWER to WRITER."""
    
    sections_to_revise: list[str] = Field(
        default_factory=list,
        description="List of section types that need revision"
    )
    critique_items: list[ReviewCritiqueItem] = Field(
        default_factory=list,
        description="Critique items to address"
    )
    revision_instructions: str = Field(
        default="",
        max_length=3000,
        description="Overall revision guidance"
    )
    iteration_count: int = Field(
        default=1,
        ge=1,
        description="Current revision iteration"
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        description="Maximum allowed iterations"
    )
    priority_order: list[str] = Field(
        default_factory=list,
        description="Sections in order of revision priority"
    )
    
    def is_final_iteration(self) -> bool:
        """Check if this is the final allowed iteration."""
        return self.iteration_count >= self.max_iterations


class ReviewerOutput(BaseModel):
    """Output from the REVIEWER node."""
    
    # The critique
    critique: ReviewCritique = Field(
        ...,
        description="The detailed critique"
    )
    
    # Decision and routing
    decision: str = Field(
        ...,
        description="Review decision: approve, revise, reject"
    )
    
    # Revision request (if decision is revise)
    revision_request: RevisionRequest | None = Field(
        default=None,
        description="Revision request if revisions needed"
    )
    
    # Human approval tracking
    human_approved: bool = Field(
        default=False,
        description="Whether human has approved the decision"
    )
    human_feedback: str | None = Field(
        default=None,
        description="Optional feedback from human reviewer"
    )
    
    # Final output (if approved)
    final_paper: str | None = Field(
        default=None,
        description="Final paper content in markdown (if approved)"
    )
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="When the review was created"
    )


# Quality thresholds for review decisions
REVIEW_THRESHOLDS = {
    "approve": 7.0,      # Score >= 7.0 -> approve
    "revise_min": 4.0,   # Score >= 4.0 and < 7.0 -> revise
    "reject": 4.0,       # Score < 4.0 -> reject
}

# Dimension weights for overall score calculation
DIMENSION_WEIGHTS = {
    "contribution": 0.25,
    "methodology": 0.25,
    "evidence": 0.20,
    "coherence": 0.15,
    "writing": 0.15,
}


def calculate_overall_score(dimension_scores: list[QualityScore]) -> float:
    """Calculate weighted overall score from dimension scores."""
    total_weight = 0.0
    weighted_sum = 0.0
    
    for score in dimension_scores:
        weight = DIMENSION_WEIGHTS.get(score.dimension, 0.1)
        weighted_sum += score.score * weight
        total_weight += weight
    
    if total_weight == 0:
        return 5.0  # Default middle score
    
    return weighted_sum / total_weight


def determine_review_decision(overall_score: float, has_critical_items: bool) -> str:
    """Determine review decision based on score and critical items."""
    # Critical items always force revision at minimum
    if has_critical_items:
        if overall_score < REVIEW_THRESHOLDS["reject"]:
            return "reject"
        return "revise"
    
    # Score-based decision
    if overall_score >= REVIEW_THRESHOLDS["approve"]:
        return "approve"
    elif overall_score >= REVIEW_THRESHOLDS["revise_min"]:
        return "revise"
    else:
        return "reject"
