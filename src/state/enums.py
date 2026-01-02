"""Enums and constants for GIA Agentic v2 workflow state."""

from enum import Enum


class ResearchStatus(str, Enum):
    """Status of the research workflow."""
    
    # Initial state
    INITIALIZED = "initialized"
    
    # Intake phase
    INTAKE_PENDING = "intake_pending"
    INTAKE_COMPLETE = "intake_complete"
    
    # Data exploration phase
    DATA_EXPLORING = "data_exploring"
    DATA_EXPLORED = "data_explored"
    DATA_QUALITY_ISSUES = "data_quality_issues"
    
    # Literature review phase
    LITERATURE_REVIEWING = "literature_reviewing"
    LITERATURE_REVIEW_COMPLETE = "literature_review_complete"
    
    # Gap identification phase
    GAP_IDENTIFYING = "gap_identifying"
    GAP_IDENTIFICATION_COMPLETE = "gap_identification_complete"
    
    # Planning phase
    PLANNING = "planning"
    PLANNING_COMPLETE = "planning_complete"
    
    # Analysis phase
    ANALYZING = "analyzing"
    ANALYSIS_COMPLETE = "analysis_complete"
    
    # Writing phase
    WRITING = "writing"
    WRITING_COMPLETE = "writing_complete"
    
    # Review phase
    REVIEWING = "reviewing"
    REVISION_NEEDED = "revision_needed"
    
    # Terminal states
    COMPLETED = "completed"
    FAILED = "failed"


class CritiqueSeverity(str, Enum):
    """Severity levels for critique items."""
    
    CRITICAL = "critical"  # Blocks approval; must be fixed
    MAJOR = "major"        # Significant issue; should be fixed
    MINOR = "minor"        # Polish issue; nice to fix
    SUGGESTION = "suggestion"  # Optional improvement


class EvidenceStrength(str, Enum):
    """Strength of evidence supporting a claim."""
    
    STRONG = "strong"          # Multiple independent sources agree
    MODERATE = "moderate"      # 2-3 sources agree
    WEAK = "weak"              # Single source or conflicting info
    INSUFFICIENT = "insufficient"  # Not enough evidence
    GAP = "gap"                # No relevant evidence found


class PaperType(str, Enum):
    """Types of academic papers."""
    
    SHORT_ARTICLE = "short_article"      # 5-10 pages
    FULL_PAPER = "full_paper"            # 30-45 pages
    WORKING_PAPER = "working_paper"      # Pre-publication
    REVIEW = "review"                    # Literature review
    PERSPECTIVE = "perspective"          # Opinion/viewpoint
    COMMENTARY = "commentary"            # Response to other work
    CASE_STUDY = "case_study"            # Single case analysis


class ResearchType(str, Enum):
    """Types of research methodology."""
    
    EMPIRICAL = "empirical"              # Data-driven analysis
    THEORETICAL = "theoretical"          # Model-based/conceptual
    MIXED = "mixed"                      # Empirical + theoretical
    LITERATURE_REVIEW = "literature_review"  # Survey of existing work
    EXPERIMENTAL = "experimental"        # Controlled study
    META_ANALYSIS = "meta_analysis"      # Analysis of prior studies
    CASE_STUDY = "case_study"            # In-depth single case


class TargetJournal(str, Enum):
    """Target academic journals for finance papers."""
    
    RFS = "RFS"              # Review of Financial Studies
    JFE = "JFE"              # Journal of Financial Economics
    JF = "JF"                # Journal of Finance
    JFQA = "JFQA"            # Journal of Financial and Quantitative Analysis
    MANAGEMENT_SCIENCE = "Management Science"
    OTHER = "Other"


class DataQualityLevel(str, Enum):
    """Data quality assessment levels."""
    
    NOT_ASSESSED = "not_assessed"  # Not yet analyzed
    EXCELLENT = "excellent"  # No issues, ready for analysis
    GOOD = "good"            # Minor issues, can proceed
    FAIR = "fair"            # Some issues, needs cleaning
    POOR = "poor"            # Major issues, may not be usable
    UNUSABLE = "unusable"    # Cannot be used as-is


class ColumnType(str, Enum):
    """Detected column types in data files."""
    
    NUMERIC = "numeric"
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    STRING = "string"
    TEXT = "text"
    DATE = "date"
    DATETIME = "datetime"
    TIMEDELTA = "timedelta"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"  # IDs, codes
    UNKNOWN = "unknown"


class GapType(str, Enum):
    """Types of research gaps identified in literature."""
    
    METHODOLOGICAL = "methodological"  # New methods or approaches needed
    EMPIRICAL = "empirical"            # Untested contexts, populations, or settings
    THEORETICAL = "theoretical"        # Unexplained phenomena, missing frameworks
    CONTEXTUAL = "contextual"          # Not tested in specific context/region
    TEMPORAL = "temporal"              # Not studied in recent time periods
    CONFLICTING = "conflicting"        # Contradictory findings need resolution


class GapSignificance(str, Enum):
    """Significance level of identified research gaps."""
    
    HIGH = "high"        # Major gap with strong potential impact
    MEDIUM = "medium"    # Notable gap worth addressing
    LOW = "low"          # Minor gap, incremental contribution


class MethodologyType(str, Enum):
    """Types of research methodology for the PLANNER node."""
    
    # Quantitative Methods
    REGRESSION_ANALYSIS = "regression_analysis"
    EVENT_STUDY = "event_study"
    PANEL_DATA = "panel_data"
    TIME_SERIES = "time_series"
    CROSS_SECTIONAL = "cross_sectional"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    INSTRUMENTAL_VARIABLES = "instrumental_variables"
    PROPENSITY_SCORE_MATCHING = "propensity_score_matching"
    
    # Qualitative Methods
    CASE_STUDY = "case_study"
    CONTENT_ANALYSIS = "content_analysis"
    GROUNDED_THEORY = "grounded_theory"
    THEMATIC_ANALYSIS = "thematic_analysis"
    
    # Mixed Methods
    SEQUENTIAL_MIXED = "sequential_mixed"
    CONCURRENT_MIXED = "concurrent_mixed"
    
    # Theoretical Methods
    ANALYTICAL_MODEL = "analytical_model"
    SIMULATION = "simulation"
    CONCEPTUAL_FRAMEWORK = "conceptual_framework"
    
    # Review Methods
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    NARRATIVE_REVIEW = "narrative_review"
    REPLICATION = "replication"
    
    # Other
    OTHER = "other"


class AnalysisApproach(str, Enum):
    """Analysis approaches for research methodology."""
    
    # Statistical Analysis
    DESCRIPTIVE_STATISTICS = "descriptive_statistics"
    INFERENTIAL_STATISTICS = "inferential_statistics"
    MULTIVARIATE_ANALYSIS = "multivariate_analysis"
    BAYESIAN_ANALYSIS = "bayesian_analysis"
    
    # Econometric Analysis
    OLS_REGRESSION = "ols_regression"
    FIXED_EFFECTS = "fixed_effects"
    RANDOM_EFFECTS = "random_effects"
    GMM = "gmm"  # Generalized Method of Moments
    TWO_STAGE_LEAST_SQUARES = "2sls"
    
    # Finance-Specific
    ASSET_PRICING_TESTS = "asset_pricing_tests"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    OPTION_PRICING_MODELS = "option_pricing_models"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    
    # Qualitative Analysis
    CODING_ANALYSIS = "coding_analysis"
    NARRATIVE_ANALYSIS = "narrative_analysis"
    DISCOURSE_ANALYSIS = "discourse_analysis"
    
    # Other
    OTHER = "other"


class PlanApprovalStatus(str, Enum):
    """Status of research plan approval."""
    
    PENDING = "pending"
    APPROVED = "approved"
    REVISION_REQUESTED = "revision_requested"
    REJECTED = "rejected"
