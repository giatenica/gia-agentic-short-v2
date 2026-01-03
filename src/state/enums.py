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


class FindingType(str, Enum):
    """Types of research findings."""
    
    MAIN_RESULT = "main_result"      # Primary finding addressing research question
    SUPPORTING = "supporting"        # Supporting evidence for main findings
    UNEXPECTED = "unexpected"        # Unexpected or novel findings
    NULL_RESULT = "null_result"      # No significant effect found
    ROBUSTNESS = "robustness"        # Robustness check results
    SENSITIVITY = "sensitivity"      # Sensitivity analysis results
    EXPLORATORY = "exploratory"      # Exploratory analysis findings


class AnalysisStatus(str, Enum):
    """Status of analysis execution."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    PARTIAL = "partial"  # Some analyses complete, others failed


class StatisticalTestType(str, Enum):
    """Types of statistical tests."""
    
    # Parametric tests
    T_TEST = "t_test"
    PAIRED_T_TEST = "paired_t_test"
    ANOVA = "anova"
    F_TEST = "f_test"
    
    # Non-parametric tests
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    KRUSKAL_WALLIS = "kruskal_wallis"
    CHI_SQUARE = "chi_square"
    
    # Correlation tests
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    
    # Regression diagnostics
    HETEROSKEDASTICITY = "heteroskedasticity"
    AUTOCORRELATION = "autocorrelation"
    NORMALITY = "normality"
    
    # Other
    OTHER = "other"


class ConceptType(str, Enum):
    """Types of theoretical concepts."""
    
    CONSTRUCT = "construct"          # Theoretical construct
    VARIABLE = "variable"            # Measurable variable
    MECHANISM = "mechanism"          # Causal mechanism
    MODERATOR = "moderator"          # Moderating factor
    MEDIATOR = "mediator"            # Mediating factor
    BOUNDARY_CONDITION = "boundary_condition"  # Scope condition
    OUTCOME = "outcome"              # Dependent/outcome concept


class RelationshipType(str, Enum):
    """Types of conceptual relationships."""
    
    CAUSAL = "causal"                # X causes Y
    CORRELATIONAL = "correlational"  # X correlates with Y
    MODERATING = "moderating"        # X moderates Y-Z relationship
    MEDIATING = "mediating"          # X mediates Y-Z relationship
    CONDITIONAL = "conditional"      # X depends on condition
    RECIPROCAL = "reciprocal"        # X and Y mutually influence
    HIERARCHICAL = "hierarchical"    # X is higher-order than Y


class PropositionStatus(str, Enum):
    """Status of theoretical propositions."""
    
    PROPOSED = "proposed"            # Newly proposed
    SUPPORTED = "supported"          # Has empirical support
    PARTIALLY_SUPPORTED = "partially_supported"
    UNTESTED = "untested"            # Not yet tested
    REJECTED = "rejected"            # Empirically rejected


# =============================================================================
# Sprint 6: WRITER Node Enums
# =============================================================================


class SectionType(str, Enum):
    """Types of paper sections."""
    
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    LITERATURE_REVIEW = "literature_review"
    METHODS = "methods"
    DATA = "data"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    APPENDIX = "appendix"
    REFERENCES = "references"


class WritingStatus(str, Enum):
    """Status of section writing."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DRAFT_COMPLETE = "draft_complete"
    STYLE_CHECKED = "style_checked"
    CITATIONS_VERIFIED = "citations_verified"
    FINALIZED = "finalized"


class StyleViolationType(str, Enum):
    """Types of style guide violations."""
    
    BANNED_WORD = "banned_word"
    INFORMAL_TONE = "informal_tone"
    MISSING_HEDGE = "missing_hedge"
    VAGUE_TERM = "vague_term"
    CITATION_FORMAT = "citation_format"
    OVERCLAIM = "overclaim"
    JOURNAL_MISMATCH = "journal_mismatch"
    CONTRACTION = "contraction"
    PASSIVE_VOICE_OVERUSE = "passive_voice_overuse"
    WORD_COUNT_VIOLATION = "word_count_violation"


class CitationStyle(str, Enum):
    """Citation formatting styles."""
    
    CHICAGO_AUTHOR_DATE = "chicago_author_date"
    APA = "apa"
    MLA = "mla"
    HARVARD = "harvard"


class JournalTarget(str, Enum):
    """Target academic journals with specific style requirements."""
    
    RFS = "rfs"              # Review of Financial Studies
    JFE = "jfe"              # Journal of Financial Economics
    JF = "jf"                # Journal of Finance
    JFQA = "jfqa"            # Journal of Financial and Quantitative Analysis
    MANAGEMENT_SCIENCE = "management_science"
    GENERIC = "generic"      # Generic academic style


# =============================================================================
# Sprint 7: REVIEWER Node Enums
# =============================================================================


class ReviewDecision(str, Enum):
    """Decision from the REVIEWER node."""
    
    APPROVE = "approve"      # Paper meets quality standards
    REVISE = "revise"        # Paper needs revision
    REJECT = "reject"        # Paper has fundamental issues


class ReviewDimension(str, Enum):
    """Dimensions of paper quality to evaluate."""
    
    CONTRIBUTION = "contribution"    # Novelty, significance, clarity
    METHODOLOGY = "methodology"      # Rigor, appropriateness, reproducibility
    EVIDENCE = "evidence"            # Data quality, analysis validity
    COHERENCE = "coherence"          # Logical flow, argument structure
    WRITING = "writing"              # Academic tone, clarity, style


class RevisionPriority(str, Enum):
    """Priority of revision items."""
    
    CRITICAL = "critical"    # Must fix before approval
    HIGH = "high"            # Strongly recommended
    MEDIUM = "medium"        # Should address
    LOW = "low"              # Nice to have


# =============================================================================
# Sprint 12: Enhanced Data Explorer Enums
# =============================================================================


class QualityFlag(str, Enum):
    """Data quality flags for data exploration.
    
    These flags indicate specific issues detected during data profiling
    that may affect analysis or require user attention.
    """
    
    # Missing data issues
    MISSING_VALUES = "missing_values"              # Significant missing data detected
    HIGH_MISSING_RATE = "high_missing_rate"        # >20% missing in one or more columns
    
    # File/format issues
    UNREADABLE_FILE = "unreadable_file"            # File could not be parsed
    ENCODING_ERROR = "encoding_error"              # Character encoding issues
    MALFORMED_FILE = "malformed_file"              # Structural issues with file
    UNSUPPORTED_FORMAT = "unsupported_format"      # File format not supported
    
    # Schema issues
    SCHEMA_MISMATCH = "schema_mismatch"            # Inconsistent schema across files
    DUPLICATE_COLUMNS = "duplicate_columns"        # Duplicate column names
    
    # Date/time issues
    DATE_PARSING_FAILED = "date_parsing_failed"    # Could not parse date columns
    INCONSISTENT_DATES = "inconsistent_dates"      # Mixed date formats
    FUTURE_DATES = "future_dates"                  # Dates in the future detected
    
    # Data quality issues
    LOW_SAMPLE_SIZE = "low_sample_size"            # Fewer than 30 observations
    HIGH_CARDINALITY = "high_cardinality"          # Categorical with too many unique values
    CONSTANT_COLUMN = "constant_column"            # Column has single value
    ALL_NULL_COLUMN = "all_null_column"            # Column is entirely null
    DUPLICATE_ROWS = "duplicate_rows"              # Significant duplicate rows
    
    # Statistical issues
    OUTLIERS_DETECTED = "outliers_detected"        # Extreme values detected
    HIGHLY_SKEWED = "highly_skewed"                # Distribution highly non-normal
    MULTICOLLINEARITY = "multicollinearity"        # High correlation between variables
    
    # Panel/time series specific
    UNBALANCED_PANEL = "unbalanced_panel"          # Panel data not balanced
    GAPS_IN_TIME_SERIES = "gaps_in_time_series"    # Missing time periods
    
    # General
    OTHER = "other"                                # Other quality issue


class DataStructureType(str, Enum):
    """Detected structure of the dataset."""
    
    CROSS_SECTIONAL = "cross_sectional"    # Single point in time, multiple entities
    TIME_SERIES = "time_series"            # Single entity over time
    PANEL = "panel"                        # Multiple entities over time (longitudinal)
    POOLED = "pooled"                      # Pooled cross-sections
    HIERARCHICAL = "hierarchical"          # Nested/multilevel structure
    UNKNOWN = "unknown"                    # Structure could not be determined


# =============================================================================
# Sprint 14: Data Acquisition Enums
# =============================================================================


class DataRequirementPriority(str, Enum):
    """Priority level for data requirements."""
    
    REQUIRED = "required"      # Must have this data for analysis
    PREFERRED = "preferred"    # Strongly desired but can proceed without
    OPTIONAL = "optional"      # Nice to have but not essential


class AcquisitionStatus(str, Enum):
    """Status of a data acquisition task."""
    
    PENDING = "pending"              # Not yet attempted
    IN_PROGRESS = "in_progress"      # Currently acquiring
    SUCCESS = "success"              # Successfully acquired
    PARTIAL = "partial"              # Partially acquired (some data missing)
    FAILED = "failed"                # Failed to acquire
    SKIPPED = "skipped"              # Skipped (e.g., already available)


class CodeExecutionStatus(str, Enum):
    """Status of code execution."""
    
    NOT_EXECUTED = "not_executed"    # Not yet run
    RUNNING = "running"              # Currently executing
    SUCCESS = "success"              # Executed successfully
    ERROR = "error"                  # Runtime error
    TIMEOUT = "timeout"              # Exceeded time limit
    VALIDATION_FAILED = "validation_failed"  # Code failed safety checks


# =============================================================================
# Sprint 15: Visualization & Table Generation Enums
# =============================================================================


class ArtifactFormat(str, Enum):
    """Output format for table artifacts."""
    
    LATEX = "LATEX"        # LaTeX tabular format
    MARKDOWN = "MARKDOWN"  # Markdown table format
    HTML = "HTML"          # HTML table format


class FigureFormat(str, Enum):
    """Output format for figure artifacts."""
    
    PNG = "PNG"    # PNG image (default)
    PDF = "PDF"    # PDF vector format
    SVG = "SVG"    # SVG vector format
