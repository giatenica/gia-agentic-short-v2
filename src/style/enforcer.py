"""Main style enforcer that combines all style checks.

Integrates BannedWordsFilter, AcademicToneChecker, HedgingLanguageChecker,
PrecisionChecker, and JournalStyleMatcher into a single interface.
"""

from dataclasses import dataclass

from src.state.models import StyleViolation
from src.style.banned_words import BannedWordsFilter
from src.style.academic_tone import AcademicToneChecker
from src.style.hedging import HedgingLanguageChecker
from src.style.precision import PrecisionChecker
from src.style.journal_style import JournalStyleMatcher


@dataclass
class StyleCheckResult:
    """Result of a comprehensive style check."""
    
    violations: list[StyleViolation]
    total_count: int
    by_type: dict[str, int]
    by_severity: dict[str, int]
    score: float  # 0-100, higher is better
    passed: bool
    summary: str


class StyleEnforcer:
    """
    Comprehensive style enforcement for academic writing.
    
    Combines all style checkers:
    - BannedWordsFilter: Marketing buzzwords and informal language
    - AcademicToneChecker: Contractions, colloquialisms, overclaims
    - HedgingLanguageChecker: Appropriate hedging for claims
    - PrecisionChecker: Vague terms and imprecise language
    - JournalStyleMatcher: Journal-specific requirements
    """
    
    def __init__(
        self,
        target_journal: str = "generic",
        sensitivity: str = "medium",
        check_banned_words: bool = True,
        check_tone: bool = True,
        check_hedging: bool = True,
        check_precision: bool = True,
        check_journal_style: bool = True,
    ):
        """
        Initialize the style enforcer.
        
        Args:
            target_journal: Target journal (rfs, jfe, jf, jfqa, generic).
            sensitivity: Sensitivity level (low, medium, high).
            check_banned_words: Enable banned words checking.
            check_tone: Enable academic tone checking.
            check_hedging: Enable hedging language checking.
            check_precision: Enable precision checking.
            check_journal_style: Enable journal-specific checks.
        """
        self.target_journal = target_journal.lower()
        self.sensitivity = sensitivity
        
        # Initialize checkers
        self.banned_words_filter = BannedWordsFilter() if check_banned_words else None
        self.tone_checker = AcademicToneChecker() if check_tone else None
        self.hedging_checker = (
            HedgingLanguageChecker(sensitivity=sensitivity) if check_hedging else None
        )
        self.precision_checker = (
            PrecisionChecker(sensitivity=sensitivity) if check_precision else None
        )
        self.journal_matcher = (
            JournalStyleMatcher(target_journal) if check_journal_style else None
        )
        
        # Track enabled checkers
        self.enabled_checkers = {
            "banned_words": check_banned_words,
            "tone": check_tone,
            "hedging": check_hedging,
            "precision": check_precision,
            "journal_style": check_journal_style,
        }
    
    @classmethod
    def from_guide(cls, guide_path: str, target_journal: str = "generic") -> "StyleEnforcer":
        """
        Create a StyleEnforcer from a style guide file.
        
        Args:
            guide_path: Path to style guide markdown file.
            target_journal: Target journal.
            
        Returns:
            Configured StyleEnforcer.
        """
        # In a full implementation, this would parse the guide file
        # For now, we use hardcoded values that match the guide
        return cls(target_journal=target_journal)
    
    def check(
        self,
        text: str,
        abstract: str | None = None,
        section_name: str | None = None,
    ) -> list[StyleViolation]:
        """
        Check text against all enabled style rules.
        
        Args:
            text: Text to check.
            abstract: Abstract text (for word count check).
            section_name: Name of section (for context).
            
        Returns:
            List of StyleViolation objects.
        """
        violations: list[StyleViolation] = []
        
        # Banned words check
        if self.banned_words_filter:
            violations.extend(self.banned_words_filter.check(text))
        
        # Academic tone check
        if self.tone_checker:
            violations.extend(self.tone_checker.check(text))
        
        # Hedging check
        if self.hedging_checker:
            violations.extend(self.hedging_checker.check(text))
        
        # Precision check
        if self.precision_checker:
            violations.extend(self.precision_checker.check(text))
        
        # Journal style check
        if self.journal_matcher:
            violations.extend(self.journal_matcher.check(text, abstract))
        
        # Add section context to violations if provided
        if section_name:
            for v in violations:
                if v.location and not v.location.startswith(section_name):
                    v.location = f"{section_name}, {v.location}"
        
        return violations
    
    def check_full(
        self,
        text: str,
        abstract: str | None = None,
    ) -> StyleCheckResult:
        """
        Perform comprehensive style check with summary.
        
        Args:
            text: Text to check.
            abstract: Abstract text.
            
        Returns:
            StyleCheckResult with violations and summary.
        """
        violations = self.check(text, abstract)
        
        # Count by type
        by_type: dict[str, int] = {}
        for v in violations:
            by_type[v.violation_type] = by_type.get(v.violation_type, 0) + 1
        
        # Count by severity
        by_severity: dict[str, int] = {
            "critical": 0,
            "major": 0,
            "minor": 0,
            "suggestion": 0,
        }
        for v in violations:
            severity_key = v.severity.value if hasattr(v.severity, "value") else str(v.severity)
            if severity_key in by_severity:
                by_severity[severity_key] += 1
        
        # Calculate score (simple heuristic)
        # Start at 100, deduct points for violations
        word_count = len(text.split())
        violations_per_100 = (len(violations) / max(word_count, 1)) * 100
        
        # Weighted deductions
        score = 100.0
        score -= by_severity["critical"] * 20
        score -= by_severity["major"] * 10
        score -= by_severity["minor"] * 2
        score -= by_severity["suggestion"] * 0.5
        score = max(0.0, min(100.0, score))
        
        # Pass threshold
        passed = score >= 70 and by_severity["critical"] == 0
        
        # Generate summary
        summary_parts = []
        if len(violations) == 0:
            summary_parts.append("No style issues found.")
        else:
            summary_parts.append(f"Found {len(violations)} style issue(s).")
            
            if by_severity["critical"] > 0:
                summary_parts.append(f"{by_severity['critical']} critical issue(s) must be fixed.")
            if by_severity["major"] > 0:
                summary_parts.append(f"{by_severity['major']} major issue(s) should be addressed.")
            
            # Top violation types
            if by_type:
                top_types = sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:3]
                type_summary = ", ".join(f"{t}: {c}" for t, c in top_types)
                summary_parts.append(f"Top issues: {type_summary}")
        
        summary = " ".join(summary_parts)
        
        return StyleCheckResult(
            violations=violations,
            total_count=len(violations),
            by_type=by_type,
            by_severity=by_severity,
            score=round(score, 1),
            passed=passed,
            summary=summary,
        )
    
    def auto_fix(self, text: str) -> tuple[str, int]:
        """
        Auto-fix violations that can be fixed automatically.
        
        Args:
            text: Text to fix.
            
        Returns:
            Tuple of (fixed text, number of fixes).
        """
        total_fixes = 0
        result = text
        
        # Fix banned words
        if self.banned_words_filter:
            result, count = self.banned_words_filter.replace_banned_words(result)
            total_fixes += count
        
        # Expand contractions
        if self.tone_checker:
            result, count = self.tone_checker.expand_contractions(result)
            total_fixes += count
        
        return result, total_fixes
    
    def get_violation_summary(self, violations: list[StyleViolation]) -> str:
        """
        Generate a human-readable summary of violations.
        
        Args:
            violations: List of violations.
            
        Returns:
            Summary string.
        """
        if not violations:
            return "No style violations found."
        
        lines = [f"Found {len(violations)} style violation(s):"]
        
        # Group by type
        by_type: dict[str, list[StyleViolation]] = {}
        for v in violations:
            by_type.setdefault(v.violation_type, []).append(v)
        
        for vtype, vlist in by_type.items():
            lines.append(f"\n{vtype.replace('_', ' ').title()} ({len(vlist)}):")
            for v in vlist[:5]:  # Show first 5 of each type
                lines.append(f"  - {v.location}: '{v.original_text}' -> {v.suggestion}")
            if len(vlist) > 5:
                lines.append(f"  ... and {len(vlist) - 5} more")
        
        return "\n".join(lines)
