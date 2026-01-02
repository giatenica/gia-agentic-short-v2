"""Journal style matcher for target finance journals.

Matches writing style to specific journal conventions:
- Review of Financial Studies (RFS)
- Journal of Financial Economics (JFE)
- Journal of Finance (JF)
- Journal of Financial and Quantitative Analysis (JFQA)
"""

import re
from dataclasses import dataclass
from enum import Enum

from src.state.models import StyleViolation
from src.state.enums import CritiqueSeverity


class Journal(str, Enum):
    """Target finance journals."""
    
    RFS = "rfs"    # Review of Financial Studies
    JFE = "jfe"    # Journal of Financial Economics
    JF = "jf"      # Journal of Finance
    JFQA = "jfqa"  # Journal of Financial and Quantitative Analysis
    GENERIC = "generic"


@dataclass
class JournalStyle:
    """Style specifications for a journal."""
    
    name: str
    abbreviation: str
    max_abstract_words: int
    max_pages_short: int
    max_pages_full: int
    citation_style: str
    requires_jel_codes: bool
    requires_keywords: bool
    allows_online_appendix: bool
    table_notes_format: str
    significance_format: str  # How to show significance stars
    section_numbering: bool
    em_dash_allowed: bool = False


# Journal style specifications
JOURNAL_STYLES: dict[str, JournalStyle] = {
    "rfs": JournalStyle(
        name="Review of Financial Studies",
        abbreviation="RFS",
        max_abstract_words=100,
        max_pages_short=10,
        max_pages_full=45,
        citation_style="chicago_author_date",
        requires_jel_codes=True,
        requires_keywords=True,
        allows_online_appendix=True,
        table_notes_format="below_table",
        significance_format="*** p<0.01, ** p<0.05, * p<0.1",
        section_numbering=True,
    ),
    "jfe": JournalStyle(
        name="Journal of Financial Economics",
        abbreviation="JFE",
        max_abstract_words=100,
        max_pages_short=10,
        max_pages_full=40,
        citation_style="chicago_author_date",
        requires_jel_codes=True,
        requires_keywords=True,
        allows_online_appendix=True,
        table_notes_format="below_table",
        significance_format="*** p<0.01, ** p<0.05, * p<0.1",
        section_numbering=True,
    ),
    "jf": JournalStyle(
        name="Journal of Finance",
        abbreviation="JF",
        max_abstract_words=100,
        max_pages_short=10,
        max_pages_full=50,
        citation_style="chicago_author_date",
        requires_jel_codes=False,
        requires_keywords=True,
        allows_online_appendix=True,
        table_notes_format="below_table",
        significance_format="*** p<0.01, ** p<0.05, * p<0.1",
        section_numbering=True,
    ),
    "jfqa": JournalStyle(
        name="Journal of Financial and Quantitative Analysis",
        abbreviation="JFQA",
        max_abstract_words=100,
        max_pages_short=10,
        max_pages_full=40,
        citation_style="chicago_author_date",
        requires_jel_codes=True,
        requires_keywords=True,
        allows_online_appendix=True,
        table_notes_format="below_table",
        significance_format="*** p<0.01, ** p<0.05, * p<0.1",
        section_numbering=True,
    ),
    "generic": JournalStyle(
        name="Generic Academic",
        abbreviation="GENERIC",
        max_abstract_words=150,
        max_pages_short=15,
        max_pages_full=50,
        citation_style="chicago_author_date",
        requires_jel_codes=False,
        requires_keywords=False,
        allows_online_appendix=True,
        table_notes_format="below_table",
        significance_format="*** p<0.01, ** p<0.05, * p<0.1",
        section_numbering=True,
    ),
}


@dataclass
class JournalStyleIssue:
    """A journal style issue found in text."""
    
    issue_type: str
    description: str
    location: str
    suggestion: str


class JournalStyleMatcher:
    """Match text to journal-specific style requirements."""
    
    def __init__(self, journal: str = "generic"):
        """
        Initialize the journal style matcher.
        
        Args:
            journal: Target journal (rfs, jfe, jf, jfqa, generic).
        """
        self.journal = journal.lower()
        self.style = JOURNAL_STYLES.get(self.journal, JOURNAL_STYLES["generic"])
    
    def check_abstract_length(self, abstract: str) -> JournalStyleIssue | None:
        """Check if abstract meets word count limit."""
        word_count = len(abstract.split())
        
        if word_count > self.style.max_abstract_words:
            return JournalStyleIssue(
                issue_type="word_count_violation",
                description=f"Abstract is {word_count} words; {self.style.abbreviation} limit is {self.style.max_abstract_words}",
                location="abstract",
                suggestion=f"Reduce abstract to {self.style.max_abstract_words} words or fewer",
            )
        
        return None
    
    def check_page_count(self, page_count: int, paper_type: str = "short") -> JournalStyleIssue | None:
        """Check if page count is within limits."""
        limit = self.style.max_pages_short if paper_type == "short" else self.style.max_pages_full
        
        if page_count > limit:
            return JournalStyleIssue(
                issue_type="word_count_violation",
                description=f"Paper is {page_count} pages; {self.style.abbreviation} limit for {paper_type} papers is {limit}",
                location="document",
                suggestion=f"Reduce to {limit} pages or fewer",
            )
        
        return None
    
    def check_em_dashes(self, text: str) -> list[JournalStyleIssue]:
        """Check for em dashes (should be avoided per project guidelines)."""
        issues = []
        
        # Find em dashes (—) and en dashes (–)
        em_dash_pattern = re.compile(r'[—–]')
        
        lines = text.split('\n')
        for line_num, line in enumerate(lines, 1):
            for _ in em_dash_pattern.finditer(line):
                issues.append(JournalStyleIssue(
                    issue_type="journal_mismatch",
                    description="Em dash or en dash found; use semicolons, colons, or periods instead",
                    location=f"line {line_num}",
                    suggestion="Replace with semicolon, colon, or separate sentences",
                ))
        
        return issues
    
    def check_section_numbering(self, text: str) -> list[JournalStyleIssue]:
        """Check for proper section numbering format."""
        issues = []
        
        if not self.style.section_numbering:
            return issues
        
        # Look for section headers
        section_pattern = re.compile(r'^#+\s*(\d+\.?\d*\.?\d*\.?)?\s*([A-Z])', re.MULTILINE)
        
        # This is a basic check - could be expanded
        matches = list(section_pattern.finditer(text))
        
        # Check if sections are numbered when they should be
        unnumbered_sections = [m for m in matches if not m.group(1)]
        
        if unnumbered_sections and self.style.section_numbering:
            for match in unnumbered_sections[:3]:  # Report first 3
                issues.append(JournalStyleIssue(
                    issue_type="journal_mismatch",
                    description=f"{self.style.abbreviation} requires numbered sections",
                    location=f"section starting with '{match.group(0)[:30]}'",
                    suggestion="Add section numbers (e.g., '1. Introduction')",
                ))
        
        return issues
    
    def check_citation_format(self, text: str) -> list[JournalStyleIssue]:
        """Check for proper citation format."""
        issues = []
        
        # Check for common citation format issues
        # Wrong formats to flag
        wrong_formats = [
            (r'\[\d+\]', "Numbered citations; use author-date format"),
            (r'\(\d{4}\s*[,;]\s*p\.?\s*\d+\)', "Page number format; use (Author Year, page)"),
            (r'et\.\s+al\.', "Use 'et al.' without space after 'et'"),
        ]
        
        lines = text.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, description in wrong_formats:
                if re.search(pattern, line):
                    issues.append(JournalStyleIssue(
                        issue_type="citation_format",
                        description=description,
                        location=f"line {line_num}",
                        suggestion=f"Use Chicago Author-Date format: (Author Year)",
                    ))
                    break  # One issue per line
        
        return issues
    
    def check(self, text: str, abstract: str | None = None) -> list[StyleViolation]:
        """
        Check text for journal style issues.
        
        Args:
            text: Main text to check.
            abstract: Abstract text (optional).
            
        Returns:
            List of StyleViolation objects.
        """
        violations: list[StyleViolation] = []
        issues: list[JournalStyleIssue] = []
        
        # Check abstract length
        if abstract:
            issue = self.check_abstract_length(abstract)
            if issue:
                issues.append(issue)
        
        # Check em dashes
        issues.extend(self.check_em_dashes(text))
        
        # Check citation format
        issues.extend(self.check_citation_format(text))
        
        # Convert to StyleViolations
        for issue in issues:
            severity = (
                CritiqueSeverity.MAJOR if issue.issue_type == "word_count_violation"
                else CritiqueSeverity.MINOR
            )
            
            violations.append(StyleViolation(
                violation_type=issue.issue_type,
                severity=severity,
                location=issue.location,
                original_text=issue.description,
                suggestion=issue.suggestion,
                rule_reference=f"docs/writing_style_guide.md - {self.style.abbreviation} Guidelines",
                auto_fixable=False,
            ))
        
        return violations
    
    def get_style_summary(self) -> dict:
        """Get a summary of style requirements for the target journal."""
        return {
            "journal": self.style.name,
            "abbreviation": self.style.abbreviation,
            "abstract_max_words": self.style.max_abstract_words,
            "short_paper_max_pages": self.style.max_pages_short,
            "full_paper_max_pages": self.style.max_pages_full,
            "citation_style": self.style.citation_style,
            "requires_jel_codes": self.style.requires_jel_codes,
            "requires_keywords": self.style.requires_keywords,
            "allows_online_appendix": self.style.allows_online_appendix,
            "section_numbering": self.style.section_numbering,
        }
