"""Precision checker for academic writing.

Checks for vague terms and encourages specific, quantified language.
Academic writing requires precision and specificity.
"""

import re
from dataclasses import dataclass

from src.state.models import StyleViolation
from src.state.enums import CritiqueSeverity


# Vague terms to flag
VAGUE_TERMS: dict[str, str] = {
    "many": "Specify the number or percentage",
    "some": "Specify how many or provide a range",
    "several": "Specify the number",
    "few": "Specify the number or percentage",
    "various": "List the specific items",
    "numerous": "Provide the actual count",
    "a number of": "Specify the number",
    "a lot of": "Quantify specifically",
    "lots of": "Quantify specifically",
    "most": "Specify the percentage (e.g., '75%')",
    "majority": "Specify the percentage",
    "minority": "Specify the percentage",
    "significant": "Define 'significant' (statistical or economic)",
    "substantial": "Quantify the magnitude",
    "considerable": "Quantify the amount",
    "large": "Specify the size or magnitude",
    "small": "Specify the size or magnitude",
    "big": "Specify the size or magnitude",
    "huge": "Quantify the magnitude",
    "tiny": "Quantify the size",
    "high": "Specify the value or range",
    "low": "Specify the value or range",
    "good": "Define criteria for 'good'",
    "bad": "Define criteria or use specific descriptor",
    "important": "Explain why it is important",
    "interesting": "Explain the significance",
    "notable": "Specify what makes it notable",
    "remarkable": "Explain what is remarkable",
    "recently": "Specify the time period",
    "soon": "Specify the timeline",
    "often": "Quantify frequency",
    "sometimes": "Quantify frequency",
    "rarely": "Quantify frequency",
    "usually": "Quantify frequency (e.g., '80% of cases')",
    "generally": "Specify conditions or frequency",
    "typically": "Quantify or specify conditions",
    "roughly": "Provide precise figure with confidence interval",
    "approximately": "Provide precise figure with bounds",
    "about": "Provide precise figure or range",
    "around": "Provide precise figure or range",
    "almost": "Quantify (e.g., '95%')",
    "nearly": "Quantify (e.g., '90%')",
    "somewhat": "Quantify the degree",
    "quite": "Quantify the degree",
    "very": "Use a stronger adjective or quantify",
    "really": "Remove or use specific descriptor",
    "fairly": "Quantify the degree",
    "rather": "Quantify or remove",
    "relatively": "Specify what it is relative to",
    "things": "Name the specific items",
    "stuff": "Name the specific items",
    "etc.": "List all items or use 'among others'",
    "and so on": "List all items or be more specific",
    "et cetera": "List all items or use 'among others'",
}

# Phrases that indicate imprecision
IMPRECISE_PHRASES: list[str] = [
    "a growing body of",
    "increasing evidence",
    "widespread",
    "common",
    "well-known",
    "well documented",
    "extensively studied",
    "long been recognized",
    "in recent years",
    "over the years",
    "in the literature",
    "previous research",
    "prior studies",
    "existing work",
]


@dataclass
class PrecisionIssue:
    """A precision issue found in text."""
    
    term: str
    suggestion: str
    position: int
    line_number: int
    context: str


class PrecisionChecker:
    """Check for vague terms and imprecision in academic text."""
    
    def __init__(
        self,
        vague_terms: dict[str, str] | None = None,
        sensitivity: str = "medium",  # low, medium, high
    ):
        """
        Initialize the precision checker.
        
        Args:
            vague_terms: Dictionary of vague terms and suggestions.
            sensitivity: How strict to be (low, medium, high).
        """
        self.vague_terms = vague_terms or VAGUE_TERMS
        self.sensitivity = sensitivity
        
        # High sensitivity checks all terms
        # Medium sensitivity skips some common ones
        # Low sensitivity only checks most problematic terms
        if sensitivity == "low":
            self.check_terms = {
                k: v for k, v in self.vague_terms.items()
                if k in ["many", "some", "various", "significant", "etc."]
            }
        elif sensitivity == "medium":
            self.check_terms = {
                k: v for k, v in self.vague_terms.items()
                if k not in ["very", "really", "quite", "rather", "fairly"]
            }
        else:
            self.check_terms = self.vague_terms
        
        # Build pattern
        sorted_terms = sorted(self.check_terms.keys(), key=len, reverse=True)
        escaped = [re.escape(t) for t in sorted_terms]
        pattern = r'\b(' + '|'.join(escaped) + r')\b'
        self._pattern = re.compile(pattern, re.IGNORECASE)
    
    def find_issues(self, text: str) -> list[PrecisionIssue]:
        """
        Find precision issues in text.
        
        Args:
            text: Text to check.
            
        Returns:
            List of PrecisionIssue objects.
        """
        issues: list[PrecisionIssue] = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for match in self._pattern.finditer(line):
                term = match.group(1).lower()
                
                # Get suggestion
                suggestion = None
                for key, sugg in self.check_terms.items():
                    if key.lower() == term:
                        suggestion = sugg
                        break
                
                if suggestion:
                    # Get context
                    start = max(0, match.start() - 20)
                    end = min(len(line), match.end() + 20)
                    context = line[start:end]
                    if start > 0:
                        context = "..." + context
                    if end < len(line):
                        context = context + "..."
                    
                    issues.append(PrecisionIssue(
                        term=match.group(1),
                        suggestion=suggestion,
                        position=match.start(),
                        line_number=line_num,
                        context=context,
                    ))
        
        return issues
    
    def check(self, text: str) -> list[StyleViolation]:
        """
        Check text for precision issues and return violations.
        
        Args:
            text: Text to check.
            
        Returns:
            List of StyleViolation objects.
        """
        violations: list[StyleViolation] = []
        issues = self.find_issues(text)
        
        for issue in issues:
            violations.append(StyleViolation(
                violation_type="vague_term",
                severity=CritiqueSeverity.MINOR,
                location=f"line {issue.line_number}, position {issue.position}",
                original_text=issue.term,
                suggestion=issue.suggestion,
                rule_reference="docs/writing_style_guide.md - Precision",
                auto_fixable=False,  # Precision requires domain knowledge
            ))
        
        return violations
    
    def count_vague_terms(self, text: str) -> dict[str, int]:
        """
        Count occurrences of each vague term.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Dictionary of term -> count.
        """
        counts: dict[str, int] = {}
        
        for match in self._pattern.finditer(text.lower()):
            term = match.group(1)
            counts[term] = counts.get(term, 0) + 1
        
        return counts
    
    def precision_score(self, text: str) -> float:
        """
        Calculate a precision score for the text.
        
        Higher score = more precise language.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Score from 0 to 1.
        """
        word_count = len(text.split())
        if word_count == 0:
            return 1.0
        
        issues = self.find_issues(text)
        issue_count = len(issues)
        
        # Score based on issues per 100 words
        issues_per_100 = (issue_count / word_count) * 100
        
        # Map to 0-1 score (0 issues = 1.0, 10+ issues per 100 words = 0.0)
        score = max(0.0, 1.0 - (issues_per_100 / 10))
        
        return round(score, 2)
