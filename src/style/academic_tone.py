"""Academic tone checker for formal academic writing.

Checks text for informal language, contractions, colloquialisms,
and other tone issues.
"""

import re
from dataclasses import dataclass

from src.state.models import StyleViolation
from src.state.enums import CritiqueSeverity


# Common contractions to flag
CONTRACTIONS: dict[str, str] = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "what's": "what is",
    "who's": "who is",
    "let's": "let us",
    "I'm": "I am",
    "we're": "we are",
    "they're": "they are",
    "you're": "you are",
    "I've": "I have",
    "we've": "we have",
    "they've": "they have",
    "you've": "you have",
    "I'll": "I will",
    "we'll": "we will",
    "they'll": "they will",
    "you'll": "you will",
    "I'd": "I would",
    "we'd": "we would",
    "they'd": "they would",
    "you'd": "you would",
}

# Informal/colloquial phrases to flag
INFORMAL_PHRASES: dict[str, str] = {
    "a lot of": "many",
    "lots of": "many",
    "kind of": "somewhat",
    "sort of": "somewhat",
    "pretty much": "largely",
    "a bit": "somewhat",
    "stuff": "items",
    "things": "factors",  # When used vaguely
    "get": "obtain",
    "got": "obtained",
    "big": "large",
    "huge": "substantial",
    "okay": "acceptable",
    "ok": "acceptable",
    "etc.": "and others",
    "basically": "",  # Remove filler
    "actually": "",  # Remove filler
    "really": "",  # Remove filler
    "just": "",  # Remove filler (often)
    "very": "",  # Remove or use stronger word
    "quite": "",  # Remove filler
    "anyway": "",  # Remove filler
    "anyways": "",  # Remove filler
    "in order to": "to",
    "due to the fact that": "because",
    "the fact that": "that",
    "at this point in time": "now",
    "at the present time": "now",
    "in the event that": "if",
    "in the case of": "for",
    "with regard to": "regarding",
    "with respect to": "regarding",
}

# Phrases that indicate overclaiming
OVERCLAIM_PHRASES: list[str] = [
    "it is obvious that",
    "it is clear that",
    "clearly",
    "obviously",
    "of course",
    "undoubtedly",
    "without doubt",
    "certainly",
    "definitely",
    "absolutely",
    "always",
    "never",
    "proves",
    "proven",
    "conclusively",
]

# First person singular (acceptable sparingly in some contexts)
FIRST_PERSON_SINGULAR: list[str] = ["I ", "I'm", "I've", "I'll", "my ", "mine "]

# Preferred first person plural
FIRST_PERSON_PLURAL: list[str] = ["we ", "we're", "we've", "we'll", "our ", "ours "]


@dataclass
class ToneIssue:
    """A tone issue found in text."""
    
    issue_type: str  # contraction, informal, overclaim, first_person
    original: str
    suggestion: str | None
    position: int
    line_number: int
    context: str


class AcademicToneChecker:
    """Check text for academic tone issues."""
    
    def __init__(
        self,
        check_contractions: bool = True,
        check_informal: bool = True,
        check_overclaims: bool = True,
        check_first_person: bool = False,  # First person plural is OK
    ):
        """
        Initialize the academic tone checker.
        
        Args:
            check_contractions: Check for contractions.
            check_informal: Check for informal language.
            check_overclaims: Check for overclaiming language.
            check_first_person: Check for first person singular.
        """
        self.check_contractions = check_contractions
        self.check_informal = check_informal
        self.check_overclaims = check_overclaims
        self.check_first_person = check_first_person
        
        # Build patterns
        self._contraction_pattern = self._build_pattern(CONTRACTIONS.keys())
        self._informal_pattern = self._build_pattern(INFORMAL_PHRASES.keys())
        self._overclaim_pattern = self._build_pattern(OVERCLAIM_PHRASES)
    
    def _build_pattern(self, phrases: list[str] | dict) -> re.Pattern:
        """Build regex pattern from phrases."""
        if isinstance(phrases, dict):
            phrases = list(phrases.keys())
        sorted_phrases = sorted(phrases, key=len, reverse=True)
        escaped = [re.escape(p) for p in sorted_phrases]
        pattern = r'\b(' + '|'.join(escaped) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def find_issues(self, text: str) -> list[ToneIssue]:
        """
        Find all tone issues in text.
        
        Args:
            text: Text to check.
            
        Returns:
            List of ToneIssue objects.
        """
        issues: list[ToneIssue] = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check contractions
            if self.check_contractions:
                for match in self._contraction_pattern.finditer(line):
                    word = match.group(1).lower()
                    # Handle case variations
                    for key in CONTRACTIONS:
                        if key.lower() == word:
                            suggestion = CONTRACTIONS[key]
                            break
                    else:
                        suggestion = None
                    
                    issues.append(ToneIssue(
                        issue_type="contraction",
                        original=match.group(1),
                        suggestion=suggestion,
                        position=match.start(),
                        line_number=line_num,
                        context=self._get_context(line, match.start(), match.end()),
                    ))
            
            # Check informal phrases
            if self.check_informal:
                for match in self._informal_pattern.finditer(line):
                    phrase = match.group(1).lower()
                    suggestion = INFORMAL_PHRASES.get(phrase, None)
                    
                    issues.append(ToneIssue(
                        issue_type="informal",
                        original=match.group(1),
                        suggestion=suggestion if suggestion else "Consider removing or rephrasing",
                        position=match.start(),
                        line_number=line_num,
                        context=self._get_context(line, match.start(), match.end()),
                    ))
            
            # Check overclaims
            if self.check_overclaims:
                for match in self._overclaim_pattern.finditer(line):
                    issues.append(ToneIssue(
                        issue_type="overclaim",
                        original=match.group(1),
                        suggestion="Use hedging language instead (e.g., 'suggests', 'indicates')",
                        position=match.start(),
                        line_number=line_num,
                        context=self._get_context(line, match.start(), match.end()),
                    ))
            
            # Check first person singular
            if self.check_first_person:
                for phrase in FIRST_PERSON_SINGULAR:
                    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                    for match in pattern.finditer(line):
                        issues.append(ToneIssue(
                            issue_type="first_person",
                            original=match.group(0),
                            suggestion="Use first person plural ('we') or passive voice",
                            position=match.start(),
                            line_number=line_num,
                            context=self._get_context(line, match.start(), match.end()),
                        ))
        
        return issues
    
    def _get_context(self, line: str, start: int, end: int, window: int = 20) -> str:
        """Get surrounding context for a match."""
        ctx_start = max(0, start - window)
        ctx_end = min(len(line), end + window)
        context = line[ctx_start:ctx_end]
        if ctx_start > 0:
            context = "..." + context
        if ctx_end < len(line):
            context = context + "..."
        return context
    
    def check(self, text: str) -> list[StyleViolation]:
        """
        Check text for tone issues and return violations.
        
        Args:
            text: Text to check.
            
        Returns:
            List of StyleViolation objects.
        """
        violations: list[StyleViolation] = []
        issues = self.find_issues(text)
        
        for issue in issues:
            # Map issue type to violation type and severity
            if issue.issue_type == "contraction":
                violation_type = "contraction"
                severity = CritiqueSeverity.MINOR
                rule_ref = "docs/writing_style_guide.md - Academic Language"
            elif issue.issue_type == "informal":
                violation_type = "informal_tone"
                severity = CritiqueSeverity.MINOR
                rule_ref = "docs/writing_style_guide.md - Academic Language"
            elif issue.issue_type == "overclaim":
                violation_type = "overclaim"
                severity = CritiqueSeverity.MAJOR
                rule_ref = "docs/writing_style_guide.md - Hedging Language"
            else:  # first_person
                violation_type = "informal_tone"
                severity = CritiqueSeverity.MINOR
                rule_ref = "docs/writing_style_guide.md - Voice and Person"
            
            violations.append(StyleViolation(
                violation_type=violation_type,
                severity=severity,
                location=f"line {issue.line_number}, position {issue.position}",
                original_text=issue.original,
                suggestion=issue.suggestion or "Consider revising",
                rule_reference=rule_ref,
                auto_fixable=issue.suggestion is not None and issue.suggestion != "",
            ))
        
        return violations
    
    def expand_contractions(self, text: str) -> tuple[str, int]:
        """
        Expand all contractions in text.
        
        Args:
            text: Text to process.
            
        Returns:
            Tuple of (processed text, number of expansions).
        """
        expansion_count = 0
        
        def replace_match(match: re.Match) -> str:
            nonlocal expansion_count
            contraction = match.group(1)
            
            # Find the expansion (case-insensitive lookup)
            for key, expansion in CONTRACTIONS.items():
                if key.lower() == contraction.lower():
                    # Preserve capitalization
                    if contraction[0].isupper():
                        expansion = expansion[0].upper() + expansion[1:]
                    expansion_count += 1
                    return expansion
            
            return contraction
        
        processed = self._contraction_pattern.sub(replace_match, text)
        return processed, expansion_count
