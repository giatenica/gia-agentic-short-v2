"""Hedging language checker for academic writing.

Ensures appropriate use of hedging language for claims and findings.
Academic writing requires careful qualification of claims.
"""

import re
from dataclasses import dataclass

from src.state.models import StyleViolation
from src.state.enums import CritiqueSeverity


# Strong claim indicators that may need hedging
STRONG_CLAIM_VERBS: list[str] = [
    "proves",
    "demonstrates",
    "shows",
    "confirms",
    "establishes",
    "determines",
    "reveals",
    "indicates",
    "causes",
    "leads to",
    "results in",
    "affects",
    "influences",
    "impacts",
]

# Appropriate hedging phrases
HEDGING_PHRASES: list[str] = [
    "suggests that",
    "indicates that",
    "appears to",
    "seems to",
    "may",
    "might",
    "could",
    "possibly",
    "potentially",
    "likely",
    "probably",
    "tends to",
    "is consistent with",
    "is associated with",
    "the results suggest",
    "the evidence suggests",
    "the findings indicate",
    "one interpretation is",
    "one possible explanation",
    "this may reflect",
]

# Absolute terms that need qualification in most contexts
ABSOLUTE_TERMS: list[str] = [
    "always",
    "never",
    "all",
    "none",
    "every",
    "no one",
    "everyone",
    "completely",
    "totally",
    "absolutely",
    "certainly",
    "definitely",
    "undoubtedly",
    "unquestionably",
    "without exception",
    "in all cases",
    "invariably",
]

# Suggested hedging alternatives
HEDGING_ALTERNATIVES: dict[str, str] = {
    "proves": "suggests",
    "proves that": "provides evidence that",
    "demonstrates": "indicates",
    "shows": "suggests",
    "confirms": "supports",
    "establishes": "suggests",
    "determines": "suggests",
    "reveals": "indicates",
    "causes": "is associated with",
    "leads to": "may lead to",
    "results in": "may result in",
    "affects": "appears to affect",
    "influences": "may influence",
    "impacts": "may affect",
    "always": "typically",
    "never": "rarely",
    "all": "most",
    "none": "few",
    "certainly": "likely",
    "definitely": "probably",
    "undoubtedly": "likely",
}


@dataclass
class HedgingIssue:
    """A hedging issue found in text."""
    
    issue_type: str  # missing_hedge, absolute_term
    original: str
    suggestion: str
    position: int
    line_number: int
    context: str
    sentence: str


class HedgingLanguageChecker:
    """Check for appropriate hedging language in academic text."""
    
    def __init__(
        self,
        check_strong_claims: bool = True,
        check_absolute_terms: bool = True,
        sensitivity: str = "medium",  # low, medium, high
    ):
        """
        Initialize the hedging language checker.
        
        Args:
            check_strong_claims: Check for strong claim verbs.
            check_absolute_terms: Check for absolute terms.
            sensitivity: How strict to be (low, medium, high).
        """
        self.check_strong_claims = check_strong_claims
        self.check_absolute_terms = check_absolute_terms
        self.sensitivity = sensitivity
        
        # Build patterns
        self._strong_claim_pattern = self._build_pattern(STRONG_CLAIM_VERBS)
        self._absolute_pattern = self._build_pattern(ABSOLUTE_TERMS)
        self._hedging_pattern = self._build_pattern(HEDGING_PHRASES)
    
    def _build_pattern(self, phrases: list[str]) -> re.Pattern:
        """Build regex pattern from phrases."""
        sorted_phrases = sorted(phrases, key=len, reverse=True)
        escaped = [re.escape(p) for p in sorted_phrases]
        pattern = r'\b(' + '|'.join(escaped) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences with position info."""
        # Simple sentence splitter
        sentences = []
        pattern = re.compile(r'[.!?]+\s*')
        pos = 0
        
        for match in pattern.finditer(text):
            sentence = text[pos:match.end()].strip()
            if sentence:
                sentences.append((sentence, pos, match.end()))
            pos = match.end()
        
        # Handle final sentence without ending punctuation
        if pos < len(text):
            final = text[pos:].strip()
            if final:
                sentences.append((final, pos, len(text)))
        
        return sentences
    
    def _sentence_has_hedging(self, sentence: str) -> bool:
        """Check if a sentence contains hedging language."""
        return bool(self._hedging_pattern.search(sentence))
    
    def find_issues(self, text: str) -> list[HedgingIssue]:
        """
        Find hedging issues in text.
        
        Args:
            text: Text to check.
            
        Returns:
            List of HedgingIssue objects.
        """
        issues: list[HedgingIssue] = []
        lines = text.split('\n')
        
        # Track line positions for accurate line numbers
        line_starts = [0]
        for line in lines:
            line_starts.append(line_starts[-1] + len(line) + 1)
        
        def get_line_number(pos: int) -> int:
            for i, start in enumerate(line_starts):
                if start > pos:
                    return i
            return len(lines)
        
        sentences = self._split_sentences(text)
        
        for sentence, sent_start, sent_end in sentences:
            line_num = get_line_number(sent_start)
            
            # Check for strong claims without hedging
            if self.check_strong_claims:
                strong_matches = list(self._strong_claim_pattern.finditer(sentence))
                
                for match in strong_matches:
                    # Check if sentence already has hedging
                    if not self._sentence_has_hedging(sentence):
                        word = match.group(1).lower()
                        suggestion = HEDGING_ALTERNATIVES.get(word, f"Consider hedging: '{word}'")
                        
                        # Only flag if sensitivity allows
                        should_flag = (
                            self.sensitivity == "high" or
                            (self.sensitivity == "medium" and word in ["proves", "demonstrates", "confirms", "causes"]) or
                            (self.sensitivity == "low" and word in ["proves", "causes"])
                        )
                        
                        if should_flag:
                            issues.append(HedgingIssue(
                                issue_type="missing_hedge",
                                original=match.group(1),
                                suggestion=f"Consider using '{suggestion}' or adding hedging language",
                                position=sent_start + match.start(),
                                line_number=line_num,
                                context=sentence[:100] + "..." if len(sentence) > 100 else sentence,
                                sentence=sentence,
                            ))
            
            # Check for absolute terms
            if self.check_absolute_terms:
                for match in self._absolute_pattern.finditer(sentence):
                    word = match.group(1).lower()
                    suggestion = HEDGING_ALTERNATIVES.get(word, f"Consider qualifying '{word}'")
                    
                    issues.append(HedgingIssue(
                        issue_type="absolute_term",
                        original=match.group(1),
                        suggestion=f"Consider using '{suggestion}' instead",
                        position=sent_start + match.start(),
                        line_number=line_num,
                        context=sentence[:100] + "..." if len(sentence) > 100 else sentence,
                        sentence=sentence,
                    ))
        
        return issues
    
    def check(self, text: str) -> list[StyleViolation]:
        """
        Check text for hedging issues and return violations.
        
        Args:
            text: Text to check.
            
        Returns:
            List of StyleViolation objects.
        """
        violations: list[StyleViolation] = []
        issues = self.find_issues(text)
        
        for issue in issues:
            if issue.issue_type == "missing_hedge":
                violation_type = "missing_hedge"
                severity = CritiqueSeverity.MINOR
            else:  # absolute_term
                violation_type = "overclaim"
                severity = CritiqueSeverity.MINOR
            
            violations.append(StyleViolation(
                violation_type=violation_type,
                severity=severity,
                location=f"line {issue.line_number}",
                original_text=issue.original,
                suggestion=issue.suggestion,
                rule_reference="docs/writing_style_guide.md - Hedging Language",
                auto_fixable=False,  # Hedging requires context
            ))
        
        return violations
    
    def suggest_hedging(self, sentence: str) -> str:
        """
        Suggest hedged version of a sentence.
        
        Args:
            sentence: Original sentence.
            
        Returns:
            Suggested hedged version.
        """
        result = sentence
        
        # Apply replacements
        for original, replacement in HEDGING_ALTERNATIVES.items():
            pattern = re.compile(r'\b' + re.escape(original) + r'\b', re.IGNORECASE)
            if pattern.search(result):
                # Preserve case
                def replace_preserve_case(match: re.Match) -> str:
                    if match.group(0)[0].isupper():
                        return replacement[0].upper() + replacement[1:]
                    return replacement
                result = pattern.sub(replace_preserve_case, result)
        
        return result
