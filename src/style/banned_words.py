"""Banned words filter for academic writing.

Filters marketing buzzwords and informal language from academic text.
Based on the banned words list in docs/writing_style_guide.md.
"""

import re
from dataclasses import dataclass

from src.state.models import StyleViolation
from src.state.enums import CritiqueSeverity


# Banned words from docs/writing_style_guide.md and .github/copilot-instructions.md
BANNED_WORDS: set[str] = {
    # Marketing/buzzwords
    "delve", "realm", "harness", "unlock", "tapestry", "paradigm",
    "cutting-edge", "revolutionize", "landscape", "intricate",
    "showcasing", "crucial", "pivotal", "surpass", "meticulously",
    "vibrant", "unparalleled", "underscore", "leverage", "synergy",
    "innovative", "game-changer", "testament", "commendable",
    "meticulous", "highlight", "emphasize", "boast", "groundbreaking",
    "align", "foster", "showcase", "enhance", "holistic", "garner",
    "accentuate", "pioneering", "trailblazing", "unleash", "versatile",
    "transformative", "redefine", "seamless", "optimize", "scalable",
    "breakthrough", "empower", "streamline", "intelligent", "smart",
    "next-gen", "frictionless", "elevate", "adaptive", "effortless",
    "data-driven", "insightful", "proactive", "mission-critical",
    "visionary", "disruptive", "reimagine", "agile", "customizable",
    "personalized", "unprecedented", "intuitive", "leading-edge",
    "synergize", "democratize", "automate", "accelerate",
    "state-of-the-art", "reliable", "efficient", "cloud-native",
    "immersive", "predictive", "transparent", "proprietary",
    "integrated", "plug-and-play", "turnkey", "future-proof",
    "open-ended", "AI-powered", "next-generation", "always-on",
    "hyper-personalized", "results-driven", "machine-first",
    "paradigm-shifting", "novel", "unique", "utilize", "impactful",
    # Context-dependent (non-technical usage)
    "robust",  # OK in statistical context
    "dynamic",  # OK in technical context
    "potential",  # Often vague
    "findings",  # Prefer specific terms
}

# Words that are OK in specific contexts
CONTEXT_DEPENDENT_WORDS: dict[str, str] = {
    "robust": "statistical",  # OK when describing statistical robustness
    "dynamic": "technical",   # OK when describing dynamic systems
}

# Suggested replacements for banned words
WORD_REPLACEMENTS: dict[str, str] = {
    "delve": "examine",
    "utilize": "use",
    "novel": "new",
    "unique": "distinctive",
    "crucial": "important",
    "pivotal": "important",
    "showcase": "show",
    "showcasing": "showing",
    "leverage": "use",
    "highlight": "note",
    "emphasize": "note",
    "enhance": "improve",
    "optimize": "improve",
    "streamline": "simplify",
    "innovative": "new",
    "groundbreaking": "significant",
    "breakthrough": "advance",
    "transformative": "significant",
    "seamless": "smooth",
    "robust": "strong",  # For non-statistical use
    "impactful": "significant",
    "garner": "gain",
    "foster": "encourage",
    "harness": "use",
    "unlock": "enable",
    "empower": "enable",
    "revolutionize": "change",
    "redefine": "change",
    "accelerate": "speed up",
    "synergy": "cooperation",
    "holistic": "comprehensive",
    "cutting-edge": "advanced",
    "state-of-the-art": "advanced",
    "next-gen": "new",
    "next-generation": "new",
    "paradigm": "approach",
    "paradigm-shifting": "significant",
    "game-changer": "significant change",
    "disruptive": "significant",
    "unprecedented": "unusual",
    "unparalleled": "exceptional",
    "meticulously": "carefully",
    "meticulous": "careful",
    "vibrant": "active",
    "intricate": "complex",
    "align": "match",
    "boast": "have",
    "accentuate": "emphasize",
    "underscore": "emphasize",
    "surpass": "exceed",
    "elevate": "raise",
    "intuitive": "easy to use",
    "agile": "flexible",
    "scalable": "expandable",
    "efficient": "effective",
    "reliable": "dependable",
    "transparent": "clear",
    "integrated": "combined",
    "adaptive": "flexible",
    "proactive": "active",
    "insightful": "informative",
    "visionary": "forward-looking",
    "effortless": "easy",
    "frictionless": "smooth",
    "turnkey": "ready-to-use",
    "future-proof": "long-lasting",
    "data-driven": "based on data",
    "AI-powered": "using AI",
    "machine-first": "automated",
    "results-driven": "focused on results",
    "mission-critical": "essential",
    "pioneering": "early",
    "trailblazing": "leading",
    "commendable": "good",
    "testament": "evidence",
    "versatile": "flexible",
    "realm": "area",
    "tapestry": "combination",
    "landscape": "field",
}


@dataclass
class BannedWordMatch:
    """A match of a banned word in text."""
    
    word: str
    position: int
    line_number: int
    context: str  # Surrounding text
    suggestion: str | None


class BannedWordsFilter:
    """Filter banned words from academic text."""
    
    def __init__(
        self,
        banned_words: set[str] | None = None,
        replacements: dict[str, str] | None = None,
        case_sensitive: bool = False,
    ):
        """
        Initialize the banned words filter.
        
        Args:
            banned_words: Set of banned words. Defaults to BANNED_WORDS.
            replacements: Dictionary of word replacements. Defaults to WORD_REPLACEMENTS.
            case_sensitive: Whether to match case-sensitively.
        """
        self.banned_words = banned_words or BANNED_WORDS
        self.replacements = replacements or WORD_REPLACEMENTS
        self.case_sensitive = case_sensitive
        
        # Build regex pattern for efficient matching
        # Sort by length (longest first) to match multi-word phrases first
        sorted_words = sorted(self.banned_words, key=len, reverse=True)
        escaped_words = [re.escape(w) for w in sorted_words]
        pattern = r'\b(' + '|'.join(escaped_words) + r')\b'
        flags = 0 if case_sensitive else re.IGNORECASE
        self._pattern = re.compile(pattern, flags)
    
    def find_banned_words(self, text: str) -> list[BannedWordMatch]:
        """
        Find all banned words in the text.
        
        Args:
            text: Text to check.
            
        Returns:
            List of BannedWordMatch objects.
        """
        matches: list[BannedWordMatch] = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for match in self._pattern.finditer(line):
                word = match.group(1).lower()
                
                # Get context (surrounding text)
                start = max(0, match.start() - 20)
                end = min(len(line), match.end() + 20)
                context = line[start:end]
                if start > 0:
                    context = "..." + context
                if end < len(line):
                    context = context + "..."
                
                matches.append(BannedWordMatch(
                    word=match.group(1),
                    position=match.start(),
                    line_number=line_num,
                    context=context,
                    suggestion=self.replacements.get(word),
                ))
        
        return matches
    
    def check(self, text: str) -> list[StyleViolation]:
        """
        Check text for banned words and return violations.
        
        Args:
            text: Text to check.
            
        Returns:
            List of StyleViolation objects.
        """
        violations: list[StyleViolation] = []
        matches = self.find_banned_words(text)
        
        for match in matches:
            suggestion = match.suggestion or f"Avoid using '{match.word}'"
            if match.suggestion:
                suggestion = f"Consider using '{match.suggestion}' instead of '{match.word}'"
            
            violations.append(StyleViolation(
                violation_type="banned_word",
                severity=CritiqueSeverity.MINOR,
                location=f"line {match.line_number}, position {match.position}",
                original_text=match.word,
                suggestion=suggestion,
                rule_reference="docs/writing_style_guide.md - Banned Words",
                auto_fixable=match.suggestion is not None,
            ))
        
        return violations
    
    def replace_banned_words(self, text: str) -> tuple[str, int]:
        """
        Replace banned words with suggested alternatives.
        
        Args:
            text: Text to process.
            
        Returns:
            Tuple of (processed text, number of replacements).
        """
        replacement_count = 0
        
        def replace_match(match: re.Match) -> str:
            nonlocal replacement_count
            word = match.group(1)
            word_lower = word.lower()
            
            if word_lower in self.replacements:
                replacement = self.replacements[word_lower]
                # Preserve case of first letter
                if word[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                replacement_count += 1
                return replacement
            
            return word
        
        processed = self._pattern.sub(replace_match, text)
        return processed, replacement_count
    
    def is_word_banned(self, word: str) -> bool:
        """Check if a specific word is banned."""
        check_word = word if self.case_sensitive else word.lower()
        return check_word in self.banned_words or check_word in {w.lower() for w in self.banned_words}
    
    def get_suggestion(self, word: str) -> str | None:
        """Get replacement suggestion for a word."""
        return self.replacements.get(word.lower())
