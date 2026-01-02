"""Style enforcement module for academic writing.

This module provides tools to enforce academic writing style guidelines
from the writing_style_guide.md document.
"""

from src.style.banned_words import BannedWordsFilter, BANNED_WORDS, WORD_REPLACEMENTS
from src.style.academic_tone import AcademicToneChecker
from src.style.hedging import HedgingLanguageChecker
from src.style.precision import PrecisionChecker
from src.style.journal_style import JournalStyleMatcher
from src.style.enforcer import StyleEnforcer

__all__ = [
    "BannedWordsFilter",
    "BANNED_WORDS",
    "WORD_REPLACEMENTS",
    "AcademicToneChecker",
    "HedgingLanguageChecker",
    "PrecisionChecker",
    "JournalStyleMatcher",
    "StyleEnforcer",
]
