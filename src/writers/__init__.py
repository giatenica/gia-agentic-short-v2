"""Section writers module for academic paper writing.

Provides individual section writers for each paper component:
- Introduction
- Literature Review
- Methods
- Results
- Discussion
- Conclusion
- Abstract (written last)
"""

from src.writers.base import BaseSectionWriter, SectionWriterConfig
from src.writers.introduction import IntroductionWriter
from src.writers.literature_review import LiteratureReviewWriter
from src.writers.methods import MethodsWriter
from src.writers.results import ResultsWriter
from src.writers.discussion import DiscussionWriter
from src.writers.conclusion import ConclusionWriter
from src.writers.abstract import AbstractWriter
from src.writers.argument import ArgumentManager

__all__ = [
    "BaseSectionWriter",
    "SectionWriterConfig",
    "IntroductionWriter",
    "LiteratureReviewWriter",
    "MethodsWriter",
    "ResultsWriter",
    "DiscussionWriter",
    "ConclusionWriter",
    "AbstractWriter",
    "ArgumentManager",
]
