"""Section writers module for academic paper writing.

Provides individual section writers for each paper component:
- Introduction
- Literature Review
- Methods
- Results
- Discussion
- Conclusion
- Abstract (written last)

Sprint 16: Added artifact helpers for table/figure integration.
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
from src.writers.artifact_helpers import (
    format_table_reference,
    format_figure_reference,
    generate_table_summary,
    generate_figure_summary,
    format_data_exploration_for_methods,
    generate_results_artifacts_prompt,
    get_table_labels,
    get_figure_labels,
)

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
    # Sprint 16: Artifact helpers
    "format_table_reference",
    "format_figure_reference",
    "generate_table_summary",
    "generate_figure_summary",
    "format_data_exploration_for_methods",
    "generate_results_artifacts_prompt",
    "get_table_labels",
    "get_figure_labels",
]
