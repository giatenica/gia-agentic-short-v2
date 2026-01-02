"""Citation management module for academic writing.

Provides Chicago Author-Date citation formatting for finance journals.
"""

from src.citations.formatter import (
    format_inline_citation,
    format_narrative_citation,
    format_multiple_citations,
)
from src.citations.manager import CitationManager
from src.citations.reference_list import ReferenceListGenerator

__all__ = [
    "format_inline_citation",
    "format_narrative_citation",
    "format_multiple_citations",
    "CitationManager",
    "ReferenceListGenerator",
]
