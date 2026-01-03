"""Review module for GIA Agentic v2.

This module provides evaluation criteria and functions for the REVIEWER node
to assess paper quality across multiple dimensions.
"""

from src.review.criteria import (
    evaluate_contribution,
    evaluate_methodology,
    evaluate_evidence,
    evaluate_coherence,
    evaluate_writing,
    evaluate_paper,
    EVALUATION_DIMENSIONS,
)

__all__ = [
    "evaluate_contribution",
    "evaluate_methodology",
    "evaluate_evidence",
    "evaluate_coherence",
    "evaluate_writing",
    "evaluate_paper",
    "EVALUATION_DIMENSIONS",
]
