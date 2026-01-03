"""Quality metrics for evaluating GIA research outputs.

This module defines metrics for assessing the quality of:
- Literature reviews
- Gap analysis
- Research methodology
- Written output
- Citations
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class MetricType(str, Enum):
    """Types of evaluation metrics."""
    
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    CITATION_QUALITY = "citation_quality"
    METHODOLOGY = "methodology"
    WRITING_QUALITY = "writing_quality"


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""
    
    metric: MetricType
    score: float  # 0.0 to 1.0
    details: dict[str, Any] = field(default_factory=dict)
    passed: bool = True
    feedback: str = ""
    
    @property
    def percentage(self) -> float:
        """Return score as percentage."""
        return self.score * 100


@dataclass
class EvaluationResult:
    """Complete evaluation result for a research output."""
    
    query_id: str
    metrics: list[MetricResult] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = True
    timestamp: str = ""
    
    def add_metric(self, result: MetricResult) -> None:
        """Add a metric result."""
        self.metrics.append(result)
        self._update_overall()
    
    def _update_overall(self) -> None:
        """Update overall score from metrics."""
        if self.metrics:
            self.overall_score = sum(m.score for m in self.metrics) / len(self.metrics)
            self.passed = all(m.passed for m in self.metrics)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "timestamp": self.timestamp,
            "metrics": [
                {
                    "metric": m.metric.value,
                    "score": m.score,
                    "passed": m.passed,
                    "feedback": m.feedback,
                }
                for m in self.metrics
            ],
        }


# =============================================================================
# Metric Evaluators
# =============================================================================


def evaluate_completeness(state: dict[str, Any], expected: dict[str, Any]) -> MetricResult:
    """Evaluate completeness of research output.
    
    Checks that all expected sections/fields are present and non-empty.
    
    Args:
        state: Workflow state with outputs
        expected: Expected outputs specification
        
    Returns:
        MetricResult with completeness score
    """
    required_fields = [
        "literature_review_results",
        "gap_analysis",
        "research_plan",
        "writer_output",
    ]
    
    present = sum(1 for f in required_fields if state.get(f))
    total = len(required_fields)
    score = present / total if total > 0 else 0.0
    
    missing = [f for f in required_fields if not state.get(f)]
    
    return MetricResult(
        metric=MetricType.COMPLETENESS,
        score=score,
        passed=score >= 0.8,
        details={"present": present, "total": total, "missing": missing},
        feedback=f"Completeness: {present}/{total} required fields present",
    )


def evaluate_theme_coverage(
    state: dict[str, Any],
    expected_themes: list[str],
) -> MetricResult:
    """Evaluate coverage of expected themes.
    
    Args:
        state: Workflow state with literature synthesis
        expected_themes: List of themes that should be covered
        
    Returns:
        MetricResult with theme coverage score
    """
    synthesis = state.get("literature_synthesis", {})
    found_themes = synthesis.get("themes", [])
    
    # Simple word matching (case-insensitive)
    expected_lower = [t.lower() for t in expected_themes]
    found_lower = [t.lower() for t in found_themes]
    
    covered = 0
    for expected in expected_lower:
        for found in found_lower:
            if expected in found or found in expected:
                covered += 1
                break
    
    score = covered / len(expected_themes) if expected_themes else 0.0
    
    return MetricResult(
        metric=MetricType.RELEVANCE,
        score=score,
        passed=score >= 0.6,
        details={
            "expected_themes": expected_themes,
            "found_themes": found_themes,
            "covered": covered,
        },
        feedback=f"Theme coverage: {covered}/{len(expected_themes)} expected themes found",
    )


def evaluate_citation_quality(state: dict[str, Any]) -> MetricResult:
    """Evaluate quality of citations.
    
    Args:
        state: Workflow state with literature results
        
    Returns:
        MetricResult with citation quality score
    """
    lit_results = state.get("literature_review_results", {})
    papers_found = lit_results.get("papers_found", 0)
    seminal_works = lit_results.get("seminal_works", [])
    
    # Scoring criteria
    score = 0.0
    feedback_parts = []
    
    # Papers found (40% weight)
    if papers_found >= 20:
        score += 0.4
        feedback_parts.append("Sufficient papers found")
    elif papers_found >= 10:
        score += 0.2
        feedback_parts.append("Moderate papers found")
    else:
        feedback_parts.append(f"Only {papers_found} papers found")
    
    # Seminal works identified (30% weight)
    if len(seminal_works) >= 3:
        score += 0.3
        feedback_parts.append("Key seminal works identified")
    elif len(seminal_works) >= 1:
        score += 0.15
        feedback_parts.append("Some seminal works identified")
    else:
        feedback_parts.append("No seminal works identified")
    
    # Has synthesis (30% weight)
    if state.get("literature_synthesis"):
        score += 0.3
        feedback_parts.append("Literature synthesis present")
    else:
        feedback_parts.append("No literature synthesis")
    
    return MetricResult(
        metric=MetricType.CITATION_QUALITY,
        score=score,
        passed=score >= 0.6,
        details={
            "papers_found": papers_found,
            "seminal_works_count": len(seminal_works),
        },
        feedback="; ".join(feedback_parts),
    )


def evaluate_methodology_quality(
    state: dict[str, Any],
    expected_methodology: list[str],
) -> MetricResult:
    """Evaluate quality of research methodology.
    
    Args:
        state: Workflow state with research plan
        expected_methodology: Expected methodology approaches
        
    Returns:
        MetricResult with methodology quality score
    """
    plan = state.get("research_plan", {})
    methodology = plan.get("methodology", "")
    
    # Check for methodology match
    methodology_lower = methodology.lower() if methodology else ""
    expected_lower = [m.lower() for m in expected_methodology]
    
    matched = any(exp in methodology_lower for exp in expected_lower)
    
    # Score components
    score = 0.0
    feedback_parts = []
    
    if matched:
        score += 0.5
        feedback_parts.append("Appropriate methodology selected")
    else:
        feedback_parts.append("Methodology may not match expected approach")
    
    # Has detailed plan
    if plan.get("analysis_approach") or plan.get("variables"):
        score += 0.3
        feedback_parts.append("Analysis approach defined")
    
    # Has success criteria
    if plan.get("success_criteria"):
        score += 0.2
        feedback_parts.append("Success criteria defined")
    
    return MetricResult(
        metric=MetricType.METHODOLOGY,
        score=score,
        passed=score >= 0.5,
        details={
            "methodology": methodology,
            "expected": expected_methodology,
            "matched": matched,
        },
        feedback="; ".join(feedback_parts),
    )


def evaluate_writing_quality(state: dict[str, Any]) -> MetricResult:
    """Evaluate quality of written output.
    
    Args:
        state: Workflow state with writer output
        
    Returns:
        MetricResult with writing quality score
    """
    writer_output = state.get("writer_output", {})
    
    expected_sections = [
        "abstract",
        "introduction",
        "literature_review",
        "methods",
        "results",
        "discussion",
        "conclusion",
    ]
    
    present_sections = [s for s in expected_sections if writer_output.get(s)]
    score = len(present_sections) / len(expected_sections)
    
    # Check minimum word counts
    min_words = {
        "abstract": 100,
        "introduction": 200,
        "methods": 150,
    }
    
    word_count_issues = []
    for section, min_count in min_words.items():
        content = writer_output.get(section, "")
        word_count = len(content.split()) if content else 0
        if word_count < min_count:
            word_count_issues.append(f"{section}: {word_count}/{min_count} words")
    
    # Reduce score for word count issues
    if word_count_issues:
        score *= 0.8
    
    return MetricResult(
        metric=MetricType.WRITING_QUALITY,
        score=score,
        passed=score >= 0.7,
        details={
            "present_sections": present_sections,
            "total_sections": len(expected_sections),
            "word_count_issues": word_count_issues,
        },
        feedback=f"Writing: {len(present_sections)}/{len(expected_sections)} sections present",
    )


def evaluate_coherence(state: dict[str, Any]) -> MetricResult:
    """Evaluate coherence between research stages.
    
    Checks that gap analysis connects to literature, methodology addresses gap, etc.
    
    Args:
        state: Workflow state
        
    Returns:
        MetricResult with coherence score
    """
    score = 0.0
    feedback_parts = []
    
    # Gap connects to literature
    gap = state.get("gap_analysis", {})
    lit = state.get("literature_synthesis", {})
    
    if gap and lit:
        score += 0.3
        feedback_parts.append("Gap analysis builds on literature")
    
    # Plan addresses gap
    plan = state.get("research_plan", {})
    if plan and gap:
        score += 0.3
        feedback_parts.append("Research plan addresses gap")
    
    # Analysis follows plan
    analysis = state.get("data_analyst_output") or state.get("conceptual_synthesis_output")
    if analysis and plan:
        score += 0.2
        feedback_parts.append("Analysis follows plan")
    
    # Writing integrates all
    writer = state.get("writer_output", {})
    if writer and analysis:
        score += 0.2
        feedback_parts.append("Writing integrates findings")
    
    return MetricResult(
        metric=MetricType.COHERENCE,
        score=score,
        passed=score >= 0.6,
        details={
            "has_gap": bool(gap),
            "has_literature": bool(lit),
            "has_plan": bool(plan),
            "has_analysis": bool(analysis),
            "has_writing": bool(writer),
        },
        feedback="; ".join(feedback_parts) if feedback_parts else "Low coherence",
    )


# =============================================================================
# Full Evaluation
# =============================================================================


def evaluate_research_output(
    state: dict[str, Any],
    query_spec: dict[str, Any],
) -> EvaluationResult:
    """Run full evaluation on research output.
    
    Args:
        state: Complete workflow state
        query_spec: Query specification with expected values
        
    Returns:
        EvaluationResult with all metrics
    """
    from datetime import datetime, timezone
    
    result = EvaluationResult(
        query_id=query_spec.get("id", "unknown"),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    
    # Run all evaluations
    result.add_metric(evaluate_completeness(state, query_spec))
    
    if query_spec.get("expected_themes"):
        result.add_metric(evaluate_theme_coverage(state, query_spec["expected_themes"]))
    
    result.add_metric(evaluate_citation_quality(state))
    
    if query_spec.get("expected_methodology"):
        result.add_metric(evaluate_methodology_quality(state, query_spec["expected_methodology"]))
    
    result.add_metric(evaluate_writing_quality(state))
    result.add_metric(evaluate_coherence(state))
    
    return result
