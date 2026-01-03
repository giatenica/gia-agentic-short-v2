"""Tests for evaluation metrics module."""

import pytest

from evaluation.metrics import (
    MetricType,
    MetricResult,
    EvaluationResult,
    evaluate_completeness,
    evaluate_theme_coverage,
    evaluate_citation_quality,
    evaluate_methodology_quality,
    evaluate_writing_quality,
    evaluate_coherence,
    evaluate_research_output,
)


class TestMetricResult:
    """Tests for MetricResult dataclass."""
    
    def test_create_metric_result(self):
        """Test creating a MetricResult."""
        result = MetricResult(
            metric=MetricType.COMPLETENESS,
            score=0.8,
            passed=True,
            feedback="Test feedback",
        )
        
        assert result.metric == MetricType.COMPLETENESS
        assert result.score == 0.8
        assert result.passed is True
        assert result.feedback == "Test feedback"
    
    def test_percentage_property(self):
        """Test percentage calculation."""
        result = MetricResult(
            metric=MetricType.COMPLETENESS,
            score=0.75,
        )
        
        assert result.percentage == 75.0
    
    def test_details_default(self):
        """Test that details defaults to empty dict."""
        result = MetricResult(
            metric=MetricType.ACCURACY,
            score=0.9,
        )
        
        assert result.details == {}


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    def test_create_evaluation_result(self):
        """Test creating an EvaluationResult."""
        result = EvaluationResult(query_id="test-001")
        
        assert result.query_id == "test-001"
        assert result.metrics == []
        assert result.overall_score == 0.0
        assert result.passed is True
    
    def test_add_metric(self):
        """Test adding a metric result."""
        eval_result = EvaluationResult(query_id="test-001")
        metric = MetricResult(
            metric=MetricType.COMPLETENESS,
            score=0.8,
            passed=True,
        )
        
        eval_result.add_metric(metric)
        
        assert len(eval_result.metrics) == 1
        assert eval_result.overall_score == 0.8
    
    def test_add_multiple_metrics(self):
        """Test overall score with multiple metrics."""
        eval_result = EvaluationResult(query_id="test-001")
        
        eval_result.add_metric(MetricResult(
            metric=MetricType.COMPLETENESS,
            score=0.8,
            passed=True,
        ))
        eval_result.add_metric(MetricResult(
            metric=MetricType.ACCURACY,
            score=0.6,
            passed=True,
        ))
        
        assert len(eval_result.metrics) == 2
        assert eval_result.overall_score == pytest.approx(0.7)
    
    def test_passed_false_if_any_metric_fails(self):
        """Test passed is False if any metric fails."""
        eval_result = EvaluationResult(query_id="test-001")
        
        eval_result.add_metric(MetricResult(
            metric=MetricType.COMPLETENESS,
            score=0.8,
            passed=True,
        ))
        eval_result.add_metric(MetricResult(
            metric=MetricType.ACCURACY,
            score=0.3,
            passed=False,
        ))
        
        assert eval_result.passed is False
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        eval_result = EvaluationResult(
            query_id="test-001",
            timestamp="2026-01-15T10:00:00Z",
        )
        eval_result.add_metric(MetricResult(
            metric=MetricType.COMPLETENESS,
            score=0.8,
            passed=True,
            feedback="Good completeness",
        ))
        
        d = eval_result.to_dict()
        
        assert d["query_id"] == "test-001"
        assert d["timestamp"] == "2026-01-15T10:00:00Z"
        assert len(d["metrics"]) == 1
        assert d["metrics"][0]["metric"] == "completeness"


class TestEvaluateCompleteness:
    """Tests for evaluate_completeness function."""
    
    def test_complete_state(self):
        """Test completeness with all fields present."""
        state = {
            "literature_review_results": {"papers": 10},
            "gap_analysis": {"gaps": ["gap1"]},
            "research_plan": {"methodology": "regression"},
            "writer_output": {"abstract": "..."},
        }
        
        result = evaluate_completeness(state, {})
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_partial_state(self):
        """Test completeness with some fields missing."""
        state = {
            "literature_review_results": {"papers": 10},
            "gap_analysis": {"gaps": ["gap1"]},
            "research_plan": None,
            "writer_output": None,
        }
        
        result = evaluate_completeness(state, {})
        
        assert result.score == 0.5
        assert result.passed is False
    
    def test_empty_state(self):
        """Test completeness with empty state."""
        state = {}
        
        result = evaluate_completeness(state, {})
        
        assert result.score == 0.0
        assert result.passed is False


class TestEvaluateThemeCoverage:
    """Tests for evaluate_theme_coverage function."""
    
    def test_all_themes_found(self):
        """Test when all expected themes are found."""
        state = {
            "literature_synthesis": {
                "themes": ["AI adoption", "productivity metrics", "firm size"],
            }
        }
        expected = ["AI", "productivity", "firm"]
        
        result = evaluate_theme_coverage(state, expected)
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_some_themes_found(self):
        """Test when some themes are found."""
        state = {
            "literature_synthesis": {
                "themes": ["AI adoption"],
            }
        }
        expected = ["AI", "productivity", "firm"]
        
        result = evaluate_theme_coverage(state, expected)
        
        assert result.score == pytest.approx(0.333, rel=0.01)
        assert result.passed is False
    
    def test_no_themes(self):
        """Test when no themes are found."""
        state = {
            "literature_synthesis": {
                "themes": [],
            }
        }
        expected = ["AI", "productivity"]
        
        result = evaluate_theme_coverage(state, expected)
        
        assert result.score == 0.0
    
    def test_missing_synthesis(self):
        """Test when literature_synthesis is missing."""
        state = {}
        expected = ["AI", "productivity"]
        
        result = evaluate_theme_coverage(state, expected)
        
        assert result.score == 0.0


class TestEvaluateCitationQuality:
    """Tests for evaluate_citation_quality function."""
    
    def test_high_quality_citations(self):
        """Test state with good citations."""
        state = {
            "literature_review_results": {
                "papers_found": 30,
                "seminal_works": ["Work1", "Work2", "Work3"],
            },
            "literature_synthesis": {"themes": ["theme1"]},
        }
        
        result = evaluate_citation_quality(state)
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_low_paper_count(self):
        """Test state with few papers."""
        state = {
            "literature_review_results": {
                "papers_found": 5,
                "seminal_works": [],
            },
            "literature_synthesis": None,
        }
        
        result = evaluate_citation_quality(state)
        
        assert result.score < 0.5
        assert result.passed is False
    
    def test_missing_literature_results(self):
        """Test state without literature results."""
        state = {}
        
        result = evaluate_citation_quality(state)
        
        assert result.score <= 0.3


class TestEvaluateMethodologyQuality:
    """Tests for evaluate_methodology_quality function."""
    
    def test_methodology_matches(self):
        """Test when methodology matches expected."""
        state = {
            "research_plan": {
                "methodology": "panel regression analysis",
                "analysis_approach": "Fixed effects",
                "variables": ["var1", "var2"],
                "success_criteria": ["criterion1"],
            }
        }
        expected = ["regression", "panel"]
        
        result = evaluate_methodology_quality(state, expected)
        
        assert result.score >= 0.8
        assert result.passed is True
    
    def test_methodology_mismatch(self):
        """Test when methodology doesn't match."""
        state = {
            "research_plan": {
                "methodology": "case study analysis",
            }
        }
        expected = ["regression", "quantitative"]
        
        result = evaluate_methodology_quality(state, expected)
        
        assert result.score < 0.5
    
    def test_missing_plan(self):
        """Test when research plan is missing."""
        state = {}
        expected = ["regression"]
        
        result = evaluate_methodology_quality(state, expected)
        
        assert result.score == 0.0


class TestEvaluateWritingQuality:
    """Tests for evaluate_writing_quality function."""
    
    def test_all_sections_present(self):
        """Test with all sections present."""
        state = {
            "writer_output": {
                "abstract": "This study " + " ".join(["word"] * 100),
                "introduction": "Introduction " + " ".join(["word"] * 200),
                "literature_review": "Literature " + " ".join(["word"] * 100),
                "methods": "Methods " + " ".join(["word"] * 150),
                "results": "Results " + " ".join(["word"] * 100),
                "discussion": "Discussion " + " ".join(["word"] * 100),
                "conclusion": "Conclusion " + " ".join(["word"] * 50),
            }
        }
        
        result = evaluate_writing_quality(state)
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_missing_sections(self):
        """Test with some sections missing."""
        state = {
            "writer_output": {
                "abstract": "This study examines...",
                "introduction": "Introduction text...",
            }
        }
        
        result = evaluate_writing_quality(state)
        
        assert result.score < 1.0
    
    def test_word_count_issues(self):
        """Test with sections below minimum word count."""
        state = {
            "writer_output": {
                "abstract": "Short",  # Below 100 words
                "introduction": "Short",  # Below 200 words
                "methods": "Short",  # Below 150 words
            }
        }
        
        result = evaluate_writing_quality(state)
        
        # Score reduced for word count issues
        assert result.score < 1.0


class TestEvaluateCoherence:
    """Tests for evaluate_coherence function."""
    
    def test_full_coherence(self):
        """Test state with full workflow coherence."""
        state = {
            "literature_synthesis": {"themes": ["theme1"]},
            "gap_analysis": {"gaps": ["gap1"]},
            "research_plan": {"methodology": "regression"},
            "data_analyst_output": {"results": "findings"},
            "writer_output": {"abstract": "..."},
        }
        
        result = evaluate_coherence(state)
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_partial_coherence(self):
        """Test state with partial coherence."""
        state = {
            "literature_synthesis": {"themes": ["theme1"]},
            "gap_analysis": {"gaps": ["gap1"]},
        }
        
        result = evaluate_coherence(state)
        
        assert result.score == 0.3
    
    def test_no_coherence(self):
        """Test state with no coherence."""
        state = {}
        
        result = evaluate_coherence(state)
        
        assert result.score == 0.0


class TestEvaluateResearchOutput:
    """Tests for evaluate_research_output function."""
    
    def test_full_evaluation(self):
        """Test full evaluation with all metrics."""
        state = {
            "literature_review_results": {"papers_found": 25, "seminal_works": ["a", "b", "c"]},
            "literature_synthesis": {"themes": ["AI", "productivity"]},
            "gap_analysis": {"gaps": ["gap1"]},
            "research_plan": {"methodology": "regression", "analysis_approach": "fixed effects"},
            "data_analyst_output": {"results": "findings"},
            "writer_output": {
                "abstract": " ".join(["word"] * 150),
                "introduction": " ".join(["word"] * 250),
                "literature_review": " ".join(["word"] * 100),
                "methods": " ".join(["word"] * 200),
                "results": " ".join(["word"] * 100),
                "discussion": " ".join(["word"] * 100),
                "conclusion": " ".join(["word"] * 50),
            },
        }
        
        query_spec = {
            "id": "test-001",
            "expected_themes": ["AI", "productivity"],
            "expected_methodology": ["regression"],
        }
        
        result = evaluate_research_output(state, query_spec)
        
        assert result.query_id == "test-001"
        assert len(result.metrics) >= 5
        assert result.overall_score > 0
    
    def test_evaluation_with_minimal_spec(self):
        """Test evaluation with minimal query spec."""
        state = {
            "literature_review_results": {"papers_found": 10},
        }
        
        query_spec = {"id": "minimal"}
        
        result = evaluate_research_output(state, query_spec)
        
        assert result.query_id == "minimal"
        assert len(result.metrics) >= 3  # At least completeness, citation, writing
