"""Tests for the REVIEWER node and review criteria.

Sprint 7: Paper Quality Evaluation and Revision Loop.
"""

import pytest
from unittest.mock import patch

from src.review.criteria import (
    evaluate_contribution,
    evaluate_methodology,
    evaluate_evidence,
    evaluate_coherence,
    evaluate_writing,
    evaluate_paper,
    EVALUATION_DIMENSIONS,
)
from src.nodes.reviewer import (
    reviewer_node,
    route_after_reviewer,
    _generate_paper_markdown,
)
from src.state.models import (
    QualityScore,
    ReviewCritiqueItem,
    ReviewCritique,
    RevisionRequest,
    calculate_overall_score,
    determine_review_decision,
    DIMENSION_WEIGHTS,
)
from src.state.enums import ResearchStatus


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_writer_output():
    """Sample writer output for testing."""
    return {
        "title": "The Impact of Market Sentiment on Stock Returns",
        "sections": [
            {
                "section_type": "abstract",
                "title": "Abstract",
                "content": "This paper examines the relationship between market sentiment and stock returns. We find that sentiment indicators significantly predict short-term returns, particularly during periods of high volatility. Our results suggest important implications for market efficiency and investor behavior.",
                "order": 1,
            },
            {
                "section_type": "introduction",
                "title": "Introduction",
                "content": "Market sentiment has long been recognized as a potential driver of asset prices. Despite extensive research, significant gaps remain in understanding how sentiment affects returns across different market conditions. This paper contributes to the literature by examining the relationship between sentiment and returns. We find that sentiment indicators provide valuable predictive information. These findings have important implications for both academics and practitioners.",
                "order": 2,
            },
            {
                "section_type": "methods",
                "title": "Methods",
                "content": "We employ a panel regression approach using daily data from 2010 to 2020. Our sample includes all stocks in the S&P 500. The dependent variable is daily returns, and the key independent variable is a sentiment index. We control for market factors and firm characteristics. Our specification includes fixed effects for time and firm. We address potential endogeneity using instrumental variables. Limitations of our approach include the reliance on a single sentiment measure.",
                "order": 3,
            },
            {
                "section_type": "results",
                "title": "Results",
                "content": "Table 1 presents our main results. The coefficient on sentiment is positive and statistically significant (p < 0.01). A one standard deviation increase in sentiment is associated with a 0.5% increase in returns. The magnitude of this effect is economically significant. Robustness checks using alternative specifications confirm our findings. Subsample analysis reveals the effect is stronger during high volatility periods.",
                "order": 4,
            },
            {
                "section_type": "discussion",
                "title": "Discussion",
                "content": "Our findings suggest that market sentiment plays an important role in determining stock returns. These results are consistent with behavioral finance theories. However, we cannot rule out alternative explanations. The implications for market efficiency are significant. Practitioners may find these results useful for timing strategies.",
                "order": 5,
            },
            {
                "section_type": "conclusion",
                "title": "Conclusion",
                "content": "This paper examines the relationship between market sentiment and stock returns. We find evidence of a significant positive relationship. The implications for future research include examining other asset classes. Policymakers should consider the role of sentiment in market dynamics.",
                "order": 6,
            },
        ],
        "reference_list": {
            "entries": [
                {"citation_key": "baker2006", "formatted": "Baker, M., & Wurgler, J. (2006). Investor sentiment and the cross-section of stock returns. Journal of Finance, 61(4), 1645-1680."},
                {"citation_key": "tetlock2007", "formatted": "Tetlock, P. C. (2007). Giving content to investor sentiment. Journal of Finance, 62(3), 1139-1168."},
            ]
        },
    }


@pytest.fixture
def poor_writer_output():
    """Writer output with quality issues for testing."""
    return {
        "title": "A Study",
        "sections": [
            {
                "section_type": "abstract",
                "title": "Abstract",
                "content": "We delve into stuff and things. It's really revolutionary and groundbreaking. This paper showcases novel findings.",
                "order": 1,
            },
            {
                "section_type": "introduction",
                "title": "Introduction",
                "content": "This is an introduction. We study something important. The results are obviously true.",
                "order": 2,
            },
            {
                "section_type": "results",
                "title": "Results",
                "content": "The results clearly show everything works perfectly. It's undoubtedly the best approach.",
                "order": 4,
            },
        ],
    }


@pytest.fixture
def sample_dimension_scores():
    """Sample dimension scores for testing."""
    return [
        QualityScore(
            dimension="contribution",
            score=7.5,
            justification="Good contribution with clear gap identification.",
            strengths=["Clear research gap", "Novel contribution"],
            weaknesses=[],
        ),
        QualityScore(
            dimension="methodology",
            score=6.8,
            justification="Solid methodology with minor issues.",
            strengths=["Appropriate design"],
            weaknesses=["Limited reproducibility details"],
        ),
        QualityScore(
            dimension="evidence",
            score=7.2,
            justification="Good evidence presentation.",
            strengths=["Statistical results clear"],
            weaknesses=[],
        ),
        QualityScore(
            dimension="coherence",
            score=7.0,
            justification="Good logical flow.",
            strengths=["Consistent argument"],
            weaknesses=[],
        ),
        QualityScore(
            dimension="writing",
            score=6.5,
            justification="Academic tone maintained.",
            strengths=["Formal language"],
            weaknesses=["Some informal words"],
        ),
    ]


@pytest.fixture
def sample_state(sample_writer_output):
    """Sample workflow state for testing."""
    return {
        "writer_output": sample_writer_output,
        "original_query": "How does market sentiment affect stock returns?",
        "identified_gaps": ["Temporal variation in sentiment effects"],
        "research_plan": {"methodology_type": "regression_analysis"},
        "target_journal": "jfe",
        "revision_count": 0,
        "max_revisions": 3,
        "errors": [],
    }


# =============================================================================
# Test Evaluation Dimensions Configuration
# =============================================================================


class TestEvaluationDimensions:
    """Test evaluation dimensions configuration."""
    
    def test_all_dimensions_defined(self):
        """Test that all expected dimensions are defined."""
        expected = {"contribution", "methodology", "evidence", "coherence", "writing"}
        assert set(EVALUATION_DIMENSIONS.keys()) == expected
    
    def test_dimension_weights_sum_to_one(self):
        """Test that dimension weights sum to 1.0."""
        total = sum(DIMENSION_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001
    
    def test_all_dimensions_have_weights(self):
        """Test that all dimensions have weights."""
        for dim in EVALUATION_DIMENSIONS:
            assert dim in DIMENSION_WEIGHTS


# =============================================================================
# Test Individual Evaluation Functions
# =============================================================================


class TestEvaluateContribution:
    """Test contribution evaluation."""
    
    def test_evaluate_contribution_good_paper(self, sample_writer_output):
        """Test contribution evaluation on a good paper."""
        score = evaluate_contribution(sample_writer_output, "research question", ["gap"])
        
        assert isinstance(score, QualityScore)
        assert score.dimension == "contribution"
        assert 1.0 <= score.score <= 10.0
        assert len(score.justification) > 0
    
    def test_evaluate_contribution_identifies_strengths(self, sample_writer_output):
        """Test that strengths are identified."""
        score = evaluate_contribution(sample_writer_output)
        
        # Good paper should have some strengths
        assert len(score.strengths) > 0 or score.score >= 7.0
    
    def test_evaluate_contribution_poor_paper(self, poor_writer_output):
        """Test contribution evaluation on a poor paper."""
        score = evaluate_contribution(poor_writer_output)
        
        # Poor paper should have lower score and weaknesses
        assert score.score < 7.0
        assert len(score.weaknesses) > 0


class TestEvaluateMethodology:
    """Test methodology evaluation."""
    
    def test_evaluate_methodology_good_paper(self, sample_writer_output):
        """Test methodology evaluation on a good paper."""
        score = evaluate_methodology(sample_writer_output)
        
        assert isinstance(score, QualityScore)
        assert score.dimension == "methodology"
        assert 1.0 <= score.score <= 10.0
    
    def test_evaluate_methodology_with_research_plan(self, sample_writer_output):
        """Test methodology evaluation with research plan context."""
        plan = {"methodology_type": "regression_analysis"}
        score = evaluate_methodology(sample_writer_output, plan)
        
        assert score.score > 0
    
    def test_evaluate_methodology_missing_sections(self, poor_writer_output):
        """Test methodology evaluation with missing methods section."""
        score = evaluate_methodology(poor_writer_output)
        
        # Missing methods should result in lower score
        assert score.score < 6.0


class TestEvaluateEvidence:
    """Test evidence evaluation."""
    
    def test_evaluate_evidence_good_paper(self, sample_writer_output):
        """Test evidence evaluation on a good paper."""
        score = evaluate_evidence(sample_writer_output)
        
        assert isinstance(score, QualityScore)
        assert score.dimension == "evidence"
        assert 1.0 <= score.score <= 10.0
    
    def test_evaluate_evidence_detects_overclaims(self, poor_writer_output):
        """Test that overclaiming language is detected."""
        score = evaluate_evidence(poor_writer_output)
        
        # Paper with "obviously", "clearly", "undoubtedly" should have weaknesses
        assert any("overstated" in w.lower() or "overclaim" in w.lower() for w in score.weaknesses) or score.score < 6.0


class TestEvaluateCoherence:
    """Test coherence evaluation."""
    
    def test_evaluate_coherence_complete_paper(self, sample_writer_output):
        """Test coherence evaluation on a complete paper."""
        score = evaluate_coherence(sample_writer_output)
        
        assert isinstance(score, QualityScore)
        assert score.dimension == "coherence"
        assert 1.0 <= score.score <= 10.0
    
    def test_evaluate_coherence_missing_sections(self, poor_writer_output):
        """Test coherence evaluation with missing sections."""
        score = evaluate_coherence(poor_writer_output)
        
        # Missing sections should be detected
        assert len(score.weaknesses) > 0 or "missing" in score.justification.lower()


class TestEvaluateWriting:
    """Test writing evaluation."""
    
    def test_evaluate_writing_good_paper(self, sample_writer_output):
        """Test writing evaluation on a good paper."""
        score = evaluate_writing(sample_writer_output)
        
        assert isinstance(score, QualityScore)
        assert score.dimension == "writing"
        assert 1.0 <= score.score <= 10.0
    
    def test_evaluate_writing_detects_banned_words(self, poor_writer_output):
        """Test that banned words are detected."""
        score = evaluate_writing(poor_writer_output)
        
        # Paper with "delve", "revolutionary", "groundbreaking", "showcases", "novel"
        assert any("banned" in w.lower() for w in score.weaknesses) or score.score < 6.0
    
    def test_evaluate_writing_detects_informal_language(self, poor_writer_output):
        """Test that informal language is detected."""
        score = evaluate_writing(poor_writer_output)
        
        # Paper with "stuff", "things", "really"
        assert any("informal" in w.lower() for w in score.weaknesses) or score.score < 6.0


# =============================================================================
# Test Comprehensive Paper Evaluation
# =============================================================================


class TestEvaluatePaper:
    """Test comprehensive paper evaluation."""
    
    def test_evaluate_paper_returns_all_dimensions(self, sample_writer_output):
        """Test that all dimensions are evaluated."""
        scores, critiques = evaluate_paper(sample_writer_output)
        
        assert len(scores) == 5
        dimensions = {s.dimension for s in scores}
        assert dimensions == {"contribution", "methodology", "evidence", "coherence", "writing"}
    
    def test_evaluate_paper_generates_critiques(self, poor_writer_output):
        """Test that critiques are generated for poor papers."""
        scores, critiques = evaluate_paper(poor_writer_output)
        
        # Poor paper should have critique items
        assert len(critiques) > 0
    
    def test_critique_items_have_required_fields(self, sample_writer_output):
        """Test that critique items have all required fields."""
        _, critiques = evaluate_paper(sample_writer_output)
        
        for critique in critiques:
            assert isinstance(critique, ReviewCritiqueItem)
            assert critique.section
            assert critique.issue
            assert critique.severity
            assert critique.suggestion


# =============================================================================
# Test Score Calculation and Decision Logic
# =============================================================================


class TestScoreCalculation:
    """Test overall score calculation."""
    
    def test_calculate_overall_score(self, sample_dimension_scores):
        """Test weighted score calculation."""
        score = calculate_overall_score(sample_dimension_scores)
        
        assert 1.0 <= score <= 10.0
    
    def test_calculate_overall_score_uses_weights(self, sample_dimension_scores):
        """Test that weights are applied correctly."""
        score = calculate_overall_score(sample_dimension_scores)
        
        # Manual calculation
        expected = sum(
            s.score * DIMENSION_WEIGHTS.get(s.dimension, 0.1)
            for s in sample_dimension_scores
        )
        
        assert abs(score - expected) < 0.001
    
    def test_calculate_overall_score_empty_list(self):
        """Test score calculation with empty list."""
        score = calculate_overall_score([])
        
        assert score == 5.0  # Default middle score


class TestReviewDecision:
    """Test review decision logic."""
    
    def test_approve_decision_high_score(self):
        """Test approve decision for high score."""
        decision = determine_review_decision(8.0, has_critical_items=False)
        assert decision == "approve"
    
    def test_revise_decision_medium_score(self):
        """Test revise decision for medium score."""
        decision = determine_review_decision(5.5, has_critical_items=False)
        assert decision == "revise"
    
    def test_reject_decision_low_score(self):
        """Test reject decision for low score."""
        decision = determine_review_decision(3.0, has_critical_items=False)
        assert decision == "reject"
    
    def test_critical_items_force_revise(self):
        """Test that critical items force at least revise."""
        # High score but with critical items
        decision = determine_review_decision(8.0, has_critical_items=True)
        assert decision == "revise"
    
    def test_critical_items_with_low_score_reject(self):
        """Test that critical items with low score result in reject."""
        decision = determine_review_decision(3.0, has_critical_items=True)
        assert decision == "reject"
    
    def test_threshold_boundaries(self):
        """Test decisions at threshold boundaries."""
        # At approve threshold
        assert determine_review_decision(7.0, False) == "approve"
        
        # Just below approve threshold
        assert determine_review_decision(6.99, False) == "revise"
        
        # At reject threshold
        assert determine_review_decision(4.0, False) == "revise"
        
        # Just below reject threshold
        assert determine_review_decision(3.99, False) == "reject"


# =============================================================================
# Test Review Models
# =============================================================================


class TestReviewCritique:
    """Test ReviewCritique model."""
    
    def test_review_critique_creation(self, sample_dimension_scores):
        """Test creating a ReviewCritique."""
        critique = ReviewCritique(
            overall_score=7.2,
            decision="approve",
            dimension_scores=sample_dimension_scores,
            critique_items=[],
            summary="Paper meets standards.",
        )
        
        assert critique.overall_score == 7.2
        assert critique.decision == "approve"
        assert len(critique.dimension_scores) == 5
    
    def test_get_score_by_dimension(self, sample_dimension_scores):
        """Test getting score by dimension."""
        critique = ReviewCritique(
            overall_score=7.0,
            decision="approve",
            dimension_scores=sample_dimension_scores,
        )
        
        score = critique.get_score_by_dimension("contribution")
        assert score == 7.5
        
        # Non-existent dimension
        assert critique.get_score_by_dimension("nonexistent") is None
    
    def test_get_critical_items(self):
        """Test getting critical items."""
        items = [
            ReviewCritiqueItem(section="intro", issue="This is a major problem that needs attention", severity="critical", suggestion="Address this significant issue promptly"),
            ReviewCritiqueItem(section="methods", issue="This is a minor issue to review", severity="minor", suggestion="Polish this section for better clarity"),
        ]
        critique = ReviewCritique(
            overall_score=5.0,
            decision="revise",
            critique_items=items,
        )
        
        critical = critique.get_critical_items()
        assert len(critical) == 1
        assert critical[0].severity == "critical"
    
    def test_has_critical_issues(self):
        """Test checking for critical issues."""
        # No critical items
        critique1 = ReviewCritique(
            overall_score=7.0,
            decision="approve",
            critique_items=[],
        )
        assert not critique1.has_critical_issues()
        
        # With critical item
        critique2 = ReviewCritique(
            overall_score=5.0,
            decision="revise",
            critique_items=[
                ReviewCritiqueItem(section="intro", issue="This is a critical problem", severity="critical", suggestion="Fix this issue immediately"),
            ],
        )
        assert critique2.has_critical_issues()


class TestRevisionRequest:
    """Test RevisionRequest model."""
    
    def test_revision_request_creation(self):
        """Test creating a RevisionRequest."""
        request = RevisionRequest(
            sections_to_revise=["introduction", "methods"],
            critique_items=[],
            iteration_count=1,
            max_iterations=3,
        )
        
        assert len(request.sections_to_revise) == 2
        assert request.iteration_count == 1
    
    def test_is_final_iteration(self):
        """Test checking if final iteration."""
        request1 = RevisionRequest(iteration_count=1, max_iterations=3)
        assert not request1.is_final_iteration()
        
        request2 = RevisionRequest(iteration_count=3, max_iterations=3)
        assert request2.is_final_iteration()


# =============================================================================
# Test REVIEWER Node
# =============================================================================


class TestReviewerNode:
    """Test the REVIEWER node."""
    
    def test_reviewer_node_no_writer_output(self):
        """Test reviewer node with no writer output."""
        state = {"errors": []}
        result = reviewer_node(state)
        
        assert result["status"] == ResearchStatus.FAILED
        assert len(result["errors"]) > 0
    
    @patch("src.nodes.reviewer.interrupt")
    def test_reviewer_node_approve_flow(self, mock_interrupt, sample_state):
        """Test reviewer node approve flow."""
        mock_interrupt.return_value = {"approved": True}
        
        result = reviewer_node(sample_state)
        
        assert "review_critique" in result
        assert "review_decision" in result
        assert result["human_approved"] is True
    
    @patch("src.nodes.reviewer.interrupt")
    def test_reviewer_node_revise_flow(self, mock_interrupt, sample_state):
        """Test reviewer node revise flow."""
        # Force a lower score by using poor output
        poor_state = sample_state.copy()
        poor_state["writer_output"] = {
            "title": "Test",
            "sections": [
                {"section_type": "abstract", "content": "Short.", "order": 1}
            ],
        }
        mock_interrupt.return_value = {"approved": True}
        
        result = reviewer_node(poor_state)
        
        # Should have review results
        assert "review_critique" in result
    
    @patch("src.nodes.reviewer.interrupt")
    def test_reviewer_node_human_override(self, mock_interrupt, sample_state):
        """Test human can override decision."""
        mock_interrupt.return_value = {
            "approved": True,
            "override_decision": "approve",
            "feedback": "Approved with minor notes",
        }
        
        result = reviewer_node(sample_state)
        
        assert result["review_decision"] == "approve"
        assert result["human_feedback"] == "Approved with minor notes"
    
    @patch("src.nodes.reviewer.interrupt")
    def test_reviewer_node_max_revisions(self, mock_interrupt, sample_state):
        """Test max revision limit handling."""
        sample_state["revision_count"] = 3
        mock_interrupt.return_value = {"approved": True}
        
        result = reviewer_node(sample_state)
        
        # Should escalate if max revisions reached
        critique = result.get("review_critique")
        if critique:
            if isinstance(critique, dict):
                summary = critique.get("summary", "")
            else:
                summary = critique.summary
            # Should mention escalation if at limit and would normally revise
            assert "escalat" in summary.lower() or result.get("review_decision") != "revise"


# =============================================================================
# Test Routing Function
# =============================================================================


class TestRouteAfterReviewer:
    """Test routing after reviewer."""
    
    def test_route_approve_to_output(self):
        """Test approved papers route to output."""
        state = {
            "review_decision": "approve",
            "human_approved": True,
        }
        assert route_after_reviewer(state) == "output"
    
    def test_route_revise_to_writer(self):
        """Test revise decision routes back to writer."""
        state = {
            "review_decision": "revise",
            "human_approved": True,
        }
        assert route_after_reviewer(state) == "writer"
    
    def test_route_reject_to_end(self):
        """Test reject decision routes to end."""
        state = {
            "review_decision": "reject",
            "human_approved": False,
        }
        assert route_after_reviewer(state) == "__end__"
    
    def test_route_approve_not_human_approved(self):
        """Test approve without human approval doesn't go to output."""
        state = {
            "review_decision": "approve",
            "human_approved": False,
        }
        # Should not go to output without human approval
        result = route_after_reviewer(state)
        assert result != "output" or result == "__end__"


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestGeneratePaperMarkdown:
    """Test markdown generation helper."""
    
    def test_generate_markdown_basic(self, sample_writer_output):
        """Test basic markdown generation."""
        markdown = _generate_paper_markdown(sample_writer_output)
        
        assert "# The Impact of Market Sentiment" in markdown
        assert "## Abstract" in markdown
        assert "## Introduction" in markdown
        assert "## Methods" in markdown
    
    def test_generate_markdown_includes_content(self, sample_writer_output):
        """Test that content is included in markdown."""
        markdown = _generate_paper_markdown(sample_writer_output)
        
        assert "market sentiment" in markdown.lower()
        assert "stock returns" in markdown.lower()
    
    def test_generate_markdown_handles_dict(self):
        """Test markdown generation with dict input."""
        output = {
            "title": "Test Paper",
            "sections": [
                {"section_type": "abstract", "title": "Abstract", "content": "Test content.", "order": 1}
            ],
        }
        markdown = _generate_paper_markdown(output)
        
        assert "# Test Paper" in markdown
        assert "Test content." in markdown


# =============================================================================
# Test Integration
# =============================================================================


class TestReviewerIntegration:
    """Integration tests for reviewer components."""
    
    @patch("src.nodes.reviewer.interrupt")
    def test_full_review_cycle(self, mock_interrupt, sample_writer_output):
        """Test a full review cycle."""
        mock_interrupt.return_value = {"approved": True}
        
        state = {
            "writer_output": sample_writer_output,
            "original_query": "How does sentiment affect returns?",
            "identified_gaps": ["Temporal effects"],
            "target_journal": "jfe",
            "revision_count": 0,
            "max_revisions": 3,
            "errors": [],
        }
        
        result = reviewer_node(state)
        
        # Should complete without errors
        assert result.get("status") != ResearchStatus.FAILED
        
        # Should have all review outputs
        assert "review_critique" in result
        assert "review_decision" in result
        assert "reviewer_output" in result
        
        # Critique should have dimension scores
        critique = result["review_critique"]
        if isinstance(critique, dict):
            assert "dimension_scores" in critique
        else:
            assert hasattr(critique, "dimension_scores")
    
    def test_quality_scores_are_meaningful(self, sample_writer_output):
        """Test that quality scores are meaningful and differentiate papers."""
        good_scores, _ = evaluate_paper(sample_writer_output)
        
        poor_output = {
            "title": "Bad",
            "sections": [
                {"section_type": "abstract", "content": "Stuff.", "order": 1}
            ],
        }
        poor_scores, _ = evaluate_paper(poor_output)
        
        # Good paper should score higher on average
        good_avg = sum(s.score for s in good_scores) / len(good_scores)
        poor_avg = sum(s.score for s in poor_scores) / len(poor_scores)
        
        assert good_avg > poor_avg
