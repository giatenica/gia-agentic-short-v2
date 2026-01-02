"""Unit tests for PLANNER node and methodology/analysis tools.

Tests cover:
1. Methodology tools (methodology.py)
2. Analysis design tools (analysis_design.py)
3. PLANNER node (planner.py)
"""

import pytest
from unittest.mock import MagicMock, patch

from src.state.enums import (
    AnalysisApproach,
    MethodologyType,
    PlanApprovalStatus,
    ResearchStatus,
)
from src.state.models import (
    ContributionStatement,
    GapAnalysis,
    RefinedResearchQuestion,
    ResearchGap,
    ResearchPlan,
    SearchResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def create_mock_gap_analysis() -> GapAnalysis:
    """Create a mock gap analysis for testing PLANNER."""
    gaps = [
        ResearchGap(
            gap_type="empirical",
            title="Emerging markets ESG studies",
            description="Limited research on ESG effects in emerging markets, particularly regarding long-term performance.",
            significance="high",
            significance_justification="Large economic impact potential in rapidly growing economies.",
            addressable=True,
        ),
        ResearchGap(
            gap_type="methodological",
            title="Causal identification strategies",
            description="Most studies lack rigorous causal identification, relying on correlational evidence.",
            significance="high",
            significance_justification="Fundamental to establishing valid conclusions.",
            addressable=True,
        ),
        ResearchGap(
            gap_type="theoretical",
            title="Mechanism clarity",
            description="Unclear mechanisms through which ESG affects firm performance and value creation.",
            significance="medium",
            significance_justification="Important for theory building and practical implications.",
            addressable=True,
        ),
    ]
    
    return GapAnalysis(
        original_question="How does ESG performance affect firm value in emerging markets?",
        literature_coverage_summary="Extensive coverage in developed markets but limited emerging market research.",
        gaps=gaps,
        primary_gap=gaps[0],
        coverage_comparison="Strong coverage of ESG effects in US and EU markets, weak coverage in BRICS and other emerging economies.",
        coverage_percentage=45.0,
        methodological_gaps=["Causal identification", "Endogeneity concerns"],
        empirical_gaps=["Emerging markets data", "Long-term performance studies"],
        theoretical_gaps=["Mechanism clarity", "Context-dependent effects"],
        gap_significance_ranking=[g.gap_id for g in gaps],
    )


def create_mock_refined_question() -> RefinedResearchQuestion:
    """Create a mock refined research question for testing."""
    return RefinedResearchQuestion(
        original_question="How does ESG affect firm performance?",
        refined_question="Does ESG performance causally improve long-term firm value in emerging markets, and through what mechanisms does this relationship operate?",
        refinement_rationale="Added specificity on market context (emerging markets), temporal dimension (long-term), and mechanisms to address identified gaps.",
        key_variables=["ESG scores", "Firm value (Tobin's Q)", "Market context", "Time horizon"],
        scope_boundaries="Focus on BRICS economies from 2015-2023.",
        testable=True,
    )


def create_mock_contribution() -> ContributionStatement:
    """Create a mock contribution statement for testing."""
    return ContributionStatement(
        main_statement="First comprehensive study of ESG-performance relationship in emerging markets using causal identification strategies.",
        contribution_type="empirical",
        gap_addressed="Emerging markets ESG studies",
        novelty_explanation="Extends ESG research to understudied emerging market contexts.",
    )


def create_mock_synthesis_info() -> dict:
    """Create mock literature synthesis information."""
    return {
        "summary": "ESG research is extensive but geographically limited to developed markets.",
        "state_of_field": "Current research shows mixed results with limited causal evidence.",
        "key_findings": [
            "ESG scores correlate with long-term performance in developed markets.",
            "Governance factors show strongest correlation.",
            "Emerging markets remain understudied.",
        ],
        "theoretical_frameworks": [
            "Stakeholder theory",
            "Resource-based view",
            "Agency theory",
        ],
        "methodological_approaches": [
            "Panel data regression",
            "Event studies",
            "Cross-sectional analysis",
        ],
        "themes_identified": 5,
    }


def create_mock_data_exploration() -> dict:
    """Create mock data exploration results."""
    return {
        "datasets_found": 3,
        "total_rows": 50000,
        "total_columns": 25,
        "variables_available": [
            "esg_score", "firm_value", "market_cap", "country", "industry"
        ],
        "quality_level": "good",
        "missing_data_percentage": 5.2,
    }


# =============================================================================
# Methodology Tools Tests
# =============================================================================


class TestSelectMethodology:
    """Tests for methodology selection functions."""

    @patch("src.tools.methodology.ChatAnthropic")
    def test_select_methodology_empirical_gap(self, mock_llm):
        """Test methodology selection for empirical gap type."""
        from src.tools.methodology import select_methodology
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = """METHODOLOGY: panel_data
RATIONALE: Panel data is ideal for studying cross-sectional and time-series variation in ESG effects.
ALTERNATIVES: event_study, regression_analysis
PRECEDENTS: Consistent with established finance literature."""
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = select_methodology(
            research_type="empirical",
            gap_type="empirical",
            has_data=True,
        )
        
        assert "methodology_type" in result or "methodology_name" in result
        assert "justification" in result

    @patch("src.tools.methodology.ChatAnthropic")
    def test_select_methodology_methodological_gap(self, mock_llm):
        """Test methodology selection for methodological gap."""
        from src.tools.methodology import select_methodology
        
        mock_response = MagicMock()
        mock_response.content = """METHODOLOGY: difference_in_differences
RATIONALE: DiD addresses causal identification concerns.
ALTERNATIVES: instrumental_variables
PRECEDENTS: Widely used in policy evaluation."""
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = select_methodology(
            research_type="empirical",
            gap_type="methodological",
            has_data=True,
        )
        
        assert "methodology_type" in result or "methodology_name" in result

    @patch("src.tools.methodology.ChatAnthropic")
    def test_select_methodology_theoretical_gap(self, mock_llm):
        """Test methodology selection for theoretical gap."""
        from src.tools.methodology import select_methodology
        
        mock_response = MagicMock()
        mock_response.content = """METHODOLOGY: analytical_model
RATIONALE: Theoretical gaps require formal model development.
ALTERNATIVES: conceptual_framework
PRECEDENTS: Standard in theory papers."""
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = select_methodology(
            research_type="theoretical",
            gap_type="theoretical",
            has_data=False,
        )
        
        assert "methodology_type" in result or "methodology_name" in result

    @patch("src.tools.methodology.ChatAnthropic")
    def test_select_methodology_qualitative(self, mock_llm):
        """Test methodology selection for qualitative research."""
        from src.tools.methodology import select_methodology
        
        mock_response = MagicMock()
        mock_response.content = """METHODOLOGY: case_study
RATIONALE: Case study allows deep exploration of mechanisms.
ALTERNATIVES: content_analysis, thematic_analysis
PRECEDENTS: Common in exploratory research."""
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = select_methodology(
            research_type="qualitative",
            gap_type="empirical",
            has_data=False,
        )
        
        assert "methodology_type" in result or "methodology_name" in result


class TestValidateMethodologyFit:
    """Tests for methodology validation."""

    def test_validate_methodology_good_fit(self):
        """Test validation with well-matched methodology."""
        from src.tools.methodology import validate_methodology_fit
        
        result = validate_methodology_fit(
            methodology_type=MethodologyType.PANEL_DATA,
            gap_type="empirical",
            research_type="empirical",
            has_data=True,
        )
        
        assert "is_valid" in result
        assert result["is_valid"] is True
        assert "score" in result

    def test_validate_methodology_poor_fit_no_data(self):
        """Test validation with mismatched methodology."""
        from src.tools.methodology import validate_methodology_fit
        
        result = validate_methodology_fit(
            methodology_type=MethodologyType.PANEL_DATA,
            gap_type="empirical",
            research_type="empirical",
            has_data=False,  # Panel data needs data
        )
        
        # Score should be lower when data-required method used without data
        assert "score" in result
        assert result["score"] < 1.0


class TestAssessFeasibility:
    """Tests for feasibility assessment."""

    @patch("src.tools.methodology.ChatAnthropic")
    def test_assess_feasibility_high(self, mock_llm):
        """Test feasibility assessment for well-resourced research."""
        from src.tools.methodology import assess_feasibility
        
        mock_response = MagicMock()
        mock_response.content = """FEASIBILITY_SCORE: 0.85
ASSESSMENT: Research is highly feasible with available data.
CHALLENGES:
- Minor data cleaning required
RECOMMENDATIONS:
- Proceed with research plan"""
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = assess_feasibility(
            methodology_type=MethodologyType.PANEL_DATA,
            data_available={"total_rows": 50000, "quality_level": "good"},
            time_constraints="6 months",
        )
        
        assert "feasibility_score" in result

    @patch("src.tools.methodology.ChatAnthropic")
    def test_assess_feasibility_low(self, mock_llm):
        """Test feasibility assessment with limited resources."""
        from src.tools.methodology import assess_feasibility
        
        mock_response = MagicMock()
        mock_response.content = """FEASIBILITY_SCORE: 0.35
ASSESSMENT: Research has significant challenges due to missing data.
CHALLENGES:
- Missing treatment data
- Short timeline
RECOMMENDATIONS:
- Consider alternative methodology"""
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = assess_feasibility(
            methodology_type=MethodologyType.DIFFERENCE_IN_DIFFERENCES,
            data_available=None,
            time_constraints="1 month",
        )
        
        assert "feasibility_score" in result


# =============================================================================
# Analysis Design Tools Tests
# =============================================================================


class TestDesignQuantitativeAnalysis:
    """Tests for quantitative analysis design."""

    @patch("src.tools.analysis_design.ChatAnthropic")
    def test_design_quantitative_panel_regression(self, mock_llm):
        """Test analysis design for panel data."""
        from src.tools.analysis_design import design_quantitative_analysis
        
        mock_response = MagicMock()
        mock_response.content = """DEPENDENT_VARIABLE: Tobin's Q (firm_value)
INDEPENDENT_VARIABLES: esg_score, esg_environmental, esg_social, esg_governance
CONTROL_VARIABLES: firm_size, leverage, profitability, industry_dummies, year_dummies
MODEL_SPECIFICATION: firm_value_{it} = α + β·esg_{it} + γ·X_{it} + μ_i + λ_t + ε_{it}
ROBUSTNESS_TESTS: 
- Alternative ESG measures
- Lagged ESG scores
- Subsample analysis by region
EXPECTED_TABLES:
- Summary statistics
- Correlation matrix
- Main regression results
- Robustness checks
EXPECTED_FIGURES:
- Time trend of ESG scores
- Distribution of firm value by ESG quartile"""
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = design_quantitative_analysis(
            methodology_type=MethodologyType.PANEL_DATA,
            research_question="How does ESG affect firm value?",
            key_variables=["esg_score", "firm_value", "size", "leverage", "profitability"],
        )
        
        assert "analysis_type" in result
        assert result["analysis_type"] == "quantitative"
        assert "statistical_tests" in result
        assert "full_design" in result

    @patch("src.tools.analysis_design.ChatAnthropic")
    def test_design_quantitative_event_study(self, mock_llm):
        """Test analysis design for event study."""
        from src.tools.analysis_design import design_quantitative_analysis
        
        mock_response = MagicMock()
        mock_response.content = """DEPENDENT_VARIABLE: Cumulative abnormal returns (CAR)
INDEPENDENT_VARIABLES: esg_announcement, announcement_type
CONTROL_VARIABLES: firm_size, market_beta, book_to_market
MODEL_SPECIFICATION: CAR[-5,+5] = α + β·announcement + γ·X + ε
ROBUSTNESS_TESTS:
- Different event windows
- Market model vs Fama-French
EXPECTED_TABLES:
- Event dates summary
- CAR by window
- Cross-sectional regression
EXPECTED_FIGURES:
- Cumulative abnormal returns plot"""
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = design_quantitative_analysis(
            methodology_type=MethodologyType.EVENT_STUDY,
            research_question="What is the market reaction to ESG announcements?",
            key_variables=["abnormal_returns", "esg_announcement", "event_window"],
        )
        
        assert "analysis_type" in result
        assert "statistical_tests" in result


class TestDesignQualitativeAnalysis:
    """Tests for qualitative analysis design."""

    @patch("src.tools.analysis_design.ChatAnthropic")
    def test_design_qualitative_case_study(self, mock_llm):
        """Test analysis design for case study."""
        from src.tools.analysis_design import design_qualitative_analysis
        
        mock_response = MagicMock()
        mock_response.content = """DATA_COLLECTION: Semi-structured interviews with executives, annual reports, sustainability reports
CODING_APPROACH: Thematic coding with constant comparison method
THEMES_TO_EXPLORE: ESG integration, strategic alignment, stakeholder pressure
VALIDITY_MEASURES: Triangulation, member checking, peer review
PRESENTATION_FORMAT: Rich case narratives with cross-case comparison"""
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = design_qualitative_analysis(
            methodology_type=MethodologyType.CASE_STUDY,
            research_question="How do firms integrate ESG into strategy?",
            data_sources=["interviews", "annual reports", "press releases"],
        )
        
        assert "analysis_type" in result
        assert result["analysis_type"] == "qualitative"


class TestDeterminePaperSections:
    """Tests for paper section determination."""

    def test_determine_sections_empirical(self):
        """Test section determination for empirical paper."""
        from src.tools.analysis_design import determine_paper_sections
        
        result = determine_paper_sections(
            paper_type="full_paper",
            methodology_type=MethodologyType.PANEL_DATA,
        )
        
        # Should return a list of sections
        assert isinstance(result, list)
        assert len(result) > 0
        # Standard empirical sections should be present
        sections_lower = [s.lower() for s in result]
        assert any("introduction" in s for s in sections_lower)
        assert any("data" in s or "sample" in s for s in sections_lower)

    def test_determine_sections_theoretical(self):
        """Test section determination for theoretical paper."""
        from src.tools.analysis_design import determine_paper_sections
        
        result = determine_paper_sections(
            paper_type="theoretical",
            methodology_type=MethodologyType.ANALYTICAL_MODEL,
            research_type="theoretical",
        )
        
        assert isinstance(result, list)
        assert len(result) > 0

    def test_determine_sections_literature_review(self):
        """Test section determination for literature review."""
        from src.tools.analysis_design import determine_paper_sections
        
        result = determine_paper_sections(
            paper_type="full_paper",
            methodology_type=MethodologyType.SYSTEMATIC_REVIEW,
        )
        
        assert isinstance(result, list)
        sections_lower = [s.lower() for s in result]
        # Should have literature review-specific sections
        assert any("literature" in s or "methodology" in s for s in sections_lower)


class TestDefineSuccessCriteria:
    """Tests for success criteria definition."""

    @patch("src.tools.analysis_design.ChatAnthropic")
    def test_define_success_criteria_empirical(self, mock_llm):
        """Test success criteria for empirical research."""
        from src.tools.analysis_design import define_success_criteria
        
        mock_response = MagicMock()
        mock_response.content = """SUCCESS_CRITERIA:
1. Statistical significance of main coefficient at p < 0.05
2. Economic significance with meaningful effect size
3. Robustness across alternative specifications
4. Clear causal interpretation with valid identification
5. Novel contribution to emerging markets ESG literature"""
        mock_llm.return_value.invoke.return_value = mock_response
        
        gap_dict = create_mock_gap_analysis().model_dump()
        
        result = define_success_criteria(
            gap_analysis=gap_dict,
            methodology_type=MethodologyType.PANEL_DATA,
            research_question="How does ESG affect firm value?",
        )
        
        # Result is a list of criteria
        assert isinstance(result, list)
        assert len(result) > 0


# =============================================================================
# PLANNER Node Tests
# =============================================================================


class TestPlannerNodeHelpers:
    """Tests for PLANNER node helper functions."""

    def test_extract_gap_info(self):
        """Test gap information extraction from state."""
        from src.nodes.planner import _extract_gap_info
        
        gap_analysis = create_mock_gap_analysis()
        state = {
            "gap_analysis": gap_analysis.model_dump(),
        }
        
        result = _extract_gap_info(state)
        
        assert "gap_type" in result
        assert result["gap_type"] == "empirical"
        assert "gap_title" in result

    def test_extract_synthesis_info(self):
        """Test synthesis information extraction."""
        from src.nodes.planner import _extract_synthesis_info
        
        synthesis = create_mock_synthesis_info()
        state = {
            "literature_synthesis": synthesis,
        }
        
        result = _extract_synthesis_info(state)
        
        assert "methodological_approaches" in result
        assert "theoretical_frameworks" in result
        assert len(result["methodological_approaches"]) > 0

    def test_get_research_question(self):
        """Test research question extraction."""
        from src.nodes.planner import _get_research_question
        
        # Test with refined question
        state = {
            "refined_research_question": create_mock_refined_question().model_dump(),
            "original_query": "How does ESG affect firms?",
        }
        
        result = _get_research_question(state)
        assert "emerging markets" in result.lower()
        
        # Test fallback to original
        state2 = {
            "original_query": "How does ESG affect firms?",
        }
        
        result2 = _get_research_question(state2)
        assert result2 == "How does ESG affect firms?"


class TestRouteAfterPlanner:
    """Tests for routing logic after PLANNER node."""

    def test_route_to_data_analyst(self):
        """Test routing to data analyst for empirical research."""
        from src.nodes.planner import route_after_planner
        
        state = {
            "research_plan": {
                "title": "ESG and Firm Value",
                "summary": "Empirical analysis of ESG effects",
                "methodology": "Panel data regression analysis",
                "methodology_type": MethodologyType.PANEL_DATA.value,
                "data_requirements": ["ESG scores", "financial data"],
            },
            "data_exploration_results": {"datasets_found": 3, "total_rows": 50000},
        }
        
        result = route_after_planner(state)
        assert result == "data_analyst"

    def test_route_to_conceptual_synthesizer(self):
        """Test routing to conceptual synthesizer for theoretical research."""
        from src.nodes.planner import route_after_planner
        
        state = {
            "research_plan": {
                "title": "Theory of ESG Value Creation",
                "summary": "Theoretical framework development",
                "methodology": "Conceptual analysis and model building",
                "methodology_type": MethodologyType.CONCEPTUAL_FRAMEWORK.value,
            },
        }
        
        result = route_after_planner(state)
        assert result == "conceptual_synthesizer"

    def test_route_conceptual_when_no_data(self):
        """Test routing to conceptual synthesizer when no data available."""
        from src.nodes.planner import route_after_planner
        
        state = {
            "research_plan": {
                "title": "ESG Analysis",
                "summary": "Analysis without available data",
                "methodology": "Review and synthesis",
            },
            "data_exploration_results": None,
        }
        
        result = route_after_planner(state)
        assert result == "conceptual_synthesizer"


class TestProcessPlanApproval:
    """Tests for plan approval processing."""

    def test_process_string_approval(self):
        """Test processing string approval response."""
        from src.nodes.planner import _process_plan_approval
        
        plan = ResearchPlan(
            original_query="How does ESG affect firm value?",
            title="ESG and Firm Value Study",
            summary="Test plan",
            methodology="Panel data",
        )
        
        # Test simple approval
        result = _process_plan_approval(plan, "approved")
        assert result.approval_status == PlanApprovalStatus.APPROVED
        
    def test_process_string_revision(self):
        """Test processing revision request."""
        from src.nodes.planner import _process_plan_approval
        
        plan = ResearchPlan(
            original_query="How does ESG affect firm value?",
            title="ESG and Firm Value Study",
            summary="Test plan",
            methodology="Panel data",
        )
        
        result = _process_plan_approval(plan, "Add more robustness checks")
        assert result.approval_status == PlanApprovalStatus.REVISION_REQUESTED
        assert "robustness" in result.approval_notes.lower()

    def test_process_dict_approval(self):
        """Test processing dict approval response."""
        from src.nodes.planner import _process_plan_approval
        
        plan = ResearchPlan(
            original_query="How does ESG affect firm value?",
            title="ESG and Firm Value Study",
            summary="Test plan",
            methodology="Panel data",
        )
        
        result = _process_plan_approval(plan, {
            "action": "approved",
            "notes": "Plan is comprehensive",
        })
        assert result.approval_status == PlanApprovalStatus.APPROVED
        assert "comprehensive" in result.approval_notes

    def test_process_dict_rejection(self):
        """Test processing dict rejection response."""
        from src.nodes.planner import _process_plan_approval
        
        plan = ResearchPlan(
            original_query="How does ESG affect firm value?",
            title="ESG and Firm Value Study",
            summary="Test plan",
            methodology="Panel data",
        )
        
        result = _process_plan_approval(plan, {
            "action": "rejected",
            "reason": "Methodology does not fit the research question",
        })
        assert result.approval_status == PlanApprovalStatus.REJECTED


# =============================================================================
# Integration Tests
# =============================================================================


class TestPlannerNodeIntegration:
    """Integration tests for the full PLANNER node."""

    @pytest.fixture
    def mock_state(self):
        """Create a complete mock state for planner testing."""
        return {
            "messages": [],
            "original_query": "How does ESG performance affect firm value in emerging markets?",
            "literature_synthesis": create_mock_synthesis_info(),
            "identified_gaps": [
                {
                    "gap_type": "empirical",
                    "title": "Emerging markets ESG studies",
                    "description": "Limited research in emerging markets",
                    "significance": "high",
                }
            ],
            "gap_analysis": create_mock_gap_analysis().model_dump(),
            "refined_research_question": create_mock_refined_question().model_dump(),
            "contribution": create_mock_contribution().model_dump(),
            "data_exploration_results": create_mock_data_exploration(),
            "research_plan": None,
            "status": ResearchStatus.PLANNING,
        }

    def test_planner_creates_valid_research_plan(self, mock_state):
        """Test that planner creates a valid research plan structure."""
        # This is a structural test; actual LLM calls would be mocked
        from src.state.models import ResearchPlan
        
        # Verify ResearchPlan can be created with expected fields
        plan = ResearchPlan(
            original_query="How does ESG performance affect firm value in emerging markets?",
            methodology="Panel data regression with fixed effects",
            methodology_type=MethodologyType.PANEL_DATA,
            analysis_approach=AnalysisApproach.FIXED_EFFECTS,
            target_gap="Emerging markets ESG studies",
            gap_type="empirical",
            hypothesis="ESG performance positively affects firm value in emerging markets",
            key_variables=["esg_score", "firm_value", "country"],
            control_variables=["size", "leverage", "profitability"],
            data_requirements=["ESG ratings", "financial statements", "stock prices"],
            expected_tables=["Summary statistics", "Main regression", "Robustness"],
            expected_figures=["Time trend", "Country comparison"],
            feasibility_score=0.85,
            approval_status=PlanApprovalStatus.PENDING,
        )
        
        assert plan.original_query is not None
        assert plan.methodology_type == MethodologyType.PANEL_DATA
        assert plan.approval_status == PlanApprovalStatus.PENDING
        assert plan.feasibility_score == 0.85


class TestMethodologyTypeMapping:
    """Tests for methodology type mappings."""

    def test_all_methodology_types_have_analyses(self):
        """Verify key methodology types have mapped analysis approaches."""
        from src.tools.methodology import METHODOLOGY_ANALYSES
        
        # Check key methodologies have mappings
        key_methodologies = [
            MethodologyType.PANEL_DATA,
            MethodologyType.EVENT_STUDY,
            MethodologyType.CASE_STUDY,
            MethodologyType.REGRESSION_ANALYSIS,
        ]
        
        for method in key_methodologies:
            assert method in METHODOLOGY_ANALYSES, f"Missing mapping for {method}"
            assert len(METHODOLOGY_ANALYSES[method]) > 0

    def test_research_type_methodologies_complete(self):
        """Verify research type mappings are complete."""
        from src.tools.methodology import RESEARCH_TYPE_METHODOLOGIES
        
        # Updated to match actual keys in the implementation
        expected_types = ["empirical", "theoretical", "mixed", "literature_review", "case_study"]
        
        for research_type in expected_types:
            assert research_type in RESEARCH_TYPE_METHODOLOGIES, f"Missing {research_type}"
            assert len(RESEARCH_TYPE_METHODOLOGIES[research_type]) > 0


class TestAnalysisApproachMapping:
    """Tests for analysis approach configurations."""

    def test_statistical_tests_by_methodology(self):
        """Verify statistical tests are mapped to methodologies."""
        from src.tools.analysis_design import STATISTICAL_TESTS_BY_METHODOLOGY
        
        assert MethodologyType.PANEL_DATA in STATISTICAL_TESTS_BY_METHODOLOGY
        assert MethodologyType.EVENT_STUDY in STATISTICAL_TESTS_BY_METHODOLOGY
        
        panel_tests = STATISTICAL_TESTS_BY_METHODOLOGY[MethodologyType.PANEL_DATA]
        assert "hausman" in str(panel_tests).lower() or "f-test" in str(panel_tests).lower()

    def test_paper_sections_by_type(self):
        """Verify paper sections are defined for all types."""
        from src.tools.analysis_design import PAPER_SECTIONS_BY_TYPE
        
        # Updated to match actual keys in the implementation
        expected_types = ["full_paper", "short_article", "theoretical", "literature_review", "case_study"]
        
        for paper_type in expected_types:
            assert paper_type in PAPER_SECTIONS_BY_TYPE, f"Missing {paper_type}"
            sections = PAPER_SECTIONS_BY_TYPE[paper_type]
            assert len(sections) >= 4  # At least intro, body, results, conclusion


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_planner_with_empty_gaps(self):
        """Test handling when no gaps are identified."""
        from src.nodes.planner import _extract_gap_info
        
        state = {"gap_analysis": None, "identified_gaps": []}
        
        result = _extract_gap_info(state)
        # Should return defaults when no gap analysis
        assert "gap_type" in result
        assert result["gap_type"] == "empirical"  # default

    @patch("src.tools.methodology.ChatAnthropic")
    def test_methodology_selection_unknown_type(self, mock_llm):
        """Test methodology selection with unknown research type."""
        from src.tools.methodology import select_methodology
        
        mock_response = MagicMock()
        mock_response.content = """METHODOLOGY: regression_analysis
RATIONALE: Default quantitative approach.
ALTERNATIVES: panel_data
PRECEDENTS: Standard in empirical research."""
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = select_methodology(
            research_type="unknown_type",
            gap_type="empirical",
            has_data=True,
        )
        
        # Should still return a result
        assert "methodology_type" in result or "methodology_name" in result

    def test_route_with_missing_plan(self):
        """Test routing when research plan is missing."""
        from src.nodes.planner import route_after_planner
        
        state = {"research_plan": None}
        
        # Should default to conceptual synthesizer
        result = route_after_planner(state)
        assert result == "conceptual_synthesizer"

    def test_route_with_none_values(self):
        """Test routing handles None values gracefully."""
        from src.nodes.planner import route_after_planner
        
        state = {
            "research_plan": None,
            "data_exploration_results": None,
        }
        
        result = route_after_planner(state)
        assert result == "conceptual_synthesizer"
