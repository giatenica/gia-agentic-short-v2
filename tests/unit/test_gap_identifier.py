"""Unit tests for gap analysis tools and GAP_IDENTIFIER node.

Tests cover:
1. Gap analysis tools (gap_analysis.py)
2. Contribution tools (contribution.py)
3. GAP_IDENTIFIER node (gap_identifier.py)
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

from src.state.enums import ResearchStatus
from src.state.models import (
    ContributionStatement,
    GapAnalysis,
    LiteratureSynthesis,
    RefinedResearchQuestion,
    ResearchGap,
    SearchResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def create_mock_literature_synthesis() -> dict:
    """Create a mock literature synthesis for testing."""
    return {
        "summary": "The literature on ESG and firm performance is extensive but shows mixed results.",
        "state_of_field": "Current research shows conflicting findings on ESG-performance relationships.",
        "key_findings": [
            "ESG scores correlate positively with long-term performance in developed markets.",
            "Short-term ESG effects are unclear and context-dependent.",
            "Governance factors show strongest correlation with financial performance.",
            "Environmental factors' impact varies by industry.",
            "Social factors are understudied compared to E and G.",
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
        "contribution_opportunities": [
            "Study ESG in emerging markets",
            "Examine individual E, S, G components separately",
            "Longitudinal studies needed",
        ],
        "papers_analyzed": 25,
        "themes_identified": 5,
        "gaps_identified": 3,
    }


def create_mock_search_results() -> list[SearchResult]:
    """Create mock search results for testing."""
    return [
        SearchResult(
            query_id="q1",
            title="ESG and Corporate Performance: A Meta-Analysis",
            url="https://example.com/paper1",
            snippet="This paper analyzes 200+ studies on ESG and firm performance...",
            source_type="academic",
            citation_count=150,
            venue="Journal of Finance",
        ),
        SearchResult(
            query_id="q1",
            title="The Impact of Environmental Scores on Stock Returns",
            url="https://example.com/paper2",
            snippet="We examine environmental ratings and stock market performance...",
            source_type="academic",
            citation_count=75,
            venue="Review of Financial Studies",
        ),
        SearchResult(
            query_id="q2",
            title="Social Responsibility and Firm Value in Emerging Markets",
            url="https://example.com/paper3",
            snippet="Corporate social responsibility in developing economies...",
            source_type="academic",
            citation_count=45,
            venue="Journal of Corporate Finance",
        ),
    ]


def create_mock_gap_analysis() -> GapAnalysis:
    """Create a mock gap analysis for testing."""
    gaps = [
        ResearchGap(
            gap_type="empirical",
            title="Emerging markets ESG studies",
            description="Limited research on ESG effects in emerging markets.",
            significance="high",
            significance_justification="Large economic impact potential",
            addressable=True,
        ),
        ResearchGap(
            gap_type="methodological",
            title="Causal identification",
            description="Most studies lack rigorous causal identification.",
            significance="high",
            significance_justification="Fundamental to validity",
            addressable=True,
        ),
        ResearchGap(
            gap_type="theoretical",
            title="Mechanism clarity",
            description="Unclear mechanisms through which ESG affects performance.",
            significance="medium",
            significance_justification="Important for theory building",
            addressable=True,
        ),
    ]
    
    return GapAnalysis(
        original_question="How does ESG performance affect firm value?",
        literature_coverage_summary="Extensive but geographically limited coverage.",
        gaps=gaps,
        primary_gap=gaps[0],
        coverage_comparison="Strong coverage in developed markets, weak in emerging markets.",
        coverage_percentage=65.0,
        methodological_gaps=["Causal identification"],
        empirical_gaps=["Emerging markets ESG studies"],
        theoretical_gaps=["Mechanism clarity"],
        gap_significance_ranking=[g.gap_id for g in gaps],
    )


# =============================================================================
# Gap Analysis Tools Tests
# =============================================================================


class TestCompareCoverage:
    """Tests for compare_coverage function."""
    
    def test_parse_coverage_response_basic(self):
        """Test parsing of a basic coverage response."""
        from src.tools.gap_analysis import _parse_coverage_response
        
        content = """COVERAGE_SUMMARY: The literature covers most aspects but lacks emerging market data.

COVERAGE_PERCENTAGE: 70

WELL_COVERED:
- Developed market ESG studies
- Governance and performance links

PARTIALLY_COVERED:
- Environmental impact: lacks longitudinal data
- Social factors: limited geographic scope

NOT_COVERED:
- Emerging market ESG effects
- Real-time ESG measurement"""
        
        result = _parse_coverage_response(content)
        
        assert "coverage_summary" in result
        assert result["coverage_percentage"] == 70.0
        assert len(result["well_covered"]) == 2
        assert len(result["partially_covered"]) == 2
        assert len(result["not_covered"]) == 2
    
    def test_parse_coverage_handles_missing_sections(self):
        """Test parsing handles missing sections gracefully."""
        from src.tools.gap_analysis import _parse_coverage_response
        
        content = """COVERAGE_SUMMARY: Limited coverage available.

COVERAGE_PERCENTAGE: 30"""
        
        result = _parse_coverage_response(content)
        
        assert result["coverage_summary"] == "Limited coverage available."
        assert result["coverage_percentage"] == 30.0
        assert result["well_covered"] == []
        assert result["not_covered"] == []


class TestIdentifyGaps:
    """Tests for gap identification functions."""
    
    def test_parse_gap_response_methodological(self):
        """Test parsing methodological gap response."""
        from src.tools.gap_analysis import _parse_gap_response
        
        content = """GAP: Causal identification weakness
DESCRIPTION: Most studies use correlational designs without proper causal identification.
CURRENT_METHODS: OLS regression, fixed effects
LIMITATION: Cannot establish causality
SIGNIFICANCE: high
ADDRESSABLE: yes
HOW_TO_ADDRESS: Use instrumental variables or natural experiments

GAP: Data quality issues
DESCRIPTION: ESG ratings from different providers show low correlation.
CURRENT_METHODS: Single-source ESG data
LIMITATION: Results may be provider-specific
SIGNIFICANCE: medium
ADDRESSABLE: yes
HOW_TO_ADDRESS: Use multiple ESG providers or construct composite scores"""
        
        gaps = _parse_gap_response(content, "methodological")
        
        assert len(gaps) == 2
        assert gaps[0]["title"] == "Causal identification weakness"
        assert gaps[0]["gap_type"] == "methodological"
        assert gaps[0]["significance"] == "high"
        assert gaps[0]["addressable"] is True
        assert gaps[1]["title"] == "Data quality issues"
        assert gaps[1]["significance"] == "medium"
    
    def test_parse_gap_response_empirical(self):
        """Test parsing empirical gap response."""
        from src.tools.gap_analysis import _parse_gap_response
        
        content = """GAP: Emerging market evidence
DESCRIPTION: Very few studies examine ESG in emerging economies.
EXISTING_EVIDENCE: Mostly developed market studies
MISSING_EVIDENCE: Emerging market data
CONTEXT: BRICS countries, Southeast Asia
SIGNIFICANCE: high
ADDRESSABLE: yes
DATA_NEEDED: ESG scores and financial data for emerging market firms"""
        
        gaps = _parse_gap_response(content, "empirical")
        
        assert len(gaps) == 1
        assert gaps[0]["title"] == "Emerging market evidence"
        assert gaps[0]["gap_type"] == "empirical"
        assert "BRICS" in gaps[0].get("context", "")
    
    def test_parse_gap_response_handles_empty(self):
        """Test parsing handles empty or minimal response."""
        from src.tools.gap_analysis import _parse_gap_response
        
        content = "No clear gaps identified in this area."
        
        gaps = _parse_gap_response(content, "theoretical")
        
        assert gaps == []


class TestAssessGapSignificance:
    """Tests for gap significance assessment."""
    
    def test_parse_significance_response_basic(self):
        """Test parsing significance assessment response."""
        from src.tools.gap_analysis import _parse_significance_response
        
        gaps = [
            {"title": "Gap 1", "gap_type": "empirical"},
            {"title": "Gap 2", "gap_type": "methodological"},
        ]
        
        content = """GAP_NUMBER: 1
ACADEMIC_IMPACT: 8
PRACTICAL_RELEVANCE: 7
FEASIBILITY: 9
ALIGNMENT: 8
OVERALL_SCORE: 8.0
REVISED_SIGNIFICANCE: high
JUSTIFICATION: High impact potential for emerging market research.

GAP_NUMBER: 2
ACADEMIC_IMPACT: 9
PRACTICAL_RELEVANCE: 5
FEASIBILITY: 6
ALIGNMENT: 7
OVERALL_SCORE: 7.2
REVISED_SIGNIFICANCE: medium
JUSTIFICATION: Important but challenging to address.

RANKING: 1, 2
PRIMARY_GAP: 1
PRIMARY_GAP_REASON: Best balance of impact and feasibility."""
        
        result = _parse_significance_response(content, gaps)
        
        assert len(result) == 2
        assert result[0]["academic_impact"] == 8
        assert result[0]["significance"] == "high"
        assert result[0]["is_primary"] is True
        assert result[1]["significance"] == "medium"


class TestPerformGapAnalysis:
    """Tests for the comprehensive gap analysis function."""
    
    @patch("src.tools.gap_analysis.compare_coverage")
    @patch("src.tools.gap_analysis.identify_methodological_gaps")
    @patch("src.tools.gap_analysis.identify_empirical_gaps")
    @patch("src.tools.gap_analysis.identify_theoretical_gaps")
    @patch("src.tools.gap_analysis.assess_gap_significance")
    def test_perform_gap_analysis_combines_all_types(
        self,
        mock_assess,
        mock_theoretical,
        mock_empirical,
        mock_methodological,
        mock_coverage,
    ):
        """Test that perform_gap_analysis combines all gap types."""
        from src.tools.gap_analysis import perform_gap_analysis
        
        mock_coverage.return_value = {
            "coverage_summary": "Good coverage overall",
            "coverage_percentage": 70.0,
            "well_covered": ["Topic A"],
            "partially_covered": ["Topic B"],
            "not_covered": ["Topic C"],
        }
        
        mock_methodological.return_value = [
            {"title": "Method Gap", "description": "This is a methodological gap that needs more detail for testing purposes.", "significance": "high", "addressable": True, "gap_type": "methodological"}
        ]
        mock_empirical.return_value = [
            {"title": "Empirical Gap", "description": "This is an empirical gap that requires additional evidence and testing.", "significance": "medium", "addressable": True, "gap_type": "empirical"}
        ]
        mock_theoretical.return_value = []
        
        mock_assess.return_value = [
            {"title": "Method Gap", "description": "This is a methodological gap that needs more detail for testing purposes.", "significance": "high", "addressable": True, "gap_type": "methodological", "is_primary": True},
            {"title": "Empirical Gap", "description": "This is an empirical gap that requires additional evidence and testing.", "significance": "medium", "addressable": True, "gap_type": "empirical", "is_primary": False},
        ]
        
        synthesis = create_mock_literature_synthesis()
        
        result = perform_gap_analysis(
            original_question="Test question?",
            literature_synthesis=synthesis,
        )
        
        assert isinstance(result, GapAnalysis)
        assert result.coverage_percentage == 70.0
        assert len(result.gaps) == 2
        assert len(result.methodological_gaps) == 1
        assert len(result.empirical_gaps) == 1
        assert result.primary_gap is not None


# =============================================================================
# Contribution Tools Tests
# =============================================================================


class TestGenerateContributionStatement:
    """Tests for contribution statement generation."""
    
    def test_parse_contribution_response_basic(self):
        """Test parsing of contribution response."""
        from src.tools.contribution import _parse_contribution_response
        
        content = """MAIN_STATEMENT: This paper contributes to the ESG literature by providing the first comprehensive analysis of ESG effects in emerging markets, using a novel instrumental variables approach to establish causal relationships.

CONTRIBUTION_TYPE: empirical

NOVELTY_EXPLANATION: First study to combine emerging market focus with rigorous causal identification.

GAP_ADDRESSED: Fills the empirical gap in emerging market ESG research while addressing methodological concerns about causality.

POTENTIAL_IMPACT: Informs ESG investment strategies in emerging markets and policy discussions.

TARGET_AUDIENCE: Asset managers, ESG researchers, policymakers"""
        
        result = _parse_contribution_response(content)
        
        assert "main_statement" in result
        assert "first comprehensive" in result["main_statement"]
        assert result["contribution_type"] == "empirical"
        assert "emerging market" in result["novelty_explanation"].lower()
        assert isinstance(result["target_audience"], list)
    
    def test_parse_contribution_handles_multiline_statement(self):
        """Test parsing handles multiline main statement."""
        from src.tools.contribution import _parse_contribution_response
        
        content = """MAIN_STATEMENT: This paper makes three contributions.
First, it provides new evidence on ESG in emerging markets.
Second, it uses novel identification strategies.
Third, it offers practical implications.

CONTRIBUTION_TYPE: empirical"""
        
        result = _parse_contribution_response(content)
        
        assert "three contributions" in result["main_statement"]
        assert "First" in result["main_statement"]


class TestPositionInLiterature:
    """Tests for literature positioning."""
    
    def test_parse_positioning_response_basic(self):
        """Test parsing of positioning response."""
        from src.tools.contribution import _parse_positioning_response
        
        content = """LITERATURE_STREAM: ESG and corporate finance

BUILDS_ON: Friede et al. (2015) meta-analysis, which established the positive ESG-performance link.

EXTENDS: Khan et al. (2016) by applying their materiality framework to emerging markets.

COMPLEMENTS: Recent work on ESG in Asia-Pacific markets.

CONTRASTS_WITH: Earlier studies that found no ESG effect, by using better identification.

FILLS_GAP_IN: The empirical literature on emerging market ESG.

RELATIONSHIP_SUMMARY: This paper bridges the gap between developed market ESG research and the growing need for evidence in emerging economies. Building on established frameworks, we extend the analysis to previously understudied contexts."""
        
        result = _parse_positioning_response(content)
        
        assert result["literature_stream"] == "ESG and corporate finance"
        assert "Friede" in result["builds_on"]
        assert "Khan" in result["extends"]
        assert "bridges the gap" in result["relationship_summary"]


class TestRefineResearchQuestion:
    """Tests for research question refinement."""
    
    def test_parse_refinement_response_basic(self):
        """Test parsing of refinement response."""
        from src.tools.contribution import _parse_refinement_response
        
        original = "How does ESG affect firm performance?"
        
        content = """REFINED_QUESTION: What is the causal effect of environmental, social, and governance practices on firm value in emerging market economies, and through what mechanisms does this effect operate?

REFINEMENT_RATIONALE: The original question was too broad. The refined version focuses on the identified gap (emerging markets) and addresses methodological concerns (causality).

SCOPE_CHANGES:
- Narrowed geographic focus to emerging markets
- Added emphasis on causal identification
- Included mechanism exploration

SPECIFICITY_IMPROVEMENT: The refined question specifies the context (emerging markets) and the analytical approach (causal, mechanisms).

GAP_TARGETING: Directly addresses the empirical gap in emerging market ESG research.

FEASIBILITY_ASSESSMENT: Feasible with available emerging market data and instrumental variables approach."""
        
        result = _parse_refinement_response(content, original)
        
        assert result["original_question"] == original
        assert "emerging market" in result["refined_question"].lower()
        assert len(result["scope_changes"]) == 3
        assert "causal" in result["refinement_rationale"].lower()


class TestDifferentiateFromPrior:
    """Tests for differentiation from prior work."""
    
    def test_parse_differentiation_response_basic(self):
        """Test parsing of differentiation response."""
        from src.tools.contribution import _parse_differentiation_response
        
        content = """PAPER_1_DIFF: Unlike Friede et al.'s meta-analysis of developed markets, we focus exclusively on emerging markets with original data collection.

PAPER_2_DIFF: While Khan et al. examined materiality in US firms, we test their framework's applicability in different institutional contexts.

KEY_DIFFERENCES:
- Geographic focus on emerging markets
- Novel identification strategy using regulatory changes
- Panel data vs. cross-sectional approach

METHODOLOGICAL_DIFFERENCES: We employ instrumental variables while prior work relied on correlational designs.

SCOPE_DIFFERENCES: Broader geographic scope but narrower time period.

DATA_DIFFERENCES: We use local ESG ratings rather than global providers.

THEORETICAL_DIFFERENCES: We incorporate institutional theory to explain cross-country variation.

DIFFERENTIATION_SUMMARY: This paper differs from prior work primarily through its geographic focus and methodological rigor. While previous studies established correlations in developed markets, we provide causal evidence from emerging economies using a novel identification strategy based on regulatory changes."""
        
        result = _parse_differentiation_response(content, 2)
        
        assert "paper_1" in result["paper_differences"]
        assert "paper_2" in result["paper_differences"]
        assert "emerging markets" in result["paper_differences"]["paper_1"].lower()
        # Key differences may come as single string or list depending on parsing
        assert len(result["key_differences"]) >= 1
        assert "instrumental variables" in result["methodological_differences"].lower()


# =============================================================================
# GAP_IDENTIFIER Node Tests
# =============================================================================


class TestGapIdentifierNode:
    """Tests for the GAP_IDENTIFIER node."""
    
    def test_create_minimal_synthesis_with_results(self):
        """Test creating minimal synthesis from search results."""
        from src.nodes.gap_identifier import _create_minimal_synthesis
        
        results = create_mock_search_results()
        synthesis = _create_minimal_synthesis(results)
        
        assert "3 search results" in synthesis["summary"]
        assert len(synthesis["key_findings"]) > 0
    
    def test_create_minimal_synthesis_empty(self):
        """Test creating minimal synthesis with no results."""
        from src.nodes.gap_identifier import _create_minimal_synthesis
        
        synthesis = _create_minimal_synthesis([])
        
        assert "No literature synthesis" in synthesis["summary"]
        assert synthesis["key_findings"] == []
    
    def test_select_primary_gap_with_high_significance(self):
        """Test selecting primary gap when high significance gaps exist."""
        from src.nodes.gap_identifier import select_primary_gap
        
        gap_analysis = create_mock_gap_analysis()
        gap_analysis.primary_gap = None  # Reset to test selection
        
        selected = select_primary_gap(gap_analysis)
        
        assert selected is not None
        assert selected.significance == "high"
    
    def test_select_primary_gap_uses_existing(self):
        """Test that existing primary gap is returned."""
        from src.nodes.gap_identifier import select_primary_gap
        
        gap_analysis = create_mock_gap_analysis()
        
        selected = select_primary_gap(gap_analysis)
        
        assert selected == gap_analysis.primary_gap
    
    def test_prepare_approval_request_format(self):
        """Test format of approval request for interrupt."""
        from src.nodes.gap_identifier import prepare_approval_request
        
        gap_analysis = create_mock_gap_analysis()
        
        refined = RefinedResearchQuestion(
            original_question="Original question?",
            refined_question="Refined question about emerging markets?",
            refinement_rationale="Focus on identified gap",
            gap_targeted="Emerging markets ESG studies",
            scope_changes=["Narrowed geographic focus"],
        )
        
        contribution = ContributionStatement(
            main_statement="This paper contributes new evidence...",
            contribution_type="empirical",
            gap_addressed="Emerging markets gap",
            novelty_explanation="First study in this context",
            potential_impact="Informs policy and practice",
            target_audience=["Researchers", "Practitioners"],
        )
        
        request = prepare_approval_request(
            original_question="Original question?",
            refined_question=refined,
            gap_analysis=gap_analysis,
            contribution=contribution,
        )
        
        assert request["action"] == "approve_refined_question"
        assert "approve" in request["allowed_actions"]
        assert "modify" in request["allowed_actions"]
        assert request["original_question"] == "Original question?"
        assert "emerging markets" in request["refined_question"].lower()
        assert request["gap_analysis"]["total_gaps_found"] == 3
    
    def test_process_approval_response_approve(self):
        """Test processing approval response with approve action."""
        from src.nodes.gap_identifier import process_approval_response
        
        refined = RefinedResearchQuestion(
            original_question="Original?",
            refined_question="Refined?",
        )
        contribution = ContributionStatement(
            main_statement="Contribution statement",
        )
        
        response = {"action": "approve"}
        
        question, statement = process_approval_response(
            response, refined, contribution
        )
        
        assert question == "Refined?"
        assert statement == "Contribution statement"
    
    def test_process_approval_response_modify(self):
        """Test processing approval response with modifications."""
        from src.nodes.gap_identifier import process_approval_response
        
        refined = RefinedResearchQuestion(
            original_question="Original?",
            refined_question="Refined?",
        )
        contribution = ContributionStatement(
            main_statement="Original contribution",
        )
        
        response = {
            "action": "modify",
            "refined_question": "User modified question?",
            "contribution": "User modified contribution",
        }
        
        question, statement = process_approval_response(
            response, refined, contribution
        )
        
        assert question == "User modified question?"
        assert statement == "User modified contribution"
    
    def test_process_approval_response_reject(self):
        """Test processing approval response with rejection."""
        from src.nodes.gap_identifier import process_approval_response
        
        refined = RefinedResearchQuestion(
            original_question="Original research question about ESG?",
            refined_question="Refined research question about ESG in emerging markets?",
        )
        contribution = ContributionStatement(
            main_statement="This paper contributes to the literature by providing new evidence.",
        )
        
        response = {"action": "reject"}
        
        question, statement = process_approval_response(
            response, refined, contribution
        )
        
        assert question == "Original research question about ESG?"
        assert statement == ""


class TestRouting:
    """Tests for routing functions."""
    
    def test_should_refine_further_needs_gaps(self):
        """Test that refinement is needed when no gaps exist."""
        from src.nodes.gap_identifier import should_refine_further
        
        state = {"gap_analysis": None}
        
        result = should_refine_further(state)
        
        assert result == "refine"
    
    def test_should_refine_further_needs_primary_gap(self):
        """Test that refinement is needed when no primary gap."""
        from src.nodes.gap_identifier import should_refine_further
        
        state = {
            "gap_analysis": {"primary_gap": None},
            "contribution_statement": None,
        }
        
        result = should_refine_further(state)
        
        assert result == "refine"
    
    def test_should_refine_further_can_proceed(self):
        """Test that we can proceed when gaps and contribution exist."""
        from src.nodes.gap_identifier import should_refine_further
        
        state = {
            "gap_analysis": {"primary_gap": {"title": "Test gap"}},
            "contribution_statement": "Test contribution",
        }
        
        result = should_refine_further(state)
        
        assert result == "proceed"
    
    def test_route_after_gap_identifier_success(self):
        """Test routing to planner on success."""
        from src.nodes.gap_identifier import route_after_gap_identifier
        
        state = {
            "status": ResearchStatus.GAP_IDENTIFICATION_COMPLETE,
            "refined_query": "Refined question?",
            "contribution_statement": "Contribution",
        }
        
        result = route_after_gap_identifier(state)
        
        assert result == "planner"
    
    def test_route_after_gap_identifier_failure(self):
        """Test routing to error on failure."""
        from src.nodes.gap_identifier import route_after_gap_identifier
        
        state = {
            "status": ResearchStatus.FAILED,
            "refined_query": None,
            "contribution_statement": None,
        }
        
        result = route_after_gap_identifier(state)
        
        assert result == "error"


class TestGapIdentifierTools:
    """Tests for gap identifier tool exports."""
    
    def test_get_gap_identifier_tools_returns_list(self):
        """Test that get_gap_identifier_tools returns tools."""
        from src.nodes.gap_identifier import get_gap_identifier_tools
        
        tools = get_gap_identifier_tools()
        
        assert isinstance(tools, list)
        assert len(tools) == 4  # 2 gap tools + 2 contribution tools


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for gap identification workflow."""
    
    @patch("src.nodes.gap_identifier.identify_gaps")
    @patch("src.nodes.gap_identifier.create_refined_question")
    @patch("src.nodes.gap_identifier.create_contribution_statement")
    @patch("src.nodes.gap_identifier.interrupt")
    def test_gap_identifier_node_full_flow(
        self,
        mock_interrupt,
        mock_contribution,
        mock_refined,
        mock_identify,
    ):
        """Test full gap identifier node flow."""
        from src.nodes.gap_identifier import gap_identifier_node
        
        # Setup mocks
        mock_identify.return_value = create_mock_gap_analysis()
        
        mock_refined.return_value = RefinedResearchQuestion(
            original_question="Original?",
            refined_question="Refined question about emerging markets?",
            refinement_rationale="Focus on gap",
        )
        
        mock_contribution.return_value = ContributionStatement(
            main_statement="This paper contributes...",
            contribution_type="empirical",
        )
        
        mock_interrupt.return_value = {"action": "approve"}
        
        # Create test state
        state = {
            "original_query": "How does ESG affect firm value?",
            "literature_synthesis": create_mock_literature_synthesis(),
            "data_exploration_results": None,
            "expected_contribution": None,
            "search_results": [],
        }
        
        # Execute
        result = gap_identifier_node(state)
        
        # Verify
        assert result["status"] == ResearchStatus.GAP_IDENTIFICATION_COMPLETE
        assert result["refined_query"] == "Refined question about emerging markets?"
        assert result["contribution_statement"] == "This paper contributes..."
        assert len(result["identified_gaps"]) > 0
        
        # Verify interrupt was called
        mock_interrupt.assert_called_once()
    
    def test_gap_identifier_node_no_query(self):
        """Test gap identifier node fails without query."""
        from src.nodes.gap_identifier import gap_identifier_node
        
        state = {
            "original_query": None,
            "literature_synthesis": None,
        }
        
        result = gap_identifier_node(state)
        
        assert result["status"] == ResearchStatus.FAILED
        assert len(result["errors"]) > 0
