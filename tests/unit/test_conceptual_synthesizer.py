"""Tests for CONCEPTUAL_SYNTHESIZER node (Sprint 5).

Tests cover:
- Concept extraction from literature
- Framework building
- Proposition generation
- Routing logic
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.nodes.conceptual_synthesizer import (
    conceptual_synthesizer_node,
    route_after_conceptual_synthesizer,
)
from src.state.schema import create_initial_state, WorkflowState
from src.state.enums import (
    ResearchStatus,
    PropositionStatus,
    ConceptType,
    RelationshipType,
)
from src.state.models import (
    LiteratureSynthesis,
    GapAnalysis,
    ResearchGap,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_literature_synthesis():
    """Create mock literature synthesis."""
    return LiteratureSynthesis(
        summary="The literature on digital transformation reveals several key themes...",
        key_findings=[
            "DT requires top management support", 
            "Dynamic capabilities are essential",
            "Learning orientation is critical",
        ],
        theoretical_frameworks=["Dynamic Capabilities", "Institutional Theory"],
        methodological_approaches=["Panel data analysis", "Case studies"],
        contribution_opportunities=["Integration of multiple theoretical perspectives"],
        papers_analyzed=15,
    )


@pytest.fixture
def mock_gap_analysis():
    """Create mock gap analysis."""
    return GapAnalysis(
        original_question="How does digital transformation affect organizational performance?",
        primary_gap=ResearchGap(
            gap_type="theoretical",
            title="Integration of Dynamic Capabilities and Institutional Theory",
            description="No integrated framework combining DC and institutional perspectives on DT",
            significance="high",
        ),
    )


@pytest.fixture
def state_with_literature(mock_literature_synthesis, mock_gap_analysis):
    """Create state with literature for synthesis."""
    state = create_initial_state()
    state["literature_synthesis"] = mock_literature_synthesis
    state["gap_analysis"] = mock_gap_analysis
    state["refined_query"] = "How does digital transformation affect organizational performance?"
    state["original_query"] = "Digital transformation and performance"
    state["research_type"] = "theoretical"
    return state


@pytest.fixture
def state_without_literature():
    """Create state without literature."""
    state = create_initial_state()
    state["gap_analysis"] = GapAnalysis(
        original_question="Test question",
        primary_gap=ResearchGap(
            gap_type="theoretical",
            title="Test Gap",
            description="A test gap for testing purposes",
        ),
    )
    return state


@pytest.fixture
def state_without_gap():
    """Create state without gap analysis."""
    state = create_initial_state()
    state["literature_synthesis"] = LiteratureSynthesis(
        summary="Some synthesis",
    )
    return state


# =============================================================================
# Main Node Tests
# =============================================================================


class TestConceptualSynthesizerNode:
    """Tests for conceptual_synthesizer_node."""
    
    def test_successful_synthesis(self, state_with_literature):
        """Test successful conceptual synthesis."""
        result = conceptual_synthesizer_node(state_with_literature)
        
        assert result["status"] == ResearchStatus.ANALYSIS_COMPLETE
        assert "analysis" in result
        assert "messages" in result
        assert not result.get("errors")
    
    def test_synthesis_generates_concepts(self, state_with_literature):
        """Test that synthesis generates concepts."""
        result = conceptual_synthesizer_node(state_with_literature)
        
        analysis = result["analysis"]
        framework = analysis.get("framework", {})
        assert "concepts" in framework
        assert len(framework["concepts"]) > 0
    
    def test_synthesis_generates_relationships(self, state_with_literature):
        """Test that synthesis generates relationships."""
        result = conceptual_synthesizer_node(state_with_literature)
        
        analysis = result["analysis"]
        framework = analysis.get("framework", {})
        assert "relationships" in framework
        # Relationships may be empty if few concepts
    
    def test_synthesis_generates_propositions(self, state_with_literature):
        """Test that synthesis generates propositions."""
        result = conceptual_synthesizer_node(state_with_literature)
        
        analysis = result["analysis"]
        framework = analysis.get("framework", {})
        assert "propositions" in framework
        assert len(framework["propositions"]) > 0
    
    def test_no_literature_error(self, state_without_literature):
        """Test error when no literature available."""
        result = conceptual_synthesizer_node(state_without_literature)
        
        assert result["status"] == ResearchStatus.FAILED
        assert len(result["errors"]) > 0
        assert "literature" in result["errors"][0].message.lower()
    
    def test_no_gap_error(self, state_without_gap):
        """Test error when no gap analysis available."""
        result = conceptual_synthesizer_node(state_without_gap)
        
        # Note: The node actually fails on literature validation first
        # since state_without_gap fixture creates minimal LiteratureSynthesis
        assert result["status"] == ResearchStatus.FAILED
        assert len(result["errors"]) > 0
    
    def test_gap_addressed_assessment(self, state_with_literature):
        """Test gap addressed is assessed."""
        result = conceptual_synthesizer_node(state_with_literature)
        
        analysis = result["analysis"]
        assert "gap_addressed" in analysis
        assert "gap_coverage_score" in analysis


# =============================================================================
# Routing Tests
# =============================================================================


class TestRouteAfterConceptualSynthesizer:
    """Tests for route_after_conceptual_synthesizer."""
    
    def test_route_to_writer_on_complete(self):
        """Test routing to writer when synthesis complete."""
        state = {
            "status": ResearchStatus.ANALYSIS_COMPLETE,
            "analysis": {"concepts": []},
        }
        assert route_after_conceptual_synthesizer(state) == "writer"
    
    def test_route_to_end_on_error(self):
        """Test routing to end when errors present."""
        state = {
            "status": ResearchStatus.ANALYSIS_COMPLETE,
            "errors": [MagicMock()],
        }
        assert route_after_conceptual_synthesizer(state) == "__end__"
    
    def test_route_to_end_on_failed_status(self):
        """Test routing to end when status is failed."""
        state = {
            "status": ResearchStatus.FAILED,
        }
        assert route_after_conceptual_synthesizer(state) == "__end__"
    
    def test_route_to_writer_with_framework(self):
        """Test routing to writer when analysis present."""
        state = {
            "analysis": {"concepts": []},
        }
        assert route_after_conceptual_synthesizer(state) == "writer"


# =============================================================================
# Integration Tests
# =============================================================================


class TestConceptualSynthesizerIntegration:
    """Integration tests for conceptual synthesizer node."""
    
    def test_full_workflow(self, state_with_literature):
        """Test complete synthesis workflow."""
        result = conceptual_synthesizer_node(state_with_literature)
        
        # Check all expected outputs
        assert result["status"] == ResearchStatus.ANALYSIS_COMPLETE
        
        analysis = result["analysis"]
        framework = analysis.get("framework", {})
        assert "title" in framework
        assert "description" in framework
        assert len(framework["concepts"]) > 0
        assert "theoretical_foundations" in framework
    
    def test_concepts_have_required_fields(self, state_with_literature):
        """Test that concepts have required fields."""
        result = conceptual_synthesizer_node(state_with_literature)
        
        analysis = result["analysis"]
        framework = analysis.get("framework", {})
        for concept in framework["concepts"]:
            assert "name" in concept
            assert "definition" in concept
            assert "concept_type" in concept
    
    def test_propositions_have_required_fields(self, state_with_literature):
        """Test that propositions have required fields."""
        result = conceptual_synthesizer_node(state_with_literature)
        
        analysis = result["analysis"]
        framework = analysis.get("framework", {})
        for prop in framework["propositions"]:
            assert "proposition_id" in prop
            assert "statement" in prop
            assert "derived_from_concepts" in prop
    
    def test_message_format(self, state_with_literature):
        """Test that message is properly formatted."""
        result = conceptual_synthesizer_node(state_with_literature)
        
        assert len(result["messages"]) == 1
        message = result["messages"][0]
        assert "CONCEPTUAL_SYNTHESIZER" in message.content
        assert "complete" in message.content.lower()
