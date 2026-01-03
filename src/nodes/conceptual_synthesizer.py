"""CONCEPTUAL_SYNTHESIZER node for theoretical research.

This node:
1. Takes input from PLANNER (for theoretical/conceptual research)
2. Extracts key concepts from literature synthesis
3. Builds a conceptual framework with concepts and relationships
4. Generates testable propositions
5. Grounds the framework in existing theory
6. Assesses theoretical contribution
"""

from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage

from src.state.enums import (
    ResearchStatus,
    AnalysisStatus,
    ConceptType,
    RelationshipType,
    PropositionStatus,
    EvidenceStrength,
)
from src.state.models import (
    ConceptualSynthesisResult,
    ConceptualFramework,
    Concept,
    ConceptRelationship,
    Proposition,
    WorkflowError,
)
from src.state.schema import WorkflowState
from src.tools.synthesis import assess_theoretical_contribution


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_literature_synthesis(state: WorkflowState) -> dict[str, Any]:
    """Extract literature synthesis from state."""
    synthesis = state.get("literature_synthesis")
    
    if not synthesis:
        return {
            "summary": "",
            "key_findings": [],
            "theoretical_frameworks": [],
            "methodological_approaches": [],
            "papers_analyzed": 0,
        }
    
    if isinstance(synthesis, dict):
        return {
            "summary": synthesis.get("summary", ""),
            "key_findings": synthesis.get("key_findings", []),
            "theoretical_frameworks": synthesis.get("theoretical_frameworks", []),
            "methodological_approaches": synthesis.get("methodological_approaches", []),
            "contribution_opportunities": synthesis.get("contribution_opportunities", []),
            "papers_analyzed": synthesis.get("papers_analyzed", 0),
        }
    elif hasattr(synthesis, "summary"):
        return {
            "summary": synthesis.summary,
            "key_findings": synthesis.key_findings,
            "theoretical_frameworks": synthesis.theoretical_frameworks,
            "methodological_approaches": synthesis.methodological_approaches,
            "contribution_opportunities": getattr(synthesis, "contribution_opportunities", []),
            "papers_analyzed": synthesis.papers_analyzed,
        }
    
    return {"summary": "", "key_findings": [], "theoretical_frameworks": []}


def _extract_gap_info(state: WorkflowState) -> dict[str, Any]:
    """Extract gap analysis information from state."""
    gap_analysis = state.get("gap_analysis")
    
    if not gap_analysis:
        return {
            "primary_gap": None,
            "gap_description": "",
            "gap_type": "",
        }
    
    if isinstance(gap_analysis, dict):
        primary_gap = gap_analysis.get("primary_gap", {})
        return {
            "primary_gap": primary_gap,
            "gap_description": primary_gap.get("description", "") if isinstance(primary_gap, dict) else "",
            "gap_type": primary_gap.get("gap_type", "") if isinstance(primary_gap, dict) else "",
        }
    elif hasattr(gap_analysis, "primary_gap"):
        primary = gap_analysis.primary_gap
        return {
            "primary_gap": primary.model_dump() if hasattr(primary, "model_dump") else primary,
            "gap_description": primary.description if hasattr(primary, "description") else "",
            "gap_type": primary.gap_type if hasattr(primary, "gap_type") else "",
        }
    
    return {"primary_gap": None, "gap_description": "", "gap_type": ""}


def _get_research_question(state: WorkflowState) -> str:
    """Get the research question from state."""
    if state.get("refined_query"):
        return state["refined_query"]
    
    refined_rq = state.get("refined_research_question")
    if refined_rq:
        if isinstance(refined_rq, dict):
            return refined_rq.get("refined_question", state.get("original_query", ""))
        elif hasattr(refined_rq, "refined_question"):
            return refined_rq.refined_question
    
    return state.get("original_query", "")


def _extract_concepts_from_synthesis(
    synthesis: dict[str, Any],
    research_question: str,
    gap_description: str,
) -> list[Concept]:
    """Extract and define concepts from literature synthesis."""
    concepts = []
    
    def _clean_concept_name(raw_name: str, fallback: str) -> str:
        """Clean and truncate concept name to fit within 100 char limit."""
        # Remove markdown formatting
        name = raw_name.strip()
        name = name.lstrip("-*# ").rstrip("*")
        # Remove bold/italic markers
        name = name.replace("**", "").replace("__", "")
        # Take first sentence or phrase if too long
        if len(name) > 95:
            # Try to cut at a reasonable point
            for sep in [":", ".", ",", " - "]:
                if sep in name[:95]:
                    idx = name[:95].index(sep)
                    name = name[:idx]
                    break
            else:
                name = name[:95]
        # Fallback if too short or contains no alphanumeric characters
        cleaned = name.strip()
        if len(cleaned) < 2 or not any(ch.isalnum() for ch in cleaned):
            return fallback
        return cleaned
    
    # Extract from theoretical frameworks
    frameworks = synthesis.get("theoretical_frameworks", [])
    for i, fw in enumerate(frameworks[:3]):  # Top 3 frameworks
        raw_name = fw if isinstance(fw, str) else f"Framework {i+1}"
        name = _clean_concept_name(raw_name, f"Framework_{i+1}")
        concepts.append(
            Concept(
                name=name,
                concept_type=ConceptType.CONSTRUCT,
                definition=f"Theoretical framework: {name}. A key theoretical lens identified in the literature.",
                source_literature=[],
                is_observable=False,
                abstraction_level="high",
            )
        )
    
    # Extract from key findings
    findings = synthesis.get("key_findings", [])
    for i, finding in enumerate(findings[:4]):  # Top 4 findings
        finding_text = finding if isinstance(finding, str) else f"Finding {i+1}"
        concepts.append(
            Concept(
                name=f"Concept_{i+1}",
                concept_type=ConceptType.VARIABLE,
                definition=f"Key concept derived from finding: {finding_text[:100]}...",
                source_literature=[],
                is_observable=True,
                abstraction_level="medium",
            )
        )
    
    # Add mechanism concept from gap
    if gap_description:
        concepts.append(
            Concept(
                name="Core_Mechanism",
                concept_type=ConceptType.MECHANISM,
                definition=f"Proposed mechanism to address gap: {gap_description[:100]}...",
                source_literature=[],
                is_observable=False,
                abstraction_level="high",
            )
        )
    
    return concepts


def _build_relationships(concepts: list[Concept]) -> list[ConceptRelationship]:
    """Build relationships between extracted concepts."""
    relationships = []
    
    # Separate concepts by type
    constructs = [c for c in concepts if c.concept_type == ConceptType.CONSTRUCT]
    variables = [c for c in concepts if c.concept_type == ConceptType.VARIABLE]
    mechanisms = [c for c in concepts if c.concept_type == ConceptType.MECHANISM]
    
    # Create construct → variable relationships
    for construct in constructs[:2]:
        for variable in variables[:2]:
            relationships.append(
                ConceptRelationship(
                    source_concept_id=construct.concept_id,
                    target_concept_id=variable.concept_id,
                    relationship_type=RelationshipType.CAUSAL,
                    description=f"{construct.name} influences {variable.name}",
                    strength="moderate",
                    direction="positive",
                    theoretical_basis="Based on literature synthesis",
                    empirical_support=EvidenceStrength.MODERATE,
                )
            )
    
    # Create mechanism relationships
    for mechanism in mechanisms[:1]:
        for construct in constructs[:1]:
            relationships.append(
                ConceptRelationship(
                    source_concept_id=construct.concept_id,
                    target_concept_id=mechanism.concept_id,
                    relationship_type=RelationshipType.MEDIATING,
                    description=f"{mechanism.name} mediates effects of {construct.name}",
                    strength="moderate",
                    direction="positive",
                    theoretical_basis="Proposed mechanism",
                    empirical_support=EvidenceStrength.INSUFFICIENT,
                )
            )
    
    return relationships


def _generate_framework_propositions(
    concepts: list[Concept],
    relationships: list[ConceptRelationship],
    gap_description: str,
) -> list[Proposition]:
    """Generate theoretical propositions from the framework."""
    propositions = []
    
    # Get concept name lookup
    concept_names = {c.concept_id: c.name for c in concepts}
    
    # Generate proposition from each relationship
    for i, rel in enumerate(relationships[:5]):  # Max 5 propositions
        source_name = concept_names.get(rel.source_concept_id, "X")
        target_name = concept_names.get(rel.target_concept_id, "Y")
        
        if rel.relationship_type == RelationshipType.CAUSAL:
            statement = f"P{i+1}: {source_name} has a {rel.direction} effect on {target_name}."
        elif rel.relationship_type == RelationshipType.MEDIATING:
            statement = f"P{i+1}: {target_name} mediates the relationship between {source_name} and downstream outcomes."
        elif rel.relationship_type == RelationshipType.MODERATING:
            statement = f"P{i+1}: {source_name} moderates relationships involving {target_name}."
        else:
            statement = f"P{i+1}: There is a relationship between {source_name} and {target_name}."
        
        propositions.append(
            Proposition(
                statement=statement,
                derived_from_concepts=[rel.source_concept_id, rel.target_concept_id],
                derived_from_relationships=[rel.relationship_id],
                derivation_logic=f"Derived from {rel.relationship_type.value} relationship based on literature synthesis",
                proposition_status=PropositionStatus.PROPOSED,
                is_testable=True,
                test_approach="Empirical testing through regression analysis or experimental design",
                empirical_support=EvidenceStrength.INSUFFICIENT,
                boundary_conditions=["Assumes standard conditions as described in literature"],
            )
        )
    
    return propositions


def _assess_gap_coverage(
    framework: ConceptualFramework,
    gap_info: dict[str, Any],
    research_question: str,
) -> tuple[bool, float, str]:
    """Assess how well the framework addresses the gap."""
    # Assess based on framework completeness
    n_concepts = len(framework.concepts)
    n_propositions = len(framework.propositions)
    n_testable = len([p for p in framework.propositions if p.is_testable])
    
    # Calculate coverage score
    if n_propositions == 0:
        coverage_score = 0.0
    else:
        completeness = min(1.0, n_concepts / 5)  # 5 concepts is complete
        testability = n_testable / max(n_propositions, 1)
        coverage_score = (completeness + testability) / 2
    
    gap_addressed = coverage_score >= 0.5 and n_propositions >= 2
    
    if gap_addressed:
        explanation = (
            f"The conceptual framework addresses the gap with {n_concepts} concepts "
            f"and {n_propositions} propositions ({n_testable} testable). "
            f"Coverage score: {coverage_score:.2f}."
        )
    else:
        explanation = (
            f"The framework partially addresses the gap. "
            f"Only {n_propositions} propositions generated. "
            f"Coverage score: {coverage_score:.2f}. "
            "Additional theoretical development may be needed."
        )
    
    return gap_addressed, coverage_score, explanation


# =============================================================================
# Main Node Function
# =============================================================================


def conceptual_synthesizer_node(state: WorkflowState) -> dict:
    """
    Build theoretical framework from literature.
    
    This node:
    1. Extracts key concepts from literature synthesis
    2. Maps relationships between concepts
    3. Generates testable propositions
    4. Builds a coherent conceptual framework
    5. Grounds the framework in existing theory
    6. Assesses theoretical contribution
    
    Args:
        state: Current workflow state.
        
    Returns:
        Updated state with conceptual framework.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Extract information from state
    synthesis = _extract_literature_synthesis(state)
    gap_info = _extract_gap_info(state)
    research_question = _get_research_question(state)
    
    # Validate we have literature synthesis
    if not synthesis.get("key_findings") and not synthesis.get("theoretical_frameworks"):
        error = WorkflowError(
            node="conceptual_synthesizer",
            category="validation",
            message="No literature synthesis available. CONCEPTUAL_SYNTHESIZER requires literature review output.",
            recoverable=False,
        )
        return {
            "status": ResearchStatus.FAILED,
            "errors": [error],
            "messages": [
                AIMessage(content=f"[{current_date}] CONCEPTUAL_SYNTHESIZER: Error - No literature synthesis available.")
            ],
        }
    
    gap_description = gap_info.get("gap_description", "Research gap not specified")
    
    # Step 1: Extract concepts from literature
    concepts = _extract_concepts_from_synthesis(synthesis, research_question, gap_description)
    
    # Step 2: Build relationships between concepts
    relationships = _build_relationships(concepts)
    
    # Step 3: Generate propositions
    propositions = _generate_framework_propositions(concepts, relationships, gap_description)
    
    # Step 4: Build the conceptual framework
    framework = ConceptualFramework(
        title=f"Conceptual Framework: {research_question[:50]}...",
        abstract=(
            f"This framework synthesizes insights from {synthesis.get('papers_analyzed', 0)} papers "
            f"to address the research gap: {gap_description[:100]}..."
        ),
        description=(
            f"A theoretical framework developed to address: {research_question}. "
            f"The framework integrates {len(concepts)} key concepts connected through "
            f"{len(relationships)} relationships, generating {len(propositions)} testable propositions."
        ),
        concepts=concepts,
        relationships=relationships,
        propositions=propositions,
        theoretical_foundations=synthesis.get("theoretical_frameworks", [])[:5],
        seminal_works=[],  # Would be populated from literature synthesis
        grounding_explanation=(
            f"This framework builds on {len(synthesis.get('theoretical_frameworks', []))} "
            f"established theoretical frameworks identified in the literature."
        ),
        domain="Academic Research",
        scope_conditions=[
            "Framework applies within the scope of the literature reviewed",
            "Boundary conditions may vary by empirical context",
        ],
        novelty_assessment=(
            "Novel synthesis of existing theoretical perspectives, "
            "providing an integrated framework for understanding the research question."
        ),
        theoretical_contribution=(
            f"Provides a structured framework with {len(propositions)} testable propositions "
            f"that address the identified gap in the literature."
        ),
        practical_implications=[
            "Provides guidance for empirical testing",
            "Identifies key relationships for investigation",
            "Offers a structured lens for analysis",
        ],
        limitations=[
            "Framework based on available literature",
            "Propositions require empirical validation",
            "Boundary conditions may limit generalizability",
        ],
        future_directions=[
            "Empirical testing of propositions",
            "Extension to related domains",
            "Refinement based on empirical findings",
        ],
    )
    
    # Step 5: Assess gap coverage
    gap_addressed, coverage_score, coverage_explanation = _assess_gap_coverage(
        framework, gap_info, research_question
    )
    
    # Step 6: Assess contribution
    contribution_assessment = assess_theoretical_contribution.invoke({
        "framework": framework.model_dump(),
        "gap_description": gap_description,
        "existing_frameworks": synthesis.get("theoretical_frameworks", []),
    })
    
    # Build synthesis result
    synthesis_result = ConceptualSynthesisResult(
        analysis_status=AnalysisStatus.COMPLETE,
        framework=framework,
        synthesis_approach="Systematic literature synthesis with concept extraction and relationship mapping",
        literature_base=f"Based on {synthesis.get('papers_analyzed', 0)} papers",
        papers_synthesized=synthesis.get("papers_analyzed", 0),
        key_concepts_identified=[c.name for c in concepts],
        theoretical_mechanisms=[c.name for c in concepts if c.concept_type == ConceptType.MECHANISM],
        gap_addressed=gap_addressed,
        gap_coverage_score=coverage_score,
        gap_coverage_explanation=coverage_explanation,
        contribution_type=contribution_assessment.get("contribution_type", "theoretical"),
        contribution_statement=contribution_assessment.get("contribution_statement", ""),
        overall_confidence=coverage_score,
        coherence_score=0.7 if len(relationships) >= 2 else 0.5,
        grounding_score=0.8 if len(synthesis.get("theoretical_frameworks", [])) >= 2 else 0.5,
        limitations=framework.limitations,
    )
    
    # Build summary message
    summary_parts = [
        f"[{current_date}] CONCEPTUAL_SYNTHESIZER: Framework complete.",
        f"Concepts: {len(concepts)}",
        f"Relationships: {len(relationships)}",
        f"Propositions: {len(propositions)} ({len([p for p in propositions if p.is_testable])} testable)",
        f"Gap addressed: {'Yes' if gap_addressed else 'Partially'} (score: {coverage_score:.2f})",
    ]
    
    return {
        "status": ResearchStatus.ANALYSIS_COMPLETE,
        "analysis": synthesis_result.model_dump(),
        "messages": [
            AIMessage(content=" | ".join(summary_parts))
        ],
    }


# =============================================================================
# Routing Function
# =============================================================================


def route_after_conceptual_synthesizer(state: WorkflowState) -> Literal["writer", "__end__"]:
    """
    Route after CONCEPTUAL_SYNTHESIZER to WRITER node.
    
    Args:
        state: Current workflow state.
        
    Returns:
        Next node: "writer" if synthesis complete, "__end__" if failed.
    """
    if state.get("errors"):
        return "__end__"
    
    if state.get("status") == ResearchStatus.ANALYSIS_COMPLETE:
        return "writer"
    
    if state.get("analysis"):
        return "writer"
    
    return "__end__"
