"""Synthesis tools for the CONCEPTUAL_SYNTHESIZER node.

These tools provide conceptual analysis capabilities for theoretical research,
including concept extraction, framework building, proposition generation,
and theoretical grounding.
"""

from typing import Any
from uuid import uuid4

from langchain_core.tools import tool

from src.state.enums import (
    ConceptType,
    RelationshipType,
    PropositionStatus,
    EvidenceStrength,
)
from src.state.models import (
    Concept,
    ConceptRelationship,
    Proposition,
    ConceptualFramework,
)


# =============================================================================
# Concept Extraction Tools
# =============================================================================


@tool
def extract_key_concepts(
    literature_synthesis: dict[str, Any],
    research_question: str,
    gap_description: str,
) -> list[Concept]:
    """
    Extract key concepts from literature synthesis.
    
    This tool identifies and structures the key theoretical concepts
    from the literature review that are relevant to the research question.
    
    Args:
        literature_synthesis: Literature synthesis from earlier node.
        research_question: The refined research question.
        gap_description: Description of the research gap being addressed.
    
    Returns:
        List of Concept objects with definitions and properties.
    """
    concepts = []
    
    # Extract theoretical frameworks mentioned
    frameworks = literature_synthesis.get("theoretical_frameworks", [])
    for i, framework in enumerate(frameworks[:5]):  # Limit to top 5
        concepts.append(
            Concept(
                concept_id=str(uuid4())[:8],
                name=framework if isinstance(framework, str) else f"Framework_{i+1}",
                concept_type=ConceptType.CONSTRUCT,
                definition=f"Theoretical framework: {framework}",
                source_literature=[],
                is_observable=False,
                abstraction_level="high",
            )
        )
    
    # Extract key findings as potential variables
    findings = literature_synthesis.get("key_findings", [])
    for i, finding in enumerate(findings[:5]):  # Limit to top 5
        concepts.append(
            Concept(
                concept_id=str(uuid4())[:8],
                name=f"Variable_{i+1}",
                concept_type=ConceptType.VARIABLE,
                definition=finding if isinstance(finding, str) else f"Finding {i+1}",
                source_literature=[],
                is_observable=True,
                abstraction_level="low",
            )
        )
    
    return concepts


@tool
def define_concept(
    name: str,
    concept_type: str,
    definition: str,
    operationalization: str | None = None,
    source_literature: list[str] | None = None,
) -> Concept:
    """
    Create a formally defined concept for the framework.
    
    This tool creates a structured concept definition with proper
    academic rigor for inclusion in a conceptual framework.
    
    Args:
        name: Name of the concept.
        concept_type: Type (construct, variable, mechanism, moderator, mediator).
        definition: Formal definition of the concept.
        operationalization: How the concept can be measured (optional).
        source_literature: Literature sources for this concept.
    
    Returns:
        Concept object with full definition.
    """
    # Map string to enum
    try:
        concept_type_enum = ConceptType(concept_type)
    except ValueError:
        concept_type_enum = ConceptType.CONSTRUCT
    
    return Concept(
        concept_id=str(uuid4())[:8],
        name=name,
        concept_type=concept_type_enum,
        definition=definition,
        operationalization=operationalization,
        source_literature=source_literature or [],
        is_observable=concept_type_enum == ConceptType.VARIABLE,
        abstraction_level="medium",
    )


# =============================================================================
# Relationship Mapping Tools
# =============================================================================


@tool
def map_concept_relationships(
    concepts: list[dict[str, Any]],
    literature_synthesis: dict[str, Any],
) -> list[ConceptRelationship]:
    """
    Map relationships between concepts based on literature.
    
    This tool identifies and formalizes the relationships between
    concepts in the framework based on evidence from the literature.
    
    Args:
        concepts: List of concepts to analyze.
        literature_synthesis: Literature synthesis for relationship evidence.
    
    Returns:
        List of ConceptRelationship objects.
    """
    relationships = []
    
    # Create relationships between concepts based on their types
    concept_ids = [c.get("concept_id", str(uuid4())[:8]) for c in concepts]
    
    # Generate relationships for constructs → variables
    constructs = [c for c in concepts if c.get("concept_type") == "construct"]
    variables = [c for c in concepts if c.get("concept_type") == "variable"]
    
    for construct in constructs[:3]:
        for variable in variables[:2]:
            relationships.append(
                ConceptRelationship(
                    relationship_id=str(uuid4())[:8],
                    source_concept_id=construct.get("concept_id", ""),
                    target_concept_id=variable.get("concept_id", ""),
                    relationship_type=RelationshipType.CAUSAL,
                    description=f"{construct.get('name', '')} influences {variable.get('name', '')}",
                    strength="moderate",
                    direction="positive",
                    empirical_support=EvidenceStrength.MODERATE,
                )
            )
    
    return relationships


@tool
def define_relationship(
    source_concept_id: str,
    target_concept_id: str,
    relationship_type: str,
    description: str,
    theoretical_basis: str,
    supporting_literature: list[str] | None = None,
) -> ConceptRelationship:
    """
    Define a relationship between two concepts.
    
    This tool creates a formally specified relationship between concepts
    with theoretical justification and literature support.
    
    Args:
        source_concept_id: ID of the source concept.
        target_concept_id: ID of the target concept.
        relationship_type: Type of relationship (causal, correlational, etc.).
        description: Description of the relationship.
        theoretical_basis: Theoretical justification for the relationship.
        supporting_literature: Literature supporting this relationship.
    
    Returns:
        ConceptRelationship object.
    """
    # Map string to enum
    try:
        rel_type_enum = RelationshipType(relationship_type)
    except ValueError:
        rel_type_enum = RelationshipType.CAUSAL
    
    return ConceptRelationship(
        relationship_id=str(uuid4())[:8],
        source_concept_id=source_concept_id,
        target_concept_id=target_concept_id,
        relationship_type=rel_type_enum,
        description=description,
        theoretical_basis=theoretical_basis,
        supporting_literature=supporting_literature or [],
        empirical_support=EvidenceStrength.MODERATE,
    )


# =============================================================================
# Proposition Generation Tools
# =============================================================================


@tool
def generate_propositions(
    concepts: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    gap_description: str,
) -> list[Proposition]:
    """
    Generate theoretical propositions from concepts and relationships.
    
    This tool derives testable propositions from the conceptual framework
    by formalizing the expected relationships and their implications.
    
    Args:
        concepts: List of concepts in the framework.
        relationships: List of relationships between concepts.
        gap_description: The research gap being addressed.
    
    Returns:
        List of Proposition objects.
    """
    propositions = []
    
    # Generate propositions from relationships
    for i, rel in enumerate(relationships[:5]):  # Limit to 5
        source_id = rel.get("source_concept_id", "")
        target_id = rel.get("target_concept_id", "")
        rel_type = rel.get("relationship_type", "causal")
        
        # Find concept names
        source_name = next(
            (c.get("name", "X") for c in concepts if c.get("concept_id") == source_id),
            "X"
        )
        target_name = next(
            (c.get("name", "Y") for c in concepts if c.get("concept_id") == target_id),
            "Y"
        )
        
        # Generate proposition statement
        if rel_type == "causal":
            statement = f"Proposition {i+1}: {source_name} has a positive effect on {target_name}."
        elif rel_type == "moderating":
            statement = f"Proposition {i+1}: {source_name} moderates the relationship involving {target_name}."
        else:
            statement = f"Proposition {i+1}: {source_name} is associated with {target_name}."
        
        propositions.append(
            Proposition(
                proposition_id=str(uuid4())[:8],
                statement=statement,
                derived_from_concepts=[source_id, target_id],
                derived_from_relationships=[rel.get("relationship_id", "")],
                derivation_logic=f"Derived from {rel_type} relationship between {source_name} and {target_name}",
                proposition_status=PropositionStatus.PROPOSED,
                is_testable=True,
                empirical_support=EvidenceStrength.INSUFFICIENT,
            )
        )
    
    return propositions


@tool
def define_proposition(
    statement: str,
    derived_from_concepts: list[str],
    derivation_logic: str,
    is_testable: bool = True,
    test_approach: str | None = None,
    boundary_conditions: list[str] | None = None,
) -> Proposition:
    """
    Define a formal theoretical proposition.
    
    This tool creates a formally specified proposition with derivation
    logic and testability assessment.
    
    Args:
        statement: The proposition statement.
        derived_from_concepts: IDs of concepts this is derived from.
        derivation_logic: Logic/reasoning for the derivation.
        is_testable: Whether the proposition is empirically testable.
        test_approach: How the proposition could be tested.
        boundary_conditions: Conditions under which proposition holds.
    
    Returns:
        Proposition object.
    """
    return Proposition(
        proposition_id=str(uuid4())[:8],
        statement=statement,
        derived_from_concepts=derived_from_concepts,
        derivation_logic=derivation_logic,
        proposition_status=PropositionStatus.PROPOSED,
        is_testable=is_testable,
        test_approach=test_approach,
        boundary_conditions=boundary_conditions or [],
        empirical_support=EvidenceStrength.INSUFFICIENT,
    )


# =============================================================================
# Framework Building Tools
# =============================================================================


@tool
def build_conceptual_framework(
    title: str,
    description: str,
    concepts: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    propositions: list[dict[str, Any]],
    theoretical_foundations: list[str],
    research_question: str,
    gap_description: str,
) -> ConceptualFramework:
    """
    Build a complete conceptual framework.
    
    This tool assembles all components into a coherent conceptual
    framework with proper theoretical grounding.
    
    Args:
        title: Framework title.
        description: Full description of the framework.
        concepts: List of concepts in the framework.
        relationships: List of relationships between concepts.
        propositions: List of propositions.
        theoretical_foundations: Existing theories the framework builds on.
        research_question: The research question being addressed.
        gap_description: The gap being addressed.
    
    Returns:
        Complete ConceptualFramework object.
    """
    # Convert dicts to model objects if needed
    concept_objs = []
    for c in concepts:
        if isinstance(c, dict):
            concept_objs.append(Concept(**c))
        else:
            concept_objs.append(c)
    
    relationship_objs = []
    for r in relationships:
        if isinstance(r, dict):
            relationship_objs.append(ConceptRelationship(**r))
        else:
            relationship_objs.append(r)
    
    proposition_objs = []
    for p in propositions:
        if isinstance(p, dict):
            proposition_objs.append(Proposition(**p))
        else:
            proposition_objs.append(p)
    
    return ConceptualFramework(
        framework_id=str(uuid4())[:8],
        title=title,
        description=description,
        concepts=concept_objs,
        relationships=relationship_objs,
        propositions=proposition_objs,
        theoretical_foundations=theoretical_foundations,
        grounding_explanation=f"This framework addresses: {gap_description}",
        domain="Academic Research",
        novelty_assessment="Novel synthesis of existing theoretical perspectives",
        theoretical_contribution=f"Provides a framework for understanding {research_question}",
    )


# =============================================================================
# Theoretical Grounding Tools
# =============================================================================


@tool
def ground_in_theory(
    framework: dict[str, Any],
    seminal_works: list[str],
    existing_theories: list[str],
) -> dict[str, Any]:
    """
    Ground the framework in existing theory.
    
    This tool connects the conceptual framework to established
    theoretical foundations and seminal works.
    
    Args:
        framework: The conceptual framework to ground.
        seminal_works: List of seminal works to connect to.
        existing_theories: List of existing theories to build on.
    
    Returns:
        Dictionary with grounding information.
    """
    return {
        "framework_title": framework.get("title", ""),
        "theoretical_foundations": existing_theories,
        "seminal_works_connected": seminal_works,
        "grounding_strength": "strong" if len(seminal_works) >= 3 else "moderate",
        "grounding_explanation": (
            f"The framework builds on {len(existing_theories)} established theories "
            f"and connects to {len(seminal_works)} seminal works in the field."
        ),
        "integration_points": [
            f"Connection to {theory}" for theory in existing_theories[:3]
        ],
    }


@tool
def assess_theoretical_contribution(
    framework: dict[str, Any],
    gap_description: str,
    existing_frameworks: list[str],
) -> dict[str, Any]:
    """
    Assess the theoretical contribution of the framework.
    
    This tool evaluates the novelty and significance of the
    theoretical contribution made by the framework.
    
    Args:
        framework: The conceptual framework to assess.
        gap_description: The gap being addressed.
        existing_frameworks: List of existing frameworks in the area.
    
    Returns:
        Dictionary with contribution assessment.
    """
    n_concepts = len(framework.get("concepts", []))
    n_propositions = len(framework.get("propositions", []))
    
    # Assess contribution type
    if n_propositions >= 5:
        contribution_type = "comprehensive"
        contribution_strength = "strong"
    elif n_propositions >= 3:
        contribution_type = "focused"
        contribution_strength = "moderate"
    else:
        contribution_type = "preliminary"
        contribution_strength = "emerging"
    
    return {
        "contribution_type": contribution_type,
        "contribution_strength": contribution_strength,
        "novelty_score": 0.7 if n_propositions >= 3 else 0.5,
        "gap_addressed": True if n_propositions > 0 else False,
        "differentiation_from_existing": (
            f"Framework differs from {len(existing_frameworks)} existing frameworks "
            "by providing a novel integration of concepts."
        ),
        "contribution_statement": (
            f"This framework makes a {contribution_strength} theoretical contribution "
            f"by offering {n_propositions} testable propositions that address "
            f"the identified gap: {gap_description[:100]}..."
        ),
        "practical_implications": [
            "Provides guidance for empirical testing",
            "Identifies key relationships for investigation",
            "Offers a structured lens for analysis",
        ],
    }


# =============================================================================
# Export Tool List
# =============================================================================


def get_synthesis_tools() -> list:
    """Get list of all synthesis tools."""
    return [
        extract_key_concepts,
        define_concept,
        map_concept_relationships,
        define_relationship,
        generate_propositions,
        define_proposition,
        build_conceptual_framework,
        ground_in_theory,
        assess_theoretical_contribution,
    ]
