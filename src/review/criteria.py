"""Evaluation criteria for paper review.

This module defines the evaluation functions for each quality dimension
used by the REVIEWER node to assess paper quality.

Dimensions:
- Contribution: Novelty, significance, clarity of contribution
- Methodology: Rigor, appropriateness, reproducibility
- Evidence: Data quality, analysis validity, interpretation
- Coherence: Logical flow, argument structure, consistency
- Writing: Academic tone, clarity, conciseness, style compliance
"""

import logging
import re
from typing import Any

from src.state.models import (
    QualityScore,
    ReviewCritiqueItem,
    WriterOutput,
)
from src.style.banned_words import BANNED_WORDS

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Dimensions Configuration
# =============================================================================


EVALUATION_DIMENSIONS = {
    "contribution": {
        "name": "Contribution",
        "weight": 0.25,
        "description": "Novelty, significance, and clarity of the paper's contribution",
        "criteria": [
            "Clear identification of research gap",
            "Novel contribution to the field",
            "Significance of findings",
            "Practical and theoretical implications",
        ],
    },
    "methodology": {
        "name": "Methodology",
        "weight": 0.25,
        "description": "Rigor, appropriateness, and reproducibility of methods",
        "criteria": [
            "Appropriate research design",
            "Valid data collection methods",
            "Sound analytical approach",
            "Reproducibility of methods",
            "Acknowledgment of limitations",
        ],
    },
    "evidence": {
        "name": "Evidence",
        "weight": 0.20,
        "description": "Data quality, analysis validity, and interpretation",
        "criteria": [
            "Data quality and representativeness",
            "Statistical validity",
            "Correct interpretation of results",
            "Support for claims",
            "Robustness of findings",
        ],
    },
    "coherence": {
        "name": "Coherence",
        "weight": 0.15,
        "description": "Logical flow, argument structure, and consistency",
        "criteria": [
            "Logical flow between sections",
            "Consistent argument throughout",
            "Clear connection between findings and conclusions",
            "Appropriate scope",
            "Internal consistency",
        ],
    },
    "writing": {
        "name": "Writing",
        "weight": 0.15,
        "description": "Academic tone, clarity, conciseness, and style compliance",
        "criteria": [
            "Academic tone and language",
            "Clarity of expression",
            "Conciseness",
            "Proper citations",
            "No banned words or informal language",
        ],
    },
}


# =============================================================================
# Contribution Evaluation
# =============================================================================


def evaluate_contribution(
    writer_output: WriterOutput | dict[str, Any],
    research_question: str = "",
    identified_gaps: list[str] | None = None,
) -> QualityScore:
    """
    Evaluate the contribution dimension of the paper.
    
    Assesses:
    - Clear identification of research gap
    - Novel contribution to the field
    - Significance of findings
    - Practical and theoretical implications
    
    Args:
        writer_output: Output from the WRITER node
        research_question: The original research question
        identified_gaps: Gaps identified in literature review
        
    Returns:
        QualityScore for the contribution dimension
    """
    strengths = []
    weaknesses = []
    score = 5.0  # Start at middle
    
    # Handle both Pydantic model and dict
    if isinstance(writer_output, dict):
        sections = writer_output.get("sections", [])
        title = writer_output.get("title", "")
    else:
        sections = writer_output.sections if hasattr(writer_output, "sections") else []
        title = writer_output.title if hasattr(writer_output, "title") else ""
    
    # Get key sections
    intro_content = ""
    conclusion_content = ""
    abstract_content = ""
    
    for section in sections:
        if isinstance(section, dict):
            section_type = section.get("section_type", "")
            content = section.get("content", "")
        else:
            section_type = section.section_type if hasattr(section, "section_type") else ""
            content = section.content if hasattr(section, "content") else ""
        
        if section_type == "introduction":
            intro_content = content
        elif section_type == "conclusion":
            conclusion_content = content
        elif section_type == "abstract":
            abstract_content = content
    
    # Check 1: Research gap identification
    gap_indicators = ["gap", "lacking", "limited", "few studies", "unexplored", "understudied"]
    has_gap_discussion = any(ind in intro_content.lower() for ind in gap_indicators)
    if has_gap_discussion:
        strengths.append("Clear identification of research gap in introduction")
        score += 0.5
    else:
        weaknesses.append("Research gap not clearly articulated")
        score -= 0.5
    
    # Check 2: Contribution statement
    contribution_indicators = ["contribut", "advance", "extend", "offer", "provide insight"]
    has_contribution = any(ind in intro_content.lower() for ind in contribution_indicators)
    if has_contribution:
        strengths.append("Clear contribution statement present")
        score += 0.5
    else:
        weaknesses.append("Contribution statement unclear or missing")
        score -= 0.5
    
    # Check 3: Significance discussion
    significance_indicators = ["important", "significant", "implic", "matter", "relevant"]
    has_significance = any(ind in conclusion_content.lower() for ind in significance_indicators)
    if has_significance:
        strengths.append("Significance of findings discussed")
        score += 0.5
    else:
        weaknesses.append("Significance of research not clearly stated")
        score -= 0.3
    
    # Check 4: Implications
    implication_indicators = ["implications", "practitioners", "policymakers", "future research"]
    has_implications = any(ind in conclusion_content.lower() for ind in implication_indicators)
    if has_implications:
        strengths.append("Practical and theoretical implications discussed")
        score += 0.5
    else:
        weaknesses.append("Implications not fully developed")
        score -= 0.3
    
    # Check 5: Abstract quality
    if abstract_content:
        word_count = len(abstract_content.split())
        if 50 <= word_count <= 150:
            strengths.append("Abstract is appropriately concise")
            score += 0.3
        elif word_count > 200:
            weaknesses.append("Abstract may be too long")
            score -= 0.2
    else:
        weaknesses.append("Missing abstract")
        score -= 0.5
    
    # Check 6: Title quality
    if title:
        if 5 <= len(title.split()) <= 15:
            strengths.append("Title is appropriately concise")
            score += 0.2
    
    # Clamp score to valid range
    score = max(1.0, min(10.0, score))
    
    justification = f"Contribution evaluation based on gap identification, contribution clarity, and significance. "
    if strengths:
        justification += f"Key strengths: {'; '.join(strengths[:2])}. "
    if weaknesses:
        justification += f"Areas for improvement: {'; '.join(weaknesses[:2])}."
    
    return QualityScore(
        dimension="contribution",
        score=score,
        justification=justification,
        strengths=strengths,
        weaknesses=weaknesses,
    )


# =============================================================================
# Methodology Evaluation
# =============================================================================


def evaluate_methodology(
    writer_output: WriterOutput | dict[str, Any],
    research_plan: dict[str, Any] | None = None,
) -> QualityScore:
    """
    Evaluate the methodology dimension of the paper.
    
    Assesses:
    - Appropriate research design
    - Valid data collection methods
    - Sound analytical approach
    - Reproducibility
    - Limitations acknowledgment
    
    Args:
        writer_output: Output from the WRITER node
        research_plan: Research plan from PLANNER node
        
    Returns:
        QualityScore for the methodology dimension
    """
    strengths = []
    weaknesses = []
    score = 5.0
    
    # Get methods section
    methods_content = ""
    data_content = ""
    
    if isinstance(writer_output, dict):
        sections = writer_output.get("sections", [])
    else:
        sections = writer_output.sections if hasattr(writer_output, "sections") else []
    
    for section in sections:
        if isinstance(section, dict):
            section_type = section.get("section_type", "")
            content = section.get("content", "")
        else:
            section_type = section.section_type if hasattr(section, "section_type") else ""
            content = section.content if hasattr(section, "content") else ""
        
        if section_type == "methods":
            methods_content = content
        elif section_type == "data":
            data_content = content
    
    combined_content = methods_content + " " + data_content
    
    # Check 1: Research design description
    design_indicators = ["design", "approach", "framework", "model", "method"]
    has_design = any(ind in combined_content.lower() for ind in design_indicators)
    if has_design:
        strengths.append("Research design clearly described")
        score += 0.5
    else:
        weaknesses.append("Research design not clearly articulated")
        score -= 0.5
    
    # Check 2: Data description
    data_indicators = ["sample", "dataset", "observations", "period", "source"]
    has_data_description = any(ind in combined_content.lower() for ind in data_indicators)
    if has_data_description:
        strengths.append("Data sources and characteristics described")
        score += 0.5
    else:
        weaknesses.append("Data description insufficient")
        score -= 0.5
    
    # Check 3: Analytical methods
    analysis_indicators = ["regression", "analysis", "test", "estimate", "model", "variable"]
    has_analysis = any(ind in combined_content.lower() for ind in analysis_indicators)
    if has_analysis:
        strengths.append("Analytical methods explained")
        score += 0.5
    else:
        weaknesses.append("Analytical methods unclear")
        score -= 0.5
    
    # Check 4: Reproducibility
    reproducibility_indicators = ["specification", "equation", "parameter", "stata", "python", "r", "code"]
    has_reproducibility = any(ind in combined_content.lower() for ind in reproducibility_indicators)
    if has_reproducibility:
        strengths.append("Methods described with reproducibility in mind")
        score += 0.3
    else:
        weaknesses.append("Limited reproducibility details")
        score -= 0.2
    
    # Check 5: Limitations
    limitation_indicators = ["limitation", "caveat", "constraint", "cannot", "beyond scope"]
    has_limitations = any(ind in combined_content.lower() for ind in limitation_indicators)
    if has_limitations:
        strengths.append("Limitations acknowledged")
        score += 0.5
    else:
        weaknesses.append("Limitations not discussed")
        score -= 0.4
    
    # Check 6: Robustness discussion
    robustness_indicators = ["robust", "sensitivity", "alternative", "specification"]
    has_robustness = any(ind in combined_content.lower() for ind in robustness_indicators)
    if has_robustness:
        strengths.append("Robustness considerations addressed")
        score += 0.3
    
    # Clamp score
    score = max(1.0, min(10.0, score))
    
    justification = f"Methodology evaluation based on research design, data description, and analytical rigor. "
    if strengths:
        justification += f"Key strengths: {'; '.join(strengths[:2])}. "
    if weaknesses:
        justification += f"Areas for improvement: {'; '.join(weaknesses[:2])}."
    
    return QualityScore(
        dimension="methodology",
        score=score,
        justification=justification,
        strengths=strengths,
        weaknesses=weaknesses,
    )


# =============================================================================
# Evidence Evaluation
# =============================================================================


def evaluate_evidence(
    writer_output: WriterOutput | dict[str, Any],
    analysis_results: dict[str, Any] | None = None,
) -> QualityScore:
    """
    Evaluate the evidence dimension of the paper.
    
    Assesses:
    - Data quality and representativeness
    - Statistical validity
    - Correct interpretation of results
    - Support for claims
    - Robustness of findings
    
    Args:
        writer_output: Output from the WRITER node
        analysis_results: Results from DATA_ANALYST node
        
    Returns:
        QualityScore for the evidence dimension
    """
    strengths = []
    weaknesses = []
    score = 5.0
    
    # Get results section
    results_content = ""
    discussion_content = ""
    
    if isinstance(writer_output, dict):
        sections = writer_output.get("sections", [])
    else:
        sections = writer_output.sections if hasattr(writer_output, "sections") else []
    
    for section in sections:
        if isinstance(section, dict):
            section_type = section.get("section_type", "")
            content = section.get("content", "")
        else:
            section_type = section.section_type if hasattr(section, "section_type") else ""
            content = section.content if hasattr(section, "content") else ""
        
        if section_type == "results":
            results_content = content
        elif section_type == "discussion":
            discussion_content = content
    
    # Check 1: Statistical results present
    stat_indicators = ["significant", "coefficient", "p-value", "standard error", "t-stat", "95%"]
    has_stats = any(ind in results_content.lower() for ind in stat_indicators)
    if has_stats:
        strengths.append("Statistical results clearly presented")
        score += 0.5
    else:
        weaknesses.append("Statistical results lack detail")
        score -= 0.5
    
    # Check 2: Tables or structured results
    table_indicators = ["table", "figure", "panel", "column"]
    has_tables = any(ind in results_content.lower() for ind in table_indicators)
    if has_tables:
        strengths.append("Results organized with tables/figures")
        score += 0.3
    
    # Check 3: Interpretation quality
    interpretation_indicators = ["suggest", "indicate", "consistent with", "support", "imply"]
    has_interpretation = any(ind in results_content.lower() for ind in interpretation_indicators)
    if has_interpretation:
        strengths.append("Results properly interpreted")
        score += 0.5
    else:
        weaknesses.append("Interpretation of results could be clearer")
        score -= 0.3
    
    # Check 4: Claims supported by evidence
    unsupported_indicators = ["clearly", "obviously", "certainly", "undoubtedly"]
    has_overclaims = any(ind in discussion_content.lower() for ind in unsupported_indicators)
    if has_overclaims:
        weaknesses.append("Some claims may be overstated")
        score -= 0.4
    else:
        strengths.append("Claims appropriately hedged")
        score += 0.3
    
    # Check 5: Effect sizes or economic significance
    magnitude_indicators = ["magnitude", "economic significance", "effect size", "percentage", "basis points"]
    has_magnitude = any(ind in results_content.lower() for ind in magnitude_indicators)
    if has_magnitude:
        strengths.append("Economic/practical significance discussed")
        score += 0.4
    else:
        weaknesses.append("Economic significance not fully addressed")
        score -= 0.2
    
    # Check 6: Robustness checks
    robustness_indicators = ["robust", "alternative", "sensitivity", "subsample"]
    has_robustness = any(ind in results_content.lower() for ind in robustness_indicators)
    if has_robustness:
        strengths.append("Robustness checks reported")
        score += 0.4
    else:
        weaknesses.append("Limited robustness analysis")
        score -= 0.3
    
    # Clamp score
    score = max(1.0, min(10.0, score))
    
    justification = f"Evidence evaluation based on statistical presentation, interpretation, and support for claims. "
    if strengths:
        justification += f"Key strengths: {'; '.join(strengths[:2])}. "
    if weaknesses:
        justification += f"Areas for improvement: {'; '.join(weaknesses[:2])}."
    
    return QualityScore(
        dimension="evidence",
        score=score,
        justification=justification,
        strengths=strengths,
        weaknesses=weaknesses,
    )


# =============================================================================
# Coherence Evaluation
# =============================================================================


def evaluate_coherence(
    writer_output: WriterOutput | dict[str, Any],
) -> QualityScore:
    """
    Evaluate the coherence dimension of the paper.
    
    Assesses:
    - Logical flow between sections
    - Consistent argument throughout
    - Clear connection between findings and conclusions
    - Appropriate scope
    - Internal consistency
    
    Args:
        writer_output: Output from the WRITER node
        
    Returns:
        QualityScore for the coherence dimension
    """
    strengths = []
    weaknesses = []
    score = 5.0
    
    if isinstance(writer_output, dict):
        sections = writer_output.get("sections", [])
    else:
        sections = writer_output.sections if hasattr(writer_output, "sections") else []
    
    # Check 1: All expected sections present
    expected_sections = {"abstract", "introduction", "methods", "results", "discussion", "conclusion"}
    present_sections = set()
    section_contents = {}
    
    for section in sections:
        if isinstance(section, dict):
            section_type = section.get("section_type", "")
            content = section.get("content", "")
        else:
            section_type = section.section_type if hasattr(section, "section_type") else ""
            content = section.content if hasattr(section, "content") else ""
        
        present_sections.add(section_type)
        section_contents[section_type] = content
    
    missing_sections = expected_sections - present_sections
    if not missing_sections:
        strengths.append("All key sections present")
        score += 0.5
    else:
        weaknesses.append(f"Missing sections: {', '.join(missing_sections)}")
        score -= 0.3 * len(missing_sections)
    
    # Check 2: Transition words and logical flow
    transition_words = ["however", "therefore", "furthermore", "moreover", "consequently", "thus", "additionally"]
    all_content = " ".join(section_contents.values()).lower()
    transition_count = sum(1 for word in transition_words if word in all_content)
    if transition_count >= 5:
        strengths.append("Good use of transition words for flow")
        score += 0.4
    elif transition_count < 2:
        weaknesses.append("Limited use of transition words")
        score -= 0.3
    
    # Check 3: Research question consistency
    intro = section_contents.get("introduction", "").lower()
    conclusion = section_contents.get("conclusion", "").lower()
    
    # Simple check: key terms from intro appear in conclusion
    intro_key_terms = set(re.findall(r'\b\w{6,}\b', intro)[:20])  # Long words from intro
    conclusion_key_terms = set(re.findall(r'\b\w{6,}\b', conclusion))
    overlap = len(intro_key_terms & conclusion_key_terms)
    
    if overlap >= 5:
        strengths.append("Consistent terminology between introduction and conclusion")
        score += 0.4
    elif overlap < 2:
        weaknesses.append("Potential inconsistency between introduction and conclusion")
        score -= 0.3
    
    # Check 4: Section length balance
    section_lengths = {k: len(v.split()) for k, v in section_contents.items()}
    if section_lengths:
        avg_length = sum(section_lengths.values()) / len(section_lengths)
        very_short = [k for k, v in section_lengths.items() if v < avg_length * 0.3 and k not in ["abstract"]]
        very_long = [k for k, v in section_lengths.items() if v > avg_length * 2.5]
        
        if not very_short and not very_long:
            strengths.append("Well-balanced section lengths")
            score += 0.3
        else:
            if very_short:
                weaknesses.append(f"Some sections may be underdeveloped: {', '.join(very_short)}")
                score -= 0.2
    
    # Check 5: Introduction-conclusion alignment
    intro_goals = ["examine", "investigate", "analyze", "explore", "study"]
    conclusion_outcomes = ["find", "show", "demonstrate", "reveal", "suggest"]
    
    has_intro_goals = any(g in intro for g in intro_goals)
    has_conclusion_outcomes = any(o in conclusion for o in conclusion_outcomes)
    
    if has_intro_goals and has_conclusion_outcomes:
        strengths.append("Clear alignment between research goals and findings")
        score += 0.4
    
    # Clamp score
    score = max(1.0, min(10.0, score))
    
    justification = f"Coherence evaluation based on section completeness, logical flow, and internal consistency. "
    if strengths:
        justification += f"Key strengths: {'; '.join(strengths[:2])}. "
    if weaknesses:
        justification += f"Areas for improvement: {'; '.join(weaknesses[:2])}."
    
    return QualityScore(
        dimension="coherence",
        score=score,
        justification=justification,
        strengths=strengths,
        weaknesses=weaknesses,
    )


# =============================================================================
# Writing Evaluation
# =============================================================================


def evaluate_writing(
    writer_output: WriterOutput | dict[str, Any],
    target_journal: str = "generic",
) -> QualityScore:
    """
    Evaluate the writing dimension of the paper.
    
    Assesses:
    - Academic tone and language
    - Clarity of expression
    - Conciseness
    - Proper citations
    - No banned words or informal language
    
    Args:
        writer_output: Output from the WRITER node
        target_journal: Target journal for style compliance
        
    Returns:
        QualityScore for the writing dimension
    """
    strengths = []
    weaknesses = []
    score = 5.0
    
    if isinstance(writer_output, dict):
        sections = writer_output.get("sections", [])
    else:
        sections = writer_output.sections if hasattr(writer_output, "sections") else []
    
    # Combine all content
    all_content = ""
    for section in sections:
        if isinstance(section, dict):
            all_content += " " + section.get("content", "")
        else:
            all_content += " " + (section.content if hasattr(section, "content") else "")
    
    all_content_lower = all_content.lower()
    
    # Check 1: Banned words
    banned_found = []
    for word in BANNED_WORDS:
        if word.lower() in all_content_lower:
            banned_found.append(word)
    
    if not banned_found:
        strengths.append("No banned words detected")
        score += 0.5
    else:
        weaknesses.append(f"Banned words found: {', '.join(banned_found[:5])}")
        score -= 0.1 * min(len(banned_found), 5)
    
    # Check 2: Informal language (using word boundary regex)
    informal_indicators = ["stuff", "things", "lots", "gonna", "wanna", "kinda", "really", "very", "just"]
    informal_found = [word for word in informal_indicators if re.search(rf'\b{word}\b', all_content_lower)]
    if not informal_found:
        strengths.append("Formal academic tone maintained")
        score += 0.4
    else:
        weaknesses.append(f"Informal language detected: {', '.join(informal_found[:3])}")
        score -= 0.3
    
    # Check 3: Contractions (using word boundary regex)
    contractions = ["don't", "can't", "won't", "isn't", "aren't", "doesn't", "didn't", "couldn't"]
    contractions_found = [c for c in contractions if re.search(rf'\b{re.escape(c)}\b', all_content_lower)]
    if not contractions_found:
        strengths.append("No contractions used")
        score += 0.3
    else:
        weaknesses.append("Contractions found (should use full forms)")
        score -= 0.2
    
    # Check 4: Sentence length variety
    sentences = re.split(r'[.!?]+', all_content)
    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
    
    if sentence_lengths:
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        if 15 <= avg_length <= 25:
            strengths.append("Good average sentence length")
            score += 0.3
        elif avg_length > 35:
            weaknesses.append("Sentences may be too long on average")
            score -= 0.3
        elif avg_length < 10:
            weaknesses.append("Sentences may be too short on average")
            score -= 0.2
    
    # Check 5: Citation presence
    citation_patterns = [r'\(\d{4}\)', r'\([A-Z][a-z]+,?\s*\d{4}\)', r'\([A-Z][a-z]+\s+et al\.?,?\s*\d{4}\)']
    citation_count = sum(len(re.findall(pattern, all_content)) for pattern in citation_patterns)
    
    if citation_count >= 10:
        strengths.append("Good citation density")
        score += 0.4
    elif citation_count < 3:
        weaknesses.append("Limited citations; may need more references")
        score -= 0.4
    
    # Check 6: Hedging language (using word boundary regex for accuracy)
    hedging_words = ["may", "might", "could", "suggest", "appear", "seem", "indicate"]
    hedging_count = sum(len(re.findall(rf'\b{word}\b', all_content_lower)) for word in hedging_words)
    
    if 5 <= hedging_count <= 20:
        strengths.append("Appropriate use of hedging language")
        score += 0.3
    elif hedging_count < 2:
        weaknesses.append("May need more hedging for academic caution")
        score -= 0.2
    elif hedging_count > 30:
        weaknesses.append("Excessive hedging may weaken claims")
        score -= 0.2
    
    # Check 7: First person usage (using word boundary regex for accuracy)
    first_person = ["i", "we", "my", "our"]
    first_person_count = sum(len(re.findall(rf'\b{word}\b', all_content_lower)) for word in first_person)
    
    if first_person_count <= 10:
        strengths.append("Appropriate use of first person")
        score += 0.2
    else:
        weaknesses.append("Consider reducing first person usage")
        score -= 0.2
    
    # Clamp score
    score = max(1.0, min(10.0, score))
    
    justification = f"Writing evaluation based on academic tone, clarity, and style compliance. "
    if strengths:
        justification += f"Key strengths: {'; '.join(strengths[:2])}. "
    if weaknesses:
        justification += f"Areas for improvement: {'; '.join(weaknesses[:2])}."
    
    return QualityScore(
        dimension="writing",
        score=score,
        justification=justification,
        strengths=strengths,
        weaknesses=weaknesses,
    )


# =============================================================================
# Comprehensive Paper Evaluation
# =============================================================================


def evaluate_paper(
    writer_output: WriterOutput | dict[str, Any],
    research_question: str = "",
    identified_gaps: list[str] | None = None,
    research_plan: dict[str, Any] | None = None,
    analysis_results: dict[str, Any] | None = None,
    target_journal: str = "generic",
) -> tuple[list[QualityScore], list[ReviewCritiqueItem]]:
    """
    Perform comprehensive evaluation of the paper across all dimensions.
    
    Args:
        writer_output: Output from the WRITER node
        research_question: The original research question
        identified_gaps: Gaps identified in literature review
        research_plan: Research plan from PLANNER node
        analysis_results: Results from DATA_ANALYST node
        target_journal: Target journal for style compliance
        
    Returns:
        Tuple of (dimension_scores, critique_items)
    """
    logger.info("Starting comprehensive paper evaluation")
    
    # Evaluate each dimension
    dimension_scores = [
        evaluate_contribution(writer_output, research_question, identified_gaps),
        evaluate_methodology(writer_output, research_plan),
        evaluate_evidence(writer_output, analysis_results),
        evaluate_coherence(writer_output),
        evaluate_writing(writer_output, target_journal),
    ]
    
    # Generate critique items from weaknesses
    critique_items = []
    
    for score in dimension_scores:
        for weakness in score.weaknesses:
            # Determine severity based on dimension and score
            if score.score < 4.0:
                severity = "critical"
            elif score.score < 6.0:
                severity = "major"
            elif score.score < 7.5:
                severity = "minor"
            else:
                severity = "suggestion"
            
            # Map dimension to most relevant section
            section_mapping = {
                "contribution": "introduction",
                "methodology": "methods",
                "evidence": "results",
                "coherence": "discussion",
                "writing": "abstract",
            }
            section = section_mapping.get(score.dimension, "general")
            
            # Ensure issue and suggestion meet minimum length requirements
            issue_text = weakness if len(weakness) >= 10 else f"Issue: {weakness} - needs attention"
            suggestion_text = f"Review {score.dimension} dimension and address: {weakness}"
            
            critique_items.append(
                ReviewCritiqueItem(
                    section=section,
                    issue=issue_text,
                    severity=severity,
                    suggestion=suggestion_text,
                )
            )
    
    logger.info(
        f"Evaluation complete: {len(dimension_scores)} dimensions, "
        f"{len(critique_items)} critique items"
    )
    
    return dimension_scores, critique_items
