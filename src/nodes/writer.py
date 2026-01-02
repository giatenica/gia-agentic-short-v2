"""WRITER node for paper composition.

Orchestrates section writing with:
- Sequential section generation (intro first, abstract last)
- Style enforcement and auto-correction
- Citation tracking and reference list generation
- Argument coherence validation
"""

from datetime import datetime, timezone
from typing import Literal
import logging

from src.config import settings
from src.state.schema import WorkflowState
from src.state.models import (
    PaperSection,
    WriterOutput,
    SectionWritingContext,
    get_section_word_count_target,
    ReferenceList,
)
from src.state.enums import (
    SectionType,
    WritingStatus,
    JournalTarget,
)
from src.style import StyleEnforcer
from src.citations import CitationManager
from src.writers import (
    BaseSectionWriter,
    SectionWriterConfig,
    IntroductionWriter,
    LiteratureReviewWriter,
    MethodsWriter,
    ResultsWriter,
    DiscussionWriter,
    ConclusionWriter,
    AbstractWriter,
    ArgumentManager,
)

logger = logging.getLogger(__name__)

# Section writing order (abstract written last)
SECTION_ORDER: list[SectionType] = [
    SectionType.INTRODUCTION,
    SectionType.LITERATURE_REVIEW,
    SectionType.METHODS,
    SectionType.RESULTS,
    SectionType.DISCUSSION,
    SectionType.CONCLUSION,
    SectionType.ABSTRACT,  # Always last
]


def get_section_writer(
    section_type: SectionType,
    config: SectionWriterConfig,
    style_enforcer: StyleEnforcer,
    citation_manager: CitationManager,
) -> BaseSectionWriter:
    """Get the appropriate writer for a section type."""
    writers: dict[SectionType, type[BaseSectionWriter]] = {
        SectionType.INTRODUCTION: IntroductionWriter,
        SectionType.LITERATURE_REVIEW: LiteratureReviewWriter,
        SectionType.METHODS: MethodsWriter,
        SectionType.RESULTS: ResultsWriter,
        SectionType.DISCUSSION: DiscussionWriter,
        SectionType.CONCLUSION: ConclusionWriter,
        SectionType.ABSTRACT: AbstractWriter,
    }
    
    writer_class = writers.get(section_type)
    if not writer_class:
        raise ValueError(f"No writer available for section type: {section_type}")
    
    return writer_class(
        config=config,
        style_enforcer=style_enforcer,
        citation_manager=citation_manager,
    )


def build_section_context(
    state: WorkflowState,
    section_type: SectionType,
    completed_sections: list[PaperSection],
    argument_manager: ArgumentManager,
) -> SectionWritingContext:
    """Build context for writing a specific section."""
    # Get research intake information
    research_intake = state.get("research_intake", {})
    
    # Determine paper type
    paper_type = research_intake.get("paper_type", "short_article")
    
    # Get target journal
    target_journal = "rfs"  # Default
    journal_str = research_intake.get("target_journal", "").lower()
    for journal in JournalTarget:
        if journal.value.lower() in journal_str:
            target_journal = journal.value
            break
    
    # Get word count target
    target_word_count = get_section_word_count_target(paper_type, section_type.value)
    # get_section_word_count_target returns tuple or None, get the upper bound
    target_count = None
    if target_word_count:
        target_count = target_word_count[1]  # Use upper bound
    
    # Build research question from intake
    research_question = research_intake.get("research_question", "")
    if not research_question:
        # Try to extract from intake form
        research_question = research_intake.get("question", "")
    
    # Build contribution statement
    contribution = research_intake.get("expected_contribution", "")
    if not contribution:
        contribution = research_intake.get("contribution", "")
    
    # Build findings summary from analysis nodes
    findings_summary = ""
    
    # Get from data explorer output
    data_explorer_output = state.get("data_explorer_output")
    if data_explorer_output:
        key_patterns = data_explorer_output.get("key_patterns", [])
        if key_patterns:
            findings_summary = "; ".join(key_patterns[:3])
    
    # Get from gap identifier output for literature gaps
    gap_output = state.get("gap_identifier_output")
    gap_analysis_summary = ""
    if gap_output:
        identified_gaps = gap_output.get("identified_gaps", [])
        gap_analysis_summary = "; ".join([
            g.get("gap_statement", "") 
            for g in identified_gaps 
            if g.get("gap_statement")
        ][:3])
    
    # Get methodology summary from planner
    methodology_summary = ""
    planner_output = state.get("planner_output")
    if planner_output:
        methodology_summary = planner_output.get("methodology_framework", "")
    
    # Get literature synthesis summary
    lit_synthesis = state.get("literature_synthesis", {})
    literature_synthesis_summary = ""
    if lit_synthesis:
        themes = lit_synthesis.get("themes", [])
        if themes:
            literature_synthesis_summary = "; ".join(themes[:3])
    
    # Check for quantitative/qualitative results
    has_quantitative = False
    has_qualitative = False
    data_analyst_output = state.get("data_analyst_output")
    if data_analyst_output:
        has_quantitative = True
    conceptual_output = state.get("conceptual_synthesis_output")
    if conceptual_output:
        has_qualitative = True
    
    # Get argument coherence prompt (stored for future prompt enhancement)
    _ = argument_manager.generate_coherence_prompt(section_type)
    
    return SectionWritingContext(
        section_type=section_type.value,
        paper_type=paper_type,
        target_journal=target_journal,
        research_question=research_question,
        contribution_statement=contribution,
        findings_summary=findings_summary,
        gap_analysis_summary=gap_analysis_summary,
        literature_synthesis_summary=literature_synthesis_summary,
        methodology_summary=methodology_summary,
        has_quantitative_results=has_quantitative,
        has_qualitative_results=has_qualitative,
        prior_sections=completed_sections,
        target_word_count=target_count,
    )


def writer_node(state: WorkflowState) -> dict:
    """WRITER node: Orchestrates paper section writing.
    
    Writes sections in order:
    1. Introduction
    2. Literature Review
    3. Methods
    4. Results
    5. Discussion
    6. Conclusion
    7. Abstract (last, after seeing all content)
    
    Each section undergoes style checking and citation tracking.
    """
    logger.info("WRITER node: Starting paper composition")
    
    # Get target journal from state
    research_intake = state.get("research_intake", {})
    target_journal = research_intake.get("target_journal", "generic")
    
    # Initialize components
    style_enforcer = StyleEnforcer(target_journal=target_journal)
    citation_manager = CitationManager()
    argument_manager = ArgumentManager()
    
    # Set up argument manager with core claims
    contribution = research_intake.get("expected_contribution", "")
    question = research_intake.get("research_question", "")
    
    # Extract key findings from data analysis
    findings = []
    data_output = state.get("data_explorer_output")
    if data_output:
        patterns = data_output.get("key_patterns", [])
        findings.extend(patterns[:3])
    
    argument_manager.set_core_argument(
        contribution=contribution,
        question=question,
        findings=findings,
    )
    
    # Create main argument thread
    argument_manager.create_thread(
        name="main_contribution",
        description=contribution,
        claimed_in=["introduction"],
        supported_in=["results", "discussion"],
    )
    
    # Configuration for writers
    writer_config = SectionWriterConfig(
        model_name=settings.default_model,
        target_journal=target_journal,
        auto_fix_style=True,
    )
    
    # Determine which sections to write
    sections_to_write = state.get("sections_to_write", [])
    if not sections_to_write:
        # Default: write all sections
        sections_to_write = [s.value for s in SECTION_ORDER]
    
    # Track completed sections
    completed_sections: list[PaperSection] = state.get("completed_sections", [])
    all_violations: list[StyleViolation] = []
    
    # Write each section in order
    for section_type_value in sections_to_write:
        try:
            section_type = SectionType(section_type_value)
        except ValueError:
            logger.warning(f"Unknown section type: {section_type_value}")
            continue
        
        logger.info(f"Writing section: {section_type.value}")
        
        # Build context for this section
        context = build_section_context(
            state=state,
            section_type=section_type,
            completed_sections=completed_sections,
            argument_manager=argument_manager,
        )
        
        # Get appropriate writer
        writer = get_section_writer(
            section_type=section_type,
            config=writer_config,
            style_enforcer=style_enforcer,
            citation_manager=citation_manager,
        )
        
        # Write the section
        section = writer.write(context)
        
        # Track any style violations
        all_violations.extend(section.style_violations)
        
        # Update argument manager with claims from this section
        argument_manager.register_claim(
            claim=f"Section {section_type.value} written",
            section=section_type.value,
        )
        
        # Add to completed sections
        completed_sections.append(section)
        
        logger.info(
            f"Completed section: {section_type.value} "
            f"({section.word_count} words, "
            f"{len(section.style_violations)} violations)"
        )
    
    # Collect all citation keys from sections
    all_citation_keys: list[str] = []
    for section in completed_sections:
        all_citation_keys.extend(section.citations_used)
    
    # Build reference list - for now, use placeholder entries
    # In production, this would look up full citation data using unique keys
    reference_list = ReferenceList(
        entries=[],  # Would be populated from citation database
        format_style="chicago_author_date",
    )
    
    # Calculate total statistics
    total_word_count = sum(s.word_count for s in completed_sections)
    
    # Get coherence summary
    coherence_summary = argument_manager.get_coherence_summary()
    
    # Get paper title from research intake
    paper_title = research_intake.get("title", "Untitled Paper")
    
    # Build argument thread for output
    argument_thread = ArgumentThread(
        main_thesis=question,
        claims=[contribution],
        promised_contribution=contribution,
        coherence_score=coherence_summary["coherence_score"],
    )
    
    # Build output with correct field names
    writer_output = WriterOutput(
        title=paper_title,
        sections=completed_sections,
        reference_list=reference_list,
        argument_thread=argument_thread,
        total_word_count=total_word_count,
        style_violations=all_violations,
        writing_status=WritingStatus.FINALIZED.value,
        contribution_delivered=coherence_summary["coherence_score"] >= 0.8,
        completed_at=datetime.now(timezone.utc),
    )
    
    logger.info(
        f"WRITER node complete: {len(completed_sections)} sections, "
        f"{total_word_count} total words"
    )
    
    return {
        "writer_output": writer_output.model_dump(),
        "completed_sections": completed_sections,
        "reference_list": reference_list.model_dump(),
        "style_violations": all_violations,
    }


def should_continue_writing(state: WorkflowState) -> Literal["continue", "end"]:
    """Determine if writing should continue or is complete."""
    sections_to_write = state.get("sections_to_write", [])
    completed_sections = state.get("completed_sections", [])
    
    # Check if all requested sections are complete
    completed_types = {s.section_type for s in completed_sections}
    remaining = set(sections_to_write) - completed_types
    
    if remaining:
        return "continue"
    return "end"
