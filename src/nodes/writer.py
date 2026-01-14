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
    StyleViolation,
    ArgumentThread,
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


def _get_primary_research_question(state: WorkflowState) -> str:
    refined = (state.get("refined_query") or "").strip()
    if refined:
        return refined
    original = (state.get("original_query") or "").strip()
    if original:
        return original

    research_intake = state.get("research_intake")
    if isinstance(research_intake, dict):
        intake_q = (research_intake.get("research_question") or "").strip()
        if intake_q:
            return intake_q
        intake_q = (research_intake.get("question") or "").strip()
        if intake_q:
            return intake_q

    return ""


def _get_paper_title(state: WorkflowState) -> str:
    title = (state.get("project_title") or "").strip()
    if title:
        return title
    research_intake = state.get("research_intake")
    if isinstance(research_intake, dict):
        intake_title = (research_intake.get("title") or "").strip()
        if intake_title:
            return intake_title
    return "Untitled Paper"


def _get_target_journal(state: WorkflowState) -> str:
    target = (state.get("target_journal") or "").strip()
    if target:
        return target
    research_intake = state.get("research_intake")
    if isinstance(research_intake, dict):
        intake_target = (research_intake.get("target_journal") or "").strip()
        if intake_target:
            return intake_target
    return "generic"


def _get_paper_type(state: WorkflowState) -> str:
    paper_type = (state.get("paper_type") or "").strip()
    if paper_type:
        return paper_type
    research_intake = state.get("research_intake")
    if isinstance(research_intake, dict):
        intake_paper_type = (research_intake.get("paper_type") or "").strip()
        if intake_paper_type:
            return intake_paper_type
    return "short_article"


def _get_research_type(state: WorkflowState) -> str:
    research_type = (state.get("research_type") or "").strip()
    if research_type:
        return research_type
    research_intake = state.get("research_intake")
    if isinstance(research_intake, dict):
        intake_research_type = (research_intake.get("research_type") or "").strip()
        if intake_research_type:
            return intake_research_type
    return "empirical"


def _get_contribution_statement(state: WorkflowState) -> str:
    contribution = (state.get("contribution_statement") or "").strip()
    if contribution:
        return contribution
    expected = (state.get("expected_contribution") or "").strip()
    if expected:
        return expected

    research_intake = state.get("research_intake")
    if isinstance(research_intake, dict):
        intake_contribution = (research_intake.get("expected_contribution") or "").strip()
        if intake_contribution:
            return intake_contribution
        intake_contribution = (research_intake.get("contribution") or "").strip()
        if intake_contribution:
            return intake_contribution
    return ""


def _summarize_gap(state: WorkflowState) -> str:
    parts: list[str] = []
    identified_gaps = state.get("identified_gaps")
    if isinstance(identified_gaps, list) and identified_gaps:
        parts.append("; ".join(str(g) for g in identified_gaps[:3] if g))

    gap_analysis = state.get("gap_analysis")
    if isinstance(gap_analysis, dict):
        primary = gap_analysis.get("primary_gap")
        if isinstance(primary, dict):
            title = (primary.get("title") or "").strip()
            desc = (primary.get("description") or "").strip()
            if title and desc:
                parts.append(f"Primary gap: {title}. {desc[:240]}")
            elif title:
                parts.append(f"Primary gap: {title}.")
        gaps = gap_analysis.get("gaps")
        if isinstance(gaps, list) and gaps:
            first = gaps[0]
            if isinstance(first, dict):
                title = (first.get("title") or "").strip()
                desc = (first.get("description") or "").strip()
                if title and desc:
                    parts.append(f"Example gap: {title}. {desc[:240]}")

    return "\n".join(p for p in parts if p)


def _summarize_literature(state: WorkflowState) -> str:
    lit = state.get("literature_synthesis")
    parts: list[str] = []
    if isinstance(lit, dict):
        summary = (lit.get("summary") or "").strip()
        if summary:
            parts.append(summary)

        themes = lit.get("themes")
        if isinstance(themes, list) and themes:
            parts.append("Themes: " + "; ".join(str(t) for t in themes[:6] if t))

        key_findings = lit.get("key_findings")
        if isinstance(key_findings, list) and key_findings:
            parts.append("Key findings: " + "; ".join(str(f) for f in key_findings[:5] if f))

    # Fallback: use top titles from search results for grounding
    search_results = state.get("search_results")
    if isinstance(search_results, list) and search_results:
        titles: list[str] = []
        for r in search_results[:12]:
            if isinstance(r, dict):
                t = (r.get("title") or "").strip()
            else:
                t = getattr(r, "title", "")
            if t:
                titles.append(t)
        if titles:
            parts.append("Representative papers: " + "; ".join(titles[:8]))

    return "\n".join(p for p in parts if p)


def _summarize_methodology(state: WorkflowState) -> str:
    plan = state.get("research_plan")
    if isinstance(plan, dict):
        pieces: list[str] = []
        methodology = (plan.get("methodology_overview") or plan.get("methodology") or "").strip()
        if methodology:
            pieces.append(methodology)
        analysis_approach = plan.get("analysis_approach")
        if analysis_approach:
            pieces.append(f"Analysis approach: {analysis_approach}")
        data_strategy = (plan.get("data_strategy") or "").strip()
        if data_strategy:
            pieces.append(f"Data strategy: {data_strategy}")
        return "\n".join(pieces)

    proposed = (state.get("proposed_methodology") or "").strip()
    return proposed


def _summarize_findings(state: WorkflowState) -> str:
    analysis = state.get("analysis")
    if isinstance(analysis, dict):
        parts: list[str] = []
        data_summary = (analysis.get("data_summary") or "").strip()
        if data_summary:
            parts.append(data_summary)

        hypothesis = (analysis.get("hypothesis_test_summary") or "").strip()
        if hypothesis:
            parts.append(hypothesis)

        main_findings = analysis.get("main_findings")
        if isinstance(main_findings, list) and main_findings:
            for f in main_findings[:4]:
                if isinstance(f, dict):
                    statement = (f.get("finding_statement") or f.get("statement") or "").strip()
                    if statement:
                        parts.append(statement)
                else:
                    statement = getattr(f, "finding_statement", "") or getattr(f, "statement", "")
                    if statement:
                        parts.append(str(statement))

        regressions = analysis.get("regression_analyses")
        if isinstance(regressions, list) and regressions:
            first = regressions[0]
            if isinstance(first, dict):
                dv = (first.get("dependent_variable") or "").strip()
                ivs = first.get("independent_variables")
                if dv and isinstance(ivs, list) and ivs:
                    parts.append(f"Main model: OLS {dv} on {', '.join(str(v) for v in ivs[:6])}")

        return "\n".join(p for p in parts if p)

    # Fallback: legacy key
    data_analyst_output = state.get("data_analyst_output")
    if isinstance(data_analyst_output, dict):
        return (data_analyst_output.get("data_summary") or "").strip()

    return ""


def _extract_revision_request(state: WorkflowState) -> dict | None:
    revision_request = state.get("revision_request")
    if not revision_request:
        reviewer_output = state.get("reviewer_output")
        if isinstance(reviewer_output, dict):
            revision_request = reviewer_output.get("revision_request")
        elif hasattr(reviewer_output, "revision_request"):
            revision_request = reviewer_output.revision_request

    if revision_request and hasattr(revision_request, "model_dump"):
        return revision_request.model_dump()
    if isinstance(revision_request, dict):
        return revision_request
    return None


def _format_section_critique(revision_request: dict | None, section_type: str) -> str:
    if not revision_request:
        return ""
    items = revision_request.get("critique_items")
    if not isinstance(items, list) or not items:
        return ""

    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        section = (item.get("section") or "").strip()
        if section and section != section_type:
            continue
        issue = (item.get("issue") or "").strip()
        suggestion = (item.get("suggestion") or "").strip()
        severity = (item.get("severity") or "").strip()

        if not issue and not suggestion:
            continue

        if severity:
            prefix = f"[{severity.upper()}] "
        else:
            prefix = ""
        if issue and suggestion:
            lines.append(f"- {prefix}{issue} Fix: {suggestion}")
        elif issue:
            lines.append(f"- {prefix}{issue}")
        else:
            lines.append(f"- {prefix}{suggestion}")

    return "\n".join(lines)


def _normalize_completed_sections(completed_sections: list) -> list[PaperSection]:
    normalized: list[PaperSection] = []
    for section in completed_sections:
        if isinstance(section, PaperSection):
            normalized.append(section)
        elif isinstance(section, dict):
            try:
                normalized.append(PaperSection(**section))
            except Exception:
                continue
    return normalized


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
    # Determine paper type and research type
    paper_type = _get_paper_type(state)
    research_type = _get_research_type(state)

    # Get target journal
    target_journal = _get_target_journal(state)
    
    # Get word count target
    target_word_count = get_section_word_count_target(paper_type, section_type.value)
    # get_section_word_count_target returns tuple or None, get the upper bound
    target_count = None
    if target_word_count:
        target_count = target_word_count[1]  # Use upper bound
    
    # Research question and contribution
    research_question = _get_primary_research_question(state)
    contribution = _get_contribution_statement(state)

    # Summaries grounded in prior nodes
    findings_summary = _summarize_findings(state)
    gap_analysis_summary = _summarize_gap(state)
    methodology_summary = _summarize_methodology(state)
    literature_synthesis_summary = _summarize_literature(state)
    
    # Check for quantitative/qualitative results
    has_quantitative = False
    has_qualitative = False
    if state.get("analysis") or state.get("data_analyst_output"):
        has_quantitative = True
    conceptual_output = state.get("conceptual_synthesis_output")
    if conceptual_output:
        has_qualitative = True
    
    # Sprint 16: Get tables and figures from state
    tables = state.get("tables", [])
    figures = state.get("figures", [])
    
    # Sprint 16: Get data exploration prose for methods section
    data_exploration_prose = ""
    data_exploration_summary = state.get("data_exploration_summary")
    if data_exploration_summary:
        if hasattr(data_exploration_summary, "prose_description"):
            data_exploration_prose = data_exploration_summary.prose_description
        elif isinstance(data_exploration_summary, dict):
            data_exploration_prose = data_exploration_summary.get("prose_description", "")
    
    # Get argument coherence prompt (stored for future prompt enhancement)
    _ = argument_manager.generate_coherence_prompt(section_type)
    
    revision_request = _extract_revision_request(state)
    human_feedback = (state.get("human_feedback") or "").strip()
    critique_for_section = _format_section_critique(revision_request, section_type.value)
    revision_instructions = ""
    revision_iteration = None
    is_revision = False
    if revision_request:
        is_revision = True
        revision_instructions = (revision_request.get("revision_instructions") or "").strip()
        revision_iteration = revision_request.get("iteration_count")

    return SectionWritingContext(
        section_type=section_type.value,
        paper_type=paper_type,
        target_journal=target_journal,
        research_type=research_type,
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
        tables=tables,
        figures=figures,
        data_exploration_prose=data_exploration_prose,
        is_revision=is_revision,
        revision_iteration=revision_iteration if isinstance(revision_iteration, int) else None,
        revision_instructions=revision_instructions,
        critique_for_section=critique_for_section,
        human_feedback=human_feedback,
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

    # Get primary paper metadata from canonical state
    target_journal = _get_target_journal(state)
    
    # Initialize components
    style_enforcer = StyleEnforcer(target_journal=target_journal)
    citation_manager = CitationManager()
    argument_manager = ArgumentManager()
    
    # Set up argument manager with core claims
    contribution = _get_contribution_statement(state)
    question = _get_primary_research_question(state)

    # Extract key findings from analysis
    findings: list[str] = []
    analysis = state.get("analysis")
    if isinstance(analysis, dict):
        main_findings = analysis.get("main_findings")
        if isinstance(main_findings, list):
            for f in main_findings[:3]:
                if isinstance(f, dict):
                    statement = (f.get("finding_statement") or f.get("statement") or "").strip()
                    if statement:
                        findings.append(statement)
    
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

    revision_request = _extract_revision_request(state)
    in_revision_loop = (state.get("review_decision") == "revise") or bool(revision_request)

    if (not sections_to_write) and in_revision_loop and revision_request:
        requested = revision_request.get("sections_to_revise")
        if isinstance(requested, list) and requested:
            sections_to_write = [str(s) for s in requested]

    if not sections_to_write:
        # Default: write all sections
        sections_to_write = [s.value for s in SECTION_ORDER]
    
    # Track completed sections
    completed_sections_raw = state.get("completed_sections", [])
    if not completed_sections_raw:
        writer_output = state.get("writer_output")
        if isinstance(writer_output, dict):
            completed_sections_raw = writer_output.get("sections", [])
        elif hasattr(writer_output, "sections"):
            completed_sections_raw = writer_output.sections

    completed_sections: list[PaperSection] = _normalize_completed_sections(completed_sections_raw or [])
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
        
        # Replace existing section in revision loop, otherwise append
        completed_sections = [s for s in completed_sections if s.section_type != section.section_type]
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
    
    paper_title = _get_paper_title(state)
    
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
