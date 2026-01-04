"""Regression tests for writer prompt style guide injection."""


def test_style_guide_excerpt_is_included_in_system_prompt():
    """Writers should include the project style guide excerpt in the system prompt."""

    from src.state.models import SectionWritingContext
    from src.writers import IntroductionWriter
    from src.writers.style_guide import get_style_guide_excerpt

    excerpt = get_style_guide_excerpt()
    assert excerpt, "Expected docs/writing_style_guide.md excerpt to be non-empty"

    writer = IntroductionWriter()
    context = SectionWritingContext(
        section_type="introduction",
        target_journal="rfs",
        paper_type="short_article",
        research_question="How does X affect Y?",
        contribution_statement="We provide evidence on X and its effect on Y.",
        gap_analysis_summary="Prior work does not test X in setting Z.",
        literature_synthesis_summary="Studies A and B suggest mechanisms; evidence is mixed.",
        methodology_summary="We estimate baseline OLS models with standard controls.",
    )

    system_prompt = writer.get_system_prompt(context)

    assert "PROJECT WRITING STYLE GUIDE (EXCERPT)" in system_prompt
    assert excerpt in system_prompt
