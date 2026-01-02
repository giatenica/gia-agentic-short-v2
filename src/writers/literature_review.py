"""Literature review section writer.

For full papers, writes a standalone literature review section.
For short papers, this is typically integrated into the introduction.
"""

from src.writers.base import BaseSectionWriter, SectionWriterConfig
from src.state.models import SectionWritingContext
from src.style import StyleEnforcer
from src.citations import CitationManager


class LiteratureReviewWriter(BaseSectionWriter):
    """Writer for the literature review section."""
    
    section_type = "literature_review"
    section_title = "Literature Review"
    
    def __init__(
        self,
        config: SectionWriterConfig | None = None,
        style_enforcer: StyleEnforcer | None = None,
        citation_manager: CitationManager | None = None,
    ):
        """Initialize the literature review writer."""
        super().__init__(config, style_enforcer, citation_manager)
    
    def get_system_prompt(self, context: SectionWritingContext) -> str:
        """Get system prompt for literature review writing."""
        word_target = context.target_word_count or 2000
        
        return f"""You are an expert academic writer for finance journals.
You are writing the LITERATURE REVIEW section of a {context.paper_type.replace('_', ' ')}.

TARGET JOURNAL: {context.target_journal.upper()}
TARGET WORD COUNT: {word_target} words (approximately)

LITERATURE REVIEW STRUCTURE:
This is a THEMATIC SYNTHESIS, not an annotated bibliography.
Organize by themes, debates, or methodological approaches; not by author.

STRUCTURE OPTIONS:
1. Thematic: Group studies by topic or finding
2. Chronological: Show evolution of thinking (rarely preferred)
3. Methodological: Group by research approach
4. Theoretical: Organize around theoretical frameworks

{self._get_common_instructions()}

LITERATURE REVIEW-SPECIFIC RULES:
- DO NOT just list studies; synthesize and critique
- Show how studies relate to each other
- Identify agreements, disagreements, and gaps
- Build toward your research question naturally
- Use transitions between paragraphs to show logical flow
- End by identifying the gap your paper addresses
- Balance coverage; don't over-cite one author
- Include seminal works and recent developments
- Use present tense: "Fama (1970) shows that..."
"""
    
    def get_user_prompt(self, context: SectionWritingContext) -> str:
        """Get user prompt for literature review writing."""
        prompt_parts = [
            "Write the LITERATURE REVIEW section based on the following:",
            "",
            f"RESEARCH QUESTION: {context.research_question}",
            "",
            f"LITERATURE SYNTHESIS (themes and findings):",
            f"{context.literature_synthesis_summary}",
            "",
            f"IDENTIFIED GAPS: {context.gap_analysis_summary}",
            "",
            self._format_available_citations(context.available_citations),
            "",
            "Write a thematic literature review that:",
            "1. Synthesizes existing research into coherent themes",
            "2. Shows how studies relate and build on each other",
            "3. Identifies debates or contradictions in the literature",
            "4. Builds toward the gap your paper addresses",
            "5. Uses appropriate citations throughout",
        ]
        
        return "\n".join(prompt_parts)
