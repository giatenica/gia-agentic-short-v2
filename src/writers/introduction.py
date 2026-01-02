"""Introduction section writer.

Writes the introduction section with:
- Opening hook
- Research question
- Preview of findings
- Contribution statement
- Literature positioning
- Paper roadmap
"""

from src.writers.base import BaseSectionWriter, SectionWriterConfig
from src.state.models import SectionWritingContext
from src.style import StyleEnforcer
from src.citations import CitationManager


class IntroductionWriter(BaseSectionWriter):
    """Writer for the introduction section."""
    
    section_type = "introduction"
    section_title = "Introduction"
    
    def __init__(
        self,
        config: SectionWriterConfig | None = None,
        style_enforcer: StyleEnforcer | None = None,
        citation_manager: CitationManager | None = None,
    ):
        """Initialize the introduction writer."""
        super().__init__(config, style_enforcer, citation_manager)
    
    def get_system_prompt(self, context: SectionWritingContext) -> str:
        """Get system prompt for introduction writing."""
        word_target = context.target_word_count or 650
        
        return f"""You are an expert academic writer for finance journals.
You are writing the INTRODUCTION section of a {context.paper_type.replace('_', ' ')}.

TARGET JOURNAL: {context.target_journal.upper()}
TARGET WORD COUNT: {word_target} words (approximately)

INTRODUCTION STRUCTURE (in this order):
1. OPENING HOOK (1-2 sentences): Why this research matters; economic motivation or puzzle
2. RESEARCH QUESTION (1 paragraph): Precisely what you are investigating
3. PREVIEW OF FINDINGS (1 paragraph): Main results, briefly stated
4. CONTRIBUTION (1 paragraph): How this advances knowledge; be specific
5. LITERATURE POSITIONING (1-2 paragraphs): Where this fits in existing work
6. PAPER ROADMAP (1-2 sentences): Structure of the rest of the paper

{self._get_common_instructions()}

INTRODUCTION-SPECIFIC RULES:
- Lead with economic motivation, not technical details
- State contribution clearly and specifically; avoid vague claims
- The introduction should be self-contained; a reader should understand 
  the paper's contribution without reading further
- For short papers, integrate literature review here; no separate section
- End with a brief roadmap: "The remainder of this paper proceeds as follows..."
"""
    
    def get_user_prompt(self, context: SectionWritingContext) -> str:
        """Get user prompt for introduction writing."""
        prompt_parts = [
            "Write the INTRODUCTION section based on the following research context:",
            "",
            f"RESEARCH QUESTION: {context.research_question}",
            "",
            f"CONTRIBUTION STATEMENT: {context.contribution_statement}",
            "",
            f"GAP ANALYSIS: {context.gap_analysis_summary}",
            "",
            f"LITERATURE SYNTHESIS: {context.literature_synthesis_summary[:2000] if context.literature_synthesis_summary else 'Not provided'}",
            "",
            f"METHODOLOGY SUMMARY: {context.methodology_summary}",
            "",
        ]
        
        if context.findings_summary:
            prompt_parts.extend([
                f"KEY FINDINGS TO PREVIEW: {context.findings_summary}",
                "",
            ])
        
        prompt_parts.extend([
            self._format_available_citations(context.available_citations),
            "",
            "Write the complete introduction section. Use citations where appropriate.",
            "Follow the structure: Hook -> Research Question -> Preview -> Contribution -> Literature -> Roadmap",
        ])
        
        return "\n".join(prompt_parts)
