"""Abstract section writer.

Writes the abstract LAST, after all other sections are complete.
Summarizes the entire paper in structured format.
"""

from src.writers.base import BaseSectionWriter, SectionWriterConfig
from src.state.models import SectionWritingContext
from src.style import StyleEnforcer
from src.citations import CitationManager


class AbstractWriter(BaseSectionWriter):
    """Writer for the abstract section.
    
    The abstract is written LAST to ensure it accurately
    reflects the final content of all other sections.
    """
    
    section_type = "abstract"
    section_title = "Abstract"
    
    def __init__(
        self,
        config: SectionWriterConfig | None = None,
        style_enforcer: StyleEnforcer | None = None,
        citation_manager: CitationManager | None = None,
    ):
        """Initialize the abstract writer."""
        super().__init__(config, style_enforcer, citation_manager)
    
    def get_system_prompt(self, context: SectionWritingContext) -> str:
        """Get system prompt for abstract writing."""
        # Abstracts are typically 100-150 words for short papers, 200-250 for full
        word_target = 150 if context.paper_type == "short_article" else 250
        
        return f"""You are an expert academic writer for finance journals.
You are writing the ABSTRACT for a {context.paper_type.replace('_', ' ')}.

TARGET JOURNAL: {context.target_journal.upper()}
TARGET WORD COUNT: {word_target} words (strict limit)

ABSTRACT STRUCTURE (Five Sentences Maximum):
1. MOTIVATION (1 sentence)
   - Why does this question matter?
   - What is the gap or puzzle?

2. RESEARCH QUESTION (1 sentence)
   - Clear statement of what you investigate

3. DATA AND METHOD (1 sentence)
   - Brief description of empirical approach

4. FINDINGS (1-2 sentences)
   - Key results only
   - One specific number if space permits

5. CONTRIBUTION (1 sentence)
   - What this adds to knowledge

{self._get_common_instructions()}

ABSTRACT-SPECIFIC RULES:
- NEVER exceed {word_target} words
- NEVER use citations in abstract
- NEVER use jargon without brief explanation
- Write in present tense for findings, past for methods
- Must be self-contained and understandable alone
- Each sentence serves a distinct purpose
- Front-load importance; start with why it matters
- The abstract sells the paper; make every word count
"""
    
    def get_user_prompt(self, context: SectionWritingContext) -> str:
        """Get user prompt for abstract writing."""
        # Abstract needs summaries of all prior sections
        intro_summary = ""
        methods_summary = ""
        results_summary = ""
        discussion_summary = ""
        conclusion_summary = ""
        
        for section in context.prior_sections:
            content_preview = section.content[:300] if section.content else ""
            if section.section_type == "introduction":
                intro_summary = content_preview
            elif section.section_type == "methods":
                methods_summary = content_preview
            elif section.section_type == "results":
                results_summary = content_preview
            elif section.section_type == "discussion":
                discussion_summary = content_preview
            elif section.section_type == "conclusion":
                conclusion_summary = content_preview
        
        prompt_parts = [
            "Write the ABSTRACT for the following paper:",
            "",
            f"RESEARCH QUESTION: {context.research_question}",
            "",
            f"CONTRIBUTION: {context.contribution_statement}",
            "",
            f"KEY FINDINGS: {context.findings_summary}",
            "",
        ]
        
        if intro_summary:
            prompt_parts.extend([
                f"INTRODUCTION PREVIEW: {intro_summary}...",
                "",
            ])
        
        if methods_summary:
            prompt_parts.extend([
                f"METHODS PREVIEW: {methods_summary}...",
                "",
            ])
        
        if results_summary:
            prompt_parts.extend([
                f"RESULTS PREVIEW: {results_summary}...",
                "",
            ])
        
        if conclusion_summary:
            prompt_parts.extend([
                f"CONCLUSION PREVIEW: {conclusion_summary}...",
                "",
            ])
        
        prompt_parts.extend([
            "Write an abstract that:",
            "1. Opens with motivation (why does this matter?)",
            "2. States the research question clearly",
            "3. Briefly describes data and method",
            "4. Highlights key findings with specificity",
            "5. States the contribution to knowledge",
            "",
            "Target exactly 5 sentences, one for each element.",
            "Do NOT include citations.",
        ])
        
        return "\n".join(prompt_parts)
