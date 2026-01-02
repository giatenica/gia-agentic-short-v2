"""Conclusion section writer.

Writes the conclusion section:
- Summary of findings
- Contribution restated
- Implications
- Future research directions
"""

from src.writers.base import BaseSectionWriter, SectionWriterConfig
from src.state.models import SectionWritingContext
from src.style import StyleEnforcer
from src.citations import CitationManager


class ConclusionWriter(BaseSectionWriter):
    """Writer for the conclusion section."""
    
    section_type = "conclusion"
    section_title = "Conclusion"
    
    def __init__(
        self,
        config: SectionWriterConfig | None = None,
        style_enforcer: StyleEnforcer | None = None,
        citation_manager: CitationManager | None = None,
    ):
        """Initialize the conclusion writer."""
        super().__init__(config, style_enforcer, citation_manager)
    
    def get_system_prompt(self, context: SectionWritingContext) -> str:
        """Get system prompt for conclusion writing."""
        word_target = context.target_word_count or 300
        
        return f"""You are an expert academic writer for finance journals.
You are writing the CONCLUSION section of a {context.paper_type.replace('_', ' ')}.

TARGET JOURNAL: {context.target_journal.upper()}
TARGET WORD COUNT: {word_target} words (approximately)

CONCLUSION STRUCTURE:
1. SUMMARY (2-3 sentences)
   - Brief restatement of what was done
   - Key findings (high-level, no detailed statistics)

2. CONTRIBUTION (1-2 sentences)
   - What this paper adds to knowledge
   - Must deliver what was promised in introduction

3. IMPLICATIONS (1 paragraph)
   - For practitioners or policy
   - For academic understanding

4. FUTURE RESEARCH (2-3 sentences)
   - Natural extensions
   - Open questions

{self._get_common_instructions()}

CONCLUSION-SPECIFIC RULES:
- DO NOT introduce new findings or arguments
- DO NOT repeat detailed statistics from results
- DO NOT over-claim implications
- Keep it brief and focused
- The contribution delivered must match what was promised in the introduction
- End on a forward-looking note
- For short papers, conclusion should be 0.5-1 page maximum
"""
    
    def get_user_prompt(self, context: SectionWritingContext) -> str:
        """Get user prompt for conclusion writing."""
        # Get prior sections for context
        prior_intro = None
        for section in context.prior_sections:
            if section.section_type == "introduction":
                prior_intro = section.content[:500]
                break
        
        prompt_parts = [
            "Write the CONCLUSION section based on the following:",
            "",
            f"RESEARCH QUESTION: {context.research_question}",
            "",
            f"CONTRIBUTION PROMISED: {context.contribution_statement}",
            "",
            f"KEY FINDINGS: {context.findings_summary}",
            "",
        ]
        
        if prior_intro:
            prompt_parts.extend([
                f"INTRODUCTION OPENING (for consistency): {prior_intro}...",
                "",
            ])
        
        prompt_parts.extend([
            "Write a conclusion that:",
            "1. Briefly summarizes what was done and found",
            "2. Delivers the contribution promised in the introduction",
            "3. States practical and academic implications",
            "4. Suggests directions for future research",
            "",
            "Keep it concise; do not repeat detailed statistics.",
        ])
        
        return "\n".join(prompt_parts)
