"""Discussion section writer.

Writes the discussion section:
- Interpretation of findings
- Connection to prior literature
- Limitations
- Alternative explanations
"""

from src.writers.base import BaseSectionWriter, SectionWriterConfig
from src.state.models import SectionWritingContext
from src.style import StyleEnforcer
from src.citations import CitationManager


class DiscussionWriter(BaseSectionWriter):
    """Writer for the discussion section."""
    
    section_type = "discussion"
    section_title = "Discussion"
    
    def __init__(
        self,
        config: SectionWriterConfig | None = None,
        style_enforcer: StyleEnforcer | None = None,
        citation_manager: CitationManager | None = None,
    ):
        """Initialize the discussion writer."""
        super().__init__(config, style_enforcer, citation_manager)
    
    def get_system_prompt(self, context: SectionWritingContext) -> str:
        """Get system prompt for discussion writing."""
        word_target = context.target_word_count or 400
        
        return f"""You are an expert academic writer for finance journals.
You are writing the DISCUSSION section of a {context.paper_type.replace('_', ' ')}.

TARGET JOURNAL: {context.target_journal.upper()}
TARGET WORD COUNT: {word_target} words (approximately)

DISCUSSION STRUCTURE:
1. INTERPRETATION
   - What do the findings mean?
   - Why do you see these results?
   - What is the economic intuition?

2. RELATION TO PRIOR WORK
   - How do findings relate to existing literature?
   - Do they confirm, extend, or contradict prior work?
   - Cite relevant papers

3. LIMITATIONS
   - Acknowledge methodological limitations honestly
   - Data limitations
   - Generalizability concerns
   - What you cannot conclude

4. ALTERNATIVE EXPLANATIONS
   - Could results be driven by something else?
   - Why is your interpretation preferred?

{self._get_common_instructions()}

DISCUSSION-SPECIFIC RULES:
- This is where you INTERPRET (results section just presents)
- Be intellectually honest about limitations
- Do not overstate implications
- Use hedging language appropriately: "suggests", "is consistent with"
- Connect back to the research question
- For short papers, this may be combined with results
- Address potential referee concerns proactively
"""
    
    def get_user_prompt(self, context: SectionWritingContext) -> str:
        """Get user prompt for discussion writing."""
        prompt_parts = [
            "Write the DISCUSSION section based on the following:",
            "",
            f"RESEARCH QUESTION: {context.research_question}",
            "",
            f"KEY FINDINGS: {context.findings_summary}",
            "",
            f"CONTRIBUTION STATEMENT: {context.contribution_statement}",
            "",
            f"LITERATURE CONTEXT: {context.literature_synthesis_summary[:1000] if context.literature_synthesis_summary else 'See introduction'}",
            "",
            self._format_available_citations(context.available_citations),
            "",
            "Write a discussion section that:",
            "1. Interprets the findings in economic terms",
            "2. Connects results to prior literature",
            "3. Acknowledges limitations honestly",
            "4. Addresses alternative explanations",
        ]
        
        return "\n".join(prompt_parts)
