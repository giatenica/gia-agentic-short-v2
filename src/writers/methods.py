"""Methods section writer.

Writes the data and methodology section with:
- Data sources and sample construction
- Variable definitions
- Econometric specification
- Identification strategy
"""

from src.writers.base import BaseSectionWriter, SectionWriterConfig
from src.state.models import SectionWritingContext
from src.style import StyleEnforcer
from src.citations import CitationManager


class MethodsWriter(BaseSectionWriter):
    """Writer for the methods/methodology section."""
    
    section_type = "methods"
    section_title = "Data and Methodology"
    
    def __init__(
        self,
        config: SectionWriterConfig | None = None,
        style_enforcer: StyleEnforcer | None = None,
        citation_manager: CitationManager | None = None,
    ):
        """Initialize the methods writer."""
        super().__init__(config, style_enforcer, citation_manager)
    
    def get_system_prompt(self, context: SectionWritingContext) -> str:
        """Get system prompt for methods writing."""
        word_target = context.target_word_count or 550
        
        # Adjust for research type
        if context.research_type == "theoretical":
            section_name = "Methodology and Framework"
            specific_instructions = """
THEORETICAL PAPER STRUCTURE:
1. Overview of the theoretical approach
2. Key assumptions and their justification
3. Model specification or framework construction
4. Analytical methods used
5. Scope and limitations of the approach
"""
        else:
            section_name = "Data and Methodology"
            specific_instructions = """
EMPIRICAL PAPER STRUCTURE:
1. DATA DESCRIPTION
   - Source(s) with citations
   - Sample period
   - Sample construction and filters
   - Final sample size (N)

2. VARIABLE DEFINITIONS
   - Dependent variable(s)
   - Key independent variable(s)
   - Control variables
   - Reference appendix for detailed definitions

3. METHODOLOGY
   - Econometric specification (with equation if helpful)
   - Why this method is appropriate
   - Identification strategy (how you establish causality)
   - Standard errors clustering

4. LIMITATIONS
   - Brief acknowledgment of methodological limitations
"""
        
        return f"""You are an expert academic writer for finance journals.
You are writing the {section_name} section of a {context.paper_type.replace('_', ' ')}.

TARGET JOURNAL: {context.target_journal.upper()}
TARGET WORD COUNT: {word_target} words (approximately)

{specific_instructions}

{self._get_common_instructions()}

METHODS-SPECIFIC RULES:
- Be precise about sample construction; readers should be able to replicate
- Define all variables explicitly
- Justify methodological choices with citations where appropriate
- Acknowledge limitations honestly
- Use past tense: "We collected data from..."
- For short papers, be concise; move details to online appendix
- Include equation for main regression if applicable
"""
    
    def get_user_prompt(self, context: SectionWritingContext) -> str:
        """Get user prompt for methods writing."""
        prompt_parts = [
            "Write the DATA AND METHODOLOGY section based on the following:",
            "",
            f"RESEARCH QUESTION: {context.research_question}",
            "",
            f"METHODOLOGY FROM RESEARCH PLAN:",
            f"{context.methodology_summary}",
            "",
        ]
        
        if context.has_quantitative_results:
            prompt_parts.extend([
                "DATA CHARACTERISTICS:",
                "- This is an EMPIRICAL study with quantitative data",
                "- Include data sources, sample period, and variable definitions",
                "",
            ])
        elif context.has_qualitative_results:
            prompt_parts.extend([
                "DATA CHARACTERISTICS:",
                "- This is a QUALITATIVE study",
                "- Describe data collection and analysis approach",
                "",
            ])
        else:
            prompt_parts.extend([
                "RESEARCH TYPE:",
                f"- This is a {context.research_type} study",
                "- Describe the analytical framework and methods",
                "",
            ])
        
        prompt_parts.extend([
            self._format_available_citations(context.available_citations),
            "",
            "Write the complete methods section following the structure above.",
            "Be precise and detailed enough for replication.",
        ])
        
        return "\n".join(prompt_parts)
