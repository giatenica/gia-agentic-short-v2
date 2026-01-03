"""Results section writer.

Writes the results/findings section:
- Main results with statistical significance
- Economic magnitude interpretation
- Robustness checks
- DOES NOT include interpretation (that's discussion)

Sprint 16: Enhanced with table/figure reference generation.
"""

from src.writers.base import BaseSectionWriter, SectionWriterConfig
from src.state.models import SectionWritingContext
from src.style import StyleEnforcer
from src.citations import CitationManager
from src.writers.artifact_helpers import generate_results_artifacts_prompt


class ResultsWriter(BaseSectionWriter):
    """Writer for the results section."""
    
    section_type = "results"
    section_title = "Results"
    
    def __init__(
        self,
        config: SectionWriterConfig | None = None,
        style_enforcer: StyleEnforcer | None = None,
        citation_manager: CitationManager | None = None,
    ):
        """Initialize the results writer."""
        super().__init__(config, style_enforcer, citation_manager)
    
    def get_system_prompt(self, context: SectionWritingContext) -> str:
        """Get system prompt for results writing."""
        word_target = context.target_word_count or 1000
        
        # Adjust for research type
        if context.research_type == "theoretical":
            section_name = "Findings"
            specific_instructions = """
THEORETICAL FINDINGS STRUCTURE:
1. Key propositions or insights from the framework
2. Theoretical predictions and their derivation
3. How the framework addresses the research question
4. Boundary conditions and scope
"""
        else:
            section_name = "Results"
            specific_instructions = """
EMPIRICAL RESULTS STRUCTURE:
1. SUMMARY STATISTICS
   - Reference Table X for descriptive statistics
   - Highlight key patterns in the data

2. MAIN RESULTS
   - Reference main regression table
   - Report coefficient magnitude and significance
   - Interpret economic magnitude (not just statistical)
   - Example: "A one standard deviation increase in X is associated 
     with a Y% change in the dependent variable"

3. ROBUSTNESS (for short papers, can be brief or in appendix)
   - Alternative specifications
   - Subsample analysis
   - Sensitivity tests

PRESENTATION ORDER:
- Summary statistics
- Main results
- Economic magnitude
- Robustness checks
"""
        
        return f"""You are an expert academic writer for finance journals.
You are writing the {section_name} section of a {context.paper_type.replace('_', ' ')}.

TARGET JOURNAL: {context.target_journal.upper()}
TARGET WORD COUNT: {word_target} words (approximately)

{specific_instructions}

{self._get_common_instructions()}

RESULTS-SPECIFIC RULES:
- PRESENT findings; DO NOT interpret them (interpretation is for Discussion)
- Report both statistical significance AND economic magnitude
- Reference tables and figures: "Table 2 reports..." or "As shown in Column (3)..."
- Use precise numbers: "0.045 (t=3.21)" not "significant"
- Significance stars: *** p<0.01, ** p<0.05, * p<0.1
- Do not overclaim; let the data speak
- For null results, report them honestly
- Use past/present mix: "The results show that..." or "We find that..."
"""
    
    def get_user_prompt(self, context: SectionWritingContext) -> str:
        """Get user prompt for results writing."""
        prompt_parts = [
            "Write the RESULTS section based on the following:",
            "",
            f"RESEARCH QUESTION: {context.research_question}",
            "",
            f"KEY FINDINGS:",
            f"{context.findings_summary}",
            "",
            f"METHODOLOGY USED: {context.methodology_summary[:500] if context.methodology_summary else 'See methods section'}",
            "",
        ]
        
        # Sprint 16: Add table and figure artifacts
        artifacts_prompt = generate_results_artifacts_prompt(
            tables=context.tables,
            figures=context.figures,
        )
        if artifacts_prompt:
            prompt_parts.extend([artifacts_prompt, ""])
        
        if context.has_quantitative_results:
            prompt_parts.extend([
                "RESULT TYPE: Quantitative/Empirical",
                "- Include statistical significance and economic magnitude",
                "- Reference tables for detailed results",
                "",
            ])
        elif context.has_qualitative_results:
            prompt_parts.extend([
                "RESULT TYPE: Qualitative",
                "- Present key themes and patterns",
                "- Use evidence from the analysis",
                "",
            ])
        else:
            prompt_parts.extend([
                f"RESULT TYPE: {context.research_type}",
                "- Present findings from the analysis",
                "",
            ])
        
        prompt_parts.extend([
            "Write the complete results section.",
            "Focus on PRESENTING findings, not interpreting them.",
            "Report both statistical significance and economic magnitude where applicable.",
        ])
        
        return "\n".join(prompt_parts)
