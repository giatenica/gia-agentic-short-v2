"""Base section writer class.

Provides common functionality for all section writers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import settings
from src.state.models import (
    PaperSection,
    CitationEntry,
    SectionWritingContext,
    get_section_word_count_target,
)
from src.style import StyleEnforcer
from src.citations import CitationManager


@dataclass
class SectionWriterConfig:
    """Configuration for section writers."""
    
    model_name: str = field(default_factory=lambda: settings.default_model)
    temperature: float = 0.3
    max_tokens: int = 4096
    target_journal: str = "generic"
    paper_type: str = "short_article"
    enforce_style: bool = True
    auto_fix_style: bool = False


class BaseSectionWriter(ABC):
    """
    Base class for all section writers.
    
    Provides common functionality:
    - LLM invocation
    - Style enforcement
    - Citation management
    - Word count tracking
    """
    
    # Section type (override in subclasses)
    section_type: str = "generic"
    section_title: str = "Section"
    
    def __init__(
        self,
        config: SectionWriterConfig | None = None,
        style_enforcer: StyleEnforcer | None = None,
        citation_manager: CitationManager | None = None,
    ):
        """
        Initialize the section writer.
        
        Args:
            config: Writer configuration.
            style_enforcer: StyleEnforcer instance.
            citation_manager: CitationManager instance.
        """
        self.config = config or SectionWriterConfig()
        
        # Initialize style enforcer
        self.style_enforcer = style_enforcer or StyleEnforcer(
            target_journal=self.config.target_journal,
        )
        
        # Initialize citation manager
        self.citation_manager = citation_manager or CitationManager()
        
        # Initialize LLM
        self._llm = ChatAnthropic(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=settings.anthropic_api_key,
        )
    
    @abstractmethod
    def get_system_prompt(self, context: SectionWritingContext) -> str:
        """
        Get the system prompt for this section writer.
        
        Args:
            context: Writing context.
            
        Returns:
            System prompt string.
        """
        pass
    
    @abstractmethod
    def get_user_prompt(self, context: SectionWritingContext) -> str:
        """
        Get the user prompt for this section writer.
        
        Args:
            context: Writing context.
            
        Returns:
            User prompt string.
        """
        pass
    
    def write(self, context: SectionWritingContext) -> PaperSection:
        """
        Write the section.
        
        Args:
            context: Writing context with all necessary information.
            
        Returns:
            PaperSection with content.
        """
        # Get prompts
        system_prompt = self.get_system_prompt(context)
        user_prompt = self.get_user_prompt(context)
        
        # Invoke LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        
        response = self._llm.invoke(messages)
        # Handle both string and list content types from response
        raw_content = response.content
        if isinstance(raw_content, list):
            # Extract text from structured content
            content = " ".join(
                item if isinstance(item, str) else item.get("text", "")
                for item in raw_content
            )
        else:
            content = raw_content
        
        # Post-process
        content = self._post_process(content, context)
        
        # Check style
        violations = []
        if self.config.enforce_style:
            violations = self.style_enforcer.check(content, section_name=self.section_type)
            
            # Auto-fix if enabled
            if self.config.auto_fix_style:
                content, _ = self.style_enforcer.auto_fix(content)
                # Re-check after fixing
                violations = self.style_enforcer.check(content, section_name=self.section_type)
        
        # Extract citations used
        citations_used = self.citation_manager.extract_and_record_citations(
            content, self.section_type
        )
        
        # Calculate word count
        word_count = len(content.split())
        
        # Get target word count
        target_range = get_section_word_count_target(
            context.paper_type, self.section_type
        )
        target_word_count = None
        if target_range:
            # Use midpoint of range
            target_word_count = (target_range[0] + target_range[1]) // 2
        
        # Create section
        section = PaperSection(
            section_type=self.section_type,
            title=self.section_title,
            content=content,
            order=self._get_section_order(),
            word_count=word_count,
            target_word_count=target_word_count,
            citations_used=citations_used,
            status="draft_complete",
            style_violations=violations,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        return section
    
    def _post_process(self, content: str, context: SectionWritingContext) -> str:
        """
        Post-process the generated content.
        
        Args:
            content: Generated content.
            context: Writing context.
            
        Returns:
            Processed content.
        """
        # Remove any markdown headers if present (we add our own structure)
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            # Skip lines that are just section headers matching our section
            if line.strip().lower().startswith(f"# {self.section_title.lower()}"):
                continue
            if line.strip().lower().startswith(f"## {self.section_title.lower()}"):
                continue
            processed_lines.append(line)
        
        content = '\n'.join(processed_lines).strip()
        
        return content
    
    def _get_section_order(self) -> int:
        """Get the order of this section in the paper."""
        order_map = {
            "abstract": 0,
            "introduction": 1,
            "literature_review": 2,
            "methods": 3,
            "data": 4,
            "results": 5,
            "discussion": 6,
            "conclusion": 7,
            "references": 8,
            "appendix": 9,
        }
        return order_map.get(self.section_type, 99)
    
    def _get_common_instructions(self) -> str:
        """Get common writing instructions for all sections."""
        return """
CRITICAL WRITING RULES:
1. NEVER use em dashes or en dashes; use semicolons, colons, or periods instead
2. NEVER use these banned words: delve, leverage, utilize, novel, unique, 
   cutting-edge, paradigm, transformative, groundbreaking, innovative
3. Use specific numbers and percentages instead of vague terms like "many" or "some"
4. Use hedging language for claims: "suggests", "indicates", "is consistent with"
5. Cite sources using Chicago Author-Date format: (Author Year)
6. Write in active voice where possible
7. Be precise and concise; every sentence should add value
8. Do not overclaim; use appropriate qualification
"""
    
    def _format_available_citations(self, citations: list[CitationEntry]) -> str:
        """Format available citations for the prompt."""
        if not citations:
            return "No specific citations provided."
        
        lines = ["Available citations:"]
        for c in citations[:20]:  # Limit to 20 citations
            lines.append(f"- {c.key}: {c.authors[0] if c.authors else 'Unknown'} ({c.year}). {c.title[:60]}...")
        
        return "\n".join(lines)
