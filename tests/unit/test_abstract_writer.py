"""Tests for abstract writer."""

import pytest
from unittest.mock import MagicMock, patch

from src.writers.abstract import AbstractWriter
from src.writers.base import SectionWriterConfig
from src.state.models import SectionWritingContext, PaperSection


class TestAbstractWriter:
    """Tests for AbstractWriter class."""
    
    @pytest.fixture
    def writer(self):
        """Create an AbstractWriter instance."""
        return AbstractWriter()
    
    @pytest.fixture
    def context(self):
        """Create a mock SectionWritingContext."""
        ctx = MagicMock(spec=SectionWritingContext)
        ctx.paper_type = "full_paper"
        ctx.target_journal = "JF"
        ctx.research_question = "How does AI adoption affect firm productivity?"
        ctx.prior_sections = []
        ctx.gap_analysis_summary = ""
        ctx.contribution_statement = ""
        ctx.literature_synthesis_summary = ""
        ctx.methodology_summary = ""
        ctx.findings_summary = ""
        return ctx
    
    def test_initialization(self, writer):
        """Test that AbstractWriter initializes correctly."""
        assert writer.section_type == "abstract"
        assert writer.section_title == "Abstract"
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = SectionWriterConfig(max_tokens=2000)
        writer = AbstractWriter(config=config)
        assert writer.config.max_tokens == 2000
    
    def test_get_system_prompt_short_paper(self, writer):
        """Test system prompt for short paper."""
        context = MagicMock(spec=SectionWritingContext)
        context.paper_type = "short_article"
        context.target_journal = "JF"
        
        prompt = writer.get_system_prompt(context)
        
        assert "150 words" in prompt
        assert "JF" in prompt
        assert "ABSTRACT STRUCTURE" in prompt
        assert "MOTIVATION" in prompt
    
    def test_get_system_prompt_full_paper(self, writer, context):
        """Test system prompt for full paper."""
        prompt = writer.get_system_prompt(context)
        
        assert "250 words" in prompt
        assert "JF" in prompt
    
    def test_get_user_prompt(self, writer, context):
        """Test user prompt generation."""
        context.prior_sections = []
        
        prompt = writer.get_user_prompt(context)
        
        assert "RESEARCH QUESTION" in prompt
        assert "AI adoption" in prompt
    
    def test_get_user_prompt_with_prior_sections(self, writer, context):
        """Test user prompt includes prior section summaries."""
        # Create mock prior sections
        intro_section = MagicMock(spec=PaperSection)
        intro_section.section_type = "introduction"
        intro_section.content = "This paper investigates the impact of AI..."
        
        methods_section = MagicMock(spec=PaperSection)
        methods_section.section_type = "methods"
        methods_section.content = "We use panel data regression to analyze..."
        
        context.prior_sections = [intro_section, methods_section]
        
        prompt = writer.get_user_prompt(context)
        
        # The prompt should reference the research question
        assert "RESEARCH QUESTION" in prompt


class TestAbstractWriterInstructions:
    """Tests for abstract-specific writing instructions."""
    
    @pytest.fixture
    def writer(self):
        return AbstractWriter()
    
    def test_prompt_forbids_citations(self, writer):
        """Test that system prompt forbids citations in abstract."""
        context = MagicMock(spec=SectionWritingContext)
        context.paper_type = "full_paper"
        context.target_journal = "JF"
        
        prompt = writer.get_system_prompt(context)
        
        assert "NEVER use citations" in prompt
    
    def test_prompt_includes_structure(self, writer):
        """Test that system prompt includes required structure."""
        context = MagicMock(spec=SectionWritingContext)
        context.paper_type = "full_paper"
        context.target_journal = "JF"
        
        prompt = writer.get_system_prompt(context)
        
        # All required structural elements
        assert "MOTIVATION" in prompt
        assert "RESEARCH QUESTION" in prompt
        assert "DATA AND METHOD" in prompt
        assert "FINDINGS" in prompt
        assert "CONTRIBUTION" in prompt
    
    def test_prompt_emphasizes_brevity(self, writer):
        """Test that system prompt emphasizes word limits."""
        context = MagicMock(spec=SectionWritingContext)
        context.paper_type = "full_paper"
        context.target_journal = "JF"
        
        prompt = writer.get_system_prompt(context)
        
        assert "strict limit" in prompt
        assert "NEVER exceed" in prompt


class TestAbstractWriterWordCount:
    """Tests for word count targeting."""
    
    def test_short_article_word_target(self):
        """Test word target for short articles."""
        writer = AbstractWriter()
        context = MagicMock(spec=SectionWritingContext)
        context.paper_type = "short_article"
        context.target_journal = "JF"
        
        prompt = writer.get_system_prompt(context)
        
        # Short articles should target 150 words
        assert "150 words" in prompt
    
    def test_full_paper_word_target(self):
        """Test word target for full papers."""
        writer = AbstractWriter()
        context = MagicMock(spec=SectionWritingContext)
        context.paper_type = "full_paper"
        context.target_journal = "JF"
        
        prompt = writer.get_system_prompt(context)
        
        # Full papers should target 250 words
        assert "250 words" in prompt
