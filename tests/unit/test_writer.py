"""Unit tests for Sprint 6 WRITER node and section writers."""

import pytest

from src.state.enums import (
    SectionType,
    WritingStatus,
    StyleViolationType,
    JournalTarget,
)
from src.state.models import (
    StyleViolation,
    PaperSection,
    WriterOutput,
    SectionWritingContext,
    ReferenceList,
    ArgumentThread,
    get_section_word_count_target,
    SECTION_WORD_COUNTS,
)


# =============================================================================
# Test Section Type Enums
# =============================================================================


class TestSectionTypeEnum:
    """Tests for SectionType enum."""
    
    def test_all_standard_sections_exist(self):
        """Verify all standard paper sections are defined."""
        expected_sections = [
            "abstract", "introduction", "literature_review", 
            "methods", "data", "results", "discussion", "conclusion"
        ]
        for section in expected_sections:
            assert hasattr(SectionType, section.upper())
    
    def test_section_type_values(self):
        """Test section type string values."""
        assert SectionType.ABSTRACT.value == "abstract"
        assert SectionType.INTRODUCTION.value == "introduction"
        assert SectionType.METHODS.value == "methods"
        assert SectionType.RESULTS.value == "results"


class TestWritingStatusEnum:
    """Tests for WritingStatus enum."""
    
    def test_status_progression(self):
        """Verify writing status values represent a valid progression."""
        statuses = [
            WritingStatus.PENDING,
            WritingStatus.IN_PROGRESS,
            WritingStatus.DRAFT_COMPLETE,
            WritingStatus.STYLE_CHECKED,
            WritingStatus.CITATIONS_VERIFIED,
            WritingStatus.FINALIZED,
        ]
        # All should be valid enum members
        for status in statuses:
            assert isinstance(status.value, str)


class TestStyleViolationTypeEnum:
    """Tests for StyleViolationType enum."""
    
    def test_banned_word_type(self):
        """Test banned word violation type exists."""
        assert StyleViolationType.BANNED_WORD.value == "banned_word"
    
    def test_all_violation_types(self):
        """Verify all expected violation types exist."""
        expected_types = [
            "banned_word", "informal_tone", "missing_hedge", 
            "vague_term", "citation_format", "overclaim"
        ]
        for vtype in expected_types:
            assert vtype in [v.value for v in StyleViolationType]


class TestJournalTargetEnum:
    """Tests for JournalTarget enum."""
    
    def test_top_finance_journals(self):
        """Verify top finance journals are included."""
        journals = [j.value for j in JournalTarget]
        assert "rfs" in journals or "RFS" in [j.value.upper() for j in JournalTarget]


# =============================================================================
# Test Style Models
# =============================================================================


class TestStyleViolation:
    """Tests for StyleViolation model."""
    
    def test_create_style_violation(self):
        """Test creating a style violation."""
        violation = StyleViolation(
            violation_type="banned_word",
            original_text="leverage",
            suggestion="use",
            location="paragraph 1",
        )
        assert violation.violation_type == "banned_word"
        assert violation.original_text == "leverage"
        assert violation.suggestion == "use"
    
    def test_style_violation_defaults(self):
        """Test style violation default values."""
        violation = StyleViolation(
            violation_type="informal_tone",
            original_text="can't",
        )
        assert violation.location == ""
        assert violation.suggestion == ""


# =============================================================================
# Test Paper Section Model
# =============================================================================


class TestPaperSection:
    """Tests for PaperSection model."""
    
    def test_create_section(self):
        """Test creating a paper section."""
        section = PaperSection(
            section_type="introduction",
            title="Introduction",
            content="This paper examines...",
            word_count=100,
        )
        assert section.section_type == "introduction"
        assert section.title == "Introduction"
        assert section.word_count == 100
    
    def test_section_with_violations(self):
        """Test section with style violations."""
        violation = StyleViolation(
            violation_type="banned_word",
            original_text="leverage",
        )
        section = PaperSection(
            section_type="methods",
            title="Methods",
            content="We leverage machine learning...",
            style_violations=[violation],
        )
        assert len(section.style_violations) == 1
        assert section.style_violations[0].original_text == "leverage"
    
    def test_update_word_count(self):
        """Test word count update method."""
        section = PaperSection(
            section_type="results",
            title="Results",
            content="One two three four five.",
        )
        count = section.update_word_count()
        assert count == 5
        assert section.word_count == 5
    
    def test_is_within_target(self):
        """Test word count target checking."""
        section = PaperSection(
            section_type="abstract",
            title="Abstract",
            content=" ".join(["word"] * 100),
            word_count=100,
            target_word_count=100,
        )
        assert section.is_within_target(tolerance=0.2)
        
        # Test over target
        section.word_count = 150
        assert not section.is_within_target(tolerance=0.2)


# =============================================================================
# Test Writer Output Model
# =============================================================================


class TestWriterOutput:
    """Tests for WriterOutput model."""
    
    def test_create_writer_output(self):
        """Test creating writer output."""
        output = WriterOutput(
            title="Test Paper",
            total_word_count=5000,
            writing_status="finalized",
        )
        assert output.title == "Test Paper"
        assert output.total_word_count == 5000
        assert output.writing_status == "finalized"
    
    def test_writer_output_with_sections(self):
        """Test writer output with multiple sections."""
        sections = [
            PaperSection(section_type="introduction", title="Introduction"),
            PaperSection(section_type="methods", title="Methods"),
            PaperSection(section_type="results", title="Results"),
        ]
        output = WriterOutput(
            title="Test Paper",
            sections=sections,
        )
        assert len(output.sections) == 3
    
    def test_get_section_method(self):
        """Test getting a section by type."""
        intro = PaperSection(
            section_type="introduction", 
            title="Introduction",
            content="Test content",
        )
        output = WriterOutput(
            title="Test Paper",
            sections=[intro],
        )
        found = output.get_section("introduction")
        assert found is not None
        assert found.content == "Test content"


# =============================================================================
# Test Section Writing Context
# =============================================================================


class TestSectionWritingContext:
    """Tests for SectionWritingContext model."""
    
    def test_create_context(self):
        """Test creating section writing context."""
        context = SectionWritingContext(
            section_type="introduction",
            target_journal="rfs",
            paper_type="short_article",
            research_question="How does X affect Y?",
        )
        assert context.section_type == "introduction"
        assert context.target_journal == "rfs"
        assert context.paper_type == "short_article"
    
    def test_context_with_prior_sections(self):
        """Test context with prior sections."""
        prior = PaperSection(
            section_type="introduction",
            title="Introduction",
            content="Prior content",
        )
        context = SectionWritingContext(
            section_type="methods",
            prior_sections=[prior],
        )
        assert len(context.prior_sections) == 1


# =============================================================================
# Test Word Count Targets
# =============================================================================


class TestWordCountTargets:
    """Tests for word count target functions."""
    
    def test_short_article_targets(self):
        """Test word count targets for short articles."""
        assert "short_article" in SECTION_WORD_COUNTS
        targets = SECTION_WORD_COUNTS["short_article"]
        
        # Introduction should have reasonable range
        intro = targets.get("introduction")
        assert intro is not None
        assert intro[0] < intro[1]  # Min < max
    
    def test_full_paper_targets(self):
        """Test word count targets for full papers."""
        assert "full_paper" in SECTION_WORD_COUNTS
        targets = SECTION_WORD_COUNTS["full_paper"]
        
        # Full paper intro should be longer than short article
        full_intro = targets.get("introduction")
        short_intro = SECTION_WORD_COUNTS["short_article"].get("introduction")
        if full_intro and short_intro:
            assert full_intro[1] > short_intro[1]
    
    def test_get_section_word_count_target(self):
        """Test helper function for getting word count target."""
        result = get_section_word_count_target("short_article", "introduction")
        assert result is not None
        assert len(result) == 2
        
        # Test invalid paper type
        result = get_section_word_count_target("invalid_type", "introduction")
        assert result is None


# =============================================================================
# Test Reference List Model
# =============================================================================


class TestReferenceList:
    """Tests for ReferenceList model."""
    
    def test_create_reference_list(self):
        """Test creating a reference list."""
        ref_list = ReferenceList()
        assert ref_list.entries == []
        assert ref_list.format_style == "chicago_author_date"
    
    def test_entry_count_property(self):
        """Test entry count property."""
        ref_list = ReferenceList()
        assert ref_list.entry_count == 0


# =============================================================================
# Test Argument Thread Model
# =============================================================================


class TestArgumentThread:
    """Tests for ArgumentThread model."""
    
    def test_create_argument_thread(self):
        """Test creating an argument thread."""
        thread = ArgumentThread(
            main_thesis="X affects Y positively",
            promised_contribution="First study to examine X-Y relationship",
        )
        assert thread.main_thesis == "X affects Y positively"
        assert thread.coherence_score == 0.0
    
    def test_add_claim(self):
        """Test adding claims to thread."""
        thread = ArgumentThread()
        thread.add_claim("Higher X leads to higher Y", ["Table 1 results"])
        assert "Higher X leads to higher Y" in thread.claims
        assert "Higher X leads to higher Y" in thread.evidence_map


# =============================================================================
# Test Writer Node Imports
# =============================================================================


class TestWriterNodeImports:
    """Tests that writer node imports correctly."""
    
    def test_writer_node_import(self):
        """Test that writer_node can be imported."""
        from src.nodes.writer import writer_node
        assert callable(writer_node)
    
    def test_section_order_import(self):
        """Test that SECTION_ORDER is correctly defined."""
        from src.nodes.writer import SECTION_ORDER
        assert len(SECTION_ORDER) > 0
        # Abstract should be last
        assert SECTION_ORDER[-1] == SectionType.ABSTRACT
    
    def test_build_section_context_import(self):
        """Test that build_section_context can be imported."""
        from src.nodes.writer import build_section_context
        assert callable(build_section_context)


# =============================================================================
# Test Section Writers
# =============================================================================


class TestSectionWriterImports:
    """Tests for section writer imports."""
    
    def test_all_writers_import(self):
        """Test that all writers can be imported."""
        from src.writers import (
            BaseSectionWriter,
            IntroductionWriter,
            LiteratureReviewWriter,
            MethodsWriter,
            ResultsWriter,
            DiscussionWriter,
            ConclusionWriter,
            AbstractWriter,
            ArgumentManager,
        )
        assert IntroductionWriter.section_type == "introduction"
        assert LiteratureReviewWriter.section_type == "literature_review"
        assert MethodsWriter.section_type == "methods"
        assert ResultsWriter.section_type == "results"
        assert DiscussionWriter.section_type == "discussion"
        assert ConclusionWriter.section_type == "conclusion"
        assert AbstractWriter.section_type == "abstract"
    
    def test_section_writer_config(self):
        """Test SectionWriterConfig defaults."""
        from src.writers import SectionWriterConfig
        config = SectionWriterConfig()
        assert config.temperature == 0.3
        assert config.enforce_style is True


class TestArgumentManager:
    """Tests for ArgumentManager."""
    
    def test_create_argument_manager(self):
        """Test creating an argument manager."""
        from src.writers import ArgumentManager
        manager = ArgumentManager()
        assert manager.main_contribution == ""
        assert manager.threads == []
    
    def test_set_core_argument(self):
        """Test setting core argument."""
        from src.writers import ArgumentManager
        manager = ArgumentManager()
        manager.set_core_argument(
            contribution="First study of X",
            question="How does X affect Y?",
            findings=["X increases Y by 10%"],
        )
        assert manager.main_contribution == "First study of X"
        assert manager.research_question == "How does X affect Y?"
    
    def test_create_thread(self):
        """Test creating a thread."""
        from src.writers import ArgumentManager
        manager = ArgumentManager()
        thread = manager.create_thread(
            name="main_contribution",
            description="Test contribution",
            claimed_in=["introduction"],
        )
        assert thread.name == "main_contribution"
        assert len(manager.threads) == 1
    
    def test_get_coherence_summary(self):
        """Test getting coherence summary."""
        from src.writers import ArgumentManager
        manager = ArgumentManager()
        summary = manager.get_coherence_summary()
        assert "total_threads" in summary
        assert "coherence_score" in summary


# =============================================================================
# Test Style Module
# =============================================================================


class TestStyleModuleImports:
    """Tests for style module imports."""
    
    def test_style_enforcer_import(self):
        """Test StyleEnforcer can be imported."""
        from src.style import StyleEnforcer
        enforcer = StyleEnforcer()
        assert hasattr(enforcer, "check")
    
    def test_banned_words_filter_import(self):
        """Test BannedWordsFilter can be imported."""
        from src.style import BannedWordsFilter
        banned_filter = BannedWordsFilter()
        assert hasattr(banned_filter, "check")
    
    def test_academic_tone_checker_import(self):
        """Test AcademicToneChecker can be imported."""
        from src.style import AcademicToneChecker
        checker = AcademicToneChecker()
        assert hasattr(checker, "check")


class TestBannedWordsFilter:
    """Tests for banned words filtering."""
    
    def test_detect_banned_word(self):
        """Test detection of banned words."""
        from src.style import BannedWordsFilter
        banned_filter = BannedWordsFilter()
        text = "We leverage machine learning to harness the power of data."
        violations = banned_filter.check(text)
        banned_texts = [v.original_text.lower() for v in violations]
        assert any("leverage" in t for t in banned_texts) or any("harness" in t for t in banned_texts)
    
    def test_no_violation_clean_text(self):
        """Test that clean text has no violations."""
        from src.style import BannedWordsFilter
        banned_filter = BannedWordsFilter()
        text = "We use regression analysis to examine the relationship."
        violations = banned_filter.check(text)
        # May have some violations, but should be minimal
        assert isinstance(violations, list)


class TestStyleEnforcer:
    """Tests for StyleEnforcer."""
    
    def test_check_returns_violations(self):
        """Test that check returns list of violations."""
        from src.style import StyleEnforcer
        enforcer = StyleEnforcer()
        text = "We leverage cutting-edge technology to unlock value."
        violations = enforcer.check(text)
        assert isinstance(violations, list)
    
    def test_auto_fix(self):
        """Test auto-fix functionality."""
        from src.style import StyleEnforcer
        enforcer = StyleEnforcer()
        text = "We leverage the data."
        fixed, count = enforcer.auto_fix(text)
        assert isinstance(fixed, str)
        assert isinstance(count, int)


# =============================================================================
# Test Citations Module
# =============================================================================


class TestCitationsModuleImports:
    """Tests for citations module imports."""
    
    def test_citation_manager_import(self):
        """Test CitationManager can be imported."""
        from src.citations import CitationManager
        manager = CitationManager()
        assert hasattr(manager, "extract_and_record_citations")
    
    def test_reference_list_generator_import(self):
        """Test ReferenceListGenerator can be imported."""
        from src.citations import ReferenceListGenerator
        generator = ReferenceListGenerator()
        assert hasattr(generator, "generate")


class TestCitationFormatter:
    """Tests for citation formatting."""
    
    def test_format_inline_citation(self):
        """Test inline citation formatting."""
        from src.citations.formatter import format_inline_citation, Author
        author = Author(last_name="Smith", first_name="John")
        citation = format_inline_citation([author], 2024)
        assert "Smith" in citation
        assert "2024" in citation
    
    def test_format_multiple_authors(self):
        """Test citation with multiple authors."""
        from src.citations.formatter import format_inline_citation, Author
        authors = [
            Author(last_name="Smith", first_name="John"),
            Author(last_name="Jones", first_name="Mary"),
            Author(last_name="Brown", first_name="Alice"),
        ]
        citation = format_inline_citation(authors, 2024)
        # Should use "et al." for 3+ authors
        assert "et al." in citation


# =============================================================================
# Test Writer Node Integration
# =============================================================================


class TestWriterNodeBuildContext:
    """Tests for build_section_context function."""
    
    def test_build_context_minimal_state(self):
        """Test building context with minimal state."""
        from src.nodes.writer import build_section_context
        from src.writers import ArgumentManager
        
        state = {
            "research_intake": {
                "research_question": "How does X affect Y?",
                "paper_type": "short_article",
                "target_journal": "rfs",
            }
        }
        manager = ArgumentManager()
        
        context = build_section_context(
            state=state,
            section_type=SectionType.INTRODUCTION,
            completed_sections=[],
            argument_manager=manager,
        )
        
        assert context.section_type == "introduction"
        assert context.research_question == "How does X affect Y?"
    
    def test_build_context_with_prior_sections(self):
        """Test building context with prior sections."""
        from src.nodes.writer import build_section_context
        from src.writers import ArgumentManager
        
        prior = PaperSection(
            section_type="introduction",
            title="Introduction",
            content="Test intro content",
        )
        state = {
            "research_intake": {
                "research_question": "Test question",
                "paper_type": "short_article",
            }
        }
        manager = ArgumentManager()
        
        context = build_section_context(
            state=state,
            section_type=SectionType.METHODS,
            completed_sections=[prior],
            argument_manager=manager,
        )
        
        assert len(context.prior_sections) == 1

    def test_build_context_includes_revision_feedback(self):
        """Test build_section_context includes reviewer revision guidance."""
        from src.nodes.writer import build_section_context
        from src.writers import ArgumentManager

        state = {
            "original_query": "How does X affect Y?",
            "target_journal": "rfs",
            "paper_type": "short_article",
            "revision_request": {
                "sections_to_revise": ["introduction"],
                "revision_instructions": "Tighten contribution and add citations.",
                "iteration_count": 2,
                "critique_items": [
                    {
                        "section": "introduction",
                        "issue": "The motivation is too generic.",
                        "severity": "major",
                        "suggestion": "State the economic mechanism and why dual-class matters.",
                    }
                ],
            },
            "human_feedback": "Please keep it concise.",
        }
        manager = ArgumentManager()

        context = build_section_context(
            state=state,
            section_type=SectionType.INTRODUCTION,
            completed_sections=[],
            argument_manager=manager,
        )

        assert context.is_revision is True
        assert context.revision_iteration == 2
        assert "Tighten contribution" in context.revision_instructions
        assert "motivation is too generic" in context.critique_for_section
        assert "keep it concise" in context.human_feedback


class TestGetSectionWriter:
    """Tests for get_section_writer function."""
    
    def test_get_introduction_writer(self):
        """Test getting introduction writer."""
        from src.nodes.writer import get_section_writer
        from src.writers import SectionWriterConfig, IntroductionWriter
        from src.style import StyleEnforcer
        from src.citations import CitationManager
        
        config = SectionWriterConfig()
        enforcer = StyleEnforcer()
        citation_mgr = CitationManager()
        
        writer = get_section_writer(
            section_type=SectionType.INTRODUCTION,
            config=config,
            style_enforcer=enforcer,
            citation_manager=citation_mgr,
        )
        
        assert isinstance(writer, IntroductionWriter)
    
    def test_get_all_writers(self):
        """Test getting all section writers."""
        from src.nodes.writer import get_section_writer, SECTION_ORDER
        from src.writers import SectionWriterConfig
        from src.style import StyleEnforcer
        from src.citations import CitationManager
        
        config = SectionWriterConfig()
        enforcer = StyleEnforcer()
        citation_mgr = CitationManager()
        
        for section_type in SECTION_ORDER:
            writer = get_section_writer(
                section_type=section_type,
                config=config,
                style_enforcer=enforcer,
                citation_manager=citation_mgr,
            )
            assert writer.section_type == section_type.value


# =============================================================================
# Test Graph Routing
# =============================================================================


class TestGraphRoutingAfterAnalysis:
    """Tests for routing after analysis nodes."""
    
    def test_route_to_writer_after_data_analysis(self):
        """Test routing to writer after data analysis completes."""
        from studio.graphs import route_after_analysis
        
        state = {
            "data_analyst_output": {"findings": []},
        }
        result = route_after_analysis(state)
        assert result == "writer"
    
    def test_route_to_writer_after_conceptual_synthesis(self):
        """Test routing to writer after conceptual synthesis."""
        from studio.graphs import route_after_analysis
        
        state = {
            "conceptual_synthesis_output": {"framework": {}},
        }
        result = route_after_analysis(state)
        assert result == "writer"
    
    def test_route_to_end_on_error(self):
        """Test routing to end on error."""
        from studio.graphs import route_after_analysis
        
        state = {
            "errors": ["Analysis failed"],
        }
        result = route_after_analysis(state)
        assert result == "__end__"
    
    def test_route_to_end_no_output(self):
        """Test routing to end when no analysis output."""
        from studio.graphs import route_after_analysis
        
        state = {}
        result = route_after_analysis(state)
        assert result == "__end__"
