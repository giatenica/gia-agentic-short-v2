"""Tests for citation formatter."""

import pytest

from src.citations.formatter import (
    Author,
    format_inline_citation,
    format_narrative_citation,
)


class TestAuthor:
    """Tests for Author dataclass."""
    
    def test_full_name_with_all_parts(self):
        """Test full_name with all name parts."""
        author = Author(
            last_name="Smith",
            first_name="John",
            middle_name="Robert",
            suffix="Jr."
        )
        assert author.full_name == "John Robert Smith Jr."
    
    def test_full_name_without_middle(self):
        """Test full_name without middle name."""
        author = Author(last_name="Smith", first_name="John")
        assert author.full_name == "John Smith"
    
    def test_full_name_only_last(self):
        """Test full_name with only last name."""
        author = Author(last_name="Smith")
        assert author.full_name == "Smith"
    
    def test_reference_name_full(self):
        """Test reference_name with full name parts."""
        author = Author(
            last_name="Smith",
            first_name="John",
            middle_name="Robert",
            suffix="Jr."
        )
        assert author.reference_name == "Smith, John Robert, Jr."
    
    def test_reference_name_simple(self):
        """Test reference_name with just first and last."""
        author = Author(last_name="Smith", first_name="John")
        assert author.reference_name == "Smith, John"
    
    def test_reference_name_only_last(self):
        """Test reference_name with only last name."""
        author = Author(last_name="Smith")
        assert author.reference_name == "Smith"


class TestAuthorFromString:
    """Tests for Author.from_string method."""
    
    def test_last_comma_first(self):
        """Test parsing 'Last, First' format."""
        author = Author.from_string("Smith, John")
        assert author.last_name == "Smith"
        assert author.first_name == "John"
    
    def test_last_comma_first_middle(self):
        """Test parsing 'Last, First Middle' format."""
        author = Author.from_string("Smith, John Robert")
        assert author.last_name == "Smith"
        assert author.first_name == "John"
        assert author.middle_name == "Robert"
    
    def test_first_last(self):
        """Test parsing 'First Last' format."""
        author = Author.from_string("John Smith")
        assert author.last_name == "Smith"
        assert author.first_name == "John"
    
    def test_first_middle_last(self):
        """Test parsing 'First Middle Last' format."""
        author = Author.from_string("John Robert Smith")
        assert author.last_name == "Smith"
        assert author.first_name == "John"
        assert author.middle_name == "Robert"
    
    def test_single_name(self):
        """Test parsing single name."""
        author = Author.from_string("Smith")
        assert author.last_name == "Smith"
        assert author.first_name == ""
    
    def test_with_suffix_jr(self):
        """Test parsing name with Jr. suffix."""
        author = Author.from_string("Smith, John Jr.")
        assert author.last_name == "Smith"
        assert author.first_name == "John"
        assert author.suffix == "Jr."
    
    def test_with_suffix_iii(self):
        """Test parsing name with III suffix."""
        author = Author.from_string("Smith, John III")
        assert author.last_name == "Smith"
        assert author.first_name == "John"
        assert author.suffix == "III"
    
    def test_strips_whitespace(self):
        """Test that whitespace is stripped."""
        author = Author.from_string("  Smith, John  ")
        assert author.last_name == "Smith"
        assert author.first_name == "John"


class TestFormatInlineCitation:
    """Tests for format_inline_citation function."""
    
    def test_single_author(self):
        """Test citation with single author."""
        result = format_inline_citation(["Fama, Eugene F."], 1970)
        assert result == "(Fama 1970)"
    
    def test_two_authors(self):
        """Test citation with two authors."""
        result = format_inline_citation(
            ["Fama, Eugene F.", "French, Kenneth R."],
            1993
        )
        assert result == "(Fama and French 1993)"
    
    def test_three_plus_authors(self):
        """Test citation with three or more authors uses et al."""
        result = format_inline_citation(
            ["Barberis, N.", "Shleifer, A.", "Vishny, R."],
            1998
        )
        assert result == "(Barberis et al. 1998)"
    
    def test_with_page_number(self):
        """Test citation with page number."""
        result = format_inline_citation(["Fama, Eugene F."], 1970, page="25")
        assert result == "(Fama 1970, 25)"
    
    def test_with_page_range(self):
        """Test citation with page range."""
        result = format_inline_citation(
            ["Fama, Eugene F."],
            1970,
            page="25-30"
        )
        assert result == "(Fama 1970, 25-30)"
    
    def test_empty_authors(self):
        """Test citation with no authors."""
        result = format_inline_citation([], 2020)
        assert result == "(2020)"
    
    def test_with_author_objects(self):
        """Test citation with Author objects."""
        authors = [
            Author(last_name="Fama", first_name="Eugene"),
            Author(last_name="French", first_name="Kenneth"),
        ]
        result = format_inline_citation(authors, 1993)
        assert result == "(Fama and French 1993)"


class TestFormatNarrativeCitation:
    """Tests for format_narrative_citation function."""
    
    def test_single_author(self):
        """Test narrative citation with single author."""
        result = format_narrative_citation(["Fama, Eugene F."], 1970)
        assert result == "Fama (1970)"
    
    def test_two_authors(self):
        """Test narrative citation with two authors."""
        result = format_narrative_citation(
            ["Fama, Eugene F.", "French, Kenneth R."],
            1993
        )
        assert result == "Fama and French (1993)"
    
    def test_three_plus_authors(self):
        """Test narrative citation with three or more authors."""
        result = format_narrative_citation(
            ["Barberis, N.", "Shleifer, A.", "Vishny, R."],
            1998
        )
        assert result == "Barberis et al. (1998)"
    
    def test_with_page_number(self):
        """Test narrative citation with page number."""
        result = format_narrative_citation(
            ["Fama, Eugene F."],
            1970,
            page="25"
        )
        assert result == "Fama (1970, 25)"
    
    def test_empty_authors(self):
        """Test narrative citation with no authors."""
        result = format_narrative_citation([], 2020)
        assert result == "(2020)"
