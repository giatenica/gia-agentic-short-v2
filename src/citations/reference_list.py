"""Reference list generator for Chicago Author-Date style."""

from src.state.models import CitationEntry, ReferenceList
from src.citations.formatter import format_reference_entry


class ReferenceListGenerator:
    """
    Generate formatted reference lists.
    
    Formats references according to Chicago Manual of Style,
    17th edition, Author-Date system.
    """
    
    def __init__(self, entries: list[CitationEntry] | None = None):
        """
        Initialize the generator.
        
        Args:
            entries: Optional list of CitationEntry objects.
        """
        self.entries = entries or []
    
    def add_entry(self, entry: CitationEntry) -> None:
        """Add a citation entry."""
        # Check for duplicates by key
        if not any(e.key == entry.key for e in self.entries):
            self.entries.append(entry)
    
    def add_entries(self, entries: list[CitationEntry]) -> None:
        """Add multiple entries."""
        for entry in entries:
            self.add_entry(entry)
    
    def sort_entries(self) -> list[CitationEntry]:
        """
        Sort entries alphabetically by first author's last name, then year.
        
        Returns:
            Sorted list of entries.
        """
        def sort_key(entry: CitationEntry) -> tuple[str, int]:
            # Get first author's last name
            if entry.authors:
                first_author = entry.authors[0]
                # Handle "Last, First" format
                if "," in first_author:
                    last_name = first_author.split(",")[0].strip()
                else:
                    # "First Last" format - take last word
                    last_name = first_author.split()[-1]
            else:
                last_name = "ZZZ"  # Sort unknown authors last
            
            return (last_name.lower(), entry.year)
        
        return sorted(self.entries, key=sort_key)
    
    def format_entry(self, entry: CitationEntry) -> str:
        """
        Format a single entry.
        
        Args:
            entry: CitationEntry to format.
            
        Returns:
            Formatted reference string.
        """
        return format_reference_entry(
            authors=entry.authors,
            year=entry.year,
            title=entry.title,
            journal=entry.journal,
            volume=entry.volume,
            issue=entry.issue,
            pages=entry.pages,
            publisher=entry.publisher,
            doi=entry.doi,
            url=entry.url,
            source_type=entry.source_type,
        )
    
    def generate(
        self,
        include_only: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> str:
        """
        Generate the complete reference list.
        
        Args:
            include_only: If provided, only include these keys.
            exclude: If provided, exclude these keys.
            
        Returns:
            Formatted reference list as string.
        """
        # Filter entries
        entries = self.sort_entries()
        
        if include_only:
            entries = [e for e in entries if e.key in include_only]
        
        if exclude:
            entries = [e for e in entries if e.key not in exclude]
        
        # Format each entry
        lines = []
        for entry in entries:
            lines.append(self.format_entry(entry))
        
        return "\n\n".join(lines)
    
    def generate_for_section(
        self,
        used_keys: list[str],
    ) -> str:
        """
        Generate reference list for keys used in a section.
        
        Args:
            used_keys: Citation keys used in the section.
            
        Returns:
            Formatted reference list.
        """
        return self.generate(include_only=used_keys)
    
    def to_reference_list(self) -> ReferenceList:
        """
        Convert to a ReferenceList model.
        
        Returns:
            ReferenceList model instance.
        """
        return ReferenceList(
            entries=self.entries,
            format_style="chicago_author_date",
        )
    
    @classmethod
    def from_reference_list(cls, ref_list: ReferenceList) -> "ReferenceListGenerator":
        """
        Create generator from ReferenceList model.
        
        Args:
            ref_list: ReferenceList model.
            
        Returns:
            ReferenceListGenerator instance.
        """
        return cls(entries=ref_list.entries)
    
    def get_entry_count(self) -> int:
        """Get number of entries."""
        return len(self.entries)
    
    def get_entry(self, key: str) -> CitationEntry | None:
        """Get entry by key."""
        for entry in self.entries:
            if entry.key == key:
                return entry
        return None
    
    def has_entry(self, key: str) -> bool:
        """Check if entry exists."""
        return self.get_entry(key) is not None


def create_sample_entries() -> list[CitationEntry]:
    """
    Create sample citation entries for testing.
    
    Returns:
        List of sample CitationEntry objects.
    """
    return [
        CitationEntry(
            key="Fama1970",
            authors=["Fama, Eugene F."],
            year=1970,
            title="Efficient Capital Markets: A Review of Theory and Empirical Work",
            journal="Journal of Finance",
            volume="25",
            issue="2",
            pages="383-417",
            source_type="journal",
        ),
        CitationEntry(
            key="FamaFrench1993",
            authors=["Fama, Eugene F.", "French, Kenneth R."],
            year=1993,
            title="Common Risk Factors in the Returns on Stocks and Bonds",
            journal="Journal of Financial Economics",
            volume="33",
            issue="1",
            pages="3-56",
            source_type="journal",
        ),
        CitationEntry(
            key="Jensen1986",
            authors=["Jensen, Michael C."],
            year=1986,
            title="Agency Costs of Free Cash Flow, Corporate Finance, and Takeovers",
            journal="American Economic Review",
            volume="76",
            issue="2",
            pages="323-329",
            source_type="journal",
        ),
        CitationEntry(
            key="Shleifer2000",
            authors=["Shleifer, Andrei"],
            year=2000,
            title="Inefficient Markets: An Introduction to Behavioral Finance",
            publisher="Oxford University Press",
            source_type="book",
        ),
        CitationEntry(
            key="BarberisShleifer1998",
            authors=["Barberis, Nicholas", "Shleifer, Andrei", "Vishny, Robert"],
            year=1998,
            title="A Model of Investor Sentiment",
            journal="Journal of Financial Economics",
            volume="49",
            issue="3",
            pages="307-343",
            source_type="journal",
        ),
    ]
