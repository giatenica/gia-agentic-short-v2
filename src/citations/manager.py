"""Citation manager for tracking and verifying citations."""

from dataclasses import dataclass, field

from src.state.models import CitationEntry, StyleViolation
from src.state.enums import CritiqueSeverity
from src.citations.formatter import format_inline_citation, extract_citation_keys


@dataclass
class CitationUsage:
    """Tracks where a citation is used."""
    
    key: str
    section: str
    location: str  # Line or paragraph reference
    context: str   # Surrounding text


class CitationManager:
    """
    Manage citations throughout a document.
    
    Tracks:
    - All available citations (from literature review)
    - Citations actually used in text
    - Missing or unverified citations
    """
    
    def __init__(self):
        """Initialize the citation manager."""
        self._entries: dict[str, CitationEntry] = {}
        self._usages: list[CitationUsage] = []
    
    def add_entry(self, entry: CitationEntry) -> None:
        """
        Add a citation entry to the registry.
        
        Args:
            entry: CitationEntry to add.
        """
        self._entries[entry.key] = entry
    
    def add_entries(self, entries: list[CitationEntry]) -> None:
        """Add multiple citation entries."""
        for entry in entries:
            self.add_entry(entry)
    
    def get_entry(self, key: str) -> CitationEntry | None:
        """
        Get a citation entry by key.
        
        Args:
            key: Citation key.
            
        Returns:
            CitationEntry or None if not found.
        """
        return self._entries.get(key)
    
    def has_entry(self, key: str) -> bool:
        """Check if a citation key exists."""
        return key in self._entries
    
    def record_usage(
        self,
        key: str,
        section: str,
        location: str,
        context: str,
    ) -> None:
        """
        Record a citation usage in the document.
        
        Args:
            key: Citation key.
            section: Section name.
            location: Location in section.
            context: Surrounding text.
        """
        self._usages.append(CitationUsage(
            key=key,
            section=section,
            location=location,
            context=context,
        ))
    
    def extract_and_record_citations(
        self,
        text: str,
        section: str,
    ) -> list[str]:
        """
        Extract citations from text and record usages.
        
        Args:
            text: Text to search.
            section: Section name for recording.
            
        Returns:
            List of citation keys found.
        """
        keys = extract_citation_keys(text)
        
        lines = text.split('\n')
        for line_num, line in enumerate(lines, 1):
            line_keys = extract_citation_keys(line)
            for key in line_keys:
                self.record_usage(
                    key=key,
                    section=section,
                    location=f"line {line_num}",
                    context=line[:100],
                )
        
        return keys
    
    def get_used_keys(self) -> set[str]:
        """Get set of all citation keys used in the document."""
        return {u.key for u in self._usages}
    
    def get_unused_entries(self) -> list[CitationEntry]:
        """Get entries that were added but never cited."""
        used = self.get_used_keys()
        return [e for k, e in self._entries.items() if k not in used]
    
    def get_missing_entries(self) -> list[str]:
        """Get citation keys that were used but have no entry."""
        used = self.get_used_keys()
        return [k for k in used if k not in self._entries]
    
    def verify_citations(self) -> list[StyleViolation]:
        """
        Verify all citations and return any issues.
        
        Returns:
            List of StyleViolation objects for citation issues.
        """
        violations: list[StyleViolation] = []
        
        # Check for citations without entries
        missing = self.get_missing_entries()
        for key in missing:
            usages = [u for u in self._usages if u.key == key]
            first_usage = usages[0] if usages else None
            
            violations.append(StyleViolation(
                violation_type="citation_format",
                severity=CritiqueSeverity.MAJOR,
                location=f"{first_usage.section}, {first_usage.location}" if first_usage else "unknown",
                original_text=key,
                suggestion=f"Citation '{key}' not found in reference list",
                rule_reference="docs/writing_style_guide.md - Citations",
                auto_fixable=False,
            ))
        
        return violations
    
    def format_citation(
        self,
        key: str,
        page: str | None = None,
        narrative: bool = False,
    ) -> str:
        """
        Format a citation from the registry.
        
        Args:
            key: Citation key.
            page: Optional page number.
            narrative: If True, format as narrative citation.
            
        Returns:
            Formatted citation string.
        """
        entry = self.get_entry(key)
        
        if not entry:
            return f"[{key}]"  # Placeholder for missing citation
        
        return format_inline_citation(entry.authors, entry.year, page)
    
    def get_citation_count(self) -> int:
        """Get total number of citation usages."""
        return len(self._usages)
    
    def get_entry_count(self) -> int:
        """Get total number of citation entries."""
        return len(self._entries)
    
    def get_citations_by_section(self) -> dict[str, list[str]]:
        """Get citation keys organized by section."""
        by_section: dict[str, list[str]] = {}
        for usage in self._usages:
            if usage.section not in by_section:
                by_section[usage.section] = []
            if usage.key not in by_section[usage.section]:
                by_section[usage.section].append(usage.key)
        return by_section
    
    def all_entries(self) -> list[CitationEntry]:
        """Get all citation entries."""
        return list(self._entries.values())
    
    def clear_usages(self) -> None:
        """Clear all usage records (but keep entries)."""
        self._usages = []
