"""Citation formatter for Chicago Author-Date style.

Formats citations according to Chicago Manual of Style, 17th edition,
Author-Date system, as used by major finance journals.
"""

import re
from dataclasses import dataclass


@dataclass
class Author:
    """Represents an author for citation purposes."""
    
    last_name: str
    first_name: str = ""
    middle_name: str = ""
    suffix: str = ""  # Jr., III, etc.
    
    @property
    def full_name(self) -> str:
        """Get full name in 'First Middle Last Suffix' format."""
        parts = [self.first_name, self.middle_name, self.last_name]
        if self.suffix:
            parts.append(self.suffix)
        return " ".join(p for p in parts if p)
    
    @property
    def reference_name(self) -> str:
        """Get name in 'Last, First Middle' format for reference list."""
        parts = [self.last_name]
        if self.first_name:
            name_part = self.first_name
            if self.middle_name:
                name_part += " " + self.middle_name
            parts.append(name_part)
        result = ", ".join(parts)
        if self.suffix:
            result += ", " + self.suffix
        return result
    
    @classmethod
    def from_string(cls, name: str) -> "Author":
        """
        Parse author name from string.
        
        Handles formats:
        - "Last, First"
        - "First Last"
        - "Last, First M."
        - "Last, First Middle"
        """
        name = name.strip()
        
        if "," in name:
            # "Last, First [Middle]" format
            parts = name.split(",", 1)
            last = parts[0].strip()
            rest = parts[1].strip() if len(parts) > 1 else ""
            
            # Check for suffix
            suffix = ""
            for suf in ["Jr.", "Jr", "III", "II", "IV"]:
                if rest.endswith(suf):
                    suffix = suf
                    rest = rest[:-len(suf)].strip().rstrip(",")
                    break
            
            # Split first and middle
            name_parts = rest.split()
            first = name_parts[0] if name_parts else ""
            middle = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
            
            return cls(last_name=last, first_name=first, middle_name=middle, suffix=suffix)
        else:
            # "First [Middle] Last" format
            parts = name.split()
            if len(parts) == 1:
                return cls(last_name=parts[0])
            elif len(parts) == 2:
                return cls(last_name=parts[1], first_name=parts[0])
            else:
                return cls(
                    last_name=parts[-1],
                    first_name=parts[0],
                    middle_name=" ".join(parts[1:-1])
                )


def format_inline_citation(
    authors: list[str] | list[Author],
    year: int,
    page: str | None = None,
) -> str:
    """
    Format an inline citation in Chicago Author-Date style.
    
    Args:
        authors: List of author names or Author objects.
        year: Publication year.
        page: Optional page number or range.
        
    Returns:
        Formatted citation string, e.g., "(Fama 1970)" or "(Fama and French 1993)".
        
    Examples:
        >>> format_inline_citation(["Fama, Eugene F."], 1970)
        '(Fama 1970)'
        >>> format_inline_citation(["Fama, Eugene F.", "French, Kenneth R."], 1993)
        '(Fama and French 1993)'
        >>> format_inline_citation(["Barberis, N.", "Shleifer, A.", "Vishny, R."], 1998)
        '(Barberis et al. 1998)'
    """
    if not authors:
        return f"({year})"
    
    # Convert strings to Author objects if needed
    author_objs = []
    for a in authors:
        if isinstance(a, str):
            author_objs.append(Author.from_string(a))
        else:
            author_objs.append(a)
    
    # Format author names
    if len(author_objs) == 1:
        author_str = author_objs[0].last_name
    elif len(author_objs) == 2:
        author_str = f"{author_objs[0].last_name} and {author_objs[1].last_name}"
    else:
        author_str = f"{author_objs[0].last_name} et al."
    
    # Build citation
    citation = f"({author_str} {year}"
    
    if page:
        citation += f", {page}"
    
    citation += ")"
    
    return citation


def format_narrative_citation(
    authors: list[str] | list[Author],
    year: int,
    page: str | None = None,
) -> str:
    """
    Format a narrative citation where author is part of the sentence.
    
    Args:
        authors: List of author names or Author objects.
        year: Publication year.
        page: Optional page number.
        
    Returns:
        Formatted citation, e.g., "Fama (1970)" or "Fama and French (1993)".
        
    Examples:
        >>> format_narrative_citation(["Fama, Eugene F."], 1970)
        'Fama (1970)'
        >>> format_narrative_citation(["Fama, Eugene F.", "French, Kenneth R."], 1993)
        'Fama and French (1993)'
    """
    if not authors:
        return f"({year})"
    
    # Convert strings to Author objects if needed
    author_objs = []
    for a in authors:
        if isinstance(a, str):
            author_objs.append(Author.from_string(a))
        else:
            author_objs.append(a)
    
    # Format author names
    if len(author_objs) == 1:
        author_str = author_objs[0].last_name
    elif len(author_objs) == 2:
        author_str = f"{author_objs[0].last_name} and {author_objs[1].last_name}"
    else:
        author_str = f"{author_objs[0].last_name} et al."
    
    # Build citation
    year_part = f"({year}"
    if page:
        year_part += f", {page}"
    year_part += ")"
    
    return f"{author_str} {year_part}"


def format_multiple_citations(
    citations: list[tuple[list[str], int]],
) -> str:
    """
    Format multiple citations in a single parenthetical.
    
    Args:
        citations: List of (authors, year) tuples.
        
    Returns:
        Formatted citations, e.g., "(Fama 1970; Jensen 1986)".
        
    Examples:
        >>> format_multiple_citations([
        ...     (["Fama, Eugene F."], 1970),
        ...     (["Jensen, Michael C."], 1986),
        ... ])
        '(Fama 1970; Jensen 1986)'
    """
    if not citations:
        return ""
    
    parts = []
    for authors, year in citations:
        # Get author string
        if not authors:
            parts.append(str(year))
            continue
        
        author_objs = [
            Author.from_string(a) if isinstance(a, str) else a
            for a in authors
        ]
        
        if len(author_objs) == 1:
            author_str = author_objs[0].last_name
        elif len(author_objs) == 2:
            author_str = f"{author_objs[0].last_name} and {author_objs[1].last_name}"
        else:
            author_str = f"{author_objs[0].last_name} et al."
        
        parts.append(f"{author_str} {year}")
    
    return "(" + "; ".join(parts) + ")"


def format_reference_entry(
    authors: list[str] | list[Author],
    year: int,
    title: str,
    journal: str | None = None,
    volume: str | None = None,
    issue: str | None = None,
    pages: str | None = None,
    publisher: str | None = None,
    doi: str | None = None,
    url: str | None = None,
    source_type: str = "journal",
) -> str:
    """
    Format a complete reference list entry.
    
    Args:
        authors: List of author names.
        year: Publication year.
        title: Title of the work.
        journal: Journal name (for articles).
        volume: Volume number.
        issue: Issue number.
        pages: Page range.
        publisher: Publisher (for books).
        doi: DOI identifier.
        url: URL (for online sources).
        source_type: Type of source (journal, book, chapter, working_paper).
        
    Returns:
        Formatted reference string.
    """
    # Convert to Author objects
    author_objs = [
        Author.from_string(a) if isinstance(a, str) else a
        for a in authors
    ]
    
    # Format authors
    if len(author_objs) == 0:
        author_str = "Anonymous"
    elif len(author_objs) == 1:
        author_str = author_objs[0].reference_name
    elif len(author_objs) == 2:
        author_str = f"{author_objs[0].reference_name}, and {author_objs[1].reference_name}"
    else:
        # First author in Last, First format; others in First Last format
        parts = [author_objs[0].reference_name]
        for a in author_objs[1:-1]:
            parts.append(f"{a.first_name} {a.middle_name} {a.last_name}".strip())
        parts.append(f"and {author_objs[-1].first_name} {author_objs[-1].middle_name} {author_objs[-1].last_name}".strip())
        author_str = ", ".join(parts)
    
    # Start building reference
    ref = f'{author_str}. {year}. "{title}."'
    
    if source_type == "journal" and journal:
        ref += f" {journal}"
        if volume:
            ref += f" {volume}"
            if issue:
                ref += f" ({issue})"
        if pages:
            ref += f": {pages}"
        ref += "."
    elif source_type == "book" and publisher:
        ref += f" {publisher}."
    elif source_type == "working_paper":
        ref += " Working Paper."
        if publisher:
            ref += f" {publisher}."
    elif source_type == "chapter" and publisher:
        ref += f" {publisher}."
        if pages:
            ref += f" Pages {pages}."
    
    # Add DOI or URL
    if doi:
        ref += f" https://doi.org/{doi}"
    elif url:
        ref += f" {url}"
    
    return ref


def extract_citation_keys(text: str) -> list[str]:
    """
    Extract citation keys from text.
    
    Looks for patterns like:
    - (Author Year)
    - (Author and Author Year)
    - (Author et al. Year)
    - Author (Year)
    
    Args:
        text: Text to search.
        
    Returns:
        List of citation strings found.
    """
    patterns = [
        r'\(([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?(?:\s+et\s+al\.)?)\s+(\d{4})[a-z]?(?:,\s*\d+)?\)',
        r'([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?(?:\s+et\s+al\.)?)\s+\((\d{4})[a-z]?(?:,\s*\d+)?\)',
    ]
    
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            author, year = match
            key = f"{author.split()[0]}{year}"
            if key not in citations:
                citations.append(key)
    
    return citations
