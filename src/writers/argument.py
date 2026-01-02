"""Argument thread manager.

Tracks argumentative coherence across paper sections.
Ensures the paper tells a coherent story from start to finish.
"""

from dataclasses import dataclass, field
from typing import TypedDict

from src.state.models import PaperSection
from src.state.enums import SectionType


class ClaimReference(TypedDict):
    """A reference to a claim in a specific section."""
    claim: str
    section: str
    location: str  # e.g., "paragraph 1", "opening"


@dataclass
class ThreadInfo:
    """Local tracking info for an argument thread."""
    name: str
    description: str
    claimed_in: list[str] = field(default_factory=list)
    supported_in: list[str] = field(default_factory=list)
    is_resolved: bool = False


@dataclass
class ArgumentManager:
    """Manages argumentative threads across paper sections.
    
    Tracks claims, ensures they are properly supported,
    and verifies coherence across the paper structure.
    """
    
    # Core argument structure
    main_contribution: str = ""
    research_question: str = ""
    key_findings: list[str] = field(default_factory=list)
    
    # Thread tracking (using local ThreadInfo class)
    threads: list[ThreadInfo] = field(default_factory=list)
    
    # Claim tracking
    claims_made: list[ClaimReference] = field(default_factory=list)
    claims_supported: list[ClaimReference] = field(default_factory=list)
    
    def set_core_argument(
        self,
        contribution: str,
        question: str,
        findings: list[str],
    ) -> None:
        """Set the core argument structure."""
        self.main_contribution = contribution
        self.research_question = question
        self.key_findings = findings
    
    def create_thread(
        self,
        name: str,
        description: str,
        claimed_in: list[str] | None = None,
        supported_in: list[str] | None = None,
    ) -> ThreadInfo:
        """Create a new argument thread to track."""
        thread = ThreadInfo(
            name=name,
            description=description,
            claimed_in=claimed_in or [],
            supported_in=supported_in or [],
            is_resolved=False,
        )
        self.threads.append(thread)
        return thread
    
    def register_claim(
        self,
        claim: str,
        section: str,
        location: str = "",
    ) -> None:
        """Register a claim made in a section."""
        self.claims_made.append(ClaimReference(
            claim=claim,
            section=section,
            location=location,
        ))
    
    def register_support(
        self,
        claim: str,
        section: str,
        location: str = "",
    ) -> None:
        """Register evidence supporting a claim."""
        self.claims_supported.append(ClaimReference(
            claim=claim,
            section=section,
            location=location,
        ))
    
    def get_unresolved_threads(self) -> list[ThreadInfo]:
        """Get threads that have claims but lack support."""
        unresolved = []
        for thread in self.threads:
            if thread.claimed_in and not thread.supported_in:
                unresolved.append(thread)
            elif not thread.is_resolved:
                unresolved.append(thread)
        return unresolved
    
    def check_thread_resolution(self, thread: ThreadInfo) -> bool:
        """Check if a thread is properly resolved (claimed and supported)."""
        if not thread.claimed_in:
            return False
        if not thread.supported_in:
            return False
        thread.is_resolved = True
        return True
    
    def get_section_requirements(
        self,
        section_type: SectionType | str,
    ) -> dict[str, list[str]]:
        """Get argument requirements for a section.
        
        Returns what claims should be made or supported in this section.
        """
        section = section_type.value if isinstance(section_type, SectionType) else section_type
        
        requirements: dict[str, list[str]] = {
            "should_claim": [],
            "should_support": [],
            "should_reference": [],
        }
        
        if section == "introduction":
            requirements["should_claim"] = [
                self.main_contribution,
                f"Research question: {self.research_question}",
            ]
            requirements["should_reference"] = [
                "Gap in literature",
                "Why this matters",
            ]
        
        elif section == "literature_review":
            requirements["should_support"] = [
                "Gap claim from introduction",
            ]
            requirements["should_reference"] = [
                "Related work",
                "Theoretical foundations",
            ]
        
        elif section == "methods":
            requirements["should_support"] = [
                "Appropriate methodology for question",
                "Data suitability",
            ]
        
        elif section == "results":
            requirements["should_support"] = [
                *self.key_findings,
            ]
        
        elif section == "discussion":
            requirements["should_support"] = [
                self.main_contribution,
            ]
            requirements["should_reference"] = [
                "Connect findings to literature",
                "Address limitations",
            ]
        
        elif section == "conclusion":
            requirements["should_reference"] = [
                self.main_contribution,
                "Implications",
                "Future directions",
            ]
        
        return requirements
    
    def validate_section_coherence(
        self,
        section: PaperSection,
        prior_sections: list[PaperSection],
    ) -> list[str]:
        """Validate that a section maintains argumentative coherence.
        
        Returns list of issues found.
        """
        issues: list[str] = []
        
        section_type = section.section_type
        # Get requirements for future validation expansion
        _ = self.get_section_requirements(section_type)
        
        # For introduction, check that main claim is made
        if section_type == "introduction":
            if not self.main_contribution:
                issues.append("No main contribution defined for coherence checking")
        
        # For results, check that findings are present
        elif section_type == "results":
            if not self.key_findings:
                issues.append("No key findings defined for coherence checking")
        
        # For discussion, check that contribution is delivered
        elif section_type == "discussion":
            # Check if contribution was claimed in intro
            intro_claims = [
                c for c in self.claims_made 
                if c["section"] == "introduction"
            ]
            if intro_claims and not any(
                c["section"] in ["results", "discussion"]
                for c in self.claims_supported
            ):
                issues.append(
                    "Claims from introduction may not be fully supported"
                )
        
        # For conclusion, verify no new arguments
        elif section_type == "conclusion":
            # This would need content analysis; placeholder for now
            pass
        
        return issues
    
    def get_coherence_summary(self) -> dict:
        """Get a summary of argument coherence across the paper."""
        total_threads = len(self.threads)
        resolved_threads = len([t for t in self.threads if t.is_resolved])
        
        total_claims = len(self.claims_made)
        supported_claims = len(self.claims_supported)
        
        return {
            "total_threads": total_threads,
            "resolved_threads": resolved_threads,
            "unresolved_threads": total_threads - resolved_threads,
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "unsupported_claims": total_claims - supported_claims,
            "coherence_score": (
                resolved_threads / total_threads if total_threads > 0 else 1.0
            ),
        }
    
    def generate_coherence_prompt(
        self,
        section_type: SectionType | str,
    ) -> str:
        """Generate a prompt addition for maintaining coherence.
        
        This can be added to section writer prompts to ensure
        the section maintains proper argument flow.
        """
        requirements = self.get_section_requirements(section_type)
        
        prompt_parts = [
            "ARGUMENT COHERENCE REQUIREMENTS:",
            "",
        ]
        
        if requirements["should_claim"]:
            prompt_parts.append("This section should CLAIM or ESTABLISH:")
            for claim in requirements["should_claim"]:
                prompt_parts.append(f"  - {claim}")
            prompt_parts.append("")
        
        if requirements["should_support"]:
            prompt_parts.append("This section should SUPPORT or PROVIDE EVIDENCE for:")
            for support in requirements["should_support"]:
                prompt_parts.append(f"  - {support}")
            prompt_parts.append("")
        
        if requirements["should_reference"]:
            prompt_parts.append("This section should REFERENCE or CONNECT to:")
            for ref in requirements["should_reference"]:
                prompt_parts.append(f"  - {ref}")
            prompt_parts.append("")
        
        # Add unresolved threads as reminders
        unresolved = self.get_unresolved_threads()
        if unresolved:
            prompt_parts.append("UNRESOLVED ARGUMENT THREADS to address:")
            for thread in unresolved[:3]:  # Limit to top 3
                prompt_parts.append(f"  - {thread.name}: {thread.description}")
            prompt_parts.append("")
        
        return "\n".join(prompt_parts)
