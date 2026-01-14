"""Helpers for injecting the project writing style guide into writer prompts."""

from __future__ import annotations

from pathlib import Path


_STYLE_GUIDE_RELATIVE_PATH = Path("docs") / "writing_style_guide.md"


def _project_root() -> Path:
    # src/writers/style_guide.py -> src/writers -> src -> repo root
    return Path(__file__).resolve().parents[2]


def get_style_guide_excerpt(max_chars: int = 2500) -> str:
    """Return a compact excerpt of the academic finance style guide.

    The intent is to anchor the LLM to house style and formatting without
    dumping the full document into every prompt.

    Returns an empty string if the guide cannot be found.
    """
    guide_path = _project_root() / _STYLE_GUIDE_RELATIVE_PATH
    if not guide_path.exists():
        return ""

    text = guide_path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.rstrip() for ln in text.splitlines()]

    def take_section(header: str) -> list[str]:
        header_line = f"## {header}" if not header.startswith("##") else header
        start = None
        for i, ln in enumerate(lines):
            if ln.strip() == header_line:
                start = i
                break
        if start is None:
            return []
        out: list[str] = []
        for ln in lines[start:]:
            if ln.startswith("## ") and ln.strip() != header_line and out:
                break
            out.append(ln)
        return out

    # Pull the highest signal sections for prompt grounding.
    chosen: list[str] = []

    # Title + target journals + target format are at the top.
    chosen.extend(lines[:18])

    # Key writing and formatting guidance.
    chosen.extend([""]) 
    chosen.extend(take_section("4. Writing Style"))
    chosen.extend([""])
    chosen.extend(take_section("5. Citations and References"))
    chosen.extend([""])
    chosen.extend(take_section("6. Tables and Figures"))
    chosen.extend([""])
    chosen.extend(take_section("7. Short Article Guidelines"))

    excerpt = "\n".join([ln for ln in chosen if ln is not None]).strip()
    if len(excerpt) <= max_chars:
        return excerpt
    return excerpt[: max_chars - 3].rstrip() + "..."
