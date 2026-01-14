"""LaTeX assembly and PDF compilation utilities."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LatexBuildResult:
    output_dir: str
    tex_path: str
    pdf_path: str | None
    exported_files: list[str]
    engine: str | None
    compilation_stdout: str | None
    compilation_stderr: str | None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    return cleaned.strip("_.-") or "paper"


def _escape_latex_text(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = value or ""
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


def _find_latex_engine() -> str | None:
    for candidate in ["tectonic", "latexmk", "pdflatex"]:
        if shutil.which(candidate):
            return candidate
    return None


def _compile_pdf(engine: str, tex_path: Path, output_dir: Path) -> tuple[Path | None, str, str]:
    stdout = ""
    stderr = ""

    if engine == "tectonic":
        cmd = ["tectonic", "-X", "compile", str(tex_path), "--outdir", str(output_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout = proc.stdout
        stderr = proc.stderr
        if proc.returncode != 0:
            return None, stdout, stderr
        pdf_path = output_dir / (tex_path.stem + ".pdf")
        return (pdf_path if pdf_path.exists() else None), stdout, stderr

    if engine == "latexmk":
        cmd = [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-outdir=" + str(output_dir),
            str(tex_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout = proc.stdout
        stderr = proc.stderr
        if proc.returncode != 0:
            return None, stdout, stderr
        pdf_path = output_dir / (tex_path.stem + ".pdf")
        return (pdf_path if pdf_path.exists() else None), stdout, stderr

    if engine == "pdflatex":
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(output_dir),
            str(tex_path),
        ]
        proc1 = subprocess.run(cmd, capture_output=True, text=True)
        proc2 = subprocess.run(cmd, capture_output=True, text=True)
        stdout = (proc1.stdout or "") + "\n" + (proc2.stdout or "")
        stderr = (proc1.stderr or "") + "\n" + (proc2.stderr or "")
        if proc2.returncode != 0:
            return None, stdout, stderr
        pdf_path = output_dir / (tex_path.stem + ".pdf")
        return (pdf_path if pdf_path.exists() else None), stdout, stderr

    return None, stdout, stderr


def build_paper_latex(
    *,
    title: str,
    author: str,
    sections: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    artifacts_dir_name: str = "artifacts",
) -> str:
    escaped_title = _escape_latex_text(title)
    escaped_author = _escape_latex_text(author)

    preamble = "\n".join(
        [
            r"\\documentclass[12pt]{article}",
            r"\\usepackage[margin=1in]{geometry}",
            r"\\usepackage{graphicx}",
            r"\\usepackage{booktabs}",
            r"\\usepackage{threeparttable}",
            r"\\usepackage{amsmath}",
            r"\\usepackage{hyperref}",
            r"\\usepackage{float}",
            r"\\title{" + escaped_title + r"}",
            r"\\author{" + escaped_author + r"}",
            r"\\date{}",
            r"\\begin{document}",
            r"\\maketitle",
            "",
        ]
    )

    body_parts: list[str] = [preamble]

    section_order = [
        ("abstract", "Abstract"),
        ("introduction", "Introduction"),
        ("literature_review", "Literature Review"),
        ("methods", "Methods"),
        ("results", "Results"),
        ("discussion", "Discussion"),
        ("conclusion", "Conclusion"),
    ]

    content_by_type: dict[str, str] = {}
    for s in sections:
        section_type = (s.get("section_type") or "").strip()
        content = s.get("content") or ""
        if section_type:
            content_by_type[section_type] = content

    for section_type, section_title in section_order:
        content = (content_by_type.get(section_type) or "").strip()
        if not content:
            continue
        body_parts.append(r"\\section{" + _escape_latex_text(section_title) + r"}")
        body_parts.append(content)
        body_parts.append("")

    if tables or figures:
        body_parts.append(r"\\clearpage")
        body_parts.append(r"\\section{Tables and Figures}")
        body_parts.append("")

    if tables:
        for t in tables:
            table_id = (t.get("table_id") or "").strip() or "table"
            fmt = (t.get("format") or "LATEX").lower()
            ext = ".tex" if fmt == "latex" else ".md" if fmt == "markdown" else ".html"
            rel = f"{artifacts_dir_name}/{table_id}{ext}"
            body_parts.append(r"\\input{" + rel + r"}")
            body_parts.append("")

    if figures:
        for f in figures:
            figure_id = (f.get("figure_id") or "").strip() or "figure"
            title_text = (f.get("title") or "").strip() or "Figure"
            caption = (f.get("caption") or "").strip()
            label = f"fig:{figure_id}"
            img_rel = f"{artifacts_dir_name}/{figure_id}.png"

            body_parts.append(r"\\begin{figure}[H]")
            body_parts.append(r"\\centering")
            body_parts.append(r"\\includegraphics[width=0.95\\linewidth]{" + img_rel + r"}")
            if caption:
                body_parts.append(r"\\caption{" + _escape_latex_text(caption) + r"}")
            else:
                body_parts.append(r"\\caption{" + _escape_latex_text(title_text) + r"}")
            body_parts.append(r"\\label{" + label + r"}")
            body_parts.append(r"\\end{figure}")
            body_parts.append("")

    body_parts.append(r"\\end{document}")
    body_parts.append("")

    return "\n".join(body_parts)


def build_and_compile(
    *,
    base_output_dir: str,
    run_id: str,
    title: str,
    author: str,
    sections: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    exported_files: list[str],
) -> LatexBuildResult:
    base_dir = Path(base_output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    safe_run_id = _sanitize_filename(run_id)
    output_dir = base_dir / f"{safe_run_id}-{_utc_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tex_name = _sanitize_filename(title)
    tex_path = output_dir / f"{tex_name}.tex"

    latex_doc = build_paper_latex(
        title=title,
        author=author,
        sections=sections,
        tables=tables,
        figures=figures,
    )
    tex_path.write_text(latex_doc, encoding="utf-8")

    snapshot_path = output_dir / "state_snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "title": title,
                "author": author,
                "sections": sections,
                "tables": tables,
                "figures": figures,
                "exported_files": exported_files,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    engine = _find_latex_engine()
    if not engine:
        return LatexBuildResult(
            output_dir=str(output_dir),
            tex_path=str(tex_path),
            pdf_path=None,
            exported_files=exported_files,
            engine=None,
            compilation_stdout=None,
            compilation_stderr=None,
        )

    pdf_path, stdout, stderr = _compile_pdf(engine, tex_path, output_dir)

    return LatexBuildResult(
        output_dir=str(output_dir),
        tex_path=str(tex_path),
        pdf_path=str(pdf_path) if pdf_path else None,
        exported_files=exported_files,
        engine=engine,
        compilation_stdout=stdout,
        compilation_stderr=stderr,
    )
