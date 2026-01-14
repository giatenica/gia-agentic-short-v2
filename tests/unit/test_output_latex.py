import base64
from pathlib import Path

from src.graphs.research_workflow import output_node
from src.config import settings


_PNG_1X1_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB"
    "/aVJY0kAAAAASUVORK5CYII="
)


def test_output_node_writes_latex_and_exports_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path / "outputs"))

    state = {
        "writer_output": {
            "title": "Test Paper",
            "argument_thread": {"thread_id": "abcd1234"},
            "sections": [
                {
                    "section_type": "introduction",
                    "title": "Introduction",
                    "content": "This is a LaTeX body paragraph with \\emph{emphasis}.",
                }
            ],
        },
        "completed_sections": [
            {
                "section_type": "introduction",
                "title": "Introduction",
                "content": "This is a LaTeX body paragraph with \\emph{emphasis}.",
            }
        ],
        "tables": [
            {
                "table_id": "tab_test",
                "title": "Summary Statistics",
                "format": "LATEX",
                "content": "\\begin{table}[H]\\centering\\caption{Test}\\label{tab:summary}\\end{table}",
            }
        ],
        "figures": [
            {
                "figure_id": "fig_test",
                "title": "Time Series Plot",
                "caption": "Test caption",
                "format": "PNG",
                "content_base64": _PNG_1X1_BASE64,
            }
        ],
        "errors": [],
        "human_approved": True,
        "reviewer_output": {"final_paper": "(narrative final paper)"},
    }

    result = output_node(state)

    assert result["output_dir"]
    out_dir = Path(result["output_dir"])
    assert out_dir.exists()

    tex_path = Path(result["latex_tex_path"])
    assert tex_path.exists()

    tex_text = tex_path.read_text(encoding="utf-8")
    assert "\\documentclass" in tex_text
    assert "\\section{Introduction}" in tex_text
    assert "\\input{artifacts/tab_test.tex}" in tex_text
    assert "\\includegraphics" in tex_text

    exported = result.get("exported_artifacts") or []
    assert any(p.endswith("tab_test.tex") for p in exported)
    assert any(p.endswith("fig_test.png") for p in exported)

    # PDF compilation depends on having a LaTeX engine installed.
    if result.get("latex_engine") is None:
        assert result.get("latex_pdf_path") is None
