"""Simple Flask server for the research intake form.

Serves the intake form and submits to the LangGraph workflow.
"""

import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from src.state.schema import WorkflowState, create_initial_state
from src.state.models import IntakeFormData, DataFile
from src.state.enums import PaperType, ResearchType, TargetJournal
from src.nodes import intake_node

app = Flask(__name__, static_folder="../public")
CORS(app)

# Directory for uploaded data files
UPLOAD_DIR = Path(tempfile.gettempdir()) / "gia_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Mapping from form display values to enum values
PAPER_TYPE_MAP = {
    "Short Article (5-10 pages)": PaperType.SHORT_ARTICLE,
    "Full Paper (30-45 pages)": PaperType.FULL_PAPER,
    "Working Paper": PaperType.WORKING_PAPER,
    "short_article": PaperType.SHORT_ARTICLE,
    "full_paper": PaperType.FULL_PAPER,
    "working_paper": PaperType.WORKING_PAPER,
}

RESEARCH_TYPE_MAP = {
    "Empirical": ResearchType.EMPIRICAL,
    "Theoretical": ResearchType.THEORETICAL,
    "Mixed": ResearchType.MIXED,
    "Literature Review": ResearchType.LITERATURE_REVIEW,
    "Experimental": ResearchType.EXPERIMENTAL,
    "empirical": ResearchType.EMPIRICAL,
    "theoretical": ResearchType.THEORETICAL,
    "mixed": ResearchType.MIXED,
    "literature_review": ResearchType.LITERATURE_REVIEW,
    "experimental": ResearchType.EXPERIMENTAL,
}

TARGET_JOURNAL_MAP = {
    "RFS": TargetJournal.RFS,
    "JFE": TargetJournal.JFE,
    "JF": TargetJournal.JF,
    "JFQA": TargetJournal.JFQA,
    "Other": TargetJournal.OTHER,
    "": TargetJournal.OTHER,
}


@app.route("/")
def index():
    """Serve the intake form."""
    return send_from_directory(app.static_folder, "research_intake_form.html")


@app.route("/submit", methods=["POST"])
def submit_intake():
    """Handle intake form submission."""
    try:
        # Handle file uploads
        data_files = []
        files = request.files.getlist("data_files")
        
        # Create project folder
        title = request.form.get("title", "research_project")
        project_name = title.replace(" ", "_")[:50] or "research_project"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_folder = UPLOAD_DIR / f"{project_name}_{timestamp}"
        project_folder.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            if file.filename:
                # Save file
                file_path = project_folder / file.filename
                file.save(str(file_path))
                
                # If it's a zip, extract it
                if file.filename.endswith(".zip"):
                    extract_dir = project_folder / file.filename.replace(".zip", "")
                    extract_dir.mkdir(exist_ok=True)
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    # Find parquet files in the extracted content
                    for parquet_file in extract_dir.rglob("*.parquet"):
                        data_files.append(
                            DataFile(
                                filename=parquet_file.name,
                                filepath=parquet_file,
                                size_bytes=parquet_file.stat().st_size,
                            )
                        )
                else:
                    data_files.append(
                        DataFile(
                            filename=file.filename,
                            filepath=file_path,
                            size_bytes=file_path.stat().st_size,
                        )
                    )
        
        # Convert form values to enum values
        paper_type_raw = request.form.get("paper_type", "")
        research_type_raw = request.form.get("research_type", "")
        target_journal_raw = request.form.get("target_journal", "")
        
        paper_type = PAPER_TYPE_MAP.get(paper_type_raw, PaperType.FULL_PAPER)
        research_type = RESEARCH_TYPE_MAP.get(research_type_raw, ResearchType.EMPIRICAL)
        target_journal = TARGET_JOURNAL_MAP.get(target_journal_raw, TargetJournal.OTHER)
        
        # Convert DataFile objects to file paths (strings)
        data_file_paths = [str(df.filepath) for df in data_files] if data_files else None
        
        # Check for hypothesis
        has_hypothesis = request.form.get("has_hypothesis") == "yes"
        hypothesis_text = request.form.get("hypothesis", "") if has_hypothesis else None
        
        # Check for data
        has_data = request.form.get("has_data") == "yes"
        data_description = request.form.get("data_description", "") if has_data else None
        
        # Create IntakeFormData
        intake = IntakeFormData(
            title=request.form.get("title", ""),
            research_question=request.form.get("research_question", ""),
            target_journal=target_journal,
            paper_type=paper_type,
            research_type=research_type,
            has_hypothesis=has_hypothesis,
            hypothesis=hypothesis_text or None,
            has_data=has_data,
            data_description=data_description or None,
            data_files=data_file_paths,
            data_sources=request.form.get("data_sources") or None,
            key_variables=request.form.get("key_variables") or None,
            methodology=request.form.get("methodology") or None,
            related_literature=request.form.get("related_literature") or None,
            expected_contribution=request.form.get("expected_contribution") or None,
            constraints=request.form.get("constraints") or None,
            additional_notes=request.form.get("additional_notes") or None,
        )
        
        # Create initial workflow state with form_data as raw dict
        # The intake_node expects form_data as a dict that it will parse
        research_question = request.form.get("research_question", "")
        form_data_dict = intake.model_dump()
        
        state = create_initial_state(
            form_data=form_data_dict,
            original_query=research_question,
            project_title=intake.title,
        )
        
        # Run intake node
        result = intake_node(state)
        
        # Save state to project folder
        state_file = project_folder / "workflow_state.json"
        with open(state_file, "w") as f:
            # Convert state to serializable format
            state_dict = {
                "research_query": result.get("original_query", research_question),
                "status": str(result.get("status", "")),
                "intake_form_data": intake.model_dump() if intake else None,
                "data_files": [df.model_dump() for df in data_files],
                "messages": [str(m) for m in result.get("messages", [])],
                "project_title": result.get("project_title", ""),
            }
            json.dump(state_dict, f, indent=2, default=str)
        
        return jsonify({
            "success": True,
            "project_folder": str(project_folder),
            "files_uploaded": len(data_files),
            "status": str(result.get("status", "unknown")),
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), 500


@app.route("/api/state/<path:project_folder>")
def get_state(project_folder):
    """Get the current workflow state for a project."""
    state_file = Path(project_folder) / "workflow_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "State not found"}), 404


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GIA Research Intake Form Server")
    print("=" * 60)
    print(f"\n📝 Open: http://127.0.0.1:5000")
    print(f"📁 Uploads: {UPLOAD_DIR}")
    print("\nPress Ctrl+C to stop\n")
    app.run(debug=True, port=5000, host="0.0.0.0")
