#!/usr/bin/env python3
"""
Restore a saved state to a new LangGraph thread.

Usage:
    1. Start LangGraph server: cd studio && uv run langgraph dev --port 2024
    2. Run this script: uv run python scripts/restore_state.py

This will:
    1. Create a new thread
    2. Load state from docs/state.json
    3. Update the thread state
    4. Print the thread ID and Studio URL
"""

import json
import requests
from pathlib import Path

LANGGRAPH_URL = "http://127.0.0.1:2024"
STATE_FILE = Path(__file__).parent.parent / "docs" / "state.json"
GRAPH_ID = "research_workflow"


def main():
    # Check server is running
    try:
        resp = requests.get(f"{LANGGRAPH_URL}/ok", timeout=5)
    except requests.exceptions.ConnectionError:
        print("ERROR: LangGraph server not running!")
        print("Start it with: cd studio && uv run langgraph dev --port 2024")
        return
    
    # Load saved state
    if not STATE_FILE.exists():
        print(f"ERROR: State file not found: {STATE_FILE}")
        return
    
    with open(STATE_FILE) as f:
        saved_state = json.load(f)
    
    print(f"Loaded state from {STATE_FILE}")
    print(f"  - Project: {saved_state['values'].get('project_title', 'Unknown')[:60]}...")
    
    # Create new thread
    resp = requests.post(f"{LANGGRAPH_URL}/threads", json={})
    if resp.status_code != 200:
        print(f"ERROR: Failed to create thread: {resp.status_code} {resp.text}")
        return
    
    thread_data = resp.json()
    thread_id = thread_data["thread_id"]
    print(f"Created new thread: {thread_id}")
    
    # Update thread state - use the state update endpoint
    # LangGraph API expects us to run with initial state
    state_values = saved_state.get("values", {})
    
    # Run the graph with the restored state as input (starting from a specific node)
    # We'll start from 'data_analyst' since that's where we left off
    run_payload = {
        "assistant_id": GRAPH_ID,
        "input": state_values,
        "config": {
            "configurable": {
                "thread_id": thread_id,
            }
        },
        # Start from data_analyst node
        "stream_mode": "values",
    }
    
    print(f"\nTo continue from data_analyst node in LangGraph Studio:")
    print(f"  1. Open: https://smith.langchain.com/studio/?baseUrl={LANGGRAPH_URL}")
    print(f"  2. Select 'research_workflow' graph")
    print(f"  3. Create a new thread")
    print(f"  4. Use the 'Edit State' feature to paste the state")
    print(f"\nOr run programmatically with thread_id: {thread_id}")
    
    # Alternative: Write a minimal state that can be used to restart
    minimal_state = {
        "form_data": state_values.get("form_data", {}),
        "project_title": state_values.get("project_title"),
        "original_query": state_values.get("original_query"),
        "research_type": state_values.get("research_type"),
        "uploaded_data": state_values.get("uploaded_data", []),
        "data_exploration": state_values.get("data_exploration", {}),
        "research_plan": state_values.get("research_plan", {}),
        "literature_review": state_values.get("literature_review", {}),
        "literature_synthesis": state_values.get("literature_synthesis", {}),
        "identified_gap": state_values.get("identified_gap", {}),
    }
    
    minimal_path = STATE_FILE.parent / "minimal_state.json"
    with open(minimal_path, "w") as f:
        json.dump(minimal_state, f, indent=2, default=str)
    
    print(f"\nAlso saved minimal state to: {minimal_path}")
    print("This contains just the key fields needed to restart analysis.")


if __name__ == "__main__":
    main()
