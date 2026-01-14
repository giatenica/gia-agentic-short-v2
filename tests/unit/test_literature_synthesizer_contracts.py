"""Contract tests for literature synthesizer state outputs.

These tests are intentionally lightweight and avoid LLM calls.
"""

from src.nodes.literature_synthesizer import literature_synthesizer_node
from src.state.enums import ResearchStatus


def test_empty_search_results_returns_dict_literature_synthesis():
    state = {
        "original_query": "Test research question",
        "search_results": [],
    }

    result = literature_synthesizer_node(state)

    assert result["status"] == ResearchStatus.GAP_IDENTIFICATION_COMPLETE
    assert isinstance(result.get("literature_synthesis"), dict)
    assert result["literature_synthesis"].get("papers_analyzed") == 0
    assert isinstance(result["literature_synthesis"].get("key_findings"), list)
    assert isinstance(result["literature_synthesis"].get("theoretical_frameworks"), list)
    assert isinstance(result["literature_synthesis"].get("methodological_approaches"), list)
    assert result.get("refined_query") == "Test research question"
