# Contributing to GIA Agentic Research System

Thank you for your interest in contributing to GIA! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

---

## Code of Conduct

This project follows a standard code of conduct. Be respectful, inclusive, and professional in all interactions.

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Git
- API keys for development (Anthropic, LangSmith, Tavily)

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR-USERNAME/gia-agentic-short-v2.git
cd gia-agentic-short-v2

# Add upstream remote
git remote add upstream https://github.com/giatenica/gia-agentic-short-v2.git
```

---

## Development Setup

### Install Dependencies

```bash
# Install all dependencies including dev tools
uv sync --all-extras
```

### Configure Environment

```bash
# Copy example environment
cp .env.example .env

# Edit .env with your API keys
# Note: Never commit .env to version control
```

### Verify Setup

```bash
# Run tests to verify setup
uv run pytest tests/ -v

# Run a quick smoke test
uv run python -c "from src.graphs import create_research_workflow; print('OK')"
```

---

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-new-tool` - New features
- `fix/data-loading-bug` - Bug fixes
- `docs/update-readme` - Documentation
- `refactor/cleanup-nodes` - Code refactoring
- `test/add-integration-tests` - Test additions

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Examples:

```
feat(nodes): add retry logic to literature_reviewer

Implements exponential backoff for API rate limits.
Max 3 retries with 2x backoff factor.

Closes #42
```

```
fix(tools): handle empty search results in tavily

Previously crashed on empty results. Now returns
empty list with warning log.
```

---

## Coding Standards

### Python Style

We follow PEP 8 with these additions:

```python
# Use type hints
def process_data(data: list[dict], limit: int = 10) -> dict:
    """Process data with optional limit.
    
    Args:
        data: List of data dictionaries
        limit: Maximum items to process
        
    Returns:
        Processed results dictionary
    """
    pass

# Use dataclasses for data structures
from dataclasses import dataclass

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0

# Use enums for constants
from enum import Enum

class Status(str, Enum):
    PENDING = "pending"
    COMPLETE = "complete"
```

### Import Order

```python
# 1. Standard library
import os
import sys
from datetime import datetime

# 2. Third-party packages
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
import pytest

# 3. Local imports
from src.state.schema import WorkflowState
from src.tools.academic_search import search_semantic_scholar
```

### Documentation

- All public functions must have docstrings
- Use Google-style docstring format
- Include type hints in signatures

```python
def create_research_workflow(
    config: WorkflowConfig | None = None,
) -> StateGraph:
    """Create the main research workflow graph.
    
    Assembles all nodes into the complete academic research workflow.
    
    Args:
        config: Workflow configuration. If None, uses defaults.
        
    Returns:
        Compiled StateGraph ready for execution.
        
    Raises:
        ValueError: If config contains invalid node names.
        
    Example:
        >>> workflow = create_research_workflow()
        >>> result = workflow.invoke(initial_state, config)
    """
```

### Linting and Formatting

```bash
# Run ruff linter
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check src/ tests/ --fix

# Type checking
uv run mypy src/
```

---

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
│   ├── test_nodes.py
│   ├── test_tools.py
│   └── test_state.py
├── integration/       # Integration tests (slower, full workflow)
│   ├── test_workflow.py
│   └── test_hitl.py
└── conftest.py        # Shared fixtures
```

### Writing Tests

```python
# Unit test example
import pytest
from src.tools.academic_search import search_semantic_scholar

def test_search_semantic_scholar_returns_results():
    """Test that search returns list of results."""
    results = search_semantic_scholar("machine learning", limit=5)
    
    assert isinstance(results, list)
    assert len(results) <= 5
    for result in results:
        assert "title" in result
        assert "url" in result


def test_search_semantic_scholar_empty_query():
    """Test that empty query raises ValueError."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        search_semantic_scholar("")


# Use fixtures for complex setup
@pytest.fixture
def mock_state():
    """Create mock workflow state for testing."""
    return {
        "original_query": "Test research question",
        "status": "pending",
    }


def test_intake_node(mock_state):
    """Test intake node processing."""
    from src.nodes.intake import intake_node
    
    result = intake_node(mock_state)
    
    assert result["status"] == "intake_complete"
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/unit/test_nodes.py -v

# Run with coverage
uv run pytest --cov=src tests/

# Run specific test
uv run pytest tests/unit/test_nodes.py::test_intake_node -v

# Run integration tests only
uv run pytest tests/integration/ -v

# Skip slow tests
uv run pytest tests/ -v -m "not slow"
```

### Test Coverage

We aim for >80% coverage. Check coverage with:

```bash
uv run pytest --cov=src --cov-report=html tests/
# Open htmlcov/index.html in browser
```

---

## Pull Request Process

### Before Submitting

1. **Update from upstream**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**
   ```bash
   # Tests
   uv run pytest tests/ -v
   
   # Linting
   uv run ruff check src/ tests/
   
   # Type checking
   uv run mypy src/
   ```

3. **Update documentation** if needed

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring
- [ ] Tests

## Testing
Describe testing performed.

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

1. Submit PR against `main` branch
2. Automated checks must pass
3. Request review from maintainers
4. Address feedback
5. Squash and merge when approved

---

## Release Process

### Version Numbering

We use semantic versioning (SemVer):

- `MAJOR.MINOR.PATCH`
- `1.0.0` → `1.0.1` (patch: bug fixes)
- `1.0.0` → `1.1.0` (minor: new features)
- `1.0.0` → `2.0.0` (major: breaking changes)

### Release Checklist

1. Update `CHANGELOG.md`
2. Update version in `pyproject.toml`
3. Create release PR
4. After merge, create GitHub release
5. Tag with version number

---

## Project Structure

```
src/
├── agents/          # Agent implementations
├── nodes/           # LangGraph workflow nodes
├── graphs/          # Workflow assembly
├── tools/           # LangChain tools
├── state/           # State schema and models
├── errors/          # Error handling
├── review/          # Review criteria
├── cache/           # Caching layer
├── citations/       # Citation management
├── style/           # Writing style enforcement
├── writers/         # Section writers
├── memory/          # Persistence
└── config/          # Configuration
```

### Key Files

- `src/graphs/research_workflow.py` - Main workflow factory
- `src/state/schema.py` - Workflow state definition
- `src/nodes/*.py` - Individual node implementations
- `tests/conftest.py` - Test fixtures

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email security@giatenica.com

---

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- CHANGELOG.md for significant contributions
- README.md acknowledgments section

---

*Contributing guidelines maintained by Gia Tenica. Last updated: January 2026.*
