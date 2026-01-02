"""Checkpointer configuration for conversation persistence.

Checkpointers enable:
- Thread-based conversations (resume with thread_id)
- State persistence across agent invocations
- Human-in-the-loop workflows (interrupt and resume)
"""

from pathlib import Path
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config.settings import PROJECT_ROOT


# Default SQLite database path
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "checkpoints.db"


def get_memory_saver() -> MemorySaver:
    """
    Get an in-memory checkpointer (development/testing).
    
    Data is lost when the process ends.
    
    Returns:
        MemorySaver instance.
    """
    return MemorySaver()


def get_sqlite_saver(db_path: Path | str | None = None) -> SqliteSaver:
    """
    Get a SQLite-backed checkpointer (persistent storage).
    
    Args:
        db_path: Path to SQLite database. Defaults to data/checkpoints.db
        
    Returns:
        SqliteSaver instance.
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    
    return SqliteSaver.from_conn_string(str(path))


def get_checkpointer(persistent: bool = False, db_path: Path | str | None = None):
    """
    Get appropriate checkpointer based on environment.
    
    Args:
        persistent: If True, use SQLite. If False, use in-memory.
        db_path: Custom database path for SQLite.
        
    Returns:
        Checkpointer instance.
        
    Example:
        ```python
        checkpointer = get_checkpointer(persistent=True)
        agent = create_react_agent(checkpointer=checkpointer)
        
        # First conversation turn
        result = agent.invoke(
            {"messages": [HumanMessage(content="Hi, I'm Alice")]},
            config={"configurable": {"thread_id": "user-123"}}
        )
        
        # Later - resume same conversation
        result = agent.invoke(
            {"messages": [HumanMessage(content="What's my name?")]},
            config={"configurable": {"thread_id": "user-123"}}
        )
        # Agent remembers: "Your name is Alice"
        ```
    """
    if persistent:
        return get_sqlite_saver(db_path)
    return get_memory_saver()
