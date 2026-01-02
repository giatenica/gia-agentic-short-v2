"""Long-term memory store for cross-session persistence.

The memory store enables:
- Storing facts, preferences, and summaries
- Semantic search over stored memories
- Namespaced storage (per-user, per-topic, etc.)
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from langgraph.store.memory import InMemoryStore

from src.config.settings import PROJECT_ROOT


class MemoryNamespace(str, Enum):
    """Predefined namespaces for organizing memories."""
    
    USER_FACTS = "user_facts"       # Facts about users (name, preferences)
    USER_PREFERENCES = "user_prefs"  # User preferences and settings
    CONVERSATION_SUMMARIES = "summaries"  # Summaries of past conversations
    LEARNED_KNOWLEDGE = "knowledge"  # Domain knowledge learned over time


@dataclass
class Memory:
    """A stored memory item."""
    
    namespace: str
    key: str
    value: dict[str, Any]
    
    def __str__(self) -> str:
        return f"[{self.namespace}:{self.key}] {self.value}"


def get_memory_store() -> InMemoryStore:
    """
    Get an in-memory store for long-term memory.
    
    Note: For production, consider using a persistent store like
    PostgresStore or a vector database for semantic search.
    
    Returns:
        InMemoryStore instance.
        
    Example:
        ```python
        store = get_memory_store()
        
        # Store a user fact
        store.put(
            namespace=("user_facts", "user-123"),
            key="name",
            value={"fact": "User's name is Alice", "confidence": 1.0}
        )
        
        # Retrieve memories
        memories = store.search(
            namespace=("user_facts", "user-123"),
        )
        ```
    """
    return InMemoryStore()


class MemoryManager:
    """
    Helper class for managing long-term memories.
    
    Provides a higher-level interface for common memory operations.
    """
    
    def __init__(self, store: InMemoryStore | None = None):
        self.store = store or get_memory_store()
    
    def store_user_fact(
        self, 
        user_id: str, 
        fact_key: str, 
        fact: str, 
        confidence: float = 1.0
    ) -> None:
        """Store a fact about a user."""
        self.store.put(
            namespace=(MemoryNamespace.USER_FACTS.value, user_id),
            key=fact_key,
            value={"fact": fact, "confidence": confidence}
        )
    
    def get_user_facts(self, user_id: str) -> list[dict]:
        """Retrieve all facts about a user."""
        results = self.store.search(
            namespace=(MemoryNamespace.USER_FACTS.value, user_id),
        )
        return [item.value for item in results]
    
    def store_preference(
        self, 
        user_id: str, 
        pref_key: str, 
        preference: Any
    ) -> None:
        """Store a user preference."""
        self.store.put(
            namespace=(MemoryNamespace.USER_PREFERENCES.value, user_id),
            key=pref_key,
            value={"preference": preference}
        )
    
    def get_preferences(self, user_id: str) -> dict[str, Any]:
        """Retrieve all preferences for a user."""
        results = self.store.search(
            namespace=(MemoryNamespace.USER_PREFERENCES.value, user_id),
        )
        return {item.key: item.value.get("preference") for item in results}
    
    def store_summary(
        self, 
        thread_id: str, 
        summary: str,
        turn_count: int = 0
    ) -> None:
        """Store a conversation summary."""
        self.store.put(
            namespace=(MemoryNamespace.CONVERSATION_SUMMARIES.value,),
            key=thread_id,
            value={"summary": summary, "turn_count": turn_count}
        )
    
    def get_summary(self, thread_id: str) -> str | None:
        """Retrieve a conversation summary."""
        results = self.store.search(
            namespace=(MemoryNamespace.CONVERSATION_SUMMARIES.value,),
        )
        for item in results:
            if item.key == thread_id:
                return item.value.get("summary")
        return None
