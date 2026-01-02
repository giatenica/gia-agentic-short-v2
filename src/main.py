"""Main entrypoint for running agents."""

import uuid
from langchain_core.messages import HumanMessage

from src.config import settings
from src.agents import create_react_agent, create_research_agent
from src.memory import get_checkpointer, get_memory_store


# Global memory components (persistent across CLI session)
_checkpointer = None
_store = None


def get_memory_components(persistent: bool = True):
    """Get or create memory components."""
    global _checkpointer, _store
    if _checkpointer is None:
        _checkpointer = get_checkpointer(persistent=persistent)
    if _store is None:
        _store = get_memory_store()
    return _checkpointer, _store


def run_agent(
    query: str, 
    agent_type: str = "react",
    thread_id: str | None = None,
    persistent: bool = True,
) -> str:
    """
    Run an agent with the given query.

    Args:
        query: The user's question or task.
        agent_type: Type of agent to use ("react" or "research").
        thread_id: Thread ID for conversation persistence. If None, creates new thread.
        persistent: Whether to use persistent storage (SQLite) or in-memory.

    Returns:
        The agent's response.
    """
    # Validate settings
    errors = settings.validate()
    if errors:
        return f"Configuration errors:\n" + "\n".join(f"- {e}" for e in errors)

    # Get memory components
    checkpointer, store = get_memory_components(persistent=persistent)
    
    # Create the appropriate agent with memory
    if agent_type == "research":
        agent = create_research_agent(checkpointer=checkpointer, store=store)
    else:
        agent = create_react_agent(checkpointer=checkpointer, store=store)

    # Config with thread_id for conversation persistence
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

    # Run the agent
    result = agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)

    # Extract final response
    final_message = result["messages"][-1]
    return final_message.content


def main():
    """Interactive CLI for testing agents."""
    print("=" * 60)
    print("GIA Agentic System - LangGraph + Anthropic Claude")
    print("=" * 60)

    # Check configuration
    errors = settings.validate()
    if errors:
        print("\n⚠️  Configuration Issues:")
        for error in errors:
            print(f"   - {error}")
        print("\nPlease check your .env file.")
        return

    print("\n✅ Configuration valid")
    print(f"   Model: {settings.default_model}")
    print(f"   LangSmith Tracing: {settings.langsmith_tracing}")
    print(f"   Project: {settings.langsmith_project}")
    
    # Generate a thread_id for this session
    thread_id = str(uuid.uuid4())
    print(f"   Thread ID: {thread_id[:8]}... (conversation memory enabled)")

    print("\nCommands:")
    print("  /research <query> - Use research agent")
    print("  /new - Start new conversation thread")
    print("  /quit - Exit")
    print("  <query> - Use default ReAct agent")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n🤖 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("Goodbye!")
                break
                
            if user_input.lower() == "/new":
                thread_id = str(uuid.uuid4())
                print(f"🔄 New thread started: {thread_id[:8]}...")
                continue

            if user_input.startswith("/research "):
                query = user_input[10:]
                agent_type = "research"
            else:
                query = user_input
                agent_type = "react"

            print(f"\n🔄 Processing with {agent_type} agent...")
            response = run_agent(query, agent_type, thread_id=thread_id)
            print(f"\n🤖 Agent: {response}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
