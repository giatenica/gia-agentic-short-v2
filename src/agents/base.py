"""Base agent implementation using LangGraph with Anthropic Claude."""

import operator
from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from typing_extensions import TypedDict

from src.config import settings
from src.tools import get_current_time, calculate, web_search_tool


class AgentState(TypedDict):
    """State for the agent graph."""

    messages: Annotated[list[AnyMessage], operator.add]


def create_model(
    model_name: str | None = None,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> ChatAnthropic:
    """
    Create a ChatAnthropic model instance.

    Args:
        model_name: Claude model to use (default from settings).
        temperature: Sampling temperature (0 = deterministic).
        max_tokens: Maximum tokens in response.

    Returns:
        Configured ChatAnthropic instance.
    """
    return ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=settings.anthropic_api_key,
    )


def create_react_agent(
    tools: list | None = None,
    model_name: str | None = None,
    system_prompt: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
):
    """
    Create a ReAct-style agent using LangGraph.

    This agent follows the Reasoning + Acting pattern:
    1. Receives user input
    2. Reasons about what to do
    3. Optionally calls tools
    4. Returns final response

    Args:
        tools: List of tools the agent can use. Defaults to basic tools.
        model_name: Claude model to use.
        system_prompt: Custom system prompt for the agent.
        checkpointer: Checkpointer for conversation persistence (thread memory).
        store: Store for long-term memory across sessions.

    Returns:
        Compiled LangGraph agent.
        
    Example with memory:
        ```python
        from src.memory import get_checkpointer, get_memory_store
        
        checkpointer = get_checkpointer(persistent=True)
        store = get_memory_store()
        
        agent = create_react_agent(
            checkpointer=checkpointer,
            store=store
        )
        
        # Use thread_id for conversation persistence
        config = {"configurable": {"thread_id": "user-123"}}
        result = agent.invoke({"messages": [...]}, config=config)
        ```
    """
    # Default tools if none provided
    if tools is None:
        tools = [get_current_time, calculate, web_search_tool]

    # Create model and bind tools
    model = create_model(model_name=model_name)
    model_with_tools = model.bind_tools(tools)

    # Default system prompt
    default_system = """You are a helpful AI assistant powered by Claude. You have access to tools that help you answer questions accurately.

When answering questions:
1. Think step by step about what information you need
2. Use available tools when they can help provide accurate information
3. If you use web search, cite your sources
4. Be concise but thorough in your responses

Available tools:
- get_current_time: Get the current date and time
- calculate: Perform mathematical calculations
- tavily_search_results_json: Search the web for current information"""

    system_message = SystemMessage(content=system_prompt or default_system)

    # Define the agent node
    def call_model(state: AgentState) -> dict:
        """Call the model with current state."""
        messages = [system_message] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    # Define routing logic
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        """Determine if we should continue to tools or end."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))

    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", END])
    graph.add_edge("tools", "agent")

    # Compile with optional checkpointer and store
    return graph.compile(checkpointer=checkpointer, store=store)
