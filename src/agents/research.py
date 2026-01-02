"""Research agent specialized for web search and information gathering."""

import operator
from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from src.config import settings
from src.tools import tavily_search, get_current_time


class ResearchState(TypedDict):
    """State for the research agent."""

    messages: Annotated[list[AnyMessage], operator.add]
    research_topic: str
    findings: list[str]


RESEARCH_SYSTEM_PROMPT = """You are an expert research assistant powered by Claude. Your task is to thoroughly research topics using web search and provide comprehensive, well-sourced answers.

Research Process:
1. Break down complex questions into searchable queries
2. Search for relevant, up-to-date information
3. Cross-reference multiple sources when possible
4. Synthesize findings into clear, structured responses
5. Always cite your sources with URLs

Guidelines:
- Prioritize recent and authoritative sources
- Acknowledge when information might be outdated or uncertain
- Provide balanced perspectives on controversial topics
- Format responses with clear sections and bullet points when appropriate"""


def create_research_agent(
    model_name: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
):
    """
    Create a research-focused agent with enhanced search capabilities.

    Args:
        model_name: Claude model to use.
        checkpointer: Checkpointer for conversation persistence.
        store: Store for long-term memory.

    Returns:
        Compiled LangGraph research agent.
    """
    tools = [tavily_search, get_current_time]

    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=4096,
        api_key=settings.anthropic_api_key,
    )
    model_with_tools = model.bind_tools(tools)

    system_message = SystemMessage(content=RESEARCH_SYSTEM_PROMPT)

    def call_model(state: ResearchState) -> dict:
        """Call the model with research context."""
        messages = [system_message] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: ResearchState) -> Literal["tools", "__end__"]:
        """Determine if we should continue to tools or end."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # Build graph
    graph = StateGraph(ResearchState)
    graph.add_node("researcher", call_model)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "researcher")
    graph.add_conditional_edges("researcher", should_continue, ["tools", END])
    graph.add_edge("tools", "researcher")

    return graph.compile(checkpointer=checkpointer, store=store)
