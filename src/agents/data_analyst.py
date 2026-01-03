"""Data Analyst agent for comprehensive data analysis with all new tools.

This agent has access to:
- Data loading (CSV, Parquet, Excel, Stata, SPSS, JSON, SQL, ZIP)
- Data profiling with LLM-generated descriptions
- Data transformation (filter, aggregate, merge, pivot, etc.)
- Statistical analysis (t-test, ANOVA, chi-square, regression, etc.)
- LLM interpretation for academic prose generation
"""

import operator
from datetime import datetime
from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from src.config import settings
from src.tools.data_loading import get_data_loading_tools
from src.tools.data_profiling import get_profiling_tools
from src.tools.data_transformation import get_transformation_tools
from src.tools.analysis import get_analysis_tools
from src.tools.llm_interpretation import get_interpretation_tools


class DataAnalystState(TypedDict):
    """State for the data analyst agent."""

    messages: Annotated[list[AnyMessage], operator.add]


def get_all_data_tools() -> list:
    """Get all data analysis tools combined."""
    return (
        get_data_loading_tools() +
        get_profiling_tools() +
        get_transformation_tools() +
        get_analysis_tools() +
        get_interpretation_tools()
    )


DATA_ANALYST_SYSTEM_PROMPT = """You are an expert data analyst and statistical researcher. Your role is to help researchers analyze their data rigorously and generate publication-quality results.

Current date: {current_date}

## Available Tools

### Data Loading
- `load_data`: Load data from any format (CSV, Parquet, Excel, Stata, SPSS, JSON, SQL, ZIP)
- `query_data`: Run SQL queries on loaded datasets
- `list_datasets`: Show all loaded datasets
- `get_dataset_info`: Get schema and metadata for a dataset
- `sample_data`: Get a sample of rows from a dataset

### Data Profiling
- `profile_dataset`: Generate comprehensive statistical profile
- `describe_dataset`: Generate LLM-powered narrative description
- `describe_variable`: Deep dive into a single variable

### Data Transformation
- `filter_data`: Filter rows based on conditions
- `select_columns`: Select specific columns
- `aggregate_data`: Group and aggregate data
- `merge_datasets`: Join datasets together
- `create_variable`: Create new computed columns
- `handle_missing`: Handle missing values (drop, impute, etc.)
- `encode_categorical`: Encode categorical variables
- `pivot_data`: Reshape data from long to wide
- `melt_data`: Reshape data from wide to long

### Statistical Analysis
- `execute_descriptive_stats`: Summary statistics
- `compute_correlation_matrix`: Correlation analysis
- `run_ttest`: Independent/paired t-tests with effect sizes
- `run_anova`: ANOVA with post-hoc tests
- `run_chi_square`: Chi-square test of independence
- `run_mann_whitney`: Non-parametric comparison
- `run_kruskal_wallis`: Non-parametric ANOVA alternative
- `run_normality_test`: Check distribution normality
- `run_ols_regression`: OLS regression with full diagnostics
- `run_logistic_regression`: Logistic regression with odds ratios

### LLM Interpretation
- `interpret_regression`: Generate academic interpretation of regression
- `interpret_hypothesis_test`: Generate interpretation of statistical tests
- `summarize_findings`: Synthesize multiple results into narrative
- `generate_methods_section`: Generate APA-style methods section

## Guidelines

1. **Start by loading and profiling data** - Always understand your data first
2. **Check assumptions** - Run normality tests before parametric tests
3. **Report effect sizes** - Statistical significance alone is insufficient
4. **Use appropriate tests** - Non-parametric when assumptions violated
5. **Generate interpretations** - Use LLM tools for academic prose
6. **Be rigorous** - Never fabricate or misrepresent results

## Output Format

When presenting results:
- Report exact p-values (not just < 0.05)
- Include confidence intervals where applicable
- Report effect sizes with interpretations
- Cite the statistical test used
- Note any assumption violations
"""


def create_data_analyst_agent(
    model_name: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
):
    """
    Create a data analyst agent with comprehensive analysis tools.

    Args:
        model_name: Claude model to use (defaults to claude-sonnet-4-5-20250929).
        checkpointer: Checkpointer for conversation persistence.
        store: Store for long-term memory.

    Returns:
        Compiled LangGraph data analyst agent.
    """
    tools = get_all_data_tools()
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_prompt = DATA_ANALYST_SYSTEM_PROMPT.format(current_date=current_date)

    model = ChatAnthropic(
        model=model_name or settings.default_model,
        temperature=0,
        max_tokens=8192,  # Larger for detailed analysis
        api_key=settings.anthropic_api_key,
    )
    model_with_tools = model.bind_tools(tools)

    system_message = SystemMessage(content=system_prompt)

    def call_model(state: DataAnalystState) -> dict:
        """Call the model with analysis context."""
        messages = [system_message] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: DataAnalystState) -> Literal["tools", "__end__"]:
        """Determine if we should continue to tools or end."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # Build graph
    graph = StateGraph(DataAnalystState)
    graph.add_node("analyst", call_model)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "analyst")
    graph.add_conditional_edges("analyst", should_continue, ["tools", END])
    graph.add_edge("tools", "analyst")

    return graph.compile(checkpointer=checkpointer, store=store)
