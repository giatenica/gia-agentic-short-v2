"""DATA_ACQUISITION node for intelligent external data fetching.

Sprint 14: This node analyzes data requirements from the research plan,
checks what data is available from uploads, and autonomously fetches
missing data from external sources (yfinance, FRED, CoinGecko, etc.).

Workflow:
1. Parse data requirements from research_plan
2. Check which requirements are satisfied by loaded_datasets
3. For gaps, query DataSourceRegistry for appropriate source
4. Execute acquisition tools (acquire_stock_data, acquire_economic_indicator, etc.)
5. If tools fail, generate and execute custom Python code
6. If all fails, create AcquisitionFailure and optionally interrupt for human input
7. Return updated state with acquired_datasets
"""

from datetime import datetime, timezone
from typing import Any, Literal
import logging

from langchain_core.messages import AIMessage, HumanMessage

from src.state.enums import (
    ResearchStatus,
    DataRequirementPriority,
    AcquisitionStatus,
    CodeExecutionStatus,
)
from src.state.models import (
    DataRequirement,
    DataAcquisitionPlan,
    DataAcquisitionTask,
    AcquisitionFailure,
    AcquiredDataset,
    CodeSnippet,
    TimeRange,
    WorkflowError,
)
from src.state.schema import WorkflowState
from src.tools.data_loading import get_registry

# Import Sprint 13 external data tools
from src.tools.external_data import (
    acquire_stock_data,
    acquire_economic_indicator,
    acquire_crypto_data,
    fetch_api_json,
    list_available_data_sources,
)

# Import Sprint 14 code execution
from src.tools.code_execution import (
    execute_python_code,
    validate_python_code,
)

# Import data source registry
try:
    from src.data_sources import DataSourceRegistry, get_source
    HAS_DATA_SOURCES = True
except ImportError:
    HAS_DATA_SOURCES = False


logger = logging.getLogger(__name__)


# =============================================================================
# Data Requirement Parsing
# =============================================================================


def _parse_data_requirements_from_plan(state: WorkflowState) -> list[DataRequirement]:
    """Extract data requirements from the research plan.
    
    Looks for explicit data_requirements in the research plan,
    or infers them from the methodology and analysis approach.
    """
    research_plan = state.get("research_plan")
    requirements: list[DataRequirement] = []
    
    if not research_plan:
        logger.warning("No research plan found in state")
        return requirements
    
    # Handle both dict and model forms
    if isinstance(research_plan, dict):
        plan_data = research_plan
    elif hasattr(research_plan, "model_dump"):
        plan_data = research_plan.model_dump()
    else:
        plan_data = {}
    
    # Check for explicit data_requirements field
    explicit_reqs = plan_data.get("data_requirements", [])
    if explicit_reqs:
        for req in explicit_reqs:
            if isinstance(req, dict):
                requirements.append(DataRequirement(**req))
            elif isinstance(req, DataRequirement):
                requirements.append(req)
    
    # Infer from methodology if no explicit requirements
    if not requirements:
        methodology = plan_data.get("methodology", "")
        analysis_approach = plan_data.get("analysis_approach", "")
        research_question = state.get("original_query", "")
        
        # Infer stock data requirement if methodology mentions financial analysis
        if any(kw in methodology.lower() for kw in ["stock", "equity", "return", "price"]):
            requirements.append(DataRequirement(
                variable_name="stock_prices",
                data_type="stock_prices",
                description="Stock price data for financial analysis",
                required_fields=["close", "volume"],
                priority=DataRequirementPriority.REQUIRED,
            ))
        
        # Infer economic indicators if mentioned
        if any(kw in methodology.lower() for kw in ["macro", "economic", "gdp", "inflation"]):
            requirements.append(DataRequirement(
                variable_name="economic_indicators",
                data_type="economic_indicator",
                description="Macroeconomic indicators",
                priority=DataRequirementPriority.PREFERRED,
            ))
    
    return requirements


def _check_requirements_against_uploads(
    requirements: list[DataRequirement],
    loaded_datasets: list[str],
    state: WorkflowState,
) -> tuple[list[str], list[DataRequirement]]:
    """Check which requirements are satisfied by uploaded data.
    
    Returns:
        Tuple of (satisfied_requirements, unsatisfied_requirements)
    """
    registry = get_registry()
    satisfied: list[str] = []
    unsatisfied: list[DataRequirement] = []
    
    for req in requirements:
        found = False
        
        # Check each loaded dataset
        for dataset_name in loaded_datasets:
            try:
                df = registry.get(dataset_name)
                if df is None:
                    continue
                
                columns = [c.lower() for c in df.columns]
                
                # Check if required fields are present
                required_fields = [f.lower() for f in req.required_fields]
                if required_fields:
                    if all(f in columns for f in required_fields):
                        req.matched_dataset = dataset_name
                        satisfied.append(req.requirement_id)
                        found = True
                        break
                
                # Check by data type matching
                if req.data_type == "stock_prices" and any(f in columns for f in ["close", "adj close", "price"]):
                    req.matched_dataset = dataset_name
                    satisfied.append(req.requirement_id)
                    found = True
                    break
                    
            except Exception as e:
                logger.debug(f"Error checking dataset {dataset_name}: {e}")
                continue
        
        if not found:
            unsatisfied.append(req)
    
    return satisfied, unsatisfied


# =============================================================================
# Data Acquisition Logic
# =============================================================================


def _find_acquisition_source(requirement: DataRequirement) -> tuple[str, str, dict[str, Any]] | None:
    """Find an appropriate data source for a requirement.
    
    Returns:
        Tuple of (source_name, tool_name, params) or None if no source found
    """
    data_type = requirement.data_type.lower()
    
    # Map data types to tools and sources
    if data_type in ["stock_prices", "stock_data", "equity_prices"]:
        params = {}
        if requirement.entities:
            params["ticker"] = requirement.entities[0]
        if requirement.time_range:
            params["start_date"] = requirement.time_range.start_date
            params["end_date"] = requirement.time_range.end_date
        return ("yfinance", "acquire_stock_data", params)
    
    elif data_type in ["economic_indicator", "macro_data", "fred_data"]:
        params = {"indicator": "GDP"}  # Default
        if requirement.entities:
            params["indicator"] = requirement.entities[0]
        if requirement.time_range:
            params["start_date"] = requirement.time_range.start_date
            params["end_date"] = requirement.time_range.end_date
        return ("fred", "acquire_economic_indicator", params)
    
    elif data_type in ["crypto_prices", "cryptocurrency", "bitcoin"]:
        params = {"coin": "bitcoin", "days": 365}
        if requirement.entities:
            params["coin"] = requirement.entities[0]
        return ("coingecko", "acquire_crypto_data", params)
    
    return None


def _execute_acquisition_task(task: DataAcquisitionTask) -> tuple[bool, str | None, str | None]:
    """Execute a single acquisition task.
    
    Returns:
        Tuple of (success, dataset_name, error_message)
    """
    tool_name = task.tool_to_use
    params = task.params
    
    try:
        if tool_name == "acquire_stock_data":
            result = acquire_stock_data.invoke(params)
        elif tool_name == "acquire_economic_indicator":
            result = acquire_economic_indicator.invoke(params)
        elif tool_name == "acquire_crypto_data":
            result = acquire_crypto_data.invoke(params)
        elif tool_name == "fetch_api_json":
            result = fetch_api_json.invoke(params)
        else:
            return False, None, f"Unknown tool: {tool_name}"
        
        # Check result
        if isinstance(result, dict):
            status = result.get("status", "error")
            if status == "success":
                dataset_name = result.get("dataset_name")
                return True, dataset_name, None
            else:
                error = result.get("error", "Unknown error")
                return False, None, error
        
        return False, None, "Invalid result format"
        
    except Exception as e:
        logger.error(f"Acquisition task failed: {e}")
        return False, None, str(e)


def _generate_custom_code(requirement: DataRequirement) -> str | None:
    """Generate custom Python code for data acquisition.
    
    Used when built-in tools cannot satisfy the requirement.
    """
    data_type = requirement.data_type.lower()
    
    if data_type == "stock_prices" and requirement.entities:
        ticker = requirement.entities[0]
        start = requirement.time_range.start_date if requirement.time_range else "2020-01-01"
        end = requirement.time_range.end_date if requirement.time_range else "2024-12-31"
        
        return f'''
import pandas as pd
import requests

# Alternative stock data fetch using Alpha Vantage or similar
url = f"https://www.alphavantage.co/query"
params = {{
    "function": "TIME_SERIES_DAILY_ADJUSTED",
    "symbol": "{ticker}",
    "outputsize": "full",
    "apikey": "demo"  # Replace with actual key
}}

response = requests.get(url, params=params, timeout=30)
data = response.json()

if "Time Series (Daily)" in data:
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    result = {{"status": "success", "rows": len(df)}}
else:
    result = {{"status": "error", "error": "Failed to fetch data"}}
'''
    
    return None


# =============================================================================
# Main Node Function
# =============================================================================


def data_acquisition_node(state: WorkflowState) -> dict[str, Any]:
    """Intelligent data acquisition node.
    
    This node:
    1. Parses data requirements from the research plan
    2. Checks which requirements are met by uploaded data
    3. Creates acquisition tasks for missing data
    4. Executes acquisition tools
    5. Falls back to custom code if tools fail
    6. Records successes and failures
    
    Args:
        state: Current workflow state
        
    Returns:
        State updates with acquired_datasets and acquisition_failures
    """
    logger.info("Starting data acquisition node")
    
    # Initialize results
    acquired_datasets: list[AcquiredDataset] = []
    acquisition_failures: list[AcquisitionFailure] = []
    generated_snippets: list[CodeSnippet] = []
    
    # Get loaded datasets
    loaded_datasets = state.get("loaded_datasets", [])
    
    # Parse requirements
    requirements = _parse_data_requirements_from_plan(state)
    logger.info(f"Found {len(requirements)} data requirements")
    
    if not requirements:
        # No requirements found, proceed with available data
        return {
            "data_acquisition_plan": DataAcquisitionPlan(
                requirements=[],
                available_in_upload=loaded_datasets,
                to_acquire=[],
                estimated_time="0 seconds",
            ).model_dump(),
            "acquired_datasets": [],
            "acquisition_failures": [],
            "generated_code_snippets": [],
            "messages": [AIMessage(content="No explicit data requirements found. Proceeding with uploaded data.")],
        }
    
    # Check requirements against uploads
    satisfied, unsatisfied = _check_requirements_against_uploads(
        requirements, loaded_datasets, state
    )
    logger.info(f"Requirements: {len(satisfied)} satisfied by uploads, {len(unsatisfied)} need acquisition")
    
    # Create acquisition tasks for unsatisfied requirements
    tasks: list[DataAcquisitionTask] = []
    for req in unsatisfied:
        source_info = _find_acquisition_source(req)
        if source_info:
            source, tool, params = source_info
            task = DataAcquisitionTask(
                requirement=req,
                source=source,
                tool_to_use=tool,
                params=params,
            )
            tasks.append(task)
        else:
            # No source found, mark as failure
            acquisition_failures.append(AcquisitionFailure(
                requirement=req,
                attempted_sources=[],
                error_messages=["No suitable data source found for this requirement"],
                user_action_needed=f"Please upload data for: {req.description or req.variable_name}",
            ))
    
    # Execute acquisition tasks
    for task in tasks:
        logger.info(f"Executing acquisition task: {task.tool_to_use} for {task.requirement.variable_name}")
        
        success, dataset_name, error = _execute_acquisition_task(task)
        
        if success and dataset_name:
            task.status = AcquisitionStatus.SUCCESS
            task.result_dataset = dataset_name
            
            # Record acquired dataset
            registry = get_registry()
            df = registry.get(dataset_name)
            
            acquired_datasets.append(AcquiredDataset(
                dataset_name=dataset_name,
                source=task.source,
                requirement_id=task.requirement.requirement_id,
                row_count=len(df) if df is not None else 0,
                column_count=len(df.columns) if df is not None else 0,
            ))
            
            logger.info(f"Successfully acquired: {dataset_name}")
        else:
            task.status = AcquisitionStatus.FAILED
            task.error_message = error
            
            # Try custom code as fallback
            custom_code = _generate_custom_code(task.requirement)
            if custom_code:
                logger.info(f"Trying custom code for {task.requirement.variable_name}")
                
                snippet = CodeSnippet(
                    code=custom_code,
                    description=f"Custom acquisition for {task.requirement.variable_name}",
                    requirement_id=task.requirement.requirement_id,
                )
                
                # Validate and execute
                validation = validate_python_code.invoke({"code": custom_code})
                if validation.get("valid"):
                    result = execute_python_code.invoke({
                        "code": custom_code,
                        "description": snippet.description,
                        "timeout_seconds": 30,
                    })
                    
                    if result.get("status") == "success":
                        snippet.status = CodeExecutionStatus.SUCCESS
                        snippet.execution_result = result.get("output", "")
                        generated_snippets.append(snippet)
                        logger.info("Custom code executed successfully")
                    else:
                        snippet.status = CodeExecutionStatus.ERROR
                        snippet.error_message = result.get("error", "Unknown error")
                        generated_snippets.append(snippet)
                        
                        # Record failure
                        acquisition_failures.append(AcquisitionFailure(
                            requirement=task.requirement,
                            attempted_sources=[task.source, "custom_code"],
                            error_messages=[error or "", snippet.error_message or ""],
                            user_action_needed=f"Could not acquire {task.requirement.variable_name}. Please upload the data manually.",
                        ))
                else:
                    snippet.status = CodeExecutionStatus.VALIDATION_FAILED
                    snippet.error_message = validation.get("error", "Validation failed")
                    generated_snippets.append(snippet)
                    
                    acquisition_failures.append(AcquisitionFailure(
                        requirement=task.requirement,
                        attempted_sources=[task.source],
                        error_messages=[error or ""],
                        user_action_needed=f"Could not acquire {task.requirement.variable_name}. Please upload the data manually.",
                    ))
            else:
                # No custom code fallback, record failure
                acquisition_failures.append(AcquisitionFailure(
                    requirement=task.requirement,
                    attempted_sources=[task.source],
                    error_messages=[error or "Unknown error"],
                    user_action_needed=f"Could not acquire {task.requirement.variable_name}. Please upload the data manually.",
                ))
    
    # Build acquisition plan
    plan = DataAcquisitionPlan(
        requirements=requirements,
        available_in_upload=satisfied,
        to_acquire=tasks,
        estimated_time=f"{len(tasks) * 5} seconds",
    )
    
    # Update loaded_datasets with newly acquired data
    new_datasets = [a.dataset_name for a in acquired_datasets]
    all_loaded = list(set(loaded_datasets + new_datasets))
    
    # Build summary message
    summary_parts = []
    if satisfied:
        summary_parts.append(f"{len(satisfied)} requirements satisfied by uploaded data")
    if acquired_datasets:
        summary_parts.append(f"{len(acquired_datasets)} datasets acquired from external sources")
    if acquisition_failures:
        summary_parts.append(f"{len(acquisition_failures)} requirements could not be fulfilled")
    
    summary = "Data acquisition complete. " + "; ".join(summary_parts) + "."
    
    return {
        "data_acquisition_plan": plan.model_dump(),
        "acquired_datasets": [a.model_dump() for a in acquired_datasets],
        "acquisition_failures": [f.model_dump() for f in acquisition_failures],
        "generated_code_snippets": [s.model_dump() for s in generated_snippets],
        "loaded_datasets": all_loaded,
        "messages": [AIMessage(content=summary)],
    }


# =============================================================================
# Routing Functions
# =============================================================================


def route_after_acquisition(state: WorkflowState) -> Literal["data_analyst", "human_interrupt", "__end__"]:
    """Route after data acquisition based on results.
    
    - If all required data is available: proceed to data_analyst
    - If critical data is missing: route to human_interrupt
    - If only optional data is missing: proceed to data_analyst
    """
    failures = state.get("acquisition_failures", [])
    
    if not failures:
        return "data_analyst"
    
    # Check if any required data is missing
    for failure in failures:
        if isinstance(failure, dict):
            req = failure.get("requirement", {})
            priority = req.get("priority", "required")
        else:
            priority = failure.requirement.priority.value if hasattr(failure.requirement.priority, "value") else str(failure.requirement.priority)
        
        if priority == "required":
            return "human_interrupt"
    
    # Only optional data missing, proceed
    return "data_analyst"


def should_skip_acquisition(state: WorkflowState) -> bool:
    """Check if data acquisition should be skipped.
    
    Skip if:
    - No research plan (error state)
    - Research is theoretical (no data needed)
    """
    research_plan = state.get("research_plan")
    if not research_plan:
        return True
    
    research_type = state.get("research_type", "")
    if research_type.lower() in ["theoretical", "literature_review"]:
        return True
    
    return False


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "data_acquisition_node",
    "route_after_acquisition",
    "should_skip_acquisition",
]
