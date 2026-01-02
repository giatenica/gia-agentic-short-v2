"""Basic utility tools."""

from datetime import datetime
from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """
    Get the current date and time.

    Returns:
        Current date and time as a formatted string.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2 * 3").

    Returns:
        The result of the calculation as a string.
    """
    # Safe evaluation - only allow math operations
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in expression. Only numbers and +, -, *, /, (, ) are allowed."

    try:
        result = eval(expression)  # Safe due to character filtering
        return str(result)
    except Exception as e:
        return f"Error: Could not evaluate expression - {str(e)}"
