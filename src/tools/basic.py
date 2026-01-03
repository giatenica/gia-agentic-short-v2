"""Basic utility tools."""

import ast
import operator
from datetime import datetime

from langchain_core.tools import tool


# Safe operators for expression evaluation
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float | int:
    """Safely evaluate an AST node containing only math operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _SAFE_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


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
    try:
        # Parse expression into AST and evaluate safely
        tree = ast.parse(expression, mode='eval')
        result = _safe_eval(tree)
        return str(result)
    except SyntaxError:
        return "Error: Invalid expression syntax."
    except ValueError as e:
        return f"Error: {str(e)}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error: Could not evaluate expression - {str(e)}"
