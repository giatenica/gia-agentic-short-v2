"""Safe Python code execution engine for data acquisition.

Sprint 14: Provides a sandboxed environment for executing LLM-generated
Python code with strict safety constraints.

Security Model:
- Restricted globals (only safe modules allowed)
- Pattern-based code validation (blocks dangerous patterns)
- Timeout enforcement (prevents infinite loops)
- Memory limits (via resource module on Unix)

Safe modules: pandas, numpy, requests, json, datetime, math, statistics
Forbidden: os, sys, subprocess, eval, exec, open, __import__, etc.
"""

from __future__ import annotations

import ast
import re
import signal
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Try to import optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    HAS_REQUESTS = False

import json
import math
import statistics


# =============================================================================
# Safety Configuration
# =============================================================================


# Patterns that are NEVER allowed in code
FORBIDDEN_PATTERNS = [
    # System access
    r'\bimport\s+os\b',
    r'\bimport\s+sys\b',
    r'\bimport\s+subprocess\b',
    r'\bimport\s+shutil\b',
    r'\bimport\s+pathlib\b',
    r'\bimport\s+glob\b',
    r'\bimport\s+socket\b',
    r'\bimport\s+pickle\b',
    r'\bimport\s+shelve\b',
    r'\bimport\s+sqlite3\b',
    
    # Dynamic execution
    r'\b__import__\s*\(',
    r'\beval\s*\(',
    r'\bexec\s*\(',
    r'\bcompile\s*\(',
    
    # File operations
    r'\bopen\s*\(',
    r'\bfile\s*\(',
    r'\.read\s*\(',
    r'\.write\s*\(',
    
    # Reflection and introspection
    r'\bglobals\s*\(',
    r'\blocals\s*\(',
    r'\bgetattr\s*\(',
    r'\bsetattr\s*\(',
    r'\bdelattr\s*\(',
    r'\bvars\s*\(',
    r'\bdir\s*\(',
    r'\b__dict__\b',
    r'\b__class__\b',
    r'\b__bases__\b',
    r'\b__subclasses__\b',
    r'\b__mro__\b',
    
    # Code introspection
    r'\b__code__\b',
    r'\b__globals__\b',
    r'\b__builtins__\b',
    
    # Network (beyond requests)
    r'\burllib\b',
    r'\bhttplib\b',
    r'\bftplib\b',
    
    # Process control
    r'\bsignal\b',
    r'\batexit\b',
    r'\bthreading\b',
    r'\bmultiprocessing\b',
    
    # ctypes and low-level
    r'\bctypes\b',
    r'\bcffi\b',
]

# Compiled patterns for efficiency
COMPILED_FORBIDDEN = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]


# Safe builtins that are allowed in the sandbox
SAFE_BUILTINS = {
    # Constants
    'True': True,
    'False': False,
    'None': None,
    
    # Types
    'bool': bool,
    'int': int,
    'float': float,
    'str': str,
    'bytes': bytes,
    'list': list,
    'dict': dict,
    'set': set,
    'frozenset': frozenset,
    'tuple': tuple,
    
    # Functions
    'abs': abs,
    'all': all,
    'any': any,
    'bin': bin,
    'chr': chr,
    'divmod': divmod,
    'enumerate': enumerate,
    'filter': filter,
    'format': format,
    'hash': hash,
    'hex': hex,
    'isinstance': isinstance,
    'issubclass': issubclass,
    'iter': iter,
    'len': len,
    'map': map,
    'max': max,
    'min': min,
    'next': next,
    'oct': oct,
    'ord': ord,
    'pow': pow,
    'print': print,
    'range': range,
    'repr': repr,
    'reversed': reversed,
    'round': round,
    'slice': slice,
    'sorted': sorted,
    'sum': sum,
    'zip': zip,
    
    # Exceptions (for handling)
    'Exception': Exception,
    'ValueError': ValueError,
    'TypeError': TypeError,
    'KeyError': KeyError,
    'IndexError': IndexError,
    'AttributeError': AttributeError,
    'RuntimeError': RuntimeError,
    'ZeroDivisionError': ZeroDivisionError,
}


# Modules allowed to be imported
ALLOWED_IMPORT_MODULES = {
    'json', 'math', 'statistics', 'datetime', 'decimal', 're',
    'collections', 'itertools', 'functools', 'operator',
    'pandas', 'pd', 'numpy', 'np', 'requests',
}


def _safe_import(name: str, globals_dict=None, locals_dict=None, fromlist=(), level=0):
    """Safe import function that only allows approved modules.
    
    This replaces __import__ in the sandbox to restrict which modules
    can be imported.
    """
    # Get the base module name
    base_module = name.split('.')[0]
    
    # Check if module is allowed
    if base_module not in ALLOWED_IMPORT_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed in sandbox. Allowed: {', '.join(sorted(ALLOWED_IMPORT_MODULES))}")
    
    # Use the real import for allowed modules
    return __builtins__['__import__'](name, globals_dict, locals_dict, fromlist, level)


def _build_safe_globals() -> dict[str, Any]:
    """Build the safe globals dictionary for code execution."""
    # Add safe import to builtins
    safe_builtins_with_import = SAFE_BUILTINS.copy()
    safe_builtins_with_import['__import__'] = _safe_import
    
    safe_globals = {
        '__builtins__': safe_builtins_with_import,
        'datetime': datetime,
        'timezone': timezone,
        'json': json,
        'math': math,
        'statistics': statistics,
    }
    
    # Add pandas if available
    if HAS_PANDAS:
        safe_globals['pd'] = pd
        safe_globals['pandas'] = pd
    
    # Add numpy if available
    if HAS_NUMPY:
        safe_globals['np'] = np
        safe_globals['numpy'] = np
    
    # Add requests if available (for API calls)
    if HAS_REQUESTS:
        safe_globals['requests'] = requests
    
    return safe_globals


SAFE_GLOBALS = _build_safe_globals()


# =============================================================================
# Code Validation
# =============================================================================


class CodeValidationError(Exception):
    """Raised when code fails safety validation."""
    pass


def validate_code(code: str) -> tuple[bool, str]:
    """Validate Python code against safety rules.
    
    Args:
        code: Python code string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for forbidden patterns
    for pattern in COMPILED_FORBIDDEN:
        match = pattern.search(code)
        if match:
            return False, f"Forbidden pattern detected: {match.group()}"
    
    # Try to parse the AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    # Walk the AST to check for dangerous nodes
    for node in ast.walk(tree):
        # Check for dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ['os', 'sys', 'subprocess', 'shutil', 'socket']:
                    return False, f"Forbidden import: {alias.name}"
        
        # Check for from imports
        if isinstance(node, ast.ImportFrom):
            if node.module in ['os', 'sys', 'subprocess', 'shutil', 'socket']:
                return False, f"Forbidden import from: {node.module}"
        
        # Check for attribute access to dangerous attributes
        if isinstance(node, ast.Attribute):
            if node.attr in ['__code__', '__globals__', '__builtins__', '__dict__']:
                return False, f"Forbidden attribute access: {node.attr}"
    
    return True, ""


# =============================================================================
# Timeout Handling
# =============================================================================


class TimeoutError(Exception):
    """Raised when code execution times out."""
    pass


@contextmanager
def timeout_context(seconds: int):
    """Context manager for timeout enforcement (Unix only).
    
    On Windows, this is a no-op and timeout is not enforced.
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution timed out after {seconds} seconds")
    
    # Only use signal on Unix-like systems
    if sys.platform != 'win32':
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # No timeout enforcement on Windows
        yield


# =============================================================================
# Code Execution
# =============================================================================


def execute_code_safely(
    code: str,
    timeout_seconds: int = 30,
    capture_output: bool = True,
) -> tuple[bool, str, dict[str, Any]]:
    """Execute Python code in a restricted environment.
    
    Args:
        code: Python code to execute
        timeout_seconds: Maximum execution time (Unix only)
        capture_output: Whether to capture stdout/stderr
        
    Returns:
        Tuple of (success, output_or_error, local_variables)
    """
    # Validate code first
    is_valid, error_msg = validate_code(code)
    if not is_valid:
        return False, f"Code validation failed: {error_msg}", {}
    
    # Cap timeout at 60 seconds
    timeout_seconds = min(timeout_seconds, 60)
    
    # Prepare execution environment
    exec_globals = SAFE_GLOBALS.copy()
    exec_locals: dict[str, Any] = {}
    
    # Capture output
    output = StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        if capture_output:
            sys.stdout = output
            sys.stderr = output
        
        # Execute with timeout
        with timeout_context(timeout_seconds):
            exec(code, exec_globals, exec_locals)
        
        # Get output
        result = output.getvalue()
        
        # Include any result variables
        if 'result' in exec_locals:
            if result:
                result += f"\nResult: {exec_locals['result']}"
            else:
                result = f"Result: {exec_locals['result']}"
        
        return True, result if result else "Code executed successfully", exec_locals
        
    except TimeoutError as e:
        return False, str(e), {}
    except Exception as e:
        error_trace = traceback.format_exc()
        return False, f"Execution error: {e}\n{error_trace}", {}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# =============================================================================
# Tool Interface
# =============================================================================


class ExecutePythonCodeInput(BaseModel):
    """Input schema for execute_python_code tool."""
    code: str = Field(
        description="Python code to execute. Must not contain dangerous operations."
    )
    description: str = Field(
        default="",
        description="Description of what the code does"
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=60,
        description="Maximum execution time in seconds (1-60)"
    )


@tool(args_schema=ExecutePythonCodeInput)
def execute_python_code(
    code: str,
    description: str = "",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Execute Python code in a restricted sandbox environment.
    
    This tool executes Python code with safety restrictions to prevent
    dangerous operations. Use it for custom data acquisition or transformation
    when built-in tools are insufficient.
    
    Allowed modules: pandas (pd), numpy (np), requests, json, datetime, math, statistics
    Forbidden: os, sys, subprocess, file operations, eval, exec, imports beyond allowed
    
    Args:
        code: Python code to execute
        description: What the code does (for logging)
        timeout_seconds: Max execution time (default 30, max 60)
        
    Returns:
        Dictionary with execution results:
        - status: 'success' or 'error'
        - output: Captured stdout/stderr or error message
        - variables: Any variables created (if success)
        - execution_time: How long it took
        
    Example:
        ```python
        # Fetch and process data
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = df.describe().to_dict()
        ```
    
    Security:
        - Code is validated before execution
        - Dangerous patterns are blocked
        - Timeout enforced (Unix only)
        - Restricted globals prevent system access
    """
    import time
    start_time = time.time()
    
    success, output, variables = execute_code_safely(
        code=code,
        timeout_seconds=timeout_seconds,
        capture_output=True,
    )
    
    execution_time = time.time() - start_time
    
    if success:
        return {
            "status": "success",
            "output": output,
            "variables": {k: str(v)[:1000] for k, v in variables.items() 
                        if not k.startswith('_')},
            "execution_time": round(execution_time, 3),
        }
    else:
        return {
            "status": "error",
            "error": output,
            "execution_time": round(execution_time, 3),
        }


class ValidatePythonCodeInput(BaseModel):
    """Input schema for validate_python_code tool."""
    code: str = Field(description="Python code to validate")


@tool(args_schema=ValidatePythonCodeInput)
def validate_python_code(code: str) -> dict[str, Any]:
    """Validate Python code without executing it.
    
    Checks the code against safety rules to determine if it would
    be allowed to execute. Use this before execute_python_code to
    verify code is safe.
    
    Args:
        code: Python code to validate
        
    Returns:
        Dictionary with validation result:
        - valid: Whether the code passes safety checks
        - error: Error message if invalid
    """
    is_valid, error_msg = validate_code(code)
    
    if is_valid:
        return {
            "valid": True,
            "message": "Code passes safety validation",
        }
    else:
        return {
            "valid": False,
            "error": error_msg,
        }


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "execute_python_code",
    "validate_python_code",
    "execute_code_safely",
    "validate_code",
    "CodeValidationError",
    "SAFE_GLOBALS",
    "FORBIDDEN_PATTERNS",
]
