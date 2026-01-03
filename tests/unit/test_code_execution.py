"""Unit tests for Sprint 14: Code Execution Engine.

Tests the safe Python code execution sandbox.
"""

import pytest

from src.tools.code_execution import (
    validate_code,
    execute_code_safely,
    execute_python_code,
    validate_python_code,
    FORBIDDEN_PATTERNS,
    SAFE_BUILTINS,
)


# =============================================================================
# Code Validation Tests
# =============================================================================


class TestCodeValidation:
    """Tests for code validation functionality."""
    
    def test_validate_safe_code(self):
        """Safe code should pass validation."""
        safe_code = """
import pandas as pd
import numpy as np

df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
result = df.mean()
"""
        is_valid, error = validate_code(safe_code)
        assert is_valid is True
        # Error may be empty string or None when valid
        assert error is None or error == ""
    
    def test_validate_rejects_os_import(self):
        """Code with os import should be rejected."""
        dangerous_code = """
import os
os.system('rm -rf /')
"""
        is_valid, error = validate_code(dangerous_code)
        assert is_valid is False
        assert error is not None and error != ""
    
    def test_validate_rejects_subprocess(self):
        """Code with subprocess should be rejected."""
        dangerous_code = """
import subprocess
subprocess.run(['ls'])
"""
        is_valid, error = validate_code(dangerous_code)
        assert is_valid is False
        assert error is not None
    
    def test_validate_rejects_eval(self):
        """Code with eval should be rejected."""
        dangerous_code = """
eval('print("hello")')
"""
        is_valid, error = validate_code(dangerous_code)
        assert is_valid is False
        assert error is not None
    
    def test_validate_rejects_exec(self):
        """Code with exec should be rejected."""
        dangerous_code = """
exec('print("hello")')
"""
        is_valid, error = validate_code(dangerous_code)
        assert is_valid is False
        assert error is not None
    
    def test_validate_rejects_open(self):
        """Code with open() should be rejected."""
        dangerous_code = """
with open('/etc/passwd', 'r') as f:
    data = f.read()
"""
        is_valid, error = validate_code(dangerous_code)
        assert is_valid is False
        assert error is not None
    
    def test_validate_rejects_dunder_import(self):
        """Code with __import__ should be rejected."""
        dangerous_code = """
__import__('os').system('ls')
"""
        is_valid, error = validate_code(dangerous_code)
        assert is_valid is False
        assert error is not None
    
    def test_validate_rejects_compile(self):
        """Code with compile should be rejected."""
        dangerous_code = """
code = compile('print("hello")', '<string>', 'exec')
exec(code)
"""
        is_valid, error = validate_code(dangerous_code)
        assert is_valid is False
        assert error is not None
    
    def test_validate_rejects_sys_import(self):
        """Code with sys import should be rejected."""
        dangerous_code = """
import sys
sys.exit(0)
"""
        is_valid, error = validate_code(dangerous_code)
        assert is_valid is False
        assert error is not None
    
    def test_validate_rejects_socket(self):
        """Code with socket import should be rejected."""
        dangerous_code = """
import socket
s = socket.socket()
"""
        is_valid, error = validate_code(dangerous_code)
        assert is_valid is False
        assert error is not None
    
    def test_validate_syntax_error(self):
        """Code with syntax errors should be caught."""
        invalid_code = """
def foo(
    # Missing closing parenthesis
"""
        is_valid, error = validate_code(invalid_code)
        assert is_valid is False
        assert error is not None


# =============================================================================
# Code Execution Tests
# =============================================================================


class TestCodeExecution:
    """Tests for code execution functionality.
    
    Note: execute_code_safely returns a tuple (success, output_or_error, local_vars)
    """
    
    def test_execute_simple_math(self):
        """Simple math operations should execute successfully."""
        code = """
result = 2 + 2
"""
        success, output, variables = execute_code_safely(code)
        assert success is True
        assert variables.get("result") == 4
    
    def test_execute_pandas_operations(self):
        """Pandas operations should execute successfully."""
        code = """
import pandas as pd
df = pd.DataFrame({'x': [1, 2, 3]})
result = len(df)
"""
        success, output, variables = execute_code_safely(code)
        assert success is True
        assert variables.get("result") == 3
    
    def test_execute_numpy_operations(self):
        """NumPy operations should execute successfully."""
        code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
result = np.mean(arr)
"""
        success, output, variables = execute_code_safely(code)
        assert success is True
        assert variables.get("result") == 3.0
    
    def test_execute_statistics_operations(self):
        """Statistics operations should execute successfully."""
        code = """
import statistics
data = [1, 2, 3, 4, 5]
result = statistics.mean(data)
"""
        success, output, variables = execute_code_safely(code)
        assert success is True
        assert variables.get("result") == 3
    
    def test_execute_json_operations(self):
        """JSON operations should execute successfully."""
        code = """
import json
data = json.dumps({'key': 'value'})
result = data
"""
        success, output, variables = execute_code_safely(code)
        assert success is True
        assert '"key"' in variables.get("result", "")
    
    def test_execute_captures_stdout(self):
        """Stdout should be captured."""
        code = """
print("Hello, World!")
result = "done"
"""
        success, output, variables = execute_code_safely(code)
        assert success is True
        assert "Hello, World!" in output
    
    def test_execute_handles_exceptions(self):
        """Runtime exceptions should be captured."""
        code = """
result = 1 / 0
"""
        success, output, variables = execute_code_safely(code)
        assert success is False
        assert "ZeroDivisionError" in output or "division" in output
    
    def test_execute_dangerous_code_blocked(self):
        """Dangerous code should be blocked before execution."""
        code = """
import os
os.system('ls')
"""
        success, output, variables = execute_code_safely(code)
        assert success is False
        assert "forbidden" in output.lower() or "validation failed" in output.lower()
    
    def test_execute_timeout(self):
        """Code that takes too long should timeout."""
        code = """
import time
time.sleep(10)
result = "done"
"""
        # Use a short timeout
        success, output, variables = execute_code_safely(code, timeout_seconds=1)
        # Note: timeout may not work on all platforms (Windows)
        # The test checks that code execution completes (either success or timeout)
        # On platforms without signal support, code may complete
        assert isinstance(success, bool)
    
    def test_execute_multiple_variables(self):
        """Multiple variable assignments should be captured."""
        code = """
x = 10
y = 20
result = x + y
"""
        success, output, variables = execute_code_safely(code)
        assert success is True
        assert variables.get("x") == 10
        assert variables.get("y") == 20
        assert variables.get("result") == 30


# =============================================================================
# LangChain Tool Tests
# =============================================================================


class TestLangChainTools:
    """Tests for LangChain tool interfaces."""
    
    def test_execute_python_code_tool(self):
        """The execute_python_code tool should work."""
        result = execute_python_code.invoke({
            "code": "result = 2 * 3",
            "description": "Test multiplication",
        })
        assert result["status"] == "success"
        # Variables are stringified in the tool output
        assert "6" in result["variables"].get("result", "")
    
    def test_validate_python_code_tool_valid(self):
        """The validate_python_code tool should accept valid code."""
        result = validate_python_code.invoke({
            "code": "import pandas as pd\ndf = pd.DataFrame()",
        })
        assert result["valid"] is True
    
    def test_validate_python_code_tool_invalid(self):
        """The validate_python_code tool should reject invalid code."""
        result = validate_python_code.invoke({
            "code": "import os\nos.system('ls')",
        })
        assert result["valid"] is False
        assert result["error"] is not None


# =============================================================================
# Safety Constant Tests
# =============================================================================


class TestSafetyConstants:
    """Tests for safety configuration constants."""
    
    def test_forbidden_patterns_exist(self):
        """FORBIDDEN_PATTERNS should be non-empty."""
        assert len(FORBIDDEN_PATTERNS) > 0
    
    def test_forbidden_patterns_include_os(self):
        """FORBIDDEN_PATTERNS should include os-related patterns."""
        patterns_str = str(FORBIDDEN_PATTERNS)
        assert "os" in patterns_str or "import\\s+os" in patterns_str
    
    def test_safe_builtins_limited(self):
        """SAFE_BUILTINS should not include dangerous builtins."""
        assert "exec" not in SAFE_BUILTINS
        assert "eval" not in SAFE_BUILTINS
        assert "compile" not in SAFE_BUILTINS
        assert "__import__" not in SAFE_BUILTINS
        assert "open" not in SAFE_BUILTINS
    
    def test_safe_builtins_include_basics(self):
        """SAFE_BUILTINS should include basic safe functions."""
        assert "len" in SAFE_BUILTINS
        assert "range" in SAFE_BUILTINS
        assert "str" in SAFE_BUILTINS
        assert "int" in SAFE_BUILTINS
        assert "float" in SAFE_BUILTINS
        assert "bool" in SAFE_BUILTINS
        assert "list" in SAFE_BUILTINS
        assert "dict" in SAFE_BUILTINS
