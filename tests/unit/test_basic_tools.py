"""Tests for basic utility tools."""

import pytest
from unittest.mock import patch

from src.tools.basic import (
    get_current_time,
    calculate,
    _safe_eval,
)


class TestGetCurrentTime:
    """Tests for get_current_time tool."""
    
    def test_returns_string(self):
        """Test that get_current_time returns a string."""
        result = get_current_time.invoke({})
        assert isinstance(result, str)
    
    def test_returns_formatted_datetime(self):
        """Test that result is in expected format."""
        result = get_current_time.invoke({})
        # Format: YYYY-MM-DD HH:MM:SS
        parts = result.split(" ")
        assert len(parts) == 2
        
        # Check date part
        date_parts = parts[0].split("-")
        assert len(date_parts) == 3
        assert len(date_parts[0]) == 4  # Year
        assert len(date_parts[1]) == 2  # Month
        assert len(date_parts[2]) == 2  # Day
        
        # Check time part
        time_parts = parts[1].split(":")
        assert len(time_parts) == 3
    
    @patch("src.tools.basic.datetime")
    def test_uses_current_datetime(self, mock_datetime):
        """Test that tool uses the current datetime."""
        from datetime import datetime
        mock_now = datetime(2026, 1, 15, 10, 30, 45)
        mock_datetime.now.return_value = mock_now
        
        result = get_current_time.invoke({})
        assert result == "2026-01-15 10:30:45"


class TestCalculate:
    """Tests for calculate tool."""
    
    def test_simple_addition(self):
        """Test simple addition."""
        result = calculate.invoke({"expression": "2 + 3"})
        assert result == "5"
    
    def test_simple_subtraction(self):
        """Test simple subtraction."""
        result = calculate.invoke({"expression": "10 - 4"})
        assert result == "6"
    
    def test_multiplication(self):
        """Test multiplication."""
        result = calculate.invoke({"expression": "6 * 7"})
        assert result == "42"
    
    def test_division(self):
        """Test division."""
        result = calculate.invoke({"expression": "15 / 3"})
        assert result == "5.0"
    
    def test_floor_division(self):
        """Test floor division."""
        result = calculate.invoke({"expression": "17 // 5"})
        assert result == "3"
    
    def test_modulo(self):
        """Test modulo operation."""
        result = calculate.invoke({"expression": "17 % 5"})
        assert result == "2"
    
    def test_exponentiation(self):
        """Test exponentiation."""
        result = calculate.invoke({"expression": "2 ** 8"})
        assert result == "256"
    
    def test_unary_minus(self):
        """Test unary minus."""
        result = calculate.invoke({"expression": "-5 + 10"})
        assert result == "5"
    
    def test_complex_expression(self):
        """Test complex expression with multiple operators."""
        result = calculate.invoke({"expression": "2 + 3 * 4"})
        assert result == "14"  # Follows order of operations
    
    def test_parentheses(self):
        """Test expression with parentheses."""
        result = calculate.invoke({"expression": "(2 + 3) * 4"})
        assert result == "20"
    
    def test_float_numbers(self):
        """Test expression with floating point numbers."""
        result = calculate.invoke({"expression": "3.14 * 2"})
        assert float(result) == pytest.approx(6.28)
    
    def test_division_by_zero(self):
        """Test division by zero returns error."""
        result = calculate.invoke({"expression": "10 / 0"})
        assert "Error" in result
        assert "Division by zero" in result
    
    def test_invalid_syntax(self):
        """Test invalid syntax returns error."""
        result = calculate.invoke({"expression": "2 +"})
        assert "Error" in result
    
    def test_unsupported_operation(self):
        """Test unsupported operations return error."""
        # Bitwise operations should not be supported
        result = calculate.invoke({"expression": "2 & 3"})
        assert "Error" in result
    
    def test_function_calls_not_allowed(self):
        """Test that function calls are not allowed."""
        result = calculate.invoke({"expression": "abs(-5)"})
        assert "Error" in result
    
    def test_variable_access_not_allowed(self):
        """Test that variable access is not allowed."""
        result = calculate.invoke({"expression": "x + 5"})
        assert "Error" in result


class TestSafeEval:
    """Tests for _safe_eval internal function."""
    
    def test_constant_int(self):
        """Test evaluating an integer constant."""
        import ast
        tree = ast.parse("42", mode='eval')
        result = _safe_eval(tree)
        assert result == 42
    
    def test_constant_float(self):
        """Test evaluating a float constant."""
        import ast
        tree = ast.parse("3.14", mode='eval')
        result = _safe_eval(tree)
        assert result == 3.14
    
    def test_unsupported_constant_raises(self):
        """Test that unsupported constant types raise ValueError."""
        import ast
        # Create a node with a string constant
        tree = ast.parse("'hello'", mode='eval')
        with pytest.raises(ValueError, match="Unsupported constant type"):
            _safe_eval(tree)
    
    def test_unsupported_expression_raises(self):
        """Test that unsupported expression types raise ValueError."""
        import ast
        # Create a list comprehension node (not supported)
        tree = ast.parse("[x for x in range(5)]", mode='eval')
        with pytest.raises(ValueError, match="Unsupported expression type"):
            _safe_eval(tree)
