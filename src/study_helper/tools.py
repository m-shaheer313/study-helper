"""A calculator tool, because the model predicts plausible numbers rather than
computing them.

It answered 1234 * 5678 = 7006572 (correct: 7006652) while its own intermediate
steps summed to the right value - that is prediction, not calculation.
"""

import ast
import operator
from collections.abc import Callable
from typing import Any

from agents import RunContextWrapper, function_tool

# An allow-list, not a deny-list: anything not named here is rejected.
_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# "2 ** 10**9" would hang the process computing a number nobody wants.
_MAX_EXPONENT = 1000


class CalculationError(ValueError):
    """Raised for anything the calculator will not or cannot evaluate."""


def _evaluate(node: ast.AST) -> float:
    match node:
        case ast.Expression():
            return _evaluate(node.body)
        case ast.Constant(value=bool()):
            raise CalculationError("booleans are not numbers")
        case ast.Constant(value=int() | float() as value):
            return value
        case ast.UnaryOp(op=op) if type(op) in _UNARY_OPS:
            return _UNARY_OPS[type(op)](_evaluate(node.operand))
        case ast.BinOp(op=op) if type(op) in _BINARY_OPS:
            left, right = _evaluate(node.left), _evaluate(node.right)
            if isinstance(op, ast.Pow) and abs(right) > _MAX_EXPONENT:
                raise CalculationError(f"exponent {right} is too large (max {_MAX_EXPONENT})")
            if isinstance(op, ast.Div | ast.FloorDiv | ast.Mod) and right == 0:
                raise CalculationError("division by zero")
            return _BINARY_OPS[type(op)](left, right)
        case _:
            raise CalculationError(
                f"{type(node).__name__} is not allowed here; use numbers, "
                "+ - * / // % ** and parentheses"
            )


def evaluate_expression(expression: str) -> float:
    """Safely evaluate an arithmetic expression. Never calls eval()."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as err:
        raise CalculationError(f"could not parse {expression!r}: {err.msg}") from err

    result = _evaluate(tree)
    try:
        return float(result)
    except OverflowError as err:
        raise CalculationError("result is too large to represent") from err


def _explain_failure(ctx: RunContextWrapper[Any], error: Exception) -> str:
    return (
        f"calculate could not evaluate that: {error}. Rewrite the expression using "
        "only numbers, + - * / // % ** and parentheses, then call calculate again."
    )


@function_tool(failure_error_function=_explain_failure)
def calculate(expression: str) -> float:
    """Compute the exact value of an arithmetic expression.

    Use this for every arithmetic result. Do not do the arithmetic yourself.

    Args:
        expression: Arithmetic only, such as "1234 * 5678" or "(2 + 3) ** 4 / 7".
            Allowed: numbers, + - * / // % ** and parentheses. No variables,
            no function names, no units.
    """
    return evaluate_expression(expression)
