"""LangChain tool implementations used by the agent execution node."""

import ast
import logging
import operator
from typing import Callable

from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)

_SAFE_OPERATORS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node restricted to basic arithmetic operators."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](
            _eval_ast_node(node.left), _eval_ast_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_eval_ast_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result as a string.

    Only numeric literals and the +, -, *, /, ** operators are supported
    (e.g. "2 + 2 * 3"). The expression is parsed with `ast` and evaluated
    against an operator allow-list, so no arbitrary code execution is
    possible.

    Args:
        expression: The arithmetic expression to evaluate.

    Raises:
        ValueError: If the expression is empty, malformed, or contains
            unsupported syntax.
    """
    if not expression or not expression.strip():
        raise ValueError("Calculator tool requires a non-empty expression.")
    try:
        parsed = ast.parse(expression, mode="eval")
        return str(_eval_ast_node(parsed.body))
    except Exception as exc:
        logger.warning("Calculator tool failed for expression %r: %s", expression, exc)
        raise ValueError(f"Could not evaluate expression {expression!r}: {exc}") from exc


@tool
def echo(text: str) -> str:
    """Repeat back the exact text provided as input.

    Args:
        text: The text to echo back.

    Raises:
        ValueError: If the provided text is empty.
    """
    if not text:
        raise ValueError("Echo tool requires non-empty text input.")
    return text


TOOL_REGISTRY: dict[str, BaseTool] = {t.name: t for t in (calculator, echo)}


def run_tool(tool_name: str, tool_input: str) -> str:
    """Look up and execute a registered tool by name, returning its string output.

    Args:
        tool_name: Name of the tool to run, or "none" to skip execution.
        tool_input: Raw input string passed to the tool.

    Raises:
        ValueError: If `tool_name` does not match a registered tool.
    """
    if tool_name == "none":
        return ""
    tool_obj = TOOL_REGISTRY.get(tool_name)
    if tool_obj is None:
        raise ValueError(f"Unknown tool: {tool_name!r}")
    return tool_obj.invoke(tool_input)
