"""
Calculator tool. Enforces the shared tool-call budget (see budget.py).
Safely evaluates arithmetic expressions (no raw eval/exec on strings).
"""
import ast
import operator

from google.adk.tools import ToolContext

from research_assistant.tools.budget import check_and_consume, MAX_TOOL_CALLS_PER_TURN

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](
            _eval_node(node.left), _eval_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str, tool_context: ToolContext) -> dict:
    """Evaluates a math expression and returns the numeric result.

    Supports +, -, *, /, %, ** and parentheses. Use this tool whenever the
    user asks for a calculation. Do not use it for non-math questions.

    Args:
        expression: A math expression as a string, e.g. "12 + 30 * 2".
        tool_context: Injected automatically by ADK - session state access.

    Returns:
        A dict: {"result": <number>} on success, {"error": <message>} on failure.
    """
    if not check_and_consume(tool_context):
        return {
            "error": f"Tool call budget exceeded (max {MAX_TOOL_CALLS_PER_TURN} "
            "tool calls per question). Answer using what you already know instead."
        }
    try:
        tree = ast.parse(expression, mode="eval")
        return {"result": _eval_node(tree.body)}
    except Exception as exc:
        return {"error": f"Could not evaluate '{expression}': {exc}"}