"""
Step 2 - Calculator tool.
Safely evaluates arithmetic expressions (no raw eval/exec on strings).
"""
import ast
import operator

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


def calculator(expression: str) -> dict:
    """Evaluates a math expression and returns the numeric result.

    Supports +, -, *, /, %, ** and parentheses. Use this tool whenever the
    user asks for a calculation. Do not use it for non-math questions.

    Args:
        expression: A math expression as a string, e.g. "12 + 30 * 2".

    Returns:
        A dict: {"result": <number>} on success, {"error": <message>} on failure.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return {"result": _eval_node(tree.body)}
    except Exception as exc:
        return {"error": f"Could not evaluate '{expression}': {exc}"}