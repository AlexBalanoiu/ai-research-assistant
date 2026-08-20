"""
Level 2 - Tool call budget. Prevents a runaway tool-calling loop within a
single question and gives a hard cap useful for cost estimation.

Uses the "temp:" state prefix, which ADK discards automatically at the
end of each invocation (question) - so no manual reset is needed between
questions, unlike a plain session-state counter would require.
"""

MAX_TOOL_CALLS_PER_TURN = 6
_BUDGET_KEY = "temp:tool_call_count"


def check_and_consume(tool_context) -> bool:
    """Returns True and increments the counter if under budget, False if not."""
    count = tool_context.state.get(_BUDGET_KEY, 0)
    if count >= MAX_TOOL_CALLS_PER_TURN:
        return False
    tool_context.state[_BUDGET_KEY] = count + 1
    return True