"""Shared test helpers."""


def result_for_tool(tool_calls: list[str], tool_results: list[dict], tool_name: str) -> dict | None:
    """Returns the first tool_results entry that corresponds to a call of
    tool_name, matching by position across the two parallel lists
    (tool_calls[i] was the call that produced tool_results[i])."""
    for name, result in zip(tool_calls, tool_results):
        if name == tool_name:
            return result
    return None