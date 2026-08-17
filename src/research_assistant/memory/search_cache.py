"""
Step 4 - Search cache backed by ADK session state (tool_context.state).
Prevents repeating an identical web search within the same session.
"""

CACHE_PREFIX = "search_cache:"


def _key(query: str) -> str:
    return CACHE_PREFIX + query.strip().lower()


def get_cached(tool_context, query: str):
    return tool_context.state.get(_key(query))


def store(tool_context, query: str, results: list[dict]) -> None:
    tool_context.state[_key(query)] = results