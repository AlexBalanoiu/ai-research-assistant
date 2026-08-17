"""
Web search tool with session-level caching and a small retry for
transient failures in the underlying ddgs library.
"""
import time
from ddgs import DDGS
from google.adk.tools import ToolContext

from research_assistant.memory.search_cache import get_cached, store

_MAX_ATTEMPTS = 2
_RETRY_DELAY_SECONDS = 1.5
_BACKENDS = "duckduckgo,brave,mojeek,wikipedia"


def _search(query: str, max_results: int) -> list[dict]:
    last_error = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            with DDGS() as ddgs:
                hits = list(
                    ddgs.text(query, max_results=max_results, backend=_BACKENDS)
                )
            return [
                {
                    "title": h.get("title", ""),
                    "url": h.get("href", ""),
                    "snippet": h.get("body", ""),
                }
                for h in hits
            ]
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_RETRY_DELAY_SECONDS)
    raise last_error


def web_search(query: str, tool_context: ToolContext, max_results: int = 5) -> dict:
    """Searches the web and returns a list of results.

    Use this tool for any question that needs current, factual, or
    real-world information you cannot be certain about from reasoning alone.
    Do not use it for math - use the calculator tool for that instead.
    Identical queries within this conversation are served from cache
    automatically, so you can call this freely without worrying about
    repeating a previous search.

    Args:
        query: The search query.
        tool_context: Injected automatically by ADK - session state access.
        max_results: Max number of results to return (default 5).

    Returns:
        A dict: {"results": [...], "from_cache": bool} on success,
        or {"error": <message>} on failure.
    """
    cached = get_cached(tool_context, query)
    if cached is not None:
        return {"results": cached, "from_cache": True}

    try:
        results = _search(query, max_results)
        store(tool_context, query, results)
        return {"results": results, "from_cache": False}
    except Exception as exc:
        return {"error": f"Search failed for '{query}': {exc}"}