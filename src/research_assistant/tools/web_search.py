"""
Web search tool with two-tier caching (session + persistent), a small
retry for transient failures, and a shared tool-call budget. Returns a
"count" field so the agent can detect an unhelpful (empty) search and
reformulate the query.
"""
import time
from ddgs import DDGS
from google.adk.tools import ToolContext

from research_assistant.memory.search_cache import (
    get_cached as get_session_cached,
    store as store_session,
)
from research_assistant.memory.persistent_cache import (
    get_cached as get_persistent_cached,
    store as store_persistent,
)
from research_assistant.tools.budget import check_and_consume, MAX_TOOL_CALLS_PER_TURN

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
    real-world information you cannot be certain about from reasoning
    alone (after checking past research first). Do not use it for math.
    Identical queries are served from cache (this session, or an earlier
    one). The response includes "count" - if 0 or results look
    irrelevant, reformulate your query and search again.

    Args:
        query: The search query.
        tool_context: Injected automatically by ADK - session state access.
        max_results: Max number of results to return (default 5).

    Returns:
        A dict: {"results": [...], "count": int, "from_cache": bool,
        "cache_scope": "session"|"persistent"|None} on success, or
        {"error": <message>} on failure.
    """
    if not check_and_consume(tool_context):
        return {
            "error": f"Tool call budget exceeded (max {MAX_TOOL_CALLS_PER_TURN} "
            "tool calls per question). Answer using what you already have instead."
        }

    session_hit = get_session_cached(tool_context, query)
    if session_hit is not None:
        return {"results": session_hit, "count": len(session_hit), "from_cache": True, "cache_scope": "session"}

    persistent_hit = get_persistent_cached(query)
    if persistent_hit is not None:
        store_session(tool_context, query, persistent_hit)
        return {"results": persistent_hit, "count": len(persistent_hit), "from_cache": True, "cache_scope": "persistent"}

    try:
        results = _search(query, max_results)
        store_session(tool_context, query, results)
        store_persistent(query, results)
        return {"results": results, "count": len(results), "from_cache": False, "cache_scope": None}
    except Exception as exc:
        return {"error": f"Search failed for '{query}': {exc}"}