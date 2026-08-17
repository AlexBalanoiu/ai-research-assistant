"""
Step 3 - Web search tool.
Thin wrapper over DuckDuckGo search (no API key needed for dev/testing).
"""
from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> dict:
    """Searches the web and returns a list of results.

    Use this tool for any question that needs current, factual, or
    real-world information you cannot be certain about from reasoning alone
    (news, prices, facts about specific entities, recent events, etc).
    Do not use it for math - use the calculator tool for that instead.

    Args:
        query: The search query.
        max_results: Max number of results to return (default 5).

    Returns:
        A dict: {"results": [{"title", "url", "snippet"}, ...]} on success,
        or {"error": <message>} on failure.
    """
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        results = [
            {
                "title": h.get("title", ""),
                "url": h.get("href", ""),
                "snippet": h.get("body", ""),
            }
            for h in hits
        ]
        return {"results": results}
    except Exception as exc:
        return {"error": f"Search failed for '{query}': {exc}"}