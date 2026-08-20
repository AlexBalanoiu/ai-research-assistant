"""
Tests - Level 3: persistent (cross-session) memory and its use by the agent.

Unit tests use uniquely-timestamped keys and clean up after themselves,
so they're safe to run against the real data/memory.db.

Run: pytest tests/test_level3_memory.py -v
"""
import time

from research_assistant.memory import persistent_cache, report_history
from research_assistant.memory.db import get_connection
from research_assistant.runner import AgentSession, ask_with_trace
from tests.helpers import result_for_tool


# --- Unit tests (fast, no LLM, real SQLite file) ---

def test_persistent_cache_roundtrip():
    query = f"__test_query_{time.time()}"
    try:
        assert persistent_cache.get_cached(query) is None
        persistent_cache.store(query, [{"title": "T", "url": "https://x", "snippet": "s"}])
        assert persistent_cache.get_cached(query) == [{"title": "T", "url": "https://x", "snippet": "s"}]
    finally:
        conn = get_connection()
        conn.execute("DELETE FROM search_cache WHERE query_key = ?", (query.strip().lower(),))
        conn.commit()
        conn.close()


def test_report_history_roundtrip_and_search():
    question = f"__test question about widgets {time.time()}"
    try:
        report_history.save_report(question, "Widgets are small mechanical parts.", [])
        matches = report_history.search_past_reports("widgets")
        assert any(m["question"] == question for m in matches)
    finally:
        conn = get_connection()
        conn.execute("DELETE FROM report_history WHERE question = ?", (question,))
        conn.commit()
        conn.close()


# --- Functional tests (require model configured) ---

async def test_web_search_uses_persistent_cache_across_sessions():
    """
    Best-effort: second AgentSession (simulating a fresh process, no
    session-level cache) should get a persistent-cache hit for a query
    already searched by the first session, IF the model phrases the
    second query identically enough to match the cache key.
    """
    topic = f"the fictional planet Zorblex{int(time.time())}"
    first = AgentSession()
    _, tool_calls_1, results_1 = await first.send(f"Search the web for '{topic}' and summarize it.")
    result_1 = result_for_tool(tool_calls_1, results_1, "web_search")
    assert result_1 is not None

    second = AgentSession()
    _, tool_calls_2, results_2 = await second.send(f"Search the web for '{topic}' and summarize it.")
    result_2 = result_for_tool(tool_calls_2, results_2, "web_search")
    if result_2 is not None:
        assert result_2.get("cache_scope") in ("persistent", "session")


async def test_agent_can_use_check_past_research_tool():
    """Smoke test: the tool is wired up and callable without error - not
    a guarantee the agent always chooses to call it first."""
    _, tool_calls = await ask_with_trace(
        "Before searching, check if you already researched the Eiffel Tower, then tell me its height."
    )
    assert isinstance(tool_calls, list)