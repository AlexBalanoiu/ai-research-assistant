"""
Functional tests - Step 4: session memory / search caching.

Note: the multi-turn test depends on the LLM rephrasing the second search
with a near-identical query string - it's best-effort/can be flaky with
smaller local models. The unit tests below are the reliable ones.

Run: pytest tests/test_step4_memory.py -v
"""
from research_assistant.runner import AgentSession
from research_assistant.memory.search_cache import get_cached, store
from tests.fakes import FakeToolContext


# --- Unit tests (fast, no LLM/network) ---

def test_cache_roundtrip():
    ctx = FakeToolContext()
    assert get_cached(ctx, "python") is None
    store(ctx, "python", [{"title": "Python", "url": "x", "snippet": "y"}])
    assert get_cached(ctx, "python") == [{"title": "Python", "url": "x", "snippet": "y"}]


def test_cache_is_case_and_whitespace_insensitive():
    ctx = FakeToolContext()
    store(ctx, "  Python Programming  ", [{"title": "P"}])
    assert get_cached(ctx, "python programming") is not None


# --- Functional test (requires model configured) ---

async def test_agent_does_not_repeat_identical_search_in_same_session():
    """
    'Does not repeat a search' has two acceptable forms:
    - the model skips the tool call entirely, answering from conversation
      context (the smarter outcome), or
    - it calls the tool again but the result comes from cache.
    Either way, no second real network hit happens.
    """
    session = AgentSession()

    _, tool_calls_1, results_1 = await session.send(
        "Search the web for 'Python programming language' and summarize the first result."
    )
    assert "web_search" in tool_calls_1
    assert results_1[0].get("from_cache") is False

    _, tool_calls_2, results_2 = await session.send(
        "Search again for 'Python programming language' and tell me the same thing."
    )
    if "web_search" in tool_calls_2:
        assert results_2[0].get("from_cache") is True
    # else: model answered from context without re-calling the tool - also valid.